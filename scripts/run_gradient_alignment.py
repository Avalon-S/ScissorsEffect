#!/usr/bin/env python
"""
Gradient alignment analysis.

Direct mechanism evidence for the Scissors Effect: tests whether DI's effect on
source-target gradient alignment is direction-reversed between standard and robust
source models.

Per-image quantities computed:
  g_src_noDI = grad_x L_src(x, y)
  g_src_DI   = (1/T) sum_t grad_x L_src(T_t(x), y)   (EOT over T DI transforms)
  g_tgt      = grad_x L_tgt(x, y)

Reported:
  cos(g_src_noDI, g_tgt)         alignment with target, no DI
  cos(g_src_DI,   g_tgt)         alignment with target, with DI (EOT)
  Delta_align = cos_DI - cos_noDI    DI's net effect on alignment
  cos(g_src_noDI, g_src_DI)      DI's directional bias on the source itself

Hypothesis (Scissors mechanism):
  Standard source: Delta_align > 0   (DI denoises -> better target alignment)
  Robust source:   Delta_align < 0   (DI introduces bias -> worse target alignment)

for std but bias for robust").
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.sap import di_transform
from utils import seed_everything, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=200)
    p.add_argument("--n_eot", type=int, default=10,
                   help="EOT averaging steps for DI gradient")
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str,
                   default="results/gradient_alignment")
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help="Override default source list. Comma-format: 'name:label'")
    p.add_argument("--targets", type=str, nargs="+", default=None,
                   help="Override default target list")
    p.add_argument("--filter_correct", action="store_true", default=True,
                   help="Only count samples where both source and target predict "
                        "the clean image correctly (default: True)")
    p.add_argument("--no_filter_correct", dest="filter_correct",
                   action="store_false")
    return p.parse_args()


def compute_gradient(model, x, y, di=False, n_eot=10, resize_rate=0.9):
    """
    Gradient of cross-entropy loss w.r.t. input.

    di=False: single forward/backward.
    di=True:  EOT-averaged gradient over n_eot DI transformations.

    Returns: [B, C, H, W] tensor.
    """
    if not di:
        x_in = x.clone().detach().requires_grad_(True)
        logits = model(x_in)
        loss = F.cross_entropy(logits, y, reduction="sum")
        model.zero_grad()
        loss.backward()
        return x_in.grad.detach().clone()

    # IMPORTANT: gradient must be wrt the ORIGINAL input, with grad flowing
    # through the (differentiable) DI transformation. This matches the DI-FGSM
    # update rule used by torchattacks/TransferAttack: g = grad_x L(f(T(x))).
    grad_sum = torch.zeros_like(x)
    for _ in range(n_eot):
        x_in = x.clone().detach().requires_grad_(True)
        x_t = di_transform(x_in, resize_rate=resize_rate)
        logits = model(x_t)
        loss = F.cross_entropy(logits, y, reduction="sum")
        model.zero_grad()
        if x_in.grad is not None:
            x_in.grad = None
        loss.backward()
        grad_sum += x_in.grad.detach()
    return grad_sum / n_eot


def cosine_per_sample(g1, g2):
    """Per-sample cosine similarity. Returns [B]."""
    B = g1.size(0)
    return F.cosine_similarity(g1.view(B, -1), g2.view(B, -1), dim=1)


def norm_per_sample(g):
    """Per-sample L2 norm. Returns [B]."""
    B = g.size(0)
    return g.view(B, -1).norm(dim=1)


def signed_projection_per_sample(g_src, g_tgt):
    """
    Signed scalar projection of g_src onto unit g_tgt direction.
    Captures both direction and magnitude:
        proj = (g_src . g_tgt) / |g_tgt|
    Larger positive value = more attack signal in target's gradient direction.
    Returns [B].
    """
    B = g_src.size(0)
    g_tgt_flat = g_tgt.view(B, -1)
    g_tgt_norm = g_tgt_flat.norm(dim=1).clamp(min=1e-12)
    g_src_flat = g_src.view(B, -1)
    return (g_src_flat * g_tgt_flat).sum(dim=1) / g_tgt_norm


def sign_cosine_per_sample(g1, g2):
    """
    Cosine similarity of SIGN-vectors. Equivalent to:
        (1/D) * sum_i sign(g1_i) * sign(g2_i)  =  fraction_agree*2 - 1
    This is the attack-relevant metric: DI-FGSM/MI-FGSM update uses
    sign(g_src), so the effective progress on target loss per attack step
    is approximately <g_tgt, sign(g_src)>, whose normalized directional
    component is captured by cos(sign(g_src), sign(g_tgt)).
    Returns [B] in [-1, 1]; +1 means all sign bits agree.
    """
    B = g1.size(0)
    s1 = torch.sign(g1.view(B, -1))
    s2 = torch.sign(g2.view(B, -1))
    return F.cosine_similarity(s1, s2, dim=1)


def predict(model, x, batch_size=8):
    """Get top-1 predictions, batched."""
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i + batch_size]
            preds.append(model(xb).argmax(dim=1))
    return torch.cat(preds)


DEFAULT_SOURCES = [
    ("ResNet50", "Standard (ResNet50, eps=0)"),
    ("Engstrom2019Robustness_ImageNet", "Robust (Engstrom, eps=4/255)"),
    ("Salman2020Do_R50", "Robust (Salman, eps=4/255)"),
    ("Mo2022When_ViT-B", "Robust (Mo2022, ViT-B + AT)"),  # optional, will skip if missing
]

DEFAULT_TARGETS = [
    "ViT_B_16_ImageNet",
    "Swin_B_ImageNet",
    "ConvNeXt_B_ImageNet",
]


def parse_source_list(spec_list):
    """Parse 'name:label' format. If no colon, label=name."""
    out = []
    for s in spec_list:
        if ":" in s:
            n, lbl = s.split(":", 1)
        else:
            n, lbl = s, s
        out.append((n.strip(), lbl.strip()))
    return out


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 72)
    print("Exp A: Gradient Alignment Analysis (Source vs Target, +/-DI)")
    print("=" * 72)
    print(f"  N={args.n_examples}, EOT={args.n_eot}, batch={args.batch_size}")
    print(f"  resize_rate={args.resize_rate}, seed={args.seed}, device={device}")
    print()

    sources = parse_source_list(args.sources) if args.sources else DEFAULT_SOURCES
    targets = args.targets if args.targets else DEFAULT_TARGETS

    print(f"  Sources ({len(sources)}):")
    for n, lbl in sources:
        print(f"    - {n}  [{lbl}]")
    print(f"  Targets ({len(targets)}):")
    for t in targets:
        print(f"    - {t}")
    print()

    # Load data once
    seed_everything(args.seed)
    print("[Data] Loading ImageNet validation set...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    print(f"  Loaded {x_test.size(0)} images, shape={tuple(x_test.shape[1:])}")

    results = []

    for src_name, src_label in sources:
        print(f"\n[Source] {src_name} -- {src_label}")
        try:
            src_model = get_model(src_name, dataset="imagenet", device=device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED to load source {src_name}: {e}")
            continue

        # ---- Compute source clean-correct mask (for optional filtering) ----
        if args.filter_correct:
            print("  [Pre-compute] Source clean predictions (for clean-correct filter)...")
            src_pred = predict(src_model, x_test.to(device), batch_size=args.batch_size)
            src_correct = (src_pred.cpu() == y_test).numpy()
            print(f"  [Source clean-correct] {src_correct.sum()}/{len(src_correct)} "
                  f"({100*src_correct.mean():.1f}%)")
        else:
            src_correct = np.ones(args.n_examples, dtype=bool)

        # ---- Pre-compute source gradients ONCE per source (reuse across targets) ----
        # This ensures (1) consistent magnitude/direction estimates across targets
        # for the same source, and (2) ~50% wall-clock savings since DI EOT is
        # the dominant cost.
        print("  [Pre-compute] Source gradients (no-DI + DI EOT) for all images...")
        seed_everything(args.seed)  # reset so DI sampling is deterministic per source
        src_g_noDI_chunks, src_g_DI_chunks = [], []
        t_pre = time.time()
        for i in range(0, args.n_examples, args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            g_src_noDI = compute_gradient(src_model, xb, yb, di=False)
            g_src_DI = compute_gradient(src_model, xb, yb, di=True,
                                        n_eot=args.n_eot,
                                        resize_rate=args.resize_rate)
            src_g_noDI_chunks.append(g_src_noDI.cpu())
            src_g_DI_chunks.append(g_src_DI.cpu())
        print(f"  [Pre-compute] Done in {time.time()-t_pre:.1f}s")

        # Per-source magnitude check (cleaner now since computed once).
        # Compute on filtered subset to be consistent with main metrics.
        # Use median + mean-of-norms (robust); per-image ratio mean is degenerate.
        all_norm_noDI = torch.cat([g.view(g.size(0), -1).norm(dim=1)
                                   for g in src_g_noDI_chunks]).numpy()
        all_norm_DI = torch.cat([g.view(g.size(0), -1).norm(dim=1)
                                 for g in src_g_DI_chunks]).numpy()
        keep_idx = src_correct  # source-correct subset
        if keep_idx.sum() > 0:
            n_no = all_norm_noDI[keep_idx]
            n_di = all_norm_DI[keep_idx]
            src_ratio_mean_of_norms = n_di.mean() / max(n_no.mean(), 1e-12)
            src_ratio_median = np.median(n_di / np.clip(n_no, 1e-12, None))
        else:
            src_ratio_mean_of_norms = float('nan')
            src_ratio_median = float('nan')
        print(f"  [Source-only stat] |g_DI|/|g_noDI|: "
              f"median={src_ratio_median:.3f}, "
              f"mean(|g_DI|)/mean(|g_noDI|)={src_ratio_mean_of_norms:.3f}  "
              f"(robust statistics; per-image-ratio mean is degenerate, see code comment)")

        for tgt_name in targets:
            try:
                tgt_model = get_model(tgt_name, dataset="imagenet", device=device)
                tgt_model.eval()
            except Exception as e:
                print(f"  FAILED to load target {tgt_name}: {e}")
                continue

            # Target clean-correct mask
            if args.filter_correct:
                tgt_pred = predict(tgt_model, x_test.to(device),
                                   batch_size=args.batch_size)
                tgt_correct = (tgt_pred.cpu() == y_test).numpy()
                combined_mask = src_correct & tgt_correct
            else:
                tgt_correct = np.ones(args.n_examples, dtype=bool)
                combined_mask = src_correct.copy()

            n_kept = int(combined_mask.sum())
            if n_kept < 5:
                print(f"  [SKIP] {tgt_name}: only {n_kept} clean-correct samples")
                del tgt_model
                torch.cuda.empty_cache()
                continue

            cos_no_di_all, cos_di_all, cos_drift_all = [], [], []
            norm_no_di_all, norm_di_all = [], []
            proj_no_di_all, proj_di_all = [], []
            sgn_no_di_all, sgn_di_all = [], []  # sign-cosine with target

            t_start = time.time()
            pbar = tqdm(range(0, args.n_examples, args.batch_size),
                        desc=f"  {src_name[:18]:>18} -> {tgt_name[:18]:<18}",
                        leave=False)
            for batch_idx, i in enumerate(pbar):
                xb = x_test[i:i + args.batch_size].to(device)
                yb = y_test[i:i + args.batch_size].to(device)

                # Per-batch mask (for filtering after metric compute)
                batch_mask = combined_mask[i:i + args.batch_size]
                keep = torch.from_numpy(batch_mask).to(device)

                # Re-use pre-computed source gradients (move back to GPU for ops)
                g_src_noDI = src_g_noDI_chunks[batch_idx].to(device)
                g_src_DI = src_g_DI_chunks[batch_idx].to(device)

                # Compute target gradient (this changes per target)
                g_tgt = compute_gradient(tgt_model, xb, yb, di=False)

                # All metrics, then filter by mask before extending lists
                def keep_filter(t):
                    return t[keep].cpu().tolist()

                # Cosine similarities (direction)
                cos_no_di_all.extend(
                    keep_filter(cosine_per_sample(g_src_noDI, g_tgt)))
                cos_di_all.extend(
                    keep_filter(cosine_per_sample(g_src_DI, g_tgt)))
                cos_drift_all.extend(
                    keep_filter(cosine_per_sample(g_src_noDI, g_src_DI)))

                # L2 norms (magnitude)
                norm_no_di_all.extend(keep_filter(norm_per_sample(g_src_noDI)))
                norm_di_all.extend(keep_filter(norm_per_sample(g_src_DI)))

                # Signed projection on target direction (kept for diagnostic, not main)
                proj_no_di_all.extend(
                    keep_filter(signed_projection_per_sample(g_src_noDI, g_tgt)))
                proj_di_all.extend(
                    keep_filter(signed_projection_per_sample(g_src_DI, g_tgt)))

                # Sign-cosine with target (the attack-relevant metric)
                sgn_no_di_all.extend(
                    keep_filter(sign_cosine_per_sample(g_src_noDI, g_tgt)))
                sgn_di_all.extend(
                    keep_filter(sign_cosine_per_sample(g_src_DI, g_tgt)))

            elapsed = time.time() - t_start

            no_di = np.array(cos_no_di_all)
            di = np.array(cos_di_all)
            drift = np.array(cos_drift_all)
            delta = di - no_di

            n_noDI = np.array(norm_no_di_all)
            n_DI = np.array(norm_di_all)
            # IMPORTANT: per-image ratio (n_DI[i]/n_noDI[i]) is unreliable because
            # some robust models have near-zero |g_noDI| on confident images,
            # producing per-image ratios in 10^4-10^10 range that pollute the mean.
            # Use mean-of-norms ratio (more robust) and median ratio (most robust).
            ratio_mean_of_norms = (n_DI.mean() / max(n_noDI.mean(), 1e-12))
            per_img_ratio = n_DI / np.clip(n_noDI, 1e-12, None)
            ratio_median = np.median(per_img_ratio)
            # Legacy per-image-mean (kept for backward-compat; UNRELIABLE for robust models).
            ratio = per_img_ratio

            p_noDI = np.array(proj_no_di_all)
            p_DI = np.array(proj_di_all)
            d_proj = p_DI - p_noDI

            sgn_noDI = np.array(sgn_no_di_all)
            sgn_DI = np.array(sgn_di_all)
            d_sgn = sgn_DI - sgn_noDI

            row = {
                "source": src_name,
                "source_label": src_label,
                "target": tgt_name,
                "n": len(no_di),
                "n_total": int(args.n_examples),
                "src_correct_ratio": float(src_correct.mean()),
                "tgt_correct_ratio": float(tgt_correct.mean()),
                "kept_ratio": float(combined_mask.mean()),
                # Direction metrics (D_cos)
                "cos_noDI_tgt_mean": float(no_di.mean()),
                "cos_DI_tgt_mean": float(di.mean()),
                "delta_align_mean": float(delta.mean()),
                "delta_align_sem": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                "cos_drift_mean": float(drift.mean()),
                # Sign-alignment metrics (attack-relevant, MAIN signal)
                "sign_noDI_tgt_mean": float(sgn_noDI.mean()),
                "sign_DI_tgt_mean": float(sgn_DI.mean()),
                "delta_sign_align_mean": float(d_sgn.mean()),
                "delta_sign_align_sem": float(d_sgn.std(ddof=1) / np.sqrt(len(d_sgn))),
                # Magnitude metrics (supplementary)
                "norm_noDI_mean": float(n_noDI.mean()),
                "norm_DI_mean": float(n_DI.mean()),
                "norm_ratio_mean": float(ratio.mean()),  # UNRELIABLE; see comment above
                "norm_ratio_std": float(ratio.std(ddof=1)),
                "norm_ratio_mean_of_norms": float(ratio_mean_of_norms),  # robust
                "norm_ratio_median": float(ratio_median),  # most robust
                # Signed projection (diagnostic; not reported)
                "proj_noDI_mean": float(p_noDI.mean()),
                "proj_DI_mean": float(p_DI.mean()),
                "delta_proj_mean": float(d_proj.mean()),
                "delta_proj_sem": float(d_proj.std(ddof=1) / np.sqrt(len(d_proj))),
                "runtime_sec": elapsed,
            }
            results.append(row)

            print(f"    {tgt_name:<25} "
                  f"n={row['n']}({100*row['kept_ratio']:.0f}%)  "
                  f"D_cos={row['delta_align_mean']:+.4f}({row['delta_align_sem']:.4f})  "
                  f"D_sign={row['delta_sign_align_mean']:+.5f}({row['delta_sign_align_sem']:.5f})  "
                  f"|g_DI|/|g|(med)={row['norm_ratio_median']:.3f}  "
                  f"t={elapsed:.1f}s")

            del tgt_model
            torch.cuda.empty_cache()

        del src_model
        torch.cuda.empty_cache()

    if not results:
        print("[Error] No results produced. Check model loading.")
        return

    # Save
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.results_dir, "gradient_alignment.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(args.results_dir, "gradient_alignment.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty summary
    print("\n" + "=" * 110)
    print("SUMMARY  (per source-target pair, computed on clean-correct subset)")
    print("=" * 110)
    print(f"{'Source':<32}{'Target':<22}{'n':>5}"
          f"{'D_sign':>11}{'D_cos':>11}{'|g_DI|/|g|(med)':>16}")
    print("-" * 110)
    for r in results:
        print(f"{r['source_label']:<32}{r['target']:<22}{r['n']:>5}"
              f"{r['delta_sign_align_mean']:>+11.5f}"
              f"{r['delta_align_mean']:>+11.5f}"
              f"{r['norm_ratio_median']:>16.3f}")

    # Aggregate per source-type
    df["src_type"] = df["source_label"].str.startswith("Standard").map(
        {True: "Standard", False: "Robust"})
    print("\n" + "=" * 110)
    print("AGGREGATED BY SOURCE TYPE  (averaged across targets)")
    print("=" * 110)
    agg_cols = ["delta_sign_align_mean", "delta_align_mean",
                "norm_ratio_median", "norm_ratio_mean_of_norms",
                "sign_noDI_tgt_mean", "sign_DI_tgt_mean",
                "cos_noDI_tgt_mean", "cos_DI_tgt_mean"]
    grp = df.groupby("src_type")[agg_cols].mean()
    print(grp.to_string())

    # Hypothesis check (sign-alignment is now the primary metric)
    print("\n" + "=" * 100)
    print("HYPOTHESIS CHECK")
    print("=" * 100)

    std_d_sgn = df.loc[df["src_type"] == "Standard",
                       "delta_sign_align_mean"].mean()
    rob_d_sgn = df.loc[df["src_type"] == "Robust",
                       "delta_sign_align_mean"].mean()
    print(f"\n  [PRIMARY: D_sign (attack-relevant: cos(sign(g_src), sign(g_tgt)))]")
    print(f"    Standard: {std_d_sgn:+.5f}  (predicted > 0: DI improves "
          f"sign-bit agreement with target)")
    print(f"    Robust:   {rob_d_sgn:+.5f}  (predicted ~ 0 or < 0: no improvement)")

    std_d_cos = df.loc[df["src_type"] == "Standard", "delta_align_mean"].mean()
    rob_d_cos = df.loc[df["src_type"] == "Robust", "delta_align_mean"].mean()
    print(f"\n  [SECONDARY: D_cos (raw direction)]")
    print(f"    Standard: {std_d_cos:+.5f}  (predicted > 0)")
    print(f"    Robust:   {rob_d_cos:+.5f}  (predicted ~ 0)")

    # Use ROBUST stats (median + mean-of-norms); per-image-ratio mean is degenerate
    # for robust models and produces misleading 10^3-10^4 ratios.
    std_med = df.loc[df["src_type"] == "Standard", "norm_ratio_median"].mean()
    rob_med = df.loc[df["src_type"] == "Robust", "norm_ratio_median"].mean()
    std_mn = df.loc[df["src_type"] == "Standard", "norm_ratio_mean_of_norms"].mean()
    rob_mn = df.loc[df["src_type"] == "Robust", "norm_ratio_mean_of_norms"].mean()
    asym_ratio = rob_med / max(std_med, 1e-6)
    print(f"\n  [SUPPLEMENTARY: |g_DI|/|g_noDI| (gradient magnitude scaling)]")
    print(f"    Standard:  median={std_med:.3f},  mean-of-norms={std_mn:.3f}")
    print(f"    Robust:    median={rob_med:.3f},  mean-of-norms={rob_mn:.3f}")
    print(f"    Asymmetry (median): robust/standard = {asym_ratio:.1f}x")
    print("    Note: raw norm is washed out by sign() in attack; treat as Jacobian-"
          "side evidence consistent with frequency story (Tab. 5), not main signal.")
    print("    (Reporting MEDIAN: per-image ratio MEAN is degenerate due to near-zero "
          "|g_noDI| outliers; see code comment.)")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    sgn_supports = (std_d_sgn > 0 and abs(rob_d_sgn) < std_d_sgn / 3)
    cos_supports = (std_d_cos > 0 and abs(rob_d_cos) < std_d_cos / 3)
    # Use median ratio (robust to outliers) for magnitude check
    mag_supports = (std_med < 1.0 and rob_med > 1.0)
    asym_supports = (asym_ratio > 2.0)

    print(f"  PRIMARY    Sign-alignment asymmetry (D_sign std>0, robust~0):  "
          f"{'YES' if sgn_supports else 'NO'}")
    print(f"  SECONDARY  Cos-alignment asymmetry  (D_cos std>0, robust~0):   "
          f"{'YES' if cos_supports else 'NO'}")
    print(f"  SUPPL.     Magnitude asymmetry      (std<1, robust>1):         "
          f"{'YES' if mag_supports else 'NO'}")
    print(f"  SUPPL.     Strong magnitude asym    (robust/std > 2x):          "
          f"{'YES' if asym_supports else 'NO'}  (got {asym_ratio:.1f}x)")

    if sgn_supports and cos_supports:
        verdict = ("STRONG. Both attack-relevant (sign) and direction (cos) metrics "
                   "show clean asymmetry.")
    elif sgn_supports:
        verdict = ("DEFENSIBLE. Sign-alignment (the attack-relevant metric) shows "
                   "clean asymmetry.")
    elif cos_supports:
        verdict = ("PARTIAL. Cos-alignment shows asymmetry but sign-alignment does "
                   "not -- weaker since attack uses sign(). Consider supplementary only.")
    else:
        verdict = ("WEAK. Neither sign- nor cos-alignment shows clean asymmetry. "
                   "Reconsider before including.")
    print(f"\n  >>> {verdict}")

    print(f"\n[Saved] {csv_path}")
    print(f"[Saved] {json_path}")
    print("[Exp A] Done.")


if __name__ == "__main__":
    main()
