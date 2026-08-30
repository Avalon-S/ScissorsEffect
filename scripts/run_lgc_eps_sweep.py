#!/usr/bin/env python
"""
Exp C: LGC computation on the controlled epsilon spectrum (Salman et al.)

For each source in the Exp B sweep (Standard ResNet50 + Salman eps ∈
{0.5, 1, 2, 4, 8}/255), compute Local Gradient Consistency (LGC) on N=500
ImageNet images.

This ties Exp A's gradient framework to Exp B's ASR framework:
  1. Show LGC monotonically tracks training epsilon (controlled sweep)
  2. Show LGC correlates with D_ASR (= ASR(DI) - ASR(MI), from Exp B)
  3. Show CG-DI's tau=0.92 threshold correctly identifies all Salman models
     as "robust" (LGC > tau), while a trivial training-history flag treats
     all Salman models identically regardless of epsilon.

This addresses R1's "why CG-DI beyond trivial flag" question:
  - Within the Salman family (all "Robust" by training history), LGC varies
    with epsilon, providing FINE-GRAINED regime detection that a binary
    flag cannot.

Requires: Exp B results (results/eps_sweep/eps_sweep_pivot.csv)
Outputs:  results/lgc_eps_sweep/{lgc_per_source.csv, joined_with_exp_b.csv,
                                 correlation_summary.txt}
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
from attacks.sap import estimate_p_lgc
from utils import seed_everything, get_device


# Exp B controlled spectrum + 2 paper-included literature-standard robust models
# (Engstrom and Mo2022) for outlier-isolation analysis.
# Architecture column: matters because eps=8/255 PGD-AT on ResNet50 is
# under-capacity (paper convention is WRN/ViT for eps>=8); Mo2022 ViT-B
# at eps=8 is the proper-capacity reference point.
SOURCES = [
    # name                              label                            eps    arch
    ("ResNet50",                        "Standard (eps=0, RN50)",        0.0/255),
    ("Salman_eps0.5",                   "Salman (eps=0.5/255, RN50)",    0.5/255),
    ("Salman_eps1.0",                   "Salman (eps=1/255, RN50)",      1.0/255),
    ("Salman_eps2.0",                   "Salman (eps=2/255, RN50)",      2.0/255),
    ("Salman_eps4.0",                   "Salman (eps=4/255, RN50)",      4.0/255),
    ("Engstrom2019Robustness_ImageNet", "Engstrom (eps=4/255, RN50)",    4.0/255),
    ("Mo2022When_ViT-B",                "Mo2022 (eps=8/255, ViT-B)",     8.0/255),
    ("Salman_eps8.0",                   "Salman (eps=8/255, RN50)",      8.0/255),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--K", type=int, default=5,
                   help="LGC perturbation samples (paper Algorithm 1: K=5)")
    p.add_argument("--sigma", type=float, default=1/255,
                   help="LGC perturbation scale (paper: 1/255)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="results/lgc_eps_sweep")
    p.add_argument("--exp_b_pivot", type=str,
                   default="results/eps_sweep/eps_sweep_pivot.csv",
                   help="Exp B pivot CSV (for joining with D_ASR)")
    p.add_argument("--cgdi_tau", type=float, default=0.92,
                   help="CG-DI threshold (paper default 0.92)")
    p.add_argument("--filter_correct", action="store_true", default=True,
                   help="Compute LGC only on samples where source predicts "
                        "correctly (matches Exp A protocol; default True)")
    p.add_argument("--no_filter_correct", dest="filter_correct",
                   action="store_false")
    return p.parse_args()


def predict_batched(model, x, batch_size, device):
    """Get top-1 predictions in batches."""
    preds = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i + batch_size].to(device)
            preds.append(model(xb).argmax(dim=1))
    return torch.cat(preds)


def compute_lgc_per_source(model, x, y, args, device, correct_mask=None):
    """
    Compute per-image LGC on the full evaluation set.
    If correct_mask is provided, only return LGC values for samples where
    correct_mask[i] is True (still computes on all samples for efficiency,
    then masks at the end so progress bar stays informative).

    Returns: (lgc_full [N], lgc_kept [N_correct or N])
    """
    lgc_all = []
    for i in tqdm(range(0, x.size(0), args.batch_size),
                  desc="    LGC", leave=False):
        xb = x[i:i + args.batch_size].to(device)
        yb = y[i:i + args.batch_size].to(device)
        _, lgc_batch = estimate_p_lgc(
            model, xb, yb, device,
            K=args.K, sigma=args.sigma,
        )
        lgc_all.extend(lgc_batch.detach().cpu().tolist())
    lgc_full = np.array(lgc_all)
    if correct_mask is not None:
        lgc_kept = lgc_full[correct_mask]
    else:
        lgc_kept = lgc_full
    return lgc_full, lgc_kept


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 72)
    print("Exp C: LGC computation on Salman epsilon spectrum")
    print("=" * 72)
    print(f"  N={args.n_examples}, K={args.K}, sigma={args.sigma:.4f} "
          f"({args.sigma*255:.1f}/255), seed={args.seed}")
    print(f"  CG-DI threshold tau = {args.cgdi_tau}")
    print()

    # Load data once (same seed as Exp A/B for clean comparison)
    seed_everything(args.seed)
    print(f"[Data] Loading ImageNet validation ({args.n_examples} images)...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    print(f"  Loaded {x_test.size(0)} images, shape={tuple(x_test.shape[1:])}")

    results = []
    for src_name, src_label, eps_val in SOURCES:
        print(f"\n[Source] {src_name} -- {src_label} "
              f"(eps_train={eps_val*255:.1f}/255)")
        try:
            src_model = get_model(src_name, dataset="imagenet", device=device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        # Compute clean-correct mask if requested
        correct_mask = None
        if args.filter_correct:
            preds = predict_batched(src_model, x_test, args.batch_size, device)
            correct_mask = (preds.cpu() == y_test).numpy()
            print(f"  [Clean-correct] {correct_mask.sum()}/{len(correct_mask)} "
                  f"({100*correct_mask.mean():.1f}%)")

        t_start = time.time()
        lgc_full, lgc = compute_lgc_per_source(
            src_model, x_test, y_test, args, device, correct_mask
        )
        elapsed = time.time() - t_start

        # Stats on the kept (clean-correct) subset
        lgc_mean = float(lgc.mean())
        lgc_std = float(lgc.std(ddof=1))
        lgc_sem = lgc_std / np.sqrt(len(lgc))
        lgc_median = float(np.median(lgc))

        # Stats on the full set (for diagnostic — see if filter changed result)
        lgc_full_mean = float(lgc_full.mean())
        lgc_full_median = float(np.median(lgc_full))

        # CG-DI decision uses the (clean-correct) mean
        cgdi_decision = "p=0 (skip DI)" if lgc_mean > args.cgdi_tau \
                        else "p=0.8 (apply DI)"
        flag_decision = "p=0 (skip DI)" if eps_val > 0 \
                        else "p=0.8 (apply DI)"

        print(f"  LGC (clean-correct n={len(lgc)}): mean={lgc_mean:.4f} "
              f"median={lgc_median:.4f}  SEM={lgc_sem:.4f}")
        if args.filter_correct:
            print(f"  LGC (full set n={len(lgc_full)}, diagnostic): "
                  f"mean={lgc_full_mean:.4f} median={lgc_full_median:.4f}")
        print(f"  CG-DI decision (tau={args.cgdi_tau}): {cgdi_decision}")
        print(f"  Flag-rule decision (binary):   {flag_decision}")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            "source": src_name,
            "source_label": src_label,
            "eps_train": eps_val,
            "eps_train_x255": eps_val * 255,
            "lgc_mean": lgc_mean,
            "lgc_std": lgc_std,
            "lgc_sem": lgc_sem,
            "lgc_median": lgc_median,
            "lgc_full_mean": lgc_full_mean,
            "lgc_full_median": lgc_full_median,
            "n_kept": int(len(lgc)),
            "n_total": int(len(lgc_full)),
            "kept_ratio": float(len(lgc) / len(lgc_full)),
            "cgdi_decision": cgdi_decision,
            "flag_decision": flag_decision,
            "K": args.K,
            "sigma": args.sigma,
        })

        del src_model
        torch.cuda.empty_cache()

    if not results:
        print("[Error] No results.")
        return

    df = pd.DataFrame(results)
    lgc_csv = os.path.join(args.results_dir, "lgc_per_source.csv")
    df.to_csv(lgc_csv, index=False)

    # Try to join with Exp B
    joined = None
    if os.path.exists(args.exp_b_pivot):
        exp_b = pd.read_csv(args.exp_b_pivot)
        joined = df.merge(
            exp_b[["source", "mifgsm", "difgsm", "delta_di_minus_mi"]],
            on="source", how="left",
        )
        joined_csv = os.path.join(args.results_dir, "joined_with_exp_b.csv")
        joined.to_csv(joined_csv, index=False)
        print(f"\n[Saved] {joined_csv}")
    else:
        print(f"\n[WARN] Exp B pivot not found at {args.exp_b_pivot}; "
              "skipping join.")

    # Pretty summary
    print("\n" + "=" * 100)
    print("LGC × EPS-SWEEP SUMMARY")
    print("=" * 100)
    print(f"{'Source':<24}{'eps':>9}{'LGC mean':>11}{'CG-DI':<22}{'Flag':<22}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r['source_label']:<24}{r['eps_train_x255']:>8.1f}/255"
              f"{r['lgc_mean']:>11.4f}  {r['cgdi_decision']:<22}"
              f"{r['flag_decision']:<22}")

    if joined is not None and "delta_di_minus_mi" in joined.columns:
        print("\n" + "=" * 100)
        print("LGC vs D_ASR (= ASR(DI) - ASR(MI), from Exp B)")
        print("=" * 100)
        print(f"{'Source':<24}{'eps':>9}{'LGC':>9}{'D_ASR':>10}")
        print("-" * 100)
        for _, r in joined.iterrows():
            d_asr = r.get("delta_di_minus_mi", np.nan)
            print(f"{r['source_label']:<24}{r['eps_train_x255']:>8.1f}/255"
                  f"{r['lgc_mean']:>9.4f}{d_asr*100:>+9.2f}%")

        # Spearman correlation: LGC vs D_ASR
        try:
            from scipy.stats import spearmanr, pearsonr
            valid = joined.dropna(subset=["lgc_mean", "delta_di_minus_mi"])
            if len(valid) >= 4:
                rho_s, p_s = spearmanr(valid["lgc_mean"],
                                       valid["delta_di_minus_mi"])
                rho_p, p_p = pearsonr(valid["lgc_mean"],
                                      valid["delta_di_minus_mi"])
                print(f"\n  Correlation (n={len(valid)}):")
                print(f"    Spearman rho = {rho_s:+.3f}, p = {p_s:.4f}")
                print(f"    Pearson  r   = {rho_p:+.3f}, p = {p_p:.4f}")
                print(f"    Interpretation: negative correlation = LGC predicts "
                      f"DI sensitivity (higher LGC -> more DI harm)")
        except ImportError:
            print("\n  [scipy not available; skipping correlation]")

    # Decision-rule disagreement summary
    print("\n" + "=" * 100)
    print("CG-DI vs FLAG-RULE DECISION COMPARISON")
    print("=" * 100)
    n_agree = (df["cgdi_decision"] == df["flag_decision"]).sum()
    n_disagree = len(df) - n_agree
    print(f"  Total models: {len(df)}")
    print(f"  Agreement:    {n_agree}/{len(df)}")
    print(f"  Disagreement: {n_disagree}/{len(df)}")

    if n_disagree > 0:
        print("\n  Disagreement cases (model where CG-DI != flag rule):")
        disagree = df[df["cgdi_decision"] != df["flag_decision"]]
        for _, r in disagree.iterrows():
            print(f"    {r['source_label']:<24}  CG-DI: {r['cgdi_decision']}  "
                  f"vs Flag: {r['flag_decision']}")

    # Write text summary
    summary_lines = []
    summary_lines.append("# Exp C: LGC × eps-sweep summary (auto-generated)")
    summary_lines.append("")
    summary_lines.append("## LGC for each source")
    summary_lines.append("")
    summary_lines.append("| Source | eps_train | LGC mean | CG-DI decision |")
    summary_lines.append("|---|---|---|---|")
    for _, r in df.iterrows():
        summary_lines.append(f"| {r['source_label']} | "
                             f"{r['eps_train_x255']:.1f}/255 | "
                             f"{r['lgc_mean']:.4f} | {r['cgdi_decision']} |")

    if joined is not None and "delta_di_minus_mi" in joined.columns:
        try:
            from scipy.stats import spearmanr
            valid = joined.dropna(subset=["lgc_mean", "delta_di_minus_mi"])
            rho, p = spearmanr(valid["lgc_mean"], valid["delta_di_minus_mi"])
            summary_lines.append("")
            summary_lines.append("## LGC vs D_ASR correlation")
            summary_lines.append("")
            summary_lines.append(f"Spearman rho = {rho:+.3f}, p = {p:.4f} "
                                 f"(n={len(valid)})")
        except Exception as exc:
            summary_lines.append(f"correlation not computed: {exc}")

    summary_path = os.path.join(args.results_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[Saved] {lgc_csv}")
    print(f"[Saved] {summary_path}")
    print("[Exp C] Done.")


if __name__ == "__main__":
    main()
