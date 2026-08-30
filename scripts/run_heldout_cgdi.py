#!/usr/bin/env python
"""
Held-out evaluation of the CG-DI rule.

The threshold tau_op, the probe scale, the number of probes K and the binary
choice p in {0, 0.8} are frozen at their development-set values and not tuned
here. Surrogates are used as attack sources for the first time, covering unseen
architectures, unseen adversarial-training recipes, and an unseen dataset
(--dataset cifar100).

Arms per surrogate: p=0 (MI-FGSM), p=0.8 (blind DI, the like-for-like comparison
against CG-DI's "on" state), p=1 (the setting of the main ImageNet table), and
CG-DI. CG-DI's decision is per image rather than per batch: the per-image LGC
decides, for that image, whether the DI transform may fire at all.

--p_grid replaces the arm set with one arm per diversity probability, giving a
continuous-p sweep with per-image logging.

A source that fails to load aborts the run (--no_strict_sources to override), so
a partially measured source set cannot be reported as a complete one.

Example
-------
  python scripts/run_heldout_cgdi.py --dataset imagenet --n_examples 1000 --n_seeds 5
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device

try:
    from run_recipe_panel import get_source_model as _panel_loader
except Exception:
    _panel_loader = None

# --- FROZEN development-set hyperparameters. Do not tune. -------------------
TAU_OP = 0.92
EPS_CHK = 1.0 / 255
K_PROBE = 5
P_ON = 0.8


def load_surrogate(key, dataset, device):
    if _panel_loader is not None and dataset == "imagenet":
        try:
            return _panel_loader(key, device)
        except Exception:
            pass
    return get_model(key, dataset=dataset, device=device)


def lgc_per_image(model, x, y, K=K_PROBE, sigma=EPS_CHK):
    """Eq. 1 of the paper, per image. Uniform noise, K probes."""
    xi = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xi), y, reduction="sum").backward()
    g0 = xi.grad.detach()
    model.zero_grad(set_to_none=True)
    acc = torch.zeros(x.size(0), device=x.device)
    for _ in range(K):
        xp = (x + (torch.rand_like(x) * 2 - 1) * sigma).clamp(0, 1) \
            .detach().requires_grad_(True)
        F.cross_entropy(model(xp), y, reduction="sum").backward()
        acc += F.cosine_similarity(g0.flatten(1), xp.grad.detach().flatten(1), dim=1)
        model.zero_grad(set_to_none=True)
    return acc / K


def di_transform(x, resize_rate=0.9, n_scales=None):
    """torchattacks-style DI: shrink to a random size then zero-pad back.

    n_scales coarsens the SCALE distribution only, to whatever number of
    distinct sizes is asked for, spanning the same range. At 224 with r=0.9 the
    full operator draws from 24 sizes; a 32x32 image at the same rate has only
    5. Padding offsets are left at full range so that exactly one variable
    changes.
    """
    S = x.shape[-1]
    lo = int(S * resize_rate)
    if n_scales is None:
        rnd = torch.randint(lo, S + 1, (1,)).item()
    else:
        cand = np.unique(np.linspace(lo, S, n_scales).round().astype(int))
        rnd = int(cand[torch.randint(0, len(cand), (1,)).item()])
    xr = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pt = torch.randint(0, S - rnd + 1, (1,)).item()
    pl = torch.randint(0, S - rnd + 1, (1,)).item()
    return F.pad(xr, (pl, S - rnd - pl, pt, S - rnd - pt), value=0)


def mi_di_fgsm(model, x, y, p_vec, eps, alpha, steps, resize_rate=0.9, decay=1.0,
               n_scales=None):
    """MI-FGSM with a PER-IMAGE diversity probability p_vec in [0,1].

    p_vec = 0    -> plain MI-FGSM for that image
    p_vec = 0.8  -> DI-FGSM at p=0.8 for that image
    Mixing them inside one batch is what makes CG-DI's per-image decision exact
    rather than a batch-level approximation.
    """
    x = x.detach()
    B = x.size(0)
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        xin = x + delta
        if float(p_vec.max()) > 0:
            xt = di_transform(xin, resize_rate, n_scales)
            fire = (torch.rand(B, device=x.device) < p_vec).view(B, 1, 1, 1)
            xin = torch.where(fire, xt, xin)
        loss = F.cross_entropy(model(xin), y)
        grad = torch.autograd.grad(loss, delta)[0]
        grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-10)
        momentum = decay * momentum + grad
        delta = (delta.detach() + alpha * momentum.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)
    return (x + delta).detach()


HELDOUT_IMAGENET = [
    # standard architectures never before used as sources; the three named
    # below are the decisive cases for the per-surrogate rule
    ("ViT_B_16_ImageNet", "Standard"),
    ("ConvNeXt_B_ImageNet", "Standard"),
    ("DenseNet121_Standard", "Standard"),
    ("InceptionV3", "Standard"),
    ("Swin_B_ImageNet", "Standard"),
    # unseen adversarial-training recipes
    ("Singh2023Revisiting_ViT-B-ConvStem", "Robust"),
    ("ARES_ConvNeXt_B", "Robust"),
]
IMAGENET_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet",
                    "ViT_B_16_ImageNet", "InceptionV3"]

HELDOUT_CIFAR100 = [
    ("Standard_WRN28_10", "Standard"),
    ("Rice2020Overfitting", "Robust"),
    ("Cui2023Decoupled_WRN-28-10", "Robust"),
    ("Wang2023Better_WRN-28-10", "Robust"),
    ("Hendrycks2019Using", "Robust"),
    ("Pang2022Robustness_WRN28_10", "Robust"),
    ("Rebuffi2021Fixing_28_10_cutmix_ddpm", "Robust"),
]
CIFAR100_TARGETS = ["Pang2022Robustness_WRN28_10",
                    "Rebuffi2021Fixing_28_10_cutmix_ddpm",
                    "Hendrycks2019Using"]

# CIFAR-10, mirroring the ImageNet setup: one standard and two PGD-AT sources,
# transferred to undefended targets of a different architecture family. No
# source appears among the targets.
HELDOUT_CIFAR10 = [
    ("C10_ResNet18", "Standard"),
    ("Engstrom2019Robustness", "Robust"),
    ("Rice2020Overfitting", "Robust"),
]
CIFAR10_TARGETS = ["C10_VGG16", "C10_DenseNet121", "Carmon2019Unlabeled"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="imagenet",
                   choices=["imagenet", "cifar100", "cifar10"])
    p.add_argument("--sources", type=str, nargs="+", default=None)
    p.add_argument("--targets", type=str, nargs="+", default=None)
    p.add_argument("--n_examples", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--alpha", type=float, default=2.0 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--strict_sources", action="store_true", default=True,
                   help="Abort if any source fails to load, instead of quietly "
                        "producing results for a subset.")
    p.add_argument("--no_strict_sources", dest="strict_sources",
                   action="store_false")
    p.add_argument("--p_grid", type=float, nargs="+", default=None,
                   help="Run one arm per diversity probability instead of the "
                        "standard arm set (continuous-p sweep, per-image).")
    p.add_argument("--scale_grid", type=int, nargs="+", default=None,
                   help="Granularity test: one arm per number of distinct "
                        "resize scales, at p=1 and fixed r. 24 scales is the "
                        "full operator at 224 with r=0.9; 5 matches what a "
                        "32x32 image has at the same rate.")
    p.add_argument("--arms", type=str, nargs="+", default=None,
                   help="Subset of arms to run (p0 blind_di blind_di_p1 cgdi). "
                        "Used to add an arm to an existing run without redoing it.")
    p.add_argument("--results_dir", type=str, default="results/heldout_cgdi")
    return p.parse_args()


def predict(model, x, device, bs):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    eps = args.eps if args.eps is not None else \
        (16.0 / 255 if args.dataset == "imagenet" else 8.0 / 255)

    if args.dataset == "imagenet":
        sources = HELDOUT_IMAGENET
        targets = args.targets or IMAGENET_TARGETS
    elif args.dataset == "cifar10":
        sources = HELDOUT_CIFAR10
        targets = args.targets or CIFAR10_TARGETS
    else:
        sources = HELDOUT_CIFAR100
        targets = args.targets or CIFAR100_TARGETS
    if args.sources:
        sources = [(s.split(":")[0], s.split(":")[1] if ":" in s else "?")
                   for s in args.sources]
    # Checked after --sources is applied, so it tests the list actually run.
    # The evaluation loop refuses t == src_key, so overlap is fine for the
    # held-out design; it is not fine for a p-sweep, where the overlapping
    # source would be averaged over fewer targets than the others.
    overlap = {s for s, _ in sources} & set(targets)
    if overlap and args.p_grid:
        raise SystemExit(
            f"source/target overlap in a p-sweep: {sorted(overlap)}. Each curve "
            f"must average over the same targets; pick disjoint sets.")
    if overlap:
        print(f"[note] source/target overlap {sorted(overlap)}; self-pairs skipped")
    seeds = [100 * (i + 1) for i in range(args.n_seeds)]

    print("=" * 78)
    print(f"E6: held-out CG-DI evaluation ({args.dataset})")
    print("=" * 78)
    print(f"  FROZEN: tau_op={TAU_OP}  eps_chk={EPS_CHK*255:.0f}/255  "
          f"K={K_PROBE}  p_on={P_ON}")
    print(f"  N={args.n_examples}  seeds={seeds}  eps={eps*255:.0f}/255\n")

    seed_everything(seeds[0])
    x_test, y_test = load_dataset(args.dataset, args.n_examples)
    N = x_test.size(0)

    tgt_models, tgt_correct = {}, {}
    for t in targets:
        tm = get_model(t, dataset=args.dataset, device=device)
        tm.eval()
        tgt_models[t] = tm
        tgt_correct[t] = (predict(tm, x_test, device, args.batch_size)
                          == y_test).numpy()
        print(f"[Target] {t}: clean-correct {tgt_correct[t].sum()}/{N}")

    rows, succ = [], {}
    y_np = y_test.numpy()

    for src_key, src_type in sources:
        try:
            sm = load_surrogate(src_key, args.dataset, device)
            sm.eval()
        except Exception as e:
            # A silently skipped source yields a partially measured source set that
            # still reports results. Fail loudly instead.
            if args.strict_sources:
                raise RuntimeError(
                    f"source {src_key} failed to load ({type(e).__name__}: {e}). "
                    f"Refusing to continue with an incomplete source set; pass "
                    f"--no_strict_sources to skip deliberately.") from e
            print(f"[SKIP] {src_key}: {type(e).__name__}: {str(e)[:70]}")
            continue
        src_correct = (predict(sm, x_test, device, args.batch_size)
                       == y_test).numpy()

        # ---- the CG-DI probe, once per surrogate (LGC is stable within a model)
        seed_everything(seeds[0])
        lgc_all = np.zeros(N)
        for i in range(0, N, args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            lgc_all[i:i + xb.size(0)] = lgc_per_image(sm, xb, yb).cpu().numpy()
        p_cgdi = np.where(lgc_all > TAU_OP, 0.0, P_ON)
        frac_off = float((p_cgdi == 0).mean())
        print(f"\n[Source] {src_key} ({src_type})  clean-correct "
              f"{src_correct.sum()}/{N}")
        print(f"  LGC mean={lgc_all.mean():.4f}  median={np.median(lgc_all):.4f}"
              f"   CG-DI turns DI OFF on {100*frac_off:.1f}% of images "
              f"-> model-level decision: p={0.0 if frac_off > 0.5 else P_ON}")

        # blind_di uses p=0.8 so it is the like-for-like comparison against
        # CG-DI's "on" state; blind_di_p1 uses p=1, which is what the main
        # paper's headline table reports, so the held-out numbers can be placed
        # next to it. Both are needed: comparing CG-DI against p=1 would
        # conflate the routing decision with the diversity level.
        # Each arm is (per-image p vector, n_scales). n_scales=None is the full
        # transform; an integer coarsens the scale set without changing p or r.
        arms = {"p0": (np.zeros(N), None), "blind_di": (np.full(N, P_ON), None),
                "blind_di_p1": (np.ones(N), None), "cgdi": (p_cgdi, None)}
        if args.p_grid:
            # Continuous-p sweep with per-image logging, so the argmax over p can
            # be given a bootstrap CI and "near-optimal" claimed only where the
            # optimum is statistically identifiable.
            arms = {f"p{v:g}": (np.full(N, v), None) for v in args.p_grid}
        elif args.scale_grid:
            # Granularity test: p and r fixed, only the number of distinct
            # resize scales varies. p0 is kept as the shared baseline.
            arms = {"p0": (np.zeros(N), None)}
            for n in args.scale_grid:
                arms[f"ns{n}"] = (np.ones(N), n)
        elif args.arms:
            arms = {k: v for k, v in arms.items() if k in args.arms}
        for arm, (pvec_np, n_scales) in arms.items():
            for seed in seeds:
                t0 = time.time()
                seed_everything(seed)
                pred = {t: np.zeros(N, dtype=np.int64) for t in targets}
                for i in range(0, N, args.batch_size):
                    xb = x_test[i:i + args.batch_size].to(device)
                    yb = y_test[i:i + args.batch_size].to(device)
                    pv = torch.from_numpy(
                        pvec_np[i:i + xb.size(0)]).float().to(device)
                    xadv = mi_di_fgsm(sm, xb, yb, pv, eps, args.alpha,
                                      args.steps, args.resize_rate, n_scales=n_scales)
                    with torch.no_grad():
                        for t, tm in tgt_models.items():
                            pred[t][i:i + xb.size(0)] = \
                                tm(xadv).argmax(1).cpu().numpy()
                shown = []
                for t in targets:
                    if t == src_key:      # a model cannot be its own black box
                        continue
                    s = (pred[t] != y_np)
                    succ[(src_key, arm, seed, t)] = s
                    mask = src_correct & tgt_correct[t]
                    asr = float(100 * s[mask].mean())
                    rows.append(dict(
                        dataset=args.dataset, source=src_key, src_type=src_type,
                        arm=arm, seed=seed, target=t, n_eval=int(mask.sum()),
                        lgc_mean=float(lgc_all.mean()),
                        cgdi_frac_off=frac_off, asr=asr))
                    shown.append(f"{t.split('_')[0][:7]}={asr:.1f}")
                print(f"    {arm:<9} seed={seed}  " + "  ".join(shown) +
                      f"   ({time.time()-t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(
                    os.path.join(args.results_dir,
                                 f"heldout_{args.dataset}.csv"), index=False)
        del sm
        torch.cuda.empty_cache()

    np.savez_compressed(
        os.path.join(args.results_dir, f"per_image_{args.dataset}.npz"),
        labels=y_np,
        **{f"tgt_correct__{t}": tgt_correct[t] for t in targets},
        **{"|".join(map(str, k)): v for k, v in succ.items()})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, f"heldout_{args.dataset}.csv"),
              index=False)

    # ---- outcome table: D = ASR(blind DI) - ASR(p=0), the quantity the
    # pre-registered predictions are scored against.
    piv = df.groupby(["source", "src_type", "target", "arm"])["asr"].mean() \
            .unstack("arm")
    # A run may cover only a subset of arms (--arms), e.g. adding the p=1 arm to
    # an existing run. Derive only the columns whose inputs are actually
    # present, rather than assuming the full arm set.
    if {"blind_di", "p0"} <= set(piv.columns):
        piv["D"] = piv["blind_di"] - piv["p0"]
    if {"blind_di_p1", "p0"} <= set(piv.columns):
        piv["D_p1"] = piv["blind_di_p1"] - piv["p0"]
    if {"cgdi", "blind_di", "p0"} <= set(piv.columns):
        piv["cgdi_gap_vs_best"] = piv["cgdi"] - piv[["p0", "blind_di"]].max(axis=1)
    out = os.path.join(args.results_dir, f"heldout_{args.dataset}_effects.csv")
    piv.to_csv(out)
    print(f"\n[Saved] {out}")
    with pd.option_context("display.width", 200):
        print("\n" + piv.round(2).to_string())
    print("\nD < 0 means DI HURT this surrogate; D > 0 means DI helped. Score "
          "these against prereg/E6_predictions_*.json. Report "
          "every disagreement, including any that falsify the per-surrogate "
          "theory on the standard side.")


if __name__ == "__main__":
    main()
