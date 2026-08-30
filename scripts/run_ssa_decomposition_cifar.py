#!/usr/bin/env python
"""
Exp F: SSA × Transform Decomposition on CIFAR-10 (cross-dataset replication of Exp D)

Replicates Exp D (run_ssa_decomposition.py) on CIFAR-10 to verify the
frequency-interference mechanism (resize is dominant; translation neutral)
generalizes across datasets and architectures.

Conditions (4 per source):
  C0: SSA-only       (DCT spectrum scaling, NO spatial transform)
  C1: SSA + resize   (DCT scaling + resize-then-pad-to-original)
  C2: SSA + transl   (DCT scaling + translation only, no resize)
  C3: SSA + full DI  (DCT scaling + full DI = resize + translation)

Default config (paper Tab.5 CIFAR-10 aggressive setting):
  - eps_attack = 8/255 (CIFAR convention)
  - resize_rate = 0.6  (matches paper Tab.5 aggressive CIFAR)
  - p (DI prob) = 0.8  (matches paper Tab.5)
  - SSA: sigma=16/255, rho=0.5, n_ensemble=20

Sources: Standard (WRN-28-10), Engstrom2019Robustness (Robust ε=8/255 WRN)
Target:  Carmon2019Unlabeled  (different robust recipe)

"""
import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add scripts/ so we can import the sibling Exp D module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device

# Reuse DCT helpers from run_ssa_decomposition.py
from run_ssa_decomposition import (
    dct_2d, idct_2d, HAS_TORCH_DCT,
    ssa_transform, resize_only, translation_only, full_di,
    SPATIAL_FNS, make_combined_transform, run_attack,
)


DEFAULT_SOURCES = [
    ("Standard",                  "Standard WRN-28-10 (eps=0)"),
    ("Engstrom2019Robustness",    "Engstrom Robust (eps=8/255 WRN)"),
]

CONDITIONS = ["none", "resize", "translation", "di"]
CONDITION_LABEL = {
    "none":        "SSA-only",
    "resize":      "SSA + resize",
    "translation": "SSA + translation",
    "di":          "SSA + full DI",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps_attack", type=float, default=8/255,
                   help="CIFAR-10 paper default 8/255")
    p.add_argument("--alpha", type=float, default=2/255)
    p.add_argument("--sigma_ssa", type=float, default=16.0,
                   help="SSA additive noise (in pixel scale 0-255)")
    p.add_argument("--rho_ssa", type=float, default=0.5)
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--resize_rate", type=float, default=0.6,
                   help="paper Tab.5 CIFAR aggressive default")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--target", type=str, default="Carmon2019Unlabeled")
    p.add_argument("--results_dir", type=str,
                   default="results/ssa_decomposition_cifar")
    p.add_argument("--sources", type=str, nargs="+", default=None)
    p.add_argument("--conditions", type=str, nargs="+", default=None)
    return p.parse_args()


def eval_asr_on_target(tgt_model, x_adv, y, batch_size, device):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for i in range(0, x_adv.size(0), batch_size):
            xb = x_adv[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            preds = tgt_model(xb).argmax(dim=1)
            n_correct += (preds == yb).sum().item()
            n_total += xb.size(0)
    return 1.0 - n_correct / n_total


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78)
    print("Exp F: SSA × Transform Decomposition on CIFAR-10")
    print("=" * 78)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, "
          f"n_ensemble={args.n_ensemble}")
    print(f"  steps={args.steps}, eps={args.eps_attack:.4f}, "
          f"alpha={args.alpha:.4f}")
    print(f"  SSA: sigma={args.sigma_ssa}/255, rho={args.rho_ssa}")
    print(f"  Resize rate: {args.resize_rate}")
    print(f"  Target: {args.target}")
    print(f"  torch_dct available: {HAS_TORCH_DCT}")
    print()

    sources = DEFAULT_SOURCES
    if args.sources:
        all_sources = {n: lbl for n, lbl in DEFAULT_SOURCES}
        sources = [(s, all_sources.get(s, s)) for s in args.sources]

    conditions = args.conditions if args.conditions else CONDITIONS

    print(f"  Sources ({len(sources)}):")
    for n, lbl in sources:
        print(f"    - {n}  [{lbl}]")
    print(f"  Conditions ({len(conditions)}): {conditions}")
    print()

    print(f"[Target] Loading {args.target}...")
    tgt_model = get_model(args.target, "cifar10", "Linf", device)
    tgt_model.eval()

    seed_everything(42)
    print(f"[Data] Loading CIFAR-10 ({args.n_examples} images)...")
    x_test, y_test = load_dataset("cifar10", args.n_examples)

    # Target clean acc sanity
    n_correct = 0
    with torch.no_grad():
        for i in range(0, x_test.size(0), args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            n_correct += (tgt_model(xb).argmax(dim=1) == yb).sum().item()
    target_clean_acc = n_correct / x_test.size(0)
    print(f"  Target clean acc: {target_clean_acc*100:.2f}%")

    results = []
    overall_t0 = time.time()

    for src_name, src_label in sources:
        print(f"\n[Source] {src_name} -- {src_label}")
        try:
            src_model = get_model(src_name, "cifar10", "Linf", device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        for cond in conditions:
            transform_fn = make_combined_transform(
                spatial_mode=cond,
                sigma_ssa=args.sigma_ssa,
                rho_ssa=args.rho_ssa,
                resize_rate=args.resize_rate,
            )

            asr_seeds = []
            for seed_idx in range(args.n_seeds):
                seed_everything(42 + seed_idx * 100)
                t_start = time.time()
                x_adv_chunks = []
                pbar = tqdm(range(0, x_test.size(0), args.batch_size),
                            desc=f"  {cond:<12} seed={42+seed_idx*100}",
                            leave=False)
                for i in pbar:
                    xb = x_test[i:i + args.batch_size].to(device)
                    yb = y_test[i:i + args.batch_size].to(device)
                    x_adv = run_attack(
                        src_model, xb, yb, transform_fn,
                        n_ensemble=args.n_ensemble,
                        eps=args.eps_attack, alpha=args.alpha,
                        steps=args.steps,
                    )
                    x_adv_chunks.append(x_adv.cpu())
                x_adv_all = torch.cat(x_adv_chunks, dim=0)
                asr = eval_asr_on_target(
                    tgt_model, x_adv_all, y_test, args.batch_size, device
                )
                elapsed = time.time() - t_start
                asr_seeds.append(asr)
                print(f"  {CONDITION_LABEL[cond]:<22}  "
                      f"seed={42+seed_idx*100:<5}  ASR={asr*100:6.2f}%  "
                      f"t={elapsed:.1f}s")

            mean_asr = float(np.mean(asr_seeds))
            std_asr = float(np.std(asr_seeds, ddof=1)) if len(asr_seeds) > 1 else 0.0

            results.append({
                "source": src_name,
                "source_label": src_label,
                "condition": cond,
                "condition_label": CONDITION_LABEL[cond],
                "asr_mean": mean_asr,
                "asr_std": std_asr,
                "n_seeds": len(asr_seeds),
                "n_examples": args.n_examples,
                "target": args.target,
                "target_clean_acc": target_clean_acc,
                "dataset": "cifar10",
                "resize_rate": args.resize_rate,
            })

        del src_model
        torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_t0
    if not results:
        print("[Error] No results.")
        return

    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.results_dir, "ssa_decomp_cifar_raw.csv")
    df.to_csv(raw_csv, index=False)

    pivot = df.pivot_table(
        index="source",
        columns="condition",
        values="asr_mean",
    )
    pivot = pivot[CONDITIONS] if all(c in pivot.columns for c in CONDITIONS) \
            else pivot
    pivot_csv = os.path.join(args.results_dir, "ssa_decomp_cifar_pivot.csv")
    pivot.to_csv(pivot_csv)

    print("\n" + "=" * 90)
    print(f"SUMMARY  (CIFAR-10, target={args.target}, ASR in %)")
    print("=" * 90)
    print(f"{'Source':<32}", end="")
    for cond in CONDITIONS:
        print(f"{CONDITION_LABEL[cond]:>20}", end="")
    print()
    print("-" * 90)
    for src_name, src_label in sources:
        if src_name not in df["source"].values:
            continue
        row = pivot.loc[src_name]
        print(f"{src_label:<32}", end="")
        for cond in CONDITIONS:
            v = row.get(cond, np.nan)
            print(f"{v*100:>+19.2f}%", end="")
        print()

    print("\n" + "=" * 90)
    print("HYPOTHESIS TESTS (per source)")
    print("=" * 90)
    for src_name, src_label in sources:
        if src_name not in pivot.index:
            continue
        row = pivot.loc[src_name]
        ssa_only = row.get("none", np.nan) * 100
        ssa_resize = row.get("resize", np.nan) * 100
        ssa_trans = row.get("translation", np.nan) * 100
        ssa_di = row.get("di", np.nan) * 100

        d_resize = ssa_resize - ssa_only
        d_trans = ssa_trans - ssa_only
        d_di = ssa_di - ssa_only

        print(f"\n  [{src_label}]")
        print(f"    SSA-only       = {ssa_only:.2f}%")
        print(f"    +resize        = {ssa_resize:.2f}%   (Δ vs SSA-only: {d_resize:+.2f}%)")
        print(f"    +translation   = {ssa_trans:.2f}%   (Δ vs SSA-only: {d_trans:+.2f}%)")
        print(f"    +full DI       = {ssa_di:.2f}%   (Δ vs SSA-only: {d_di:+.2f}%)")

        # Hypotheses: P1 resize >> translation magnitude; P2 translation ~ 0
        p1_pass = abs(d_resize) > 2 * abs(d_trans) if abs(d_trans) > 0.1 else abs(d_resize) > 1.0
        p2_pass = abs(d_trans) < 3.0
        print(f"    P1 (|Δresize| >> |Δtranslation|): {'PASS' if p1_pass else 'FAIL'}")
        print(f"    P2 (|Δtranslation| < 3%):         {'PASS' if p2_pass else 'FAIL'}")

    print(f"\n  Total runtime: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"\n[Saved] {raw_csv}")
    print(f"[Saved] {pivot_csv}")
    print("[Exp F] Done.")


if __name__ == "__main__":
    main()
