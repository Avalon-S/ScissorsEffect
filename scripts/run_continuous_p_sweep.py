#!/usr/bin/env python
"""
Exp E: Continuous p-sweep (CG-DI as projection of continuous LGC-curve)

For one robust source × one target, run DI-FGSM at p ∈ {0, 0.1, 0.2, ..., 1.0}
to map out the ASR(p) curve. Show that:
  1. The empirical optimum p* on this curve aligns with the LGC-predicted
     binary choice (low LGC -> p*=0.8 region; high LGC -> p*=0 region)
  2. CG-DI's binary p ∈ {0, 0.8} is a sensible projection of the continuous
     curve, not a heuristic guess
  3. The continuous LGC-D_ASR relationship (paper Sec 4.4 ρ=-0.75) plays out
     directly within a single source's p-sweep

disproof: shows LGC tracks the optimum on a continuous grid).
Also serves R1-Sug C (LGC scope: fine-grained quantitative predictor).

Default config: 1 robust source (Salman ε=2/255 — has clearest DI harm in
Exp B at -12.5%) + 1 standard source (ResNet50) + Swin-B target.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.wrappers import get_attack
from utils import seed_everything, get_device


SOURCES_DEFAULT = [
    ("ResNet50",      "Standard (ε=0)",       0.0/255),
    ("Salman_eps2.0", "Salman (ε=2/255)",     2.0/255),  # max DI harm in Exp B
    ("Engstrom2019Robustness_ImageNet", "Engstrom (ε=4/255)", 4.0/255),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps_attack", type=float, default=16/255)
    p.add_argument("--alpha", type=float, default=2/255)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--p_grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--target", type=str, default="Swin_B_ImageNet")
    p.add_argument("--results_dir", type=str, default="results/continuous_p_sweep")
    p.add_argument("--sources", type=str, nargs="+", default=None)
    return p.parse_args()


def attack_and_eval(src_model, tgt_model, x, y, p_value, args, device):
    """Run DI-FGSM with given diversity_prob, evaluate on target. Returns ASR."""
    if p_value == 0.0:
        # No DI: pure MI-FGSM
        attack = get_attack("mifgsm", src_model,
                            eps=args.eps_attack, alpha=args.alpha,
                            steps=args.steps)
    else:
        attack = get_attack("difgsm", src_model,
                            eps=args.eps_attack, alpha=args.alpha,
                            steps=args.steps,
                            resize_rate=args.resize_rate,
                            diversity_prob=p_value)

    n_correct = 0
    n_total = 0
    pbar = tqdm(range(0, x.size(0), args.batch_size),
                desc=f"    p={p_value:.1f}", leave=False)
    for i in pbar:
        xb = x[i:i + args.batch_size].to(device)
        yb = y[i:i + args.batch_size].to(device)
        x_adv = attack(xb, yb)
        with torch.no_grad():
            preds = tgt_model(x_adv).argmax(dim=1)
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)
    return 1.0 - n_correct / n_total


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    p_grid = [float(x) for x in args.p_grid.split(",")]

    sources = SOURCES_DEFAULT
    if args.sources:
        all_sources = {n: (lbl, eps) for n, lbl, eps in SOURCES_DEFAULT}
        sources = [(s, *all_sources.get(s, (s, float("nan")))) for s in args.sources]

    print("=" * 72)
    print("Exp E: Continuous p-sweep (DI-FGSM diversity_prob)")
    print("=" * 72)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, steps={args.steps}")
    print(f"  eps_attack={args.eps_attack:.4f}, alpha={args.alpha:.4f}")
    print(f"  resize_rate={args.resize_rate}, target={args.target}")
    print(f"  p_grid: {p_grid}")
    print(f"  Sources ({len(sources)}):")
    for n, lbl, eps in sources:
        print(f"    - {n}  [{lbl}, eps_train={eps*255:.1f}/255]")
    print()

    # Load target once
    print(f"[Target] Loading {args.target}...")
    tgt_model = get_model(args.target, dataset="imagenet", device=device)
    tgt_model.eval()

    # Load data once
    seed_everything(42)
    print(f"[Data] Loading ImageNet ({args.n_examples} images)...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)

    results = []
    overall_t0 = time.time()

    for src_name, src_label, eps_val in sources:
        print(f"\n[Source] {src_name} -- {src_label}")
        try:
            src_model = get_model(src_name, dataset="imagenet", device=device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for p_val in p_grid:
            asr_seeds = []
            for seed_idx in range(args.n_seeds):
                seed_everything(42 + seed_idx * 100)
                t_start = time.time()
                asr = attack_and_eval(src_model, tgt_model, x_test, y_test,
                                       p_val, args, device)
                elapsed = time.time() - t_start
                asr_seeds.append(asr)
                print(f"  p={p_val:.1f}  seed={42+seed_idx*100:<5}  "
                      f"ASR={asr*100:6.2f}%  t={elapsed:.1f}s")

                # MI-FGSM (p=0) is deterministic -- skip multi-seed
                if p_val == 0.0:
                    break

            mean_asr = float(np.mean(asr_seeds))
            std_asr = float(np.std(asr_seeds, ddof=1)) if len(asr_seeds) > 1 else 0.0

            results.append({
                "source": src_name,
                "source_label": src_label,
                "eps_train": eps_val,
                "p": p_val,
                "asr_mean": mean_asr,
                "asr_std": std_asr,
                "n_seeds": len(asr_seeds),
                "n_examples": args.n_examples,
                "target": args.target,
            })

        del src_model
        torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_t0

    if not results:
        print("[Error] No results.")
        return

    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.results_dir, "continuous_p_raw.csv")
    df.to_csv(raw_csv, index=False)

    pivot = df.pivot_table(index="source", columns="p", values="asr_mean")
    pivot_csv = os.path.join(args.results_dir, "continuous_p_pivot.csv")
    pivot.to_csv(pivot_csv)

    # Summary
    print("\n" + "=" * 100)
    print(f"SUMMARY  (target={args.target}, ASR %)")
    print("=" * 100)
    print(f"{'Source':<30}", end="")
    for p in p_grid:
        print(f"{'p='+str(p):>9}", end="")
    print()
    print("-" * 100)

    for src_name, src_label, eps_val in sources:
        if src_name not in pivot.index:
            continue
        print(f"{src_label:<30}", end="")
        for p in p_grid:
            v = pivot.loc[src_name, p] * 100 if p in pivot.columns else float("nan")
            print(f"{v:>+9.2f}", end="")
        print()

    # Find empirical p* (max ASR) per source and the CG-DI binary projection
    print("\n" + "=" * 100)
    print("EMPIRICAL p* vs CG-DI BINARY PROJECTION")
    print("=" * 100)
    for src_name, src_label, eps_val in sources:
        if src_name not in pivot.index:
            continue
        row = pivot.loc[src_name]
        p_optimal = row.idxmax()  # p with highest ASR
        asr_optimal = row.max() * 100
        asr_p0 = row.get(0.0, float("nan")) * 100
        asr_p08 = row.get(0.8, float("nan")) * 100
        # CG-DI would pick p ∈ {0, 0.8}, choose whichever is closer to optimum
        binary_choice = 0.0 if asr_p0 > asr_p08 else 0.8
        binary_asr = max(asr_p0, asr_p08)

        print(f"\n  [{src_label}]")
        print(f"    Continuous optimum:  p* = {p_optimal:.1f}, "
              f"ASR* = {asr_optimal:.2f}%")
        print(f"    CG-DI binary projection: p = {binary_choice:.1f}, "
              f"ASR = {binary_asr:.2f}%")
        gap = asr_optimal - binary_asr
        print(f"    Gap (continuous − binary): {gap:+.2f} pp")
        if gap < 1.0:
            print("    ==> binary projection within 1 pp of the best grid value")
        elif gap < 3.0:
            print("    ==> binary projection within 3 pp of the best grid value")
        else:
            print(f"    ==> Gap > 3pp — continuous p might give meaningful gain.  ?")

    print(f"\n  Total runtime: {overall_elapsed:.1f}s "
          f"({overall_elapsed/60:.1f} min)")
    print(f"\n[Saved] {raw_csv}")
    print(f"[Saved] {pivot_csv}")
    print("[Exp E] Done.")


if __name__ == "__main__":
    main()
