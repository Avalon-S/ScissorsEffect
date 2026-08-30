#!/usr/bin/env python
"""
Exp H: Same-Family (RN50→RN50) Source-Target Overlap Ablation

that the Scissors Effect could be confounded with cross-family architectural
distance, since main analyses use disjoint families RN50 → ViT-B/Swin-B/etc.).

Tests whether the Scissors asymmetry (DI helps Std, hurts Robust) survives
when source and target share architecture. Pairs:

  (1) Std RN50      → Salman ε=2 RN50    [same-family RN50→RN50, robust tgt]
  (2) Salman ε=2    → ResNet50 (Std)     [same-family RN50→RN50, std tgt]

For comparison, we also run the cross-family Swin-B target as a reproducible
baseline of Exp B (Std→Swin-B: D_ASR=+13%, Salman ε=2→Swin-B: D_ASR=-12.5%).

Expected: the same-family D_ASR signs match cross-family signs (DI helps Std
source regardless of target family; DI hurts Robust source regardless).
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


# (source, label, target, target_label, family_relation)
PAIRS_DEFAULT = [
    ("ResNet50",      "Standard RN50",       "Salman_eps2.0",                    "Salman ε=2 RN50",     "same-family RN50→RN50 (robust tgt)"),
    ("ResNet50",      "Standard RN50",       "Engstrom2019Robustness_ImageNet",  "Engstrom RN50",       "same-family RN50→RN50 (robust tgt #2)"),
    ("ResNet50",      "Standard RN50",       "Swin_B_ImageNet",                   "Swin-B",              "cross-family (baseline, Exp B)"),
    ("Salman_eps2.0", "Salman ε=2 RN50",     "ResNet50",                          "Standard RN50",       "same-family RN50→RN50 (std tgt)"),
    ("Salman_eps2.0", "Salman ε=2 RN50",     "Swin_B_ImageNet",                   "Swin-B",              "cross-family (baseline, Exp B)"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps_attack", type=float, default=16/255)
    p.add_argument("--alpha", type=float, default=2/255)
    p.add_argument("--diversity_prob", type=float, default=1.0)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="results/self_transfer_h")
    return p.parse_args()


def attack_and_eval(src_model, tgt_model, x, y, attack_name, args, device):
    if attack_name == "mifgsm":
        attack = get_attack("mifgsm", src_model,
                            eps=args.eps_attack, alpha=args.alpha,
                            steps=args.steps)
    elif attack_name == "difgsm":
        attack = get_attack("difgsm", src_model,
                            eps=args.eps_attack, alpha=args.alpha,
                            steps=args.steps,
                            resize_rate=args.resize_rate,
                            diversity_prob=args.diversity_prob)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    n_correct = 0
    n_total = 0
    pbar = tqdm(range(0, x.size(0), args.batch_size),
                desc=f"    {attack_name}", leave=False)
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

    print("=" * 78)
    print("Exp H: Same-Family (RN50→RN50) Ablation")
    print("=" * 78)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, steps={args.steps}")
    print(f"  eps_attack={args.eps_attack:.4f}, alpha={args.alpha:.4f}")
    print(f"  DI: p={args.diversity_prob}, r={args.resize_rate}")
    print()
    print(f"  Pairs ({len(PAIRS_DEFAULT)}):")
    for s, sl, t, tl, rel in PAIRS_DEFAULT:
        print(f"    [{rel}] {sl} → {tl}")
    print()

    seed_everything(42)
    print(f"[Data] Loading ImageNet ({args.n_examples} images)...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)

    # Cache loaded models to avoid redundant reload
    model_cache = {}

    def load_or_get(name):
        if name not in model_cache:
            print(f"  [Loading] {name}...")
            m = get_model(name, dataset="imagenet", device=device)
            m.eval()
            model_cache[name] = m
        return model_cache[name]

    results = []
    overall_t0 = time.time()

    for src_name, src_label, tgt_name, tgt_label, relation in PAIRS_DEFAULT:
        print(f"\n[Pair] [{relation}]")
        print(f"        Source: {src_label} ({src_name})")
        print(f"        Target: {tgt_label} ({tgt_name})")

        try:
            src_model = load_or_get(src_name)
            tgt_model = load_or_get(tgt_name)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        for attack_name in ["mifgsm", "difgsm"]:
            asr_seeds = []
            for seed_idx in range(args.n_seeds):
                seed_everything(42 + seed_idx * 100)
                t_start = time.time()
                asr = attack_and_eval(src_model, tgt_model, x_test, y_test,
                                       attack_name, args, device)
                elapsed = time.time() - t_start
                asr_seeds.append(asr)
                print(f"  {attack_name}  seed={42+seed_idx*100:<5}  "
                      f"ASR={asr*100:6.2f}%  t={elapsed:.1f}s")

                # MI-FGSM is deterministic — single seed sufficient
                if attack_name == "mifgsm":
                    break

            mean_asr = float(np.mean(asr_seeds))
            std_asr = float(np.std(asr_seeds, ddof=1)) if len(asr_seeds) > 1 else 0.0
            results.append({
                "source": src_name,
                "source_label": src_label,
                "target": tgt_name,
                "target_label": tgt_label,
                "relation": relation,
                "attack": attack_name,
                "asr_mean": mean_asr,
                "asr_std": std_asr,
                "n_seeds": len(asr_seeds),
                "n_examples": args.n_examples,
            })

    overall_elapsed = time.time() - overall_t0

    if not results:
        print("[Error] No results.")
        return

    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.results_dir, "self_transfer_raw.csv")
    df.to_csv(raw_csv, index=False)

    # Pivot: (source, target, relation) × attack
    pivot = df.pivot_table(
        index=["source_label", "target_label", "relation"],
        columns="attack",
        values="asr_mean",
    ).reset_index()
    pivot["delta_di_minus_mi"] = pivot.get("difgsm", 0) - pivot.get("mifgsm", 0)
    pivot_csv = os.path.join(args.results_dir, "self_transfer_pivot.csv")
    pivot.to_csv(pivot_csv, index=False)

    print("\n" + "=" * 100)
    print("SUMMARY  (ASR %, target-side; D_ASR = ASR(DI) − ASR(MI))")
    print("=" * 100)
    print(f"{'Relation':<42}{'Source':<22}{'Target':<22}{'MI':>8}{'DI':>8}{'D_ASR':>10}")
    print("-" * 100)
    for _, r in pivot.iterrows():
        d = r["delta_di_minus_mi"] * 100
        print(f"{r['relation']:<42}{r['source_label']:<22}{r['target_label']:<22}"
              f"{r['mifgsm']*100:>7.2f}%{r['difgsm']*100:>7.2f}%"
              f"{d:>+9.2f}%")

    print("\n" + "=" * 100)
    print("ABLATION VERDICT")
    print("=" * 100)

    # Group by source for verdict
    by_src = df.copy()
    by_src["asr_pct"] = by_src["asr_mean"] * 100
    for src_name in by_src["source"].unique():
        sub = by_src[by_src["source"] == src_name]
        sf = sub[sub["relation"].str.contains("same-family")]
        cf = sub[sub["relation"].str.contains("cross-family")]
        if sf.empty or cf.empty:
            continue
        sf_mi = sf[sf["attack"] == "mifgsm"]["asr_pct"].iloc[0]
        sf_di = sf[sf["attack"] == "difgsm"]["asr_pct"].iloc[0]
        cf_mi = cf[cf["attack"] == "mifgsm"]["asr_pct"].iloc[0]
        cf_di = cf[cf["attack"] == "difgsm"]["asr_pct"].iloc[0]
        sf_d = sf_di - sf_mi
        cf_d = cf_di - cf_mi
        sign_match = "MATCH" if (sf_d * cf_d > 0) else "MISMATCH"
        sub_label = sub["source_label"].iloc[0]
        print(f"\n  [Source: {sub_label}]")
        print(f"    Same-family    D_ASR = {sf_d:+.2f}%")
        print(f"    Cross-family   D_ASR = {cf_d:+.2f}%")
        print(f"    Sign agreement: {sign_match}")
        if sign_match == "MATCH":
            print("    ==> same-family sign matches cross-family; "
                  "overlap is not the confound")
        else:
            print("    ==> same-family sign differs from cross-family; "
                  "overlap may matter here")

    print(f"\n  Total runtime: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"\n[Saved] {raw_csv}")
    print(f"[Saved] {pivot_csv}")
    print("[Exp H] Done.")


if __name__ == "__main__":
    main()
