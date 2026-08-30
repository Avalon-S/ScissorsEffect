#!/usr/bin/env python
"""
Robustness-strength epsilon sweep.

Tests how DI's effect on transfer attack success rate evolves as the surrogate's
training epsilon increases. Uses Salman et al.'s Linf ResNet50 spectrum
(eps in {0, 0.5, 1, 2, 4, 8}/255), all sharing identical ResNet50 architecture
and the same PGD-AT recipe -- enabling a CONTROLLED sweep that isolates
robustness strength as the only varying factor.

For each source eps and each attack (MI-FGSM vs DI-FGSM with p=1):
  1. Generate adversarial examples on the source model (ResNet50_eps_X)
  2. Evaluate on a fixed target (default: Swin-B)
  3. Report ASR = fraction of adv examples that fool the target

Reports the SCISSORS curve: D_ASR(eps) = ASR(DI) - ASR(MI). Expected behavior:
  - eps = 0:           D_ASR > 0  (DI helps standard surrogate)
  - eps in [0, 4]/255: D_ASR transitions monotonically from positive to negative
  - eps >= 4/255:      D_ASR < 0  (DI harms robust surrogate, paper Tab. 1)

robustness-strength sweep"; close to Zhang et al. S&P'24 framing).
"""
import argparse
import json
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


# Source ε spectrum: standard ResNet50 + Salman ε ∈ {0.5, 1, 2, 4, 8}/255
SOURCES = [
    ("ResNet50",       "Standard (eps=0)",       0.0 / 255),
    ("Salman_eps0.5",  "Salman (eps=0.5/255)",   0.5 / 255),
    ("Salman_eps1.0",  "Salman (eps=1/255)",     1.0 / 255),
    ("Salman_eps2.0",  "Salman (eps=2/255)",     2.0 / 255),
    ("Salman_eps4.0",  "Salman (eps=4/255)",     4.0 / 255),
    ("Salman_eps8.0",  "Salman (eps=8/255)",     8.0 / 255),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10,
                   help="attack iterations (paper default: 10)")
    p.add_argument("--eps_attack", type=float, default=16/255,
                   help="attack budget (paper ImageNet default: 16/255)")
    p.add_argument("--alpha", type=float, default=2/255,
                   help="attack step size (paper default: 2/255)")
    p.add_argument("--diversity_prob", type=float, default=1.0,
                   help="DI-FGSM probability (1.0 = always apply DI)")
    p.add_argument("--resize_rate", type=float, default=0.9,
                   help="DI-FGSM resize rate (paper default 0.9)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--target", type=str, default="Swin_B_ImageNet",
                   help="target architecture for transfer eval")
    p.add_argument("--results_dir", type=str, default="results/eps_sweep")
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help="Override the default source list")
    return p.parse_args()


def attack_and_eval(src_model, tgt_model, x, y, attack_name, args, device):
    """
    Run attack on source, evaluate on target.
    Returns:
        asr:    fraction of adv examples that the target misclassifies
                (i.e., 1 - target accuracy on adv examples)
        n_eval: number of images attacked
    """
    if attack_name == "mifgsm":
        attack = get_attack(
            "mifgsm", src_model,
            eps=args.eps_attack, alpha=args.alpha, steps=args.steps,
        )
    elif attack_name == "difgsm":
        attack = get_attack(
            "difgsm", src_model,
            eps=args.eps_attack, alpha=args.alpha, steps=args.steps,
            resize_rate=args.resize_rate,
            diversity_prob=args.diversity_prob,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    n_correct = 0
    n_total = 0
    pbar = tqdm(range(0, x.size(0), args.batch_size),
                desc=f"    {attack_name}",
                leave=False)
    for i in pbar:
        xb = x[i:i + args.batch_size].to(device)
        yb = y[i:i + args.batch_size].to(device)
        x_adv = attack(xb, yb)
        with torch.no_grad():
            preds = tgt_model(x_adv).argmax(dim=1)
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)

    target_accuracy = n_correct / n_total
    asr = 1.0 - target_accuracy
    return asr, n_total


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78)
    print("Exp B: Robustness-Strength Epsilon Sweep")
    print("=" * 78)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, steps={args.steps}")
    print(f"  eps_attack={args.eps_attack:.4f}={args.eps_attack*255:.1f}/255, "
          f"alpha={args.alpha:.4f}")
    print(f"  DI: diversity_prob={args.diversity_prob}, "
          f"resize_rate={args.resize_rate}")
    print(f"  Target: {args.target}")
    print()

    # Source list
    if args.sources is not None:
        # Allow command-line override (e.g. for debugging)
        all_sources = {n: (lbl, eps) for n, lbl, eps in SOURCES}
        sources = []
        for s in args.sources:
            if s in all_sources:
                sources.append((s, all_sources[s][0], all_sources[s][1]))
            else:
                sources.append((s, s, float("nan")))
    else:
        sources = SOURCES

    print(f"  Sources ({len(sources)}):")
    for n, lbl, eps in sources:
        print(f"    - {n:<22}  [{lbl}, eps_train={eps*255:.1f}/255]")
    print()

    # Load target ONCE (reuse across all source-attack-seed combinations)
    print(f"[Target] Loading {args.target}...")
    tgt_model = get_model(args.target, dataset="imagenet", device=device)
    tgt_model.eval()

    # Load data ONCE (same images for all conditions for clean comparison)
    seed_everything(42)
    print(f"[Data] Loading ImageNet validation ({args.n_examples} images)...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    print(f"  Loaded {x_test.size(0)} images, shape={tuple(x_test.shape[1:])}")

    # Pre-compute target clean accuracy on this set (sanity check)
    print("[Sanity] Computing target clean accuracy...")
    n_correct = 0
    for i in range(0, x_test.size(0), args.batch_size):
        xb = x_test[i:i + args.batch_size].to(device)
        yb = y_test[i:i + args.batch_size].to(device)
        with torch.no_grad():
            preds = tgt_model(xb).argmax(dim=1)
        n_correct += (preds == yb).sum().item()
    target_clean_acc = n_correct / x_test.size(0)
    print(f"  Target {args.target} clean acc on this set: "
          f"{target_clean_acc*100:.2f}%")

    results = []
    overall_t0 = time.time()

    for src_name, src_label, eps_val in sources:
        print(f"\n[Source] {src_name} -- {src_label} (eps_train="
              f"{eps_val*255:.1f}/255)")
        try:
            src_model = get_model(src_name, dataset="imagenet", device=device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED to load source: {e}")
            continue

        # Source clean accuracy (sanity)
        n_correct = 0
        for i in range(0, x_test.size(0), args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            with torch.no_grad():
                preds = src_model(xb).argmax(dim=1)
            n_correct += (preds == yb).sum().item()
        src_clean_acc = n_correct / x_test.size(0)
        print(f"  Source clean acc: {src_clean_acc*100:.2f}%")

        for attack_name in ["mifgsm", "difgsm"]:
            asr_per_seed = []
            for seed_idx in range(args.n_seeds):
                attack_seed = 42 + seed_idx * 100
                seed_everything(attack_seed)
                t_start = time.time()
                asr, n_eval = attack_and_eval(
                    src_model, tgt_model, x_test, y_test,
                    attack_name, args, device,
                )
                elapsed = time.time() - t_start
                asr_per_seed.append(asr)
                print(f"  {attack_name:8s}  seed={attack_seed:5d}  "
                      f"ASR={asr*100:6.2f}%   t={elapsed:.1f}s")

                # MI-FGSM is deterministic; one seed is enough
                if attack_name == "mifgsm":
                    break

            mean_asr = float(np.mean(asr_per_seed))
            std_asr = float(np.std(asr_per_seed, ddof=1)
                            if len(asr_per_seed) > 1 else 0.0)
            sem_asr = std_asr / np.sqrt(len(asr_per_seed))

            results.append({
                "source": src_name,
                "source_label": src_label,
                "eps_train_x255": eps_val * 255,
                "eps_train": eps_val,
                "attack": attack_name,
                "asr_mean": mean_asr,
                "asr_std": std_asr,
                "asr_sem": sem_asr,
                "n_seeds": len(asr_per_seed),
                "n_examples": args.n_examples,
                "src_clean_acc": src_clean_acc,
                "target": args.target,
                "target_clean_acc": target_clean_acc,
            })

        del src_model
        torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_t0

    if not results:
        print("[Error] No results produced.")
        return

    # Save raw results
    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.results_dir, "eps_sweep_raw.csv")
    df.to_csv(raw_csv, index=False)

    # Pivot for easy reading: row per source, columns for each attack
    pivot = df.pivot_table(
        index=["source", "source_label", "eps_train", "eps_train_x255"],
        columns="attack",
        values="asr_mean",
    ).reset_index()
    pivot.columns.name = None
    if "mifgsm" in pivot.columns and "difgsm" in pivot.columns:
        pivot["delta_di_minus_mi"] = pivot["difgsm"] - pivot["mifgsm"]
        pivot = pivot.sort_values("eps_train")

    pivot_csv = os.path.join(args.results_dir, "eps_sweep_pivot.csv")
    pivot.to_csv(pivot_csv, index=False)

    # Pretty print
    print("\n" + "=" * 90)
    print("SUMMARY  (target=" + args.target + ", all values in %)")
    print("=" * 90)
    print(f"{'Source':<28}{'eps':>10}{'MI-FGSM':>12}{'DI-FGSM':>12}"
          f"{'D=DI-MI':>12}")
    print("-" * 90)
    for _, r in pivot.iterrows():
        print(f"{r['source_label']:<28}{r['eps_train_x255']:>9.1f}/255"
              f"{r['mifgsm']*100:>+12.2f}{r['difgsm']*100:>+12.2f}"
              f"{r['delta_di_minus_mi']*100:>+12.2f}")

    # Crossover detection
    print("\n" + "=" * 90)
    print("SCISSORS CURVE ANALYSIS")
    print("=" * 90)
    deltas = pivot[["eps_train_x255", "delta_di_minus_mi"]].values
    print("  eps (in /255)    D_ASR (DI - MI)")
    for eps, d in deltas:
        marker = "POS" if d > 0 else ("NEG" if d < 0 else "ZERO")
        print(f"    {eps:5.1f}            {d*100:+7.2f}%   [{marker}]")

    # Find crossover
    signs = np.sign(deltas[:, 1])
    crossover = None
    for i in range(len(signs) - 1):
        if signs[i] * signs[i + 1] < 0:
            crossover = (deltas[i, 0], deltas[i + 1, 0])
            break
    if crossover:
        print(f"\n  CROSSOVER detected: D_ASR changes sign between "
              f"eps in [{crossover[0]:.1f}, {crossover[1]:.1f}]/255")
    else:
        if all(d > 0 for _, d in deltas):
            print("\n  D_ASR > 0 for all eps  (DI uniformly helps -- unexpected)")
        elif all(d < 0 for _, d in deltas):
            print("\n  D_ASR < 0 for all eps  (DI uniformly hurts -- unexpected)")
        else:
            print("\n  No clean crossover found")

    print(f"\n  Total runtime: {overall_elapsed:.1f}s "
          f"({overall_elapsed/60:.1f} min)")
    print(f"\n[Saved] {raw_csv}")
    print(f"[Saved] {pivot_csv}")
    print("[Exp B] Done.")


if __name__ == "__main__":
    main()
