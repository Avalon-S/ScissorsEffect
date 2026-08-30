#!/usr/bin/env python
"""
Task 1 Expanded: Extreme DI Correlation with More Surrogates

Runs on CIFAR-10 with n=14 surrogates to stabilize correlation.
Uses existing LGC values where available, computes new ones for added models.
"""
import argparse
import sys
import os
import torch

# Fix for PyTorch 2.6 - must be before any model loading
try:
    import easydict
    torch.serialization.add_safe_globals([easydict.EasyDict])
except (ImportError, AttributeError):
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.sap import estimate_p_lgc
from attacks.wrappers import get_attack
from utils import seed_everything, compute_metrics, get_device


def compute_model_lgc(model, x_test, y_test, device, n_probe=50, probe_batch=25):
    """Compute LGC statistics for a model with memory-safe batching."""
    all_lgc_vals = []
    
    probe_x = x_test[:min(n_probe, len(x_test))]
    probe_y = y_test[:min(n_probe, len(y_test))]
    
    for i in range(0, len(probe_x), probe_batch):
        xb = probe_x[i:i+probe_batch].to(device)
        yb = probe_y[i:i+probe_batch].to(device)
        
        _, lgc_vals = estimate_p_lgc(model, xb, yb, device, K=5, sigma=1/255)
        all_lgc_vals.append(lgc_vals.cpu())
        
        del xb, yb, lgc_vals
        torch.cuda.empty_cache()
    
    all_lgc_vals = torch.cat(all_lgc_vals).numpy()
    
    return {
        "mean": float(np.mean(all_lgc_vals)),
        "median": float(np.median(all_lgc_vals)),
        "std": float(np.std(all_lgc_vals)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Expanded Extreme DI Correlation")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resize_rate", type=float, default=0.6)
    parser.add_argument("--resize_rates", type=str, default=None,
                        help="Comma-separated list of resize rates (e.g., '0.9,0.8,0.7,0.6'). Overrides --resize_rate.")
    parser.add_argument("--surrogates", type=str, default=None,
                        help="Comma-separated short-name filter (e.g., 'Engstrom,Rice'). "
                             "If set, only these surrogates are run. Note: torchattacks "
                             "DIFGSM needs resize_rate<1 strictly; r=1.0 degenerates on "
                             "small images, so the near-identity regime uses r in {0.95,0.97}.")
    parser.add_argument("--target", type=str, default="Standard",
                        help="Transfer target; must not be one of the "
                             "surrogates. Default is RobustBench's naturally "
                             "trained WideResNet-28-10.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="results/extreme_di_expanded")
    parser.add_argument("--lgc_csv", type=str, 
                        default="results/tier3_pilot_cifar/lgc_pstar_pilot.csv")
    return parser.parse_args()


def get_expanded_surrogates():
    """
    Expanded CIFAR-10 surrogates: n=14
    Original 9 + 5 new models
    """
    return [
        # Standard models (4). These keys resolve to genuine CIFAR-10 models
        # trained by scripts/train_cifar10_standard.py, not ImageNet checkpoints.
        ("ResNet18", "ResNet18 (Standard)", "ResNet18_Standard", "standard"),
        ("VGG16", "VGG16 (Standard)", "VGG16_Standard", "standard"),
        ("DenseNet121", "DenseNet121 (Standard)", "DenseNet121_Standard", "standard"),
        ("ResNet50", "ResNet50 (Standard)", "ResNet50_Standard", "standard"),
        # ViT-B/16 dropped from the CIFAR-10 pool: the only checkpoint available
        # is ImageNet-pretrained, and we do not train a ViT on CIFAR-10 from
        # scratch. ViT remains in the ImageNet pool, where it matters.

        # Original Robust models (6)
        ("Engstrom", "Engstrom (Robust)", "Engstrom2019Robustness", "robust"),
        ("Rice", "Rice (Robust)", "Rice2020Overfitting", "robust"),
        ("Gowal", "Gowal (Robust)", "Gowal2020Uncovering_70_16", "robust"),
        ("Carmon", "Carmon (Robust)", "Carmon2019Unlabeled", "robust"),
        ("Peng", "Peng (Robust)", "Peng2023Robust", "robust"),
        ("Wang", "Wang (Robust)", "Wang2023Better_WRN-28-10", "robust"),
        
        # NEW Robust models (3)
        ("Zhang", "Zhang (TRADES)", "Zhang2019Theoretically", "robust"),
        ("Sehwag", "Sehwag (Proxy)", "Sehwag2021Proxy", "robust"),
        ("Sehwag_R18", "Sehwag-R18 (Proxy)", "Sehwag2021Proxy_R18", "robust"),
    ]


def compute_asr(model, x_test, y_test, target_model, device, args, di_prob, resize_rate):
    """Compute ASR for a specific DI setting."""
    if di_prob == 0:
        attacker = get_attack("mifgsm", model, eps=args.eps, steps=args.steps)
    else:
        attacker = get_attack("dim", model, eps=args.eps, steps=args.steps,
                              diversity_prob=di_prob, resize_rate=resize_rate)
    
    x_adv_list = []
    for i in range(0, len(x_test), args.batch_size):
        xb = x_test[i:i+args.batch_size]
        yb = y_test[i:i+args.batch_size]
        x_adv_batch = attacker(xb, yb)
        x_adv_list.append(x_adv_batch.cpu())
    
    x_adv = torch.cat(x_adv_list)
    metrics = compute_metrics(x_adv, x_test, y_test, target_model, device)
    return metrics['asr']


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*70)
    print("Expanded Extreme DI Correlation (n=14)")
    print("="*70)
    print(f"N={args.n_examples}, Seeds={args.n_seeds}")
    
    # Parse resize rates
    if args.resize_rates:
        resize_rates = [float(r) for r in args.resize_rates.split(',')]
    else:
        resize_rates = [args.resize_rate]
    print(f"Resize rates: {resize_rates}")
    
    # Load existing LGC data
    lgc_dict = {}
    if os.path.exists(args.lgc_csv):
        try:
            lgc_df = pd.read_csv(args.lgc_csv)
            lgc_dict = dict(zip(lgc_df["surrogate"], lgc_df["lgc_mean"]))
            print(f"Loaded existing LGC data: {list(lgc_dict.keys())}")
        except Exception as e:
            print(f"Could not load LGC CSV ({e}), will compute fresh.")
    
    surrogates = get_expanded_surrogates()
    if args.surrogates:
        wanted = [s.strip() for s in args.surrogates.split(",")]
        surrogates = [s for s in surrogates if s[0] in wanted]
        print(f"Surrogate filter active: {wanted} -> {[s[0] for s in surrogates]}")
    print(f"\nTotal surrogates: {len(surrogates)}")
    
    # Load data
    seed_everything(42)
    x_test, y_test = load_dataset("cifar10", args.n_examples)
    
    # The target must be disjoint from the surrogate pool below, or that
    # surrogate's row is a white-box attack sitting among transfer results.
    target_name = args.target
    pool_keys = {s[2] for s in surrogates}
    if target_name in pool_keys:
        raise SystemExit(
            f"target {target_name} is also a surrogate: its row would be "
            f"white-box, not transfer. Choose a target outside the pool.")
    target_model = get_model(target_name, "cifar10", "Linf", device)
    print(f"Target: {target_name} (disjoint from the {len(surrogates)}-model pool)")
    
    results = []
    
    for short_name, display_name, model_key, model_type in surrogates:
        print(f"\n{'='*50}")
        print(f"Surrogate: {display_name}")
        print(f"{'='*50}")
        
        try:
            model = get_model(model_key, "cifar10", "Linf", device)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
        
        # Get or compute LGC
        if short_name in lgc_dict:
            lgc = lgc_dict[short_name]
            print(f"  Using cached LGC: {lgc:.4f}")
        else:
            print(f"  Computing LGC...")
            lgc_stats = compute_model_lgc(model, x_test[:200], y_test[:200], device)
            lgc = lgc_stats["mean"]
            print(f"  LGC = {lgc:.4f}")
        
        # Multi-seed ASR evaluation
        results_per_rate = {r: {"p0": [], "extreme": []} for r in resize_rates}
        
        for seed_idx in range(args.n_seeds):
            seed = 100 * (seed_idx + 1)
            seed_everything(seed)
            
            # p=0 baseline (independent of resize rate, using standard MI-FGSM)
            asr_p0 = compute_asr(model, x_test, y_test, target_model, device,
                                 args, di_prob=0, resize_rate=resize_rates[0])
            
            for r in resize_rates:
                asr_extreme = compute_asr(model, x_test, y_test, target_model, device,
                                          args, di_prob=0.8, resize_rate=r)
                
                results_per_rate[r]["p0"].append(asr_p0)
                results_per_rate[r]["extreme"].append(asr_extreme)
                
            print(f"  Seed {seed}: p=0: {asr_p0:.1f}%, extreme (r={resize_rates[0]}): {results_per_rate[resize_rates[0]]['extreme'][-1]:.1f}%")
        
        # Aggregate
        for r in resize_rates:
            mean_p0 = np.mean(results_per_rate[r]["p0"])
            mean_extreme = np.mean(results_per_rate[r]["extreme"])
            delta_asr = mean_extreme - mean_p0
            
            results.append({
                "surrogate": short_name,
                "display_name": display_name,
                "model_type": model_type,
                "lgc": lgc,
                "resize_rate": r,
                "asr_p0": mean_p0,
                "asr_extreme": mean_extreme,
                "delta_asr_extreme": delta_asr,
            })
            print(f"  --> r={r}: ΔASR = {delta_asr:+.2f}%")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.results_dir, "expanded_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Correlation Analysis per Resize Rate
    print("\n" + "="*70)
    print(f"Correlation Analysis (n={len(surrogates)})")
    print("="*70)
    
    for r in resize_rates:
        print(f"\n--- Resize Rate r={r} ---")
        sub_df = df[df["resize_rate"] == r]
        
        lgc_values = sub_df["lgc"].values
        delta_asr_values = sub_df["delta_asr_extreme"].values
        
        spearman_rho, spearman_p = stats.spearmanr(lgc_values, delta_asr_values)
        pearson_r, pearson_p = stats.pearsonr(lgc_values, delta_asr_values)
        
        print(f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.4f}")
        print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4f}")
        
        sig = "**" if spearman_p < 0.05 else "*" if spearman_p < 0.1 else ""
        
        # Generate Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        colors = {"standard": "blue", "robust": "red", "anomaly": "orange"}
        markers = {"standard": "o", "robust": "s", "anomaly": "D"}
        
        for _, row in sub_df.iterrows():
            c = colors.get(row["model_type"], "gray")
            m = markers.get(row["model_type"], "o")
            ax.scatter(row["lgc"], row["delta_asr_extreme"], 
                       c=c, marker=m, s=150, edgecolors='black', linewidths=1.5, zorder=5)
            ax.annotate(row["surrogate"], 
                        (row["lgc"], row["delta_asr_extreme"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Trend line
        z = np.polyfit(lgc_values, delta_asr_values, 1)
        p_trend = np.poly1d(z)
        x_line = np.linspace(min(lgc_values) - 0.02, max(lgc_values) + 0.02, 100)
        ax.plot(x_line, p_trend(x_line), "--", color="gray", alpha=0.7,
                label=f"Linear fit (r={pearson_r:.2f})")
        
        ax.set_xlabel("LGC (Local Gradient Consistency)", fontsize=12)
        ax.set_ylabel(f"ΔASR (p=0.8, r={r}) - ASR(p=0)", fontsize=12)
        ax.set_title(f"CIFAR-10: LGC vs DI Harm (r={r}, n={len(sub_df)})\n"
                     f"Spearman rho={spearman_rho:.2f}, p={spearman_p:.3f}{sig}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[t], 
                             markersize=10, label=t.capitalize(), markeredgecolor='k') 
                   for t in ["standard", "robust", "anomaly"]]
        ax.legend(handles=handles, loc="upper right")
        
        plt.tight_layout()
        plot_path = os.path.join(args.results_dir, f"lgc_vs_delta_asr_r{r}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
