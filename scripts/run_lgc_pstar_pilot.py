#!/usr/bin/env python
"""Pilot: does LGC predict the optimal DI probability p*, not just its sign?

Five surrogates: ResNet18, VGG16 and DenseNet121 (standard CNNs), ViT-B/16
(standard ViT) and Engstrom2019Robustness (robust CNN).

Outputs per-surrogate LGC, the ASR-vs-p sweep, an LGC-vs-p* scatter, and the
Spearman/Pearson correlation. The pilot criterion for pursuing a graded
predictor is |rho| >= 0.5 at p < 0.1; the correlation analysis in the paper uses
the full surrogate pools instead.
"""
import argparse
import sys
import os
import torch
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


def parse_args():
    parser = argparse.ArgumentParser(description="LGC vs p* Pilot Experiment")
    parser.add_argument("--n_examples", type=int, default=1000, help="Number of examples")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=None, help="Perturbation budget (auto: 8/255 CIFAR-10, 16/255 ImageNet)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--results_dir", type=str, default="results/tier3_pilot")
    return parser.parse_args()


def get_surrogates(dataset):
    """Define surrogate models - using ONLY locally available models."""
    if dataset == "cifar10":
        return [
            # Standard CNNs (torchvision - confirmed available)
            ("ResNet18", "ResNet18 (Standard)", "ResNet18_Standard", "Linf", "standard"),
            ("VGG16", "VGG16 (Standard)", "VGG16_Standard", "Linf", "standard"),
            ("ResNet50", "ResNet50 (Standard)", "ResNet50_Standard", "Linf", "standard"),
            
            ("DenseNet121", "DenseNet121 (Standard)", "DenseNet121_Standard", "Linf", "standard"),
            
            # Robust CNNs (RobustBench cifar10/Linf)
            # File names: Carmon2019Unlabeled.pt, Engstrom2019Robustness.pt, 
            # Gowal2020Uncovering_70_16.pt, Peng2023Robust.pt, Rice2020Overfitting.pt,
            # Wang2023Better_WRN-28-10.pt
            ("Engstrom", "Engstrom (Robust)", "Engstrom2019Robustness", "Linf", "robust"),
            ("Rice", "Rice (Robust)", "Rice2020Overfitting", "Linf", "robust"),
            ("Gowal", "Gowal (Robust)", "Gowal2020Uncovering_70_16", "Linf", "robust"),
            ("Carmon", "Carmon (Robust)", "Carmon2019Unlabeled", "Linf", "robust"),
            ("Peng", "Peng (Robust)", "Peng2023Robust", "Linf", "robust"),
            ("Wang", "Wang (Robust)", "Wang2023Better_WRN-28-10", "Linf", "robust"),
        ]
    else:  # imagenet
        return [
            # Standard models from torchvision (defined in loader.py)
            ("ResNet50", "ResNet50 (Standard)", "ResNet50", "Linf", "standard"),
            ("InceptionV3", "InceptionV3 (Standard)", "InceptionV3", "Linf", "standard"),
            ("ViT", "ViT-B/16 (Standard)", "ViT_B_16_ImageNet", "Linf", "standard"),
            ("Swin", "Swin-B (Standard)", "Swin_B_ImageNet", "Linf", "standard"),
            ("ConvNeXt", "ConvNeXt-B (Standard)", "ConvNeXt_B_ImageNet", "Linf", "standard"),
            ("DenseNet121", "DenseNet121 (Standard)", "DenseNet121_Standard", "Linf", "standard"),
            
            # Robust (RobustBench, matching models/loader.py)
            ("Engstrom", "Engstrom (Robust)", "Engstrom2019Robustness_ImageNet", "Linf", "robust"),
            ("Salman", "Salman (Robust)", "Salman2020Do_R50", "Linf", "robust"),
            ("Mo", "Mo (Robust)", "Mo2022When_ViT-B", "Linf", "robust"),
        ]


def compute_model_lgc(model, x_test, y_test, device, n_probe=50, probe_batch=25):
    """Compute LGC statistics for a model with memory-safe batching."""
    all_lgc_vals = []

    probe_x = x_test[:min(n_probe, len(x_test))]
    probe_y = y_test[:min(n_probe, len(y_test))]
    n_batches = (len(probe_x) + probe_batch - 1) // probe_batch

    # Process in small batches to avoid OOM
    for i in tqdm(range(0, len(probe_x), probe_batch),
                  total=n_batches, desc="LGC", leave=False):
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
        "min": float(np.min(all_lgc_vals)),
        "max": float(np.max(all_lgc_vals)),
    }


def run_p_sweep(model, x_test, y_test, target_model, device, args, p_values, desc_prefix=""):
    """Run p-sweep and return ASR for each p."""
    results = {}
    n_batches = (len(x_test) + args.batch_size - 1) // args.batch_size

    for p in p_values:
        attacker = get_attack("dim", model, eps=args.eps, steps=args.steps,
                              diversity_prob=p, resize_rate=0.9)

        x_adv_list = []
        for i in tqdm(range(0, len(x_test), args.batch_size),
                      total=n_batches, desc=f"{desc_prefix}p={p}", leave=False):
            xb = x_test[i:i+args.batch_size]
            yb = y_test[i:i+args.batch_size]
            x_adv_batch = attacker(xb, yb)
            x_adv_list.append(x_adv_batch.cpu())

        x_adv = torch.cat(x_adv_list)
        metrics = compute_metrics(x_adv, x_test, y_test, target_model, device)
        results[p] = metrics['asr']

    return results


def find_optimal_p(p_sweep_results):
    """Find p* (p with maximum ASR)."""
    best_p = max(p_sweep_results, key=p_sweep_results.get)
    best_asr = p_sweep_results[best_p]
    return best_p, best_asr


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    # Auto-select eps based on dataset if not explicitly provided
    if args.eps is None:
        args.eps = 16/255 if args.dataset == "imagenet" else 8/255

    print("="*70)
    print("Pilot: LGC vs p* correlation")
    print("="*70)
    print(f"Dataset: {args.dataset}, N={args.n_examples}, Seeds={args.n_seeds}, eps={args.eps:.5f}")
    
    # P-sweep values
    p_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Get surrogates
    surrogates = get_surrogates(args.dataset)
    
    # Load data
    seed_everything(42)
    try:
        x_test, y_test = load_dataset(args.dataset, args.n_examples)
    except Exception as e:
        print(f"Failed to load {args.dataset}: {e}")
        return
    
    # Load target models for transfer evaluation
    # For ImageNet: use multiple targets and average (excluding self-transfer)
    if args.dataset == "cifar10":
        target_names = ["Engstrom2019Robustness"]  # Single robust target
    else:
        # Multiple diverse targets for fair comparison
        target_names = ["InceptionV3", "ViT_B_16_ImageNet", "Swin_B_ImageNet", "ConvNeXt_B_ImageNet"]
    
    # Pre-load all target models
    target_models = {}
    for t_name in target_names:
        try:
            target_models[t_name] = get_model(t_name, args.dataset, "Linf", device)
            print(f"Loaded target: {t_name}")
        except Exception as e:
            print(f"Failed to load target {t_name}: {e}")
    
    all_results = []
    
    for short_name, display_name, model_key, threat_model, expected_type in surrogates:
        print(f"\n{'='*50}")
        print(f"Surrogate: {display_name}")
        print(f"{'='*50}")
        
        try:
            model = get_model(model_key, args.dataset, threat_model, device)
        except Exception as e:
            print(f"  Skipping {display_name}: {e}")
            continue
        
        # Compute LGC
        print("  Computing LGC...")
        lgc_stats = compute_model_lgc(model, x_test, y_test, device)
        print(f"  LGC: mean={lgc_stats['mean']:.4f}, std={lgc_stats['std']:.4f}")
        
        # P-sweep across seeds, averaging over multiple targets (excluding self-transfer)
        seed_asr_results = {p: [] for p in p_values}
        
        # Determine which targets to use (exclude self-transfer)
        # Map source short_name to potential target names to exclude
        source_to_target_map = {
            "InceptionV3": "InceptionV3",
            "ViT": "ViT_B_16_ImageNet", 
            "Swin": "Swin_B_ImageNet",
            "ConvNeXt": "ConvNeXt_B_ImageNet",
        }
        excluded_target = source_to_target_map.get(short_name, None)
        valid_targets = [t for t in target_names if t != excluded_target and t in target_models]
        
        if not valid_targets:
            print(f"  Warning: No valid targets for {short_name}, using all targets")
            valid_targets = list(target_models.keys())
        
        print(f"  Evaluating against targets: {valid_targets}")
        
        for seed_idx in range(args.n_seeds):
            seed = 100 * (seed_idx + 1)
            seed_everything(seed)
            print(f"  Seed {seed}: Running p-sweep...")
            
            # Average ASR across valid targets for each p
            p_asr_across_targets = {p: [] for p in p_values}
            
            for t_name in valid_targets:
                target_model = target_models[t_name]
                p_results = run_p_sweep(model, x_test, y_test, target_model, device, args, p_values,
                                       desc_prefix=f"seed{seed}/{t_name[:8]}/")
                for p, asr in p_results.items():
                    p_asr_across_targets[p].append(asr)
            
            # Average across targets for this seed
            for p in p_values:
                avg_asr = np.mean(p_asr_across_targets[p]) if p_asr_across_targets[p] else 0
                seed_asr_results[p].append(avg_asr)
        
        # Aggregate results across seeds
        p_sweep_mean = {p: np.mean(asrs) for p, asrs in seed_asr_results.items()}
        p_sweep_std = {p: np.std(asrs) for p, asrs in seed_asr_results.items()}
        
        # Find optimal p*
        p_star, best_asr = find_optimal_p(p_sweep_mean)
        
        print(f"  P-sweep results (avg over {len(valid_targets)} targets):")
        for p in p_values:
            marker = " <-- p*" if p == p_star else ""
            print(f"    p={p}: {p_sweep_mean[p]:.2f}% +/- {p_sweep_std[p]:.2f}{marker}")
        
        all_results.append({
            "surrogate": short_name,
            "display_name": display_name,
            "expected_type": expected_type,
            "lgc_mean": lgc_stats["mean"],
            "lgc_std": lgc_stats["std"],
            "p_star": p_star,
            "best_asr": best_asr,
            "n_targets": len(valid_targets),
            **{f"asr_p{p}": p_sweep_mean[p] for p in p_values},
            **{f"asr_std_p{p}": p_sweep_std[p] for p in p_values},
        })
        
        del model
        torch.cuda.empty_cache()
    
    # Clean up target models
    for t_name in list(target_models.keys()):
        del target_models[t_name]
    target_models.clear()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.results_dir, "lgc_pstar_pilot.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Correlation Analysis
    print("\n" + "="*70)
    print("Correlation Analysis: LGC vs p*")
    print("="*70)
    
    lgc_values = df["lgc_mean"].values
    pstar_values = df["p_star"].values
    
    # Spearman (rank-based, better for monotonic but non-linear)
    spearman_rho, spearman_p = stats.spearmanr(lgc_values, pstar_values)
    
    # Pearson (linear correlation)
    pearson_r, pearson_p = stats.pearsonr(lgc_values, pstar_values)
    
    print(f"Spearman rho = {spearman_rho:.4f}, p-value = {spearman_p:.4f}")
    print(f"Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.4f}")
    
    # Pilot criterion: is the LGC-p* relation strong enough to pursue?
    print("\n" + "="*70)
    print("Pilot criterion")
    print("="*70)
    
    go_criteria = abs(spearman_rho) >= 0.5 and spearman_p < 0.1
    
    if go_criteria:
        decision = "MET"
        reason = f"|rho|={abs(spearman_rho):.2f} >= 0.5 and p={spearman_p:.4f} < 0.1"
    else:
        decision = "NOT MET"
        if abs(spearman_rho) < 0.5:
            reason = f"|rho|={abs(spearman_rho):.2f} < 0.5 (weak correlation)"
        else:
            reason = f"p={spearman_p:.4f} >= 0.1 (not significant)"
    
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    
    # Generate Scatter Plot
    plt.figure(figsize=(8, 6))
    
    colors = {"standard": "blue", "robust": "red"}
    
    for _, row in df.iterrows():
        color = colors.get(row["expected_type"], "gray")
        plt.scatter(row["lgc_mean"], row["p_star"], 
                   c=color, s=100, edgecolors='black', linewidths=1.5)
        plt.annotate(row["surrogate"], 
                    (row["lgc_mean"], row["p_star"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=10)
    
    # Add trend line
    if len(lgc_values) > 2:
        z = np.polyfit(lgc_values, pstar_values, 1)
        p_trend = np.poly1d(z)
        x_line = np.linspace(min(lgc_values) - 0.02, max(lgc_values) + 0.02, 100)
        plt.plot(x_line, p_trend(x_line), "--", color="gray", alpha=0.7, 
                label=f"Linear fit (r={pearson_r:.2f})")
    
    plt.xlabel("LGC (Local Gradient Consistency)", fontsize=12)
    plt.ylabel("Optimal DI Probability (p*)", fontsize=12)
    plt.title(f"LGC vs p* Correlation (Spearman rho={spearman_rho:.2f}, p={spearman_p:.3f})", fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # Add color legend
    for type_name, color in colors.items():
        plt.scatter([], [], c=color, s=80, label=type_name.capitalize())
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    
    plot_path = os.path.join(args.results_dir, "lgc_pstar_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nScatter plot saved to {plot_path}")
    
    # Summary Report
    report_path = os.path.join(args.results_dir, "pilot_report.txt")
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("LGC vs p* pilot report\n")
        f.write("="*70 + "\n\n")
        
        f.write("SURROGATES:\n")
        for _, row in df.iterrows():
            f.write(f"  {row['display_name']}: LGC={row['lgc_mean']:.3f}, p*={row['p_star']}\n")
        
        f.write(f"\nCORRELATION:\n")
        f.write(f"  Spearman rho = {spearman_rho:.4f}, p-value = {spearman_p:.4f}\n")
        f.write(f"  Pearson r = {pearson_r:.4f}, p-value = {pearson_p:.4f}\n")
        
        f.write(f"\nDECISION: {decision}\n")
        f.write(f"REASON: {reason}\n")
        
        if decision == "MET":
            f.write("\nA graded LGC-to-p* predictor is worth estimating on the\n")
            f.write("full surrogate pools rather than these five.\n")
        else:
            f.write("\nThe LGC-p* relation is too weak for a graded predictor;\n")
            f.write("LGC is used only for the binary regime split.\n")
    
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
