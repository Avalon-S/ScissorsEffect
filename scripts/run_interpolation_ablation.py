#!/usr/bin/env python
"""
Interpolation Mode Ablation Experiment (ImageNet)

Tests whether the Scissors Effect harm on robust surrogates is caused by
the specific interpolation algorithm (bilinear) or is a fundamental property
of gradient geometry.

Uses torchattacks.DIFGSM as the base (identical to main paper experiments),
and only overrides the interpolation mode in input_diversity.

Compares three interpolation modes:
  - bilinear:  PyTorch default, standard DI implementation
  - bicubic:   smoother interpolation, fewer HF artifacts
  - antialias: bilinear with low-pass pre-filter (preserves LF content)

Sweeps diversity_prob p in {0.0, 0.3, 0.5, 0.7, 1.0}.
"""
import argparse
import sys
import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchattacks
from data import load_dataset
from models import get_model
from utils import seed_everything, compute_metrics, get_device


# ============================================================
# Subclass: DIFGSM with Configurable Interpolation Mode
# ============================================================

class DIFGSM_Interp(torchattacks.DIFGSM):
    """torchattacks.DIFGSM with parameterizable interpolation in input_diversity.
    
    All attack logic (gradient computation, momentum, loss, clipping) is
    inherited from torchattacks.DIFGSM -- only the interpolation call inside
    input_diversity is modified.
    """
    
    def __init__(self, model, eps=16/255, alpha=2/255, steps=10, decay=1.0,
                 resize_rate=0.9, diversity_prob=1.0, random_start=False,
                 interp_mode='bilinear', antialias=False):
        super().__init__(model, eps=eps, alpha=alpha, steps=steps, decay=decay,
                         resize_rate=resize_rate, diversity_prob=diversity_prob,
                         random_start=random_start)
        self.interp_mode = interp_mode
        self.antialias = antialias
    
    def input_diversity(self, x):
        """Override: identical logic to torchattacks.DIFGSM.input_diversity,
        but uses self.interp_mode and self.antialias instead of hardcoded bilinear.
        """
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode=self.interp_mode,
            align_corners=False, antialias=self.antialias
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x


# ============================================================
# Main Experiment
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolation Mode Ablation on ImageNet")
    parser.add_argument("--n_examples", type=int, default=1000,
                        help="Number of test images")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, 
                        default="results/paper_appendix_interp_ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Interpolation Mode Ablation Experiment (ImageNet)")
    print("  Backend: torchattacks.DIFGSM (subclassed)")
    print("  p-sweep: p in {0.0, 0.3, 0.5, 0.7, 1.0}")
    print("=" * 60)
    
    # --- Configuration (matches run_imagenet_extended.py) ---
    eps = 16 / 255
    alpha = 2 / 255
    resize_rate = 0.9   # Matches torchattacks.DIFGSM default
    decay = 1.0         # Momentum decay (MI-FGSM)
    
    sources = [
        ("Standard", "ResNet50"),
        ("Robust", "Engstrom2019Robustness_ImageNet"),
    ]
    
    targets = [
        "InceptionV3",
        "ViT_B_16_ImageNet",
        "Swin_B_ImageNet",
        "ConvNeXt_B_ImageNet",
    ]
    
    # Interpolation modes to test
    interp_configs = [
        ("Bilinear",  "bilinear", False),
        ("Bicubic",   "bicubic",  False),
        ("Antialias", "bilinear", True),
    ]
    
    # p values to sweep
    p_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    seeds = [100 * (i + 1) for i in range(args.n_seeds)]
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        seed_everything(seed)
        
        try:
            x_test, y_test = load_dataset("imagenet", args.n_examples)
        except Exception as e:
            print(f"Failed to load ImageNet data: {e}")
            return
        
        for source_type, source_name in sources:
            print(f"\n--- Source: {source_name} ({source_type}) ---")
            try:
                model = get_model(source_name, "imagenet", "Linf", device)
            except Exception as e:
                print(f"  Skipping source {source_name}: {e}")
                continue
            
            for interp_name, interp_mode, antialias in interp_configs:
                for p in p_values:
                    label = f"{interp_name} (p={p})"
                    print(f"  [{label}]...")
                    
                    attacker = DIFGSM_Interp(
                        model, eps=eps, alpha=alpha, steps=args.steps,
                        decay=decay, resize_rate=resize_rate,
                        diversity_prob=p, random_start=False,
                        interp_mode=interp_mode, antialias=antialias
                    )
                    
                    # Generate adversarial examples
                    x_adv = []
                    for i in tqdm(range(0, args.n_examples, args.batch_size),
                                  desc=label, leave=False):
                        xb = x_test[i:i + args.batch_size].to(device)
                        yb = y_test[i:i + args.batch_size].to(device)
                        x_adv.append(attacker(xb, yb).cpu())
                    x_adv = torch.cat(x_adv, dim=0)
                    
                    # Evaluate transfer to each target
                    for t_name in targets:
                        if t_name == source_name:
                            continue
                        try:
                            t_model = get_model(t_name, "imagenet", "Linf", device)
                            metrics = compute_metrics(
                                x_adv, x_test, y_test, t_model, device)
                            asr = metrics["asr"]
                            print(f"    -> {t_name}: {asr:.2f}%")
                            all_results.append({
                                "seed": seed,
                                "source": source_name,
                                "source_type": source_type,
                                "interp_mode": interp_name,
                                "diversity_prob": p,
                                "target": t_name,
                                "transfer_asr": asr,
                            })
                            del t_model
                        except Exception as e:
                            print(f"    Failed eval on {t_name}: {e}")
                    
                    del attacker
                    torch.cuda.empty_cache()
            
            del model
            torch.cuda.empty_cache()
    
    # --- Save Results ---
    df = pd.DataFrame(all_results)
    save_path = os.path.join(args.results_dir, "interpolation_ablation_psweep.csv")
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")
    
    # --- Print Summary Table ---
    print("\n" + "=" * 60)
    print("Summary: Average Transfer ASR across targets and seeds")
    print("=" * 60)
    
    summary = df.groupby(
        ["source_type", "interp_mode", "diversity_prob"]
    )["transfer_asr"].mean().reset_index()
    summary.columns = ["Source Type", "Interp Mode", "p", "Avg ASR (%)"]
    
    # Compute delta vs p=0 baseline for each (source_type, interp_mode)
    for src_type in ["Standard", "Robust"]:
        for interp in ["Bilinear", "Bicubic", "Antialias"]:
            baseline = summary[
                (summary["Source Type"] == src_type) & 
                (summary["Interp Mode"] == interp) &
                (summary["p"] == 0.0)
            ]["Avg ASR (%)"].values
            if len(baseline) > 0:
                mask = (
                    (summary["Source Type"] == src_type) & 
                    (summary["Interp Mode"] == interp)
                )
                summary.loc[mask, "Delta vs p=0"] = (
                    summary.loc[mask, "Avg ASR (%)"] - baseline[0]
                ).round(2)
    
    print(summary.to_string(index=False))
    
    # Also save summary
    summary_path = os.path.join(args.results_dir, "summary_psweep.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
