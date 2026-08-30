#!/usr/bin/env python
"""
Extended ImageNet Experiment for DI Law (Review Response)
Targets diverse architectures (ViT, Swin, ConvNeXt) and includes Robust Sources.
"""
import argparse
import sys
import os
import torch
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.sap import CG_DI
from attacks.wrappers import get_attack
from utils import seed_everything, compute_metrics, get_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=1000, help="Number of test images")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="results/imagenet_extended")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*60)
    print("Extended ImageNet Experiment: Diversity & Robustness")
    print("="*60)
    
    # 1. Broad Source Selection
    sources = [
        ("Standard", "ResNet50"),
        ("Robust", "Engstrom2019Robustness_ImageNet") 
    ]
    
    # 2. Broad Target Selection (Diverse Architectures)
    targets = [
        "InceptionV3",              # Standard CNN
        "ResNet50",                 # Standard CNN (Self-transfer check)
        "ViT_B_16_ImageNet",        # Transformer
        "Swin_B_ImageNet",          # Hierarchical Transformer
        "ConvNeXt_B_ImageNet",      # Modern CNN
        "Engstrom2019Robustness_ImageNet" # Robust Target
    ]
    
    # 3. Methods to Compare
    # Key Question: Does Diversity Help (Std) or Hurt (Robust)?
    methods = [
        ("Fixed-0.0", 0.0),    # Baseline
        ("Fixed-1.0", 1.0)     # Full DI
    ]

    all_results = []
    
    # Calibration Data Strategy
    print("Loading calibration data for CG-DI...")
    seed_everything(42)
    x_cal, y_cal = load_dataset("imagenet", 50) 
    
    seeds = [100 * (i + 1) for i in range(args.n_seeds)]
    
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_everything(seed)
        try:
            x_test, y_test = load_dataset("imagenet", args.n_examples)
        except Exception as e:
            print(f"Failed to load ImageNet data: {e}")
            return

        for source_type, source_name in sources:
            print(f"\nSource: {source_name} ({source_type})")
            try:
                model = get_model(source_name, "imagenet", "Linf", device)
            except Exception as e:
                print(f"  Skipping source {source_name}: {e}")
                continue

            # A) Fixed Baselines (MI/DI)
            for m_name, p in methods:
                print(f"  Running {m_name}...")
                attacker = get_attack("dim", model, eps=16/255, steps=args.steps, 
                                      diversity_prob=p, resize_rate=0.9)
                x_adv = []
                # Batch Generation
                for i in tqdm(range(0, args.n_examples, args.batch_size), desc=m_name, leave=False):
                    xb = x_test[i:i+args.batch_size]
                    yb = y_test[i:i+args.batch_size]
                    x_adv.append(attacker(xb, yb).cpu())
                x_adv = torch.cat(x_adv, dim=0)

                # Transfer Eval
                for t_name in targets:
                    if t_name == source_name: continue # Skip self if simple check
                    try:
                        # Load target just-in-time
                        t_model = get_model(t_name, "imagenet", "Linf", device)
                        metrics = compute_metrics(x_adv, x_test, y_test, t_model, device)
                        print(f"    -> {t_name}: {metrics['asr']:.2f}%")
                        all_results.append({
                            "seed": seed,
                            "source": source_name,
                            "source_type": source_type,
                            "method": m_name,
                            "target": t_name,
                            "transfer_asr": metrics["asr"]
                        })
                        del t_model
                    except Exception as e:
                        print(f"    Failed eval on {t_name}: {e}")

            # B) CG-DI
            print(f"  Running CG-DI...")
            sap = CG_DI(model, eps=16/255, alpha=2/255, steps=args.steps, device=device,
                             calibration_data=(x_cal, y_cal))
            x_adv = []
            for i in tqdm(range(0, args.n_examples, args.batch_size), desc="CG-DI", leave=False):
                xb = x_test[i:i+args.batch_size]
                yb = y_test[i:i+args.batch_size]
                x_adv.append(sap(xb, yb).cpu())
            x_adv = torch.cat(x_adv, dim=0)
            
            p_stats = sap.get_p_hat_stats()
            print(f"    Decided p_mean: {p_stats['mean']:.2f}")

            for t_name in targets:
                if t_name == source_name: continue
                try:
                    t_model = get_model(t_name, "imagenet", "Linf", device)
                    metrics = compute_metrics(x_adv, x_test, y_test, t_model, device)
                    print(f"    -> {t_name}: {metrics['asr']:.2f}%")
                    all_results.append({
                        "seed": seed,
                        "source": source_name,
                        "source_type": source_type,
                        "method": "CG-DI",
                        "target": t_name,
                        "transfer_asr": metrics["asr"],
                        "p_mean": p_stats["mean"]
                    })
                    del t_model
                except Exception as e:
                    pass
            
            del model
            torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(all_results)
    save_path = os.path.join(args.results_dir, "imagenet_extended.csv")
    df.to_csv(save_path, index=False)
    print(f"\nDone! Results saved to {save_path}")

if __name__ == "__main__":
    main()
