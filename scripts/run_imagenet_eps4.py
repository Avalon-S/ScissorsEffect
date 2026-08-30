#!/usr/bin/env python
"""
ImageNet Experiment with Small Perturbation Budget (epsilon=4/255)
Verifies the "Scissors Effect" holds even at lower perturbation magnitudes.
"""
import argparse
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.sap import CG_DI, estimate_p_lgc
# Assuming get_attack is available or effectively implementing DIM
from attacks.wrappers import get_attack 
from utils import seed_everything, compute_metrics, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet DI Law Experiment (Small Epsilon)")
    parser.add_argument("--n_examples", type=int, default=5000, help="Number of examples")
    parser.add_argument("--n_seeds", type=int, default=5) 
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=4/255, help="Perturbation budget")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="results/imagenet_eps4")
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*70)
    print(f"ImageNet Small Epsilon Experiment (eps={args.eps:.5f})")
    print("="*70)
    
    # Sources: One Standard, One Robust
    sources = [
        ("Standard", "ResNet50"),
        ("Robust", "Engstrom2019Robustness_ImageNet"), 
    ]
    
    # Targets for transfer
    targets = [
        "InceptionV3",
        "ResNet50", # Transfer to same arch (if source is diff) or Robust->Standard
    ]
    
    # Methods to compare
    # Fixed-0.0 (No DI) vs Fixed-1.0 (Full DI) vs CG-DI
    p_grid = [0.0, 1.0] 
    
    all_results = []
    
    seed_everything(42)
    # Load test data
    try:
        x_test, y_test = load_dataset("imagenet", args.n_examples)
    except Exception as e:
        print(f"Failed to load ImageNet: {e}")
        return

    for seed in [100 * (i + 1) for i in range(args.n_seeds)]:
        print(f"\n=== Seed {seed} ===")
        seed_everything(seed)
        
        for source_type, source_name in sources:
            print(f"\nSource: {source_name} ({source_type})")
            try:
                # Load Source
                model = get_model(source_name, "imagenet", "Linf", device)
            except Exception as e:
                print(f"Skipping {source_name}: {e}")
                continue

            # 1. Run Fixed p (0.0 and 1.0)
            x_adv_dict = {}
            
            for p in p_grid:
                print(f"  Running Fixed-p={p}...")
                # Note: eps is passed from args
                attacker = get_attack("dim", model, eps=args.eps, steps=args.steps, 
                                      diversity_prob=p, resize_rate=0.9)
                x_adv_list = []
                for i in tqdm(range(0, len(x_test), args.batch_size), desc=f"Fixed-p={p}", leave=False):
                    xb = x_test[i:i+args.batch_size]
                    yb = y_test[i:i+args.batch_size]
                    x_adv_batch = attacker(xb, yb)
                    x_adv_list.append(x_adv_batch.cpu())
                x_adv_dict[f"Fixed-{p}"] = torch.cat(x_adv_list)

            # 2. Run CG-DI
            print(f"  Running CG-DI...")
            
            # Estimate LGC ONCE per model (not per batch) - this is a model property
            probe_x = x_test[:min(50, len(x_test))].to(device)
            probe_y = y_test[:min(50, len(y_test))].to(device)
            _, lgc_vals = estimate_p_lgc(model, probe_x, probe_y, device, K=5, sigma=1/255)
            model_lgc = lgc_vals.mean().item()
            
            # Decision based on model-level LGC
            final_p = 0.0 if model_lgc > 0.92 else 0.8
            print(f"    Model LGC: {model_lgc:.3f} -> Using p={final_p}")
            
            x_adv_sap = []
            attacker_cgdi = get_attack("dim", model, eps=args.eps, steps=args.steps, 
                                       diversity_prob=final_p, resize_rate=0.9)
            
            for i in tqdm(range(0, len(x_test), args.batch_size), desc="CG-DI", leave=False):
                xb = x_test[i:i+args.batch_size]
                yb = y_test[i:i+args.batch_size]
                x_adv_batch = attacker_cgdi(xb, yb)
                x_adv_sap.append(x_adv_batch.cpu())
                
            x_adv_dict["CG-DI"] = torch.cat(x_adv_sap)

            # Evaluate Transfer
            for target_name in targets:
                if target_name == source_name: continue # Skip self-transfer if redundant
                
                print(f"  Evaluating on {target_name}...")
                try:
                    target_model = get_model(target_name, "imagenet", "Linf", device)
                    
                    for method_name, x_adv in x_adv_dict.items():
                        metrics = compute_metrics(x_adv, x_test, y_test, target_model, device)
                        asr = metrics['asr']
                        
                        all_results.append({
                            "seed": seed,
                            "source": source_name,
                            "source_type": source_type,
                            "method": method_name,
                            "target": target_name,
                            "eps": args.eps,
                            "asr": asr
                        })
                        print(f"    {method_name}: {asr:.2f}%")
                        
                    del target_model
                except Exception as e:
                    print(f"    Failed to eval {target_name}: {e}")
                    
            del model
            torch.cuda.empty_cache()

    # Save
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.results_dir, "imagenet_eps4_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()
