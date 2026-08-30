"""
Targeted Attack Experiment for Scissors Effect Validation

Verifies that the Scissors Effect persists in targeted attack setting:
- DI should still hurt robust surrogates
- DI should still help standard surrogates

Key features:
- Random target class with fixed seed (consistent across p=0 and p=0.5)
- White-box sanity check before transfer evaluation
- Multiple targets (Swin-B, InceptionV3)
- Reports both absolute and relative changes
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils import seed_everything, get_device
from models import get_model
from data import load_dataset


# =============================================================================
# Targeted MI-FGSM Attack
# =============================================================================
def di_transform(x, resize_rate=0.9):
    """DI transform: random resize + pad."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    rnd = torch.randint(low=img_resize, high=img_size, size=(1,), dtype=torch.int32).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    
    h_rem = img_size - rnd
    w_rem = img_size - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,)).item()
    pad_right = w_rem - pad_left
    
    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
    return padded


def targeted_mifgsm(model, x, y_target, eps, alpha, steps, device, use_di=False, di_prob=0.5):
    """
    Targeted MI-FGSM attack.
    
    Args:
        model: surrogate model
        x: input images [B, C, H, W]
        y_target: target class labels [B]
        eps: perturbation budget
        alpha: step size
        steps: number of iterations
        device: torch device
        use_di: whether to use DI transform
        di_prob: probability of applying DI
        
    Returns:
        x_adv: adversarial examples
    """
    x = x.clone().detach().to(device)
    y_target = y_target.to(device)
    
    x_adv = x.clone()
    momentum = torch.zeros_like(x)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # Apply DI transform probabilistically
        if use_di and random.random() < di_prob:
            x_input = di_transform(x_adv)
        else:
            x_input = x_adv
        
        logits = model(x_input)
        
        # TARGETED: maximize probability of target class
        # Use positive CE loss and SUBTRACT gradient (gradient descent on loss = maximize target prob)
        loss = F.cross_entropy(logits, y_target, reduction='sum')
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # Normalize gradient
        grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
        grad = grad / grad_norm
        
        # Momentum update
        momentum = momentum + grad
        
        # Update adversarial example: SUBTRACT to minimize CE loss (maximize target class prob)
        x_adv = x_adv.detach() - alpha * momentum.sign()
        
        # Project to epsilon ball and valid range
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv.detach()


def targeted_mifgsm_batched(model, x, y_target, eps, alpha, steps, device, use_di=False, di_prob=0.5, batch_size=32):
    """
    Batched version of targeted MI-FGSM to avoid OOM.
    """
    n_samples = x.size(0)
    x_adv_list = []
    
    for i in range(0, n_samples, batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = y_target[i:i+batch_size]
        x_adv_batch = targeted_mifgsm(model, x_batch, y_batch, eps, alpha, steps, device, use_di, di_prob)
        x_adv_list.append(x_adv_batch.cpu())  # Move to CPU to save GPU memory
        torch.cuda.empty_cache()
    
    return torch.cat(x_adv_list, dim=0)


def evaluate_targeted_asr(model, x_adv, y_target, device, batch_size=32):
    """Evaluate targeted attack success rate."""
    model.eval()
    correct = 0
    total = 0
    
    dataset = torch.utils.data.TensorDataset(x_adv, y_target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(x_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
    
    return correct / total * 100


def generate_random_targets(y_true, num_classes, seed):
    """
    Generate random target classes (different from true class).
    Uses fixed seed to ensure consistency across experiments.
    """
    rng = np.random.RandomState(seed)
    y_target = []
    for y in y_true.cpu().numpy():
        # Sample from all classes except true class
        candidates = [c for c in range(num_classes) if c != y]
        y_target.append(rng.choice(candidates))
    return torch.tensor(y_target, dtype=torch.long)


# =============================================================================
# Main Experiment
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Targeted Attack: Scissors Effect Validation")
    parser.add_argument("--n_examples", type=int, default=1000, help="Number of test images")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--eps", type=float, default=16/255, help="Perturbation budget")
    parser.add_argument("--steps", type=int, default=10, help="Attack iterations")
    parser.add_argument("--di_prob", type=float, default=0.5, help="DI probability")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="results/targeted_attack")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    N_SAMPLES = args.n_examples
    N_SEEDS = args.n_seeds
    EPS = args.eps
    ALPHA = 2 / 255
    STEPS = args.steps
    DI_PROB = args.di_prob
    NUM_CLASSES = 1000
    TARGET_SEED = 42  # Fixed seed for target class generation

    # Results directory
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 60)
    print("Targeted Attack Experiment: Scissors Effect Validation")
    print("=" * 60)
    print(f"Config: N={N_SAMPLES}, seeds={N_SEEDS}, eps={EPS*255:.0f}/255")
    
    # Surrogates
    surrogates = [
        ("ResNet50", "Standard", "ResNet50"),
        ("Engstrom", "Robust", "Engstrom2019Robustness_ImageNet"),
    ]
    
    # Targets
    targets = [
        ("Swin-B", "Swin_B_ImageNet"),
        ("IncV3", "InceptionV3"),
    ]
    
    all_results = []
    
    # Load data once
    print("\nLoading ImageNet samples...")
    x_all, y_all = load_dataset(dataset='imagenet', n_examples=N_SAMPLES)
    x_all = x_all.to(device)
    y_all = y_all.to(device)
    
    # Generate target labels (fixed seed, consistent across all experiments)
    print(f"Generating random target labels (seed={TARGET_SEED})...")
    y_target = generate_random_targets(y_all, NUM_CLASSES, TARGET_SEED).to(device)
    
    # Save target mapping for reproducibility
    target_mapping = pd.DataFrame({
        'sample_idx': range(N_SAMPLES),
        'true_class': y_all.cpu().numpy(),
        'target_class': y_target.cpu().numpy()
    })
    target_mapping.to_csv(os.path.join(results_dir, "target_mapping.csv"), index=False)
    print(f"Saved target mapping to {results_dir}/target_mapping.csv")
    
    for surr_name, surr_type, surr_model_name in surrogates:
        print(f"\n{'='*60}")
        print(f"Surrogate: {surr_name} ({surr_type})")
        print("=" * 60)
        
        # Load surrogate
        surrogate = get_model(surr_model_name, dataset='imagenet', threat_model='Linf', device=device)
        surrogate.eval()

        # White-box sanity check (small batch)
        print("\n[Sanity Check] White-box targeted attack (100 samples)...")
        seed_everything(100)
        x_adv_test = targeted_mifgsm_batched(
            surrogate, x_all[:100], y_target[:100],
            EPS, ALPHA, STEPS, device, use_di=False, batch_size=16
        )
        wb_asr = evaluate_targeted_asr(surrogate, x_adv_test, y_target[:100].cpu(), device)
        print(f"  White-box Targeted ASR: {wb_asr:.1f}%")
        
        if wb_asr < 5:
            print("  WARNING: White-box ASR very low, attack may not be working correctly!")
        
        for target_name, target_model_name in targets:
            print(f"\n--- Target: {target_name} ---")
            
            # Load target model
            target_model = get_model(target_model_name, dataset='imagenet', threat_model='Linf', device=device)
            target_model.eval()
            
            results_naked = []
            results_di = []
            
            for seed in range(N_SEEDS):
                seed_everything(100 + seed * 100)
                
                # Attack without DI (Naked) - use batched version
                x_adv_naked = targeted_mifgsm_batched(
                    surrogate, x_all, y_target,
                    EPS, ALPHA, STEPS, device, use_di=False, batch_size=16
                )
                asr_naked = evaluate_targeted_asr(target_model, x_adv_naked, y_target.cpu(), device)
                results_naked.append(asr_naked)
                
                # Attack with DI - use batched version
                x_adv_di = targeted_mifgsm_batched(
                    surrogate, x_all, y_target,
                    EPS, ALPHA, STEPS, device, use_di=True, di_prob=DI_PROB, batch_size=16
                )
                asr_di = evaluate_targeted_asr(target_model, x_adv_di, y_target.cpu(), device)
                results_di.append(asr_di)
                
                print(f"  Seed {seed}: Naked={asr_naked:.1f}%, +DI={asr_di:.1f}%")
            
            # Aggregate results
            mean_naked = np.mean(results_naked)
            mean_di = np.mean(results_di)
            std_naked = np.std(results_naked)
            std_di = np.std(results_di)
            
            delta_abs = mean_di - mean_naked
            delta_rel = (delta_abs / mean_naked * 100) if mean_naked > 0 else 0
            
            print(f"\n  Summary: Naked={mean_naked:.1f}% +/- {std_naked:.1f}%")
            print(f"           +DI  ={mean_di:.1f}% +/- {std_di:.1f}%")
            print(f"           Delta: {delta_abs:+.1f}% (abs), {delta_rel:+.1f}% (rel)")
            
            all_results.append({
                'surrogate': surr_name,
                'type': surr_type,
                'target': target_name,
                'naked_mean': mean_naked,
                'naked_std': std_naked,
                'di_mean': mean_di,
                'di_std': std_di,
                'delta_abs': delta_abs,
                'delta_rel': delta_rel,
            })
            
            # Cleanup
            del target_model
            torch.cuda.empty_cache()
        
        del surrogate
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(results_dir, "targeted_attack_results.csv"), index=False)
    
    # Generate report
    report_path = os.path.join(results_dir, "targeted_attack_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Targeted Attack Experiment Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Config: N={N_SAMPLES}, seeds={N_SEEDS}, eps={EPS*255:.0f}/255\n")
        f.write(f"Attack: Targeted MI-FGSM, T={STEPS}, DI prob={DI_PROB}\n")
        f.write(f"Target class seed: {TARGET_SEED}\n\n")
        
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        for _, row in df.iterrows():
            f.write(f"\n{row['surrogate']} ({row['type']}) -> {row['target']}\n")
            f.write(f"  Naked: {row['naked_mean']:.1f}% +/- {row['naked_std']:.1f}%\n")
            f.write(f"  +DI:   {row['di_mean']:.1f}% +/- {row['di_std']:.1f}%\n")
            f.write(f"  Delta: {row['delta_abs']:+.1f}% (abs), {row['delta_rel']:+.1f}% (rel)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Scissors Effect Summary:\n")
        f.write("-" * 60 + "\n")
        
        std_results = df[df['type'] == 'Standard']
        rob_results = df[df['type'] == 'Robust']
        
        if len(std_results) > 0:
            avg_std_delta = std_results['delta_abs'].mean()
            f.write(f"Standard surrogates: avg DI effect = {avg_std_delta:+.1f}%\n")
        
        if len(rob_results) > 0:
            avg_rob_delta = rob_results['delta_abs'].mean()
            f.write(f"Robust surrogates:   avg DI effect = {avg_rob_delta:+.1f}%\n")
        
        f.write("\nConclusion: ")
        if len(std_results) > 0 and len(rob_results) > 0:
            if avg_std_delta > 0 and avg_rob_delta < 0:
                f.write("Scissors Effect CONFIRMED in targeted attack setting.\n")
            elif avg_rob_delta < 0:
                f.write("DI harm on robust surrogates confirmed.\n")
            else:
                f.write("Results inconclusive, may need more samples.\n")
    
    print(f"\n\nResults saved to: {results_dir}")
    print(f"Report: {report_path}")
    

if __name__ == "__main__":
    main()
