#!/usr/bin/env python
"""
Supplementary Experiment: Classical Attacks on CIFAR-10 (Aligned with TransferAttack)
Evaluates classical attacks (MI, TI, SI, VMI, DI) on CIFAR-10 using TransferAttack library.
This ensures consistency with the ImageNet experiments (run_modern_attacks.py).

Attacks:
- MI-FGSM (mifgsm)
- TI-FGSM (tim)
- SI-FGSM (sim)
- VMI-FGSM (vmifgsm)
- DI-FGSM (dim) [Note: TransferAttack uses 'dim' key for DI-FGSM]
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add TransferAttack to path
TRANSFER_ATTACK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "TransferAttack"
)
sys.path.insert(0, TRANSFER_ATTACK_PATH)

from models import get_model
from data import load_dataset
from utils import seed_everything, get_device, compute_metrics

import transferattack


# ============================================================================
# TransferAttack Wrapper (Copied & Adapted from run_modern_attacks.py)
# ============================================================================

class TransferAttackWrapper:
    """
    Wrapper to use TransferAttack methods with our pre-loaded models.
    Overrides the model loading to use our models directly.
    """

    def __init__(self, attack_name, model, epsilon=8/255, steps=10, 
                 alpha=2/255, device='cuda', **kwargs):
        self.attack_name = attack_name
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = device
        self.kwargs = kwargs

        # Load attack class
        attack_class = transferattack.load_attack_class(attack_name)

        # Create attack instance with a dummy model name
        self.attack = self._create_attack(attack_class)

        # Replace the model: skip wrap_model for CIFAR-10 since
        # (1) wrap_model resizes input to 224x224 (breaks 32x32 models)
        # (2) RobustBench models already include normalization
        self.attack.model = torch.nn.Sequential(model)
        self.attack.device = device

    def _create_attack(self, attack_class):
        """Create attack with appropriate parameters."""
        # Base arguments
        kwargs = {
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'epoch': self.steps,
            'device': self.device,
        }
        
        # Add any extra kwargs passed to init (e.g., resize_rate)
        kwargs.update(self.kwargs)

        # Attack-specific hardcoded defaults for consistency
        if self.attack_name == 'mifgsm':
            kwargs['decay'] = 1.0
        elif self.attack_name == 'tim':
            kwargs['decay'] = 1.0
            kwargs['kernel_type'] = 'gaussian'
            kwargs['kernel_size'] = 5  # Smaller kernel for CIFAR-10 (32x32)
        elif self.attack_name == 'sim':
            kwargs['decay'] = 1.0
            kwargs['num_scale'] = 5
        elif self.attack_name == 'vmifgsm':
            kwargs['decay'] = 1.0
            kwargs['beta'] = 1.5
            kwargs['num_neighbor'] = 20
        elif self.attack_name == 'dim': # DI-FGSM
            kwargs['decay'] = 1.0
            if 'resize_rate' not in kwargs: kwargs['resize_rate'] = 1.15  # Must be > 1 for TransferAttack
            if 'diversity_prob' not in kwargs: kwargs['diversity_prob'] = 0.5


        # Suppress name-based model resolution; the surrogate is attached after.
        class DummyAttack(attack_class):
            def load_model(self, model_name):
                return None

        return DummyAttack(model_name='dummy', **kwargs)

    def __call__(self, x, y):
        """Generate adversarial examples."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Run attack
        delta = self.attack.forward(x, y)
        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv


# ============================================================================
# Main Experiment
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Classical Attacks on CIFAR-10 (TransferAttack)")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eps", type=float, default=8/255, help="Epsilon (CIFAR-10 default)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results_dir", type=str, default="results/paper_appendix_cifar10_classical")
    return parser.parse_args()


def run_attack_batch(attacker, x_test, y_test, batch_size, device, desc=""):
    """Run attack in batches with progress bar."""
    x_adv_list = []
    n_batches = (len(x_test) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(x_test), batch_size), total=n_batches, desc=desc, leave=False):
        xb = x_test[i:i+batch_size].to(device)
        yb = y_test[i:i+batch_size].to(device)
        x_adv_batch = attacker(xb, yb)
        x_adv_list.append(x_adv_batch.cpu())

        # Clear GPU memory
        del xb, yb, x_adv_batch
        torch.cuda.empty_cache()

    return torch.cat(x_adv_list, dim=0)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    log_file = os.path.join(args.results_dir, 'classical_attacks_cifar.txt')
    open(log_file, 'w').close()
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    log_print("=" * 70)
    log_print("Supplementary: Classical Attacks on CIFAR-10 (TransferAttack Lib)")
    log_print("=" * 70)
    log_print(f"Config: n_examples={args.n_examples}, n_seeds={args.n_seeds}")
    log_print(f"epsilon={args.eps:.4f}, steps={args.steps}, batch_size={args.batch_size}")

    seeds = [100 * (i + 1) for i in range(args.n_seeds)]

    # Attacks configuration
    # (Display Name, TransferAttack Key)
    attacks_config = [
        ("MI-FGSM",  "mifgsm"),
        ("TI-FGSM",  "tim"),
        ("SI-FGSM",  "sim"),
        ("VMI-FGSM", "vmifgsm"),
        ("DI-FGSM",  "dim"),
    ]
    
    # Sources
    sources = [
        ("Standard", "Standard"),
        ("Robust", "Engstrom2019Robustness"),
    ]

    # Targets (Same as run_sota.py selection)
    robust_targets = [
        "Engstrom2019Robustness",
        "Rice2020Overfitting",
        "Carmon2019Unlabeled",
        "Gowal2020Uncovering_70_16",
        "Wang2023Better_WRN-28-10",
    ]
    standard_targets = [
        "Standard",            # WideResNet-28-10 (default)
        "VGG16_Standard",      # VGG16 standard
        "DenseNet121_Standard",# DenseNet121 standard
        "ResNet50_Standard",   # ResNet50 standard
    ]
    all_targets_names = sorted(list(set(robust_targets + standard_targets)))

    all_results = []

    for source_key, source_name in sources:
        log_print(f"\n{'='*60}")
        log_print(f"Source: {source_name} ({source_key})")
        log_print(f"{'='*60}")

        try:
            source_model = get_model(source_name, "cifar10", "Linf", device)
        except Exception as e:
            log_print(f"Error loading source {source_name}: {e}")
            continue

        for attack_display, attack_key in attacks_config:
            log_print(f"\n  Attack: {attack_display} ({attack_key})")

            for seed in seeds:
                seed_everything(seed)
                x_test, y_test = load_dataset("cifar10", args.n_examples)

                try:
                    # Initialize attacker
                    # CIFAR-10 params: eps=8/255, alpha=2/255
                    attacker = TransferAttackWrapper(
                        attack_key, source_model,
                        epsilon=args.eps, steps=args.steps, alpha=2/255,
                        device=device
                    )

                    # Run attack
                    x_adv = run_attack_batch(
                        attacker, x_test, y_test, args.batch_size, device,
                        desc=f"{attack_display} s={seed}"
                    )

                    # Free attacker GPU memory before evaluation
                    del attacker
                    torch.cuda.empty_cache()

                    # Evaluate on targets one at a time (saves ~15GB GPU memory)
                    robust_asrs = []
                    standard_asrs = []

                    for t_name in all_targets_names:
                        if t_name == source_name:
                            continue
                        try:
                            t_model = get_model(t_name, "cifar10", "Linf", device)
                            metrics = compute_metrics(x_adv, x_test, y_test, t_model, device)
                            asr = metrics["asr"]

                            if t_name in robust_targets:
                                robust_asrs.append(asr)
                            else:
                                standard_asrs.append(asr)

                            del t_model
                            torch.cuda.empty_cache()
                        except Exception as e:
                            log_print(f"      Target {t_name} failed: {e}")

                    avg_robust = np.mean(robust_asrs) if robust_asrs else 0.0
                    avg_standard = np.mean(standard_asrs) if standard_asrs else 0.0
                    avg_all = np.mean(robust_asrs + standard_asrs) if (robust_asrs + standard_asrs) else 0.0

                    result = {
                        "seed": seed,
                        "source": source_name,
                        "source_type": source_key, # Standard/Robust
                        "method": attack_display,
                        "robust_asr": avg_robust,
                        "standard_asr": avg_standard,
                        "transfer_asr_mean": avg_all
                    }
                    all_results.append(result)

                    log_print(f"    Seed {seed}: Rob={avg_robust:.1f}%, Std={avg_standard:.1f}%")

                except Exception as e:
                    log_print(f"    Failed: {e}")
                    import traceback
                    traceback.print_exc()

        del source_model
        torch.cuda.empty_cache()

    # Save Results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.results_dir, "classical_attacks_cifar.csv"), index=False)
    
    # Generate Summaries
    log_print("\n" + "="*70)
    log_print("SUMMARY")
    log_print("="*70)
    
    if len(df) > 0:
        summary = df.groupby(["source_type", "method"])[["robust_asr", "standard_asr", "transfer_asr_mean"]].mean().reset_index()
        log_print(summary.to_string(index=False))
        summary.to_csv(os.path.join(args.results_dir, "summary_table.csv"), index=False)
        
    log_print(f"\nResults saved to: {args.results_dir}/")

if __name__ == "__main__":
    main()
