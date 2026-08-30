#!/usr/bin/env python
"""
Unified Attack Experiment for Paper Tables II/III
Evaluates 10 attacks (3 Classical + 7 Modern) with and without DI on ImageNet.

Classical: MI-FGSM, NI-FGSM, VMI-FGSM
Modern: Admix, SSA, SIA, GRA, PGN, BSR, AdaMSI

Tables II: Robust Surrogate (Engstrom -> Swin-B) - shows DI Harm
Tables III: Standard Surrogate (ResNet50 -> Swin-B) - shows DI Gain

EXCLUDED ATTACKS (see CHANGELOG.md for details):
1. TI-FGSM, SI-FGSM: Library implementation differences
   - torchattacks TIFGSM has built-in DI, TransferAttack TIM does not
   - torchattacks SINIFGSM has Nesterov, TransferAttack SIM does not
2. DeCowA, MUMODIG, GAA, OPS, etc.: Override forward() completely, bypass transform()
   - These attacks never call transform(), so our DI wrapper cannot be applied
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

# Import TransferAttack for ALL attacks (Classical + Modern)
import transferattack
from transferattack.utils import wrap_model


# ============================================================================
# Custom Attack Class that accepts our model directly
# ============================================================================

class CustomAttack:
    """
    Wrapper that creates TransferAttack-style attacks using our own models.
    Bypasses TransferAttack's model loading mechanism.
    """

    def __init__(self, attack_name, model, epsilon=16/255, alpha=None, steps=10,
                 decay=1.0, device='cuda', **kwargs):
        self.attack_name = attack_name
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha if alpha is not None else epsilon / steps * 2
        self.steps = steps
        self.decay = decay
        self.device = device
        self.kwargs = kwargs

        # Wrap model for TransferAttack compatibility
        self.wrapped_model = wrap_model(model)

    def forward(self, x, y):
        """
        Run the attack. Returns delta (perturbation).
        """
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        # Initialize delta
        delta = torch.zeros_like(x, requires_grad=True)
        momentum = torch.zeros_like(x)

        for _ in range(self.steps):
            # Transform input (can be overridden for DI)
            x_transformed = self.transform(x + delta)

            # Get logits and loss
            logits = self.wrapped_model(x_transformed)
            loss = F.cross_entropy(logits, y)

            # Compute gradient
            loss.backward()
            grad = delta.grad.detach()

            # Apply attack-specific gradient modification
            grad = self.modify_grad(grad, x, delta)

            # Update momentum (MI-FGSM style)
            grad_norm = grad.abs().mean(dim=[1,2,3], keepdim=True)
            grad = grad / (grad_norm + 1e-10)
            momentum = self.decay * momentum + grad

            # Update delta
            delta.data = delta.data + self.alpha * momentum.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.data = torch.clamp(x + delta.data, 0, 1) - x

            delta.grad.zero_()

        return delta.detach()

    def transform(self, x):
        """Input transformation. Override for DI, TI, etc."""
        return x

    def modify_grad(self, grad, x, delta):
        """Gradient modification. Override for GRA, PGN, etc."""
        return grad

    def __call__(self, x, y):
        return self.forward(x, y)


class DIAttack(CustomAttack):
    """Attack with Diverse Input (DI) transformation."""

    def __init__(self, *args, resize_rate=0.9, diversity_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def transform(self, x):
        if torch.rand(1).item() < self.diversity_prob:
            img_size = x.shape[-1]
            img_resize = int(img_size * self.resize_rate)
            rnd = torch.randint(img_resize, img_size + 1, (1,)).item()

            # Resize
            x_resized = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)

            # Random padding
            h_rem = img_size - rnd
            w_rem = img_size - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem + 1, (1,)).item()
            pad_right = w_rem - pad_left

            x = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        return x


# ============================================================================
# TransferAttack Wrapper (uses native TransferAttack implementations)
# ============================================================================

class TransferAttackWrapper:
    """
    Wrapper to use TransferAttack methods with our pre-loaded models.
    Overrides the model loading to use our models directly.
    """

    def __init__(self, attack_name, model, epsilon=16/255, steps=10,
                 add_di=False, di_prob=0.7, resize_rate=0.85, device='cuda'):
        self.attack_name = attack_name
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.add_di = add_di
        self.di_prob = di_prob
        self.resize_rate = resize_rate
        self.device = device

        # Load attack class
        attack_class = transferattack.load_attack_class(attack_name)

        # Compute alpha
        alpha = epsilon / steps * 2

        # Constructed against a placeholder name; the real model is attached below.
        self.attack = self._create_attack(attack_class, alpha)

        # Replace the model with our wrapped model
        self.attack.model = wrap_model(model)
        self.attack.device = device

        # Store original transform for DI wrapping
        self._original_transform = self.attack.transform

    def _create_attack(self, attack_class, alpha):
        """Create attack with appropriate parameters."""
        kwargs = {
            'epsilon': self.epsilon,
            'alpha': alpha,
            'epoch': self.steps,
            'device': self.device,
        }

        # Attack-specific parameters
        # --- Classical attacks ---
        if self.attack_name == 'mifgsm':
            kwargs['decay'] = 1.0
        elif self.attack_name == 'tim':
            kwargs['decay'] = 1.0
            kwargs['kernel_type'] = 'gaussian'
            kwargs['kernel_size'] = 15
        elif self.attack_name == 'sim':
            kwargs['decay'] = 1.0
            kwargs['num_scale'] = 5
        elif self.attack_name == 'vmifgsm':
            kwargs['decay'] = 1.0
            kwargs['beta'] = 1.5
            kwargs['num_neighbor'] = 20
        # --- Modern attacks ---
        elif self.attack_name == 'ssm':
            kwargs['num_spectrum'] = 20
            kwargs['rho'] = 0.5
        elif self.attack_name == 'gra':
            kwargs['num_neighbor'] = 20
            kwargs['beta'] = 3.5
        elif self.attack_name == 'pgn':
            pass  # Default params
        elif self.attack_name == 'adamsi_fgm':
            pass  # Default params
        elif self.attack_name == 'decowa':
            kwargs['mesh_width'] = 7
            kwargs['mesh_height'] = 7
        elif self.attack_name == 'mumodig':
            kwargs['N'] = 10
            kwargs['beta'] = 1.2

        # TransferAttack resolves a model from a name string at construction.
        # We suppress that and attach the already-loaded surrogate instead, so
        # every attack in the panel runs against the identical model object.
        class DummyAttack(attack_class):
            def load_model(self, model_name):
                return None

        return DummyAttack(model_name='dummy', **kwargs)

    def _di_transform(self, x, **kwargs):
        """Apply DI transformation (torchattacks style: shrink then pad) then the original transform."""
        if not hasattr(self, '_di_count'):
            self._di_count = 0
            self._di_applied = 0

        self._di_count += 1

        if torch.rand(1).item() < self.di_prob:
            self._di_applied += 1
            img_size = x.shape[-1]
            img_resize = int(img_size * self.resize_rate)  # e.g., 224 * 0.9 = 201
            rnd = torch.randint(img_resize, img_size + 1, (1,)).item()

            # Shrink
            x = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)

            # Pad back to original size
            h_rem = img_size - rnd
            w_rem = img_size - rnd
            pad_top = torch.randint(0, h_rem + 1, (1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(0, w_rem + 1, (1,)).item()
            pad_right = w_rem - pad_left

            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        return self._original_transform(x, **kwargs)

    def get_di_stats(self):
        """Return DI application statistics."""
        if hasattr(self, '_di_count') and self._di_count > 0:
            return f"DI applied {self._di_applied}/{self._di_count} times ({100*self._di_applied/self._di_count:.1f}%)"
        return "DI stats not available"

    def __call__(self, x, y):
        """Generate adversarial examples."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Apply DI wrapper if requested
        if self.add_di:
            self.attack.transform = self._di_transform
        else:
            self.attack.transform = self._original_transform

        # Run attack - TransferAttack.forward() returns delta
        delta = self.attack.forward(x, y)
        x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv


# ============================================================================
# Main Experiment
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Modern Attacks for Tables II/III")
    parser.add_argument("--n_examples", type=int, default=1000,
                        help="Number of examples per evaluation")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (adjust based on GPU memory)")
    parser.add_argument("--steps", type=int, default=10, help="Attack iterations")
    parser.add_argument("--eps", type=float, default=16/255, help="Epsilon (ImageNet default)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--results_dir", type=str,
                        default="results/paper_table2_3_modern", help="Results directory")
    return parser.parse_args()


# Fixed batch sizes for high-memory attacks (24GB GPU)
# These attacks create multiple copies of the input in transform():
#   - Admix: num_scale(5) × num_admix(3) = 15x memory
#   - SIA: num_scale = 20x memory
#   - BSR: num_scale = 20x memory
# Use fixed target batch sizes to avoid OOM while maintaining speed
ATTACK_FIXED_BATCH_SIZE = {
    "admix": 8,    # 15x memory amplification
    "sia": 4,      # 20x memory amplification
    "bsr": 4,      # 20x memory amplification
}


def get_attack_batch_size(attack_key, base_batch_size):
    """Get appropriate batch size for memory-heavy attacks."""
    if attack_key in ATTACK_FIXED_BATCH_SIZE:
        # Use fixed batch size for known high-memory attacks
        return ATTACK_FIXED_BATCH_SIZE[attack_key]
    return base_batch_size


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
    log_file = os.path.join(args.results_dir, 'modern_attacks.txt')

    # Clear log file
    open(log_file, 'w').close()

    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    # All attacks: Classical + Modern
    # Format: (Paper name, TransferAttack key, Type, Venue)
    #
    # Selection criteria (see CHANGELOG.md for detailed analysis):
    # - Must NOT override forward() completely
    # - Must call transform() so our DI wrapper can be applied
    # - Must have consistent implementation (no hidden DI or other transforms)
    #
    # EXCLUDED: TI-FGSM, SI-FGSM (library differences), DeCowA, MUMODIG, GAA, OPS (bypass transform())
    all_attacks = [
        # Classical attacks (3) - momentum/Nesterov/variance based
        ("MI-FGSM",  "mifgsm",     "Classical", "CVPR'18"),
        ("NI-FGSM",  "nifgsm",     "Classical", "ICLR'20"),  # TransferAttack: pure Nesterov, no hidden DI
        ("VMI-FGSM", "vmifgsm",    "Classical", "CVPR'21"),
        # Modern attacks (7) - input transformation based
        ("Admix",    "admix",      "Modern",    "ICCV'21"),
        ("SSA",      "ssm",        "Modern",    "ECCV'22"),
        ("SIA",      "sia",        "Modern",    "ICCV'23"),
        ("GRA",      "gra",        "Modern",    "ICCV'23"),
        ("PGN",      "pgn",        "Modern",    "NeurIPS'23"),
        ("BSR",      "bsr",        "Modern",    "CVPR'24"),
        ("AdaMSI",   "adamsi_fgm", "Modern",    "AAAI'24"),
    ]

    seeds = [100 * (i + 1) for i in range(args.n_seeds)]

    log_print("=" * 70)
    log_print("Unified Attack Experiment (Tables II/III)")
    log_print("3 Classical + 7 Modern Attacks on ImageNet")
    log_print("(See CHANGELOG.md for attack selection criteria)")
    log_print("=" * 70)
    log_print(f"Config: n_examples={args.n_examples}, n_seeds={args.n_seeds}")
    log_print(f"epsilon={args.eps:.4f}, steps={args.steps}, batch_size={args.batch_size}")
    log_print(f"Attacks: {len(all_attacks)} total (Classical + Modern)")

    # Sources: Robust and Standard on ImageNet
    sources = [
        ("Engstrom2019Robustness_ImageNet", "Robust", "Engstrom"),
        ("ResNet50", "Standard", "ResNet50"),
    ]

    # Target: Swin-B (as specified in paper Tables II/III)
    target_name = "Swin_B_ImageNet"

    all_results = []

    log_print(f"\nLoading target model: {target_name}")
    target_model = get_model(target_name, "imagenet", "Linf", device)

    # Load data once (same for all experiments with same seed)
    log_print("Loading ImageNet data...")

    for source_key, source_type, source_display in sources:
        log_print(f"\n{'='*60}")
        log_print(f"Source: {source_display} ({source_type})")
        log_print(f"{'='*60}")

        # Load source model
        source_model = get_model(source_key, "imagenet", "Linf", device)

        for attack_paper_name, attack_key, attack_type, attack_venue in all_attacks:
            # Adjust batch size for memory-heavy attacks
            attack_batch_size = get_attack_batch_size(attack_key, args.batch_size)
            if attack_batch_size != args.batch_size:
                log_print(f"\n  Attack: {attack_paper_name} ({attack_key}) [batch_size: {args.batch_size} -> {attack_batch_size} due to memory]")
            else:
                log_print(f"\n  Attack: {attack_paper_name} ({attack_key})")

            for seed in seeds:
                seed_everything(seed)

                # Load data for this seed
                x_test, y_test = load_dataset("imagenet", args.n_examples)

                # --- Naked (without DI) ---
                try:
                    # ALL attacks use TransferAttack for consistency
                    # This ensures fair "same attack ± DI" comparison
                    attacker_naked = TransferAttackWrapper(
                        attack_key, source_model,
                        epsilon=args.eps, steps=args.steps,
                        add_di=False, device=device
                    )

                    x_adv_naked = run_attack_batch(
                        attacker_naked, x_test, y_test, attack_batch_size, device,
                        desc=f"Naked seed={seed}"
                    )

                    metrics_naked = compute_metrics(x_adv_naked, x_test, y_test, target_model, device)
                    asr_naked = metrics_naked["asr"]

                except Exception as e:
                    log_print(f"    Naked failed: {e}")
                    import traceback
                    traceback.print_exc()
                    asr_naked = -1.0  # Mark as failed

                # --- With DI (Full: p=1.0, same as Table I Fixed-1.0) ---
                try:
                    # ALL attacks use TransferAttack + DI wrapper for consistency
                    # This correctly implements TI+DI, SI+DI, VMI+DI by wrapping the base attack
                    attacker_di = TransferAttackWrapper(
                        attack_key, source_model,
                        epsilon=args.eps, steps=args.steps,
                        add_di=True, di_prob=1.0, resize_rate=0.9, device=device
                    )

                    x_adv_di = run_attack_batch(
                        attacker_di, x_test, y_test, attack_batch_size, device,
                        desc=f"+DI seed={seed}"
                    )

                    metrics_di = compute_metrics(x_adv_di, x_test, y_test, target_model, device)
                    asr_di = metrics_di["asr"]

                except Exception as e:
                    log_print(f"    +DI failed: {e}")
                    import traceback
                    traceback.print_exc()
                    asr_di = -1.0  # Mark as failed

                # Calculate DI effect
                if asr_naked >= 0 and asr_di >= 0:
                    di_effect = asr_di - asr_naked
                else:
                    di_effect = 0.0

                result = {
                    "seed": seed,
                    "source": source_display,
                    "source_type": source_type,
                    "method": attack_paper_name,
                    "attack_type": attack_type,
                    "venue": attack_venue,
                    "asr_naked": asr_naked,
                    "asr_di": asr_di,
                    "di_effect": di_effect,
                }
                all_results.append(result)

                log_print(f"    Seed {seed}: Naked={asr_naked:.1f}%, +DI={asr_di:.1f}%, Effect={di_effect:+.1f}%")

                # Clear memory
                del x_test, y_test
                if 'x_adv_naked' in dir():
                    del x_adv_naked
                if 'x_adv_di' in dir():
                    del x_adv_di
                torch.cuda.empty_cache()

        del source_model
        torch.cuda.empty_cache()

    del target_model
    torch.cuda.empty_cache()

    # Save Results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.results_dir, "modern_attacks.csv"), index=False)

    # Filter out failed runs
    df_valid = df[(df["asr_naked"] >= 0) & (df["asr_di"] >= 0)]

    # Generate Table II (Robust Surrogate)
    log_print("\n" + "="*70)
    log_print("Table II: Modern Attacks on Robust Surrogate (Engstrom -> Swin-B)")
    log_print("="*70)

    robust_df = df_valid[df_valid["source_type"] == "Robust"]
    if len(robust_df) > 0:
        # Group by method while preserving attack_type and venue
        robust_summary = robust_df.groupby(["method", "attack_type", "venue"]).agg({
            "asr_naked": ["mean", "std"],
            "asr_di": ["mean", "std"],
            "di_effect": ["mean", "std"]
        }).round(2).reset_index()
        robust_summary.columns = ['Method', 'Type', 'Venue', 'Naked', 'Naked_std', '+DI(p=1.0)', 'DI_std', 'DI_Harm', 'Harm_std']
        # Sort: Classical first, then Modern, by venue year
        robust_summary = robust_summary.sort_values(['Type', 'Venue'], ascending=[False, True])
        log_print(robust_summary.to_string(index=False))
        robust_summary.to_csv(os.path.join(args.results_dir, "table2_robust.csv"), index=False)

    # Generate Table III (Standard Surrogate)
    log_print("\n" + "="*70)
    log_print("Table III: Modern Attacks on Standard Surrogate (ResNet50 -> Swin-B)")
    log_print("="*70)

    standard_df = df_valid[df_valid["source_type"] == "Standard"]
    if len(standard_df) > 0:
        # Group by method while preserving attack_type and venue
        standard_summary = standard_df.groupby(["method", "attack_type", "venue"]).agg({
            "asr_naked": ["mean", "std"],
            "asr_di": ["mean", "std"],
            "di_effect": ["mean", "std"]
        }).round(2).reset_index()
        standard_summary.columns = ['Method', 'Type', 'Venue', 'Naked', 'Naked_std', '+DI(p=1.0)', 'DI_std', 'DI_Gain', 'Gain_std']
        # Sort: Classical first, then Modern, by venue year
        standard_summary = standard_summary.sort_values(['Type', 'Venue'], ascending=[False, True])
        log_print(standard_summary.to_string(index=False))
        standard_summary.to_csv(os.path.join(args.results_dir, "table3_standard.csv"), index=False)

    log_print(f"\nResults saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
