#!/usr/bin/env python
"""
Transform Causal Decomposition Experiment
Decomposes DI's effect into: resize-only, translation-only, full DI

Goal: Determine which component of DI causes harm on Robust surrogates
- Hypothesis: Resize is harmful, Translation is neutral
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from utils import get_device, seed_everything, compute_metrics
from robustbench.utils import clean_accuracy


# ============================================================================
# Transform Components
# ============================================================================

def apply_resize_only(x, resize_rate=1.1):
    """
    Resize-only: Scale image up/down, then resize back to original size.
    No translation/padding.
    """
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    # Resize to larger size
    rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd.item(), rnd.item()], mode='bilinear', align_corners=False)
    
    # Resize back to original (center crop if larger, pad if smaller)
    return F.interpolate(rescaled, size=[img_size, img_size], mode='bilinear', align_corners=False)


def apply_translation_only(x, max_shift=25):
    """
    Translation-only: Random shift with zero-padding. No resize.
    """
    img_size = x.shape[-1]
    
    # Random shift amounts
    shift_h = torch.randint(-max_shift, max_shift+1, (1,)).item()
    shift_w = torch.randint(-max_shift, max_shift+1, (1,)).item()
    
    # Apply shift via padding and cropping
    if shift_h >= 0:
        pad_top, pad_bottom = shift_h, 0
    else:
        pad_top, pad_bottom = 0, -shift_h
    
    if shift_w >= 0:
        pad_left, pad_right = shift_w, 0
    else:
        pad_left, pad_right = 0, -shift_w
    
    # Pad
    padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], value=0)
    
    # Crop back to original size
    h_start = pad_bottom if shift_h < 0 else 0
    w_start = pad_right if shift_w < 0 else 0
    
    return padded[:, :, h_start:h_start+img_size, w_start:w_start+img_size]


def apply_full_di(x, resize_rate=1.1, diversity_prob=1.0):
    """
    Full DI: Resize + Random Padding (standard DI-FGSM transform)
    """
    if torch.rand(1).item() > diversity_prob:
        return x
    
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    
    rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd.item(), rnd.item()], mode='bilinear', align_corners=False)
    
    h_rem = img_resize - rnd.item()
    w_rem = img_resize - rnd.item()
    pad_top = torch.randint(low=0, high=max(1, h_rem), size=(1,), dtype=torch.int32).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=max(1, w_rem), size=(1,), dtype=torch.int32).item()
    pad_right = w_rem - pad_left
    
    padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bottom], value=0)
    return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)


def apply_no_transform(x):
    """Identity - no transformation"""
    return x


# ============================================================================
# Attack with configurable transform
# ============================================================================

def mi_fgsm_attack(model, x, y, epsilon=16/255, alpha=2/255, epochs=10,
                   transform_fn=None, device='cuda'):
    """
    MI-FGSM attack with configurable input transform
    """
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    decay = 1.0
    
    for _ in range(epochs):
        # Apply transform if specified
        if transform_fn is not None:
            x_transformed = transform_fn(x + delta)
        else:
            x_transformed = x + delta
        
        # Forward pass
        logits = model(x_transformed)
        loss = F.cross_entropy(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Get gradient
        grad = delta.grad.detach()
        grad = grad / (grad.abs().mean(dim=[1,2,3], keepdim=True) + 1e-10)
        
        # Update momentum
        momentum = decay * momentum + grad
        
        # Update delta
        delta.data = delta.data + alpha * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        
        delta.grad.zero_()
    
    return (x + delta).detach()


def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    """Create Gaussian kernel for TI-FGSM translation-invariant attack"""
    x_coord = torch.arange(kernel_size, dtype=torch.float32)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
        torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel


def ti_fgsm_attack(model, x, y, epsilon=16/255, alpha=2/255, epochs=10,
                   kernel_size=5, device='cuda'):
    """
    TI-FGSM: MI-FGSM with Gaussian kernel gradient smoothing.
    Translation-invariant attack from Dong et al. (CVPR 2019).
    """
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    channels = x.shape[1]

    gaussian_kernel = get_gaussian_kernel(
        kernel_size=kernel_size, channels=channels).to(device)

    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    decay = 1.0

    for _ in range(epochs):
        logits = model(x + delta)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad = delta.grad.detach()
        # TI: convolve gradient with Gaussian kernel
        grad = F.conv2d(grad, gaussian_kernel, padding=kernel_size // 2, groups=channels)
        grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-10)

        momentum = decay * momentum + grad
        delta.data = delta.data + alpha * momentum.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(x + delta.data, 0, 1) - x

        delta.grad.zero_()

    return (x + delta).detach()


# ============================================================================
# Main Experiment
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Transform Decomposition")
    parser.add_argument("--n_examples", type=int, default=500, help="Number of examples (alias for n_samples)")
    parser.add_argument("--n_samples", type=int, default=None, help="Legacy argument for number of samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eps", type=float, default=16/255)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--mode", type=str, default="default", 
                        choices=["default", "translation_impact"], 
                        help="Experiment mode")
    parser.add_argument("--results_dir", type=str, default="results/transform_decomposition")
    args = parser.parse_args()
    
    if args.n_samples is not None:
        args.n_examples = args.n_samples
        
    return args

def main():
    args = parse_args()
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    
    os.makedirs(args.results_dir, exist_ok=True)
    log_file = os.path.join(args.results_dir, 'transform_decomposition.txt')
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
            
    log_print(f"Running Transform Decomposition (Mode: {args.mode})")
    log_print(f"N={args.n_examples}, Seeds={args.n_seeds}")

    # Load Data
    # Paper uses ImageNet for this experiment (Tab V & Appendix)
    try:
        x_test, y_test = load_dataset("imagenet", args.n_examples)
    except Exception as e:
        log_print(f"ImageNet not found ({e}), falling back to CIFAR-10")
        x_test, y_test = load_dataset("cifar10", args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # Models
    # We compare Standard vs Robust surrogates
    surrogates = {
        'Standard': get_model('ResNet50', 'imagenet', 'Linf', device),
        'Robust': get_model('Engstrom2019Robustness_ImageNet', 'imagenet', 'Linf', device)
    }
    # Target (Black-box) - Paper Table V: → Swin-B
    target = get_model('Swin_B_ImageNet', 'imagenet', 'Linf', device)

    # Transforms
    all_transforms = {
        'No-Transform': apply_no_transform,
        'Resize-Only': apply_resize_only,
        'Translation-Only': apply_translation_only,
        'Full-DI': apply_full_di,
    }
    
    # Build methods to run based on mode
    if args.mode == "translation_impact":
        # Paper appendix tab:translation_impact
        # CG-DI (p=0) = Naked, DI-FGSM (p=0.5), TI-FGSM (kernel=5)
        methods_to_run = {
            'CG-DI (p=0)': lambda model, x, y: mi_fgsm_attack(
                model, x, y, epsilon=args.eps, transform_fn=None, device=device),
            'DI-FGSM (p=0.5)': lambda model, x, y: mi_fgsm_attack(
                model, x, y, epsilon=args.eps,
                transform_fn=lambda xi: apply_full_di(xi, diversity_prob=0.5),
                device=device),
            'TI-FGSM': lambda model, x, y: ti_fgsm_attack(
                model, x, y, epsilon=args.eps, device=device),
        }
    else:
        # Default (Tab V): transform decomposition
        methods_to_run = {}
        for trans_name, trans_fn in all_transforms.items():
            methods_to_run[trans_name] = lambda model, x, y, tf=trans_fn: mi_fgsm_attack(
                model, x, y, epsilon=args.eps, transform_fn=tf, device=device)

    results = []

    seeds = [100 * (i+1) for i in range(args.n_seeds)]

    for seed in seeds:
        seed_everything(seed)
        log_print(f"\n--- Seed {seed} ---")

        for surr_name, surr_model in surrogates.items():
            log_print(f"Surrogate: {surr_name}")

            for method_name, attack_fn in methods_to_run.items():
                # Attack in batches
                x_adv_list = []
                n_batches = (args.n_examples + args.batch_size - 1) // args.batch_size
                for i in tqdm(range(n_batches), desc=f"  {method_name}", leave=False):
                    start = i * args.batch_size
                    end = min(start + args.batch_size, args.n_examples)
                    xb = x_test[start:end]
                    yb = y_test[start:end]
                    x_adv_list.append(attack_fn(surr_model, xb, yb).cpu())

                x_adv = torch.cat(x_adv_list).to(device)

                # Eval
                metrics = compute_metrics(x_adv, x_test, y_test, target, device)
                asr = metrics['asr']

                log_print(f"    {method_name}: {asr:.2f}%")

                results.append({
                    'seed': seed,
                    'surrogate': surr_name,
                    'method': method_name,
                    'asr': asr
                })

    # Save
    import pandas as pd
    df = pd.DataFrame(results)
    csv_name = 'translation_impact.csv' if args.mode == "translation_impact" else 'transform_decomposition.csv'
    df.to_csv(os.path.join(args.results_dir, csv_name), index=False)
    log_print(f"Saved results to {os.path.join(args.results_dir, csv_name)}")

if __name__ == '__main__':
    main()
