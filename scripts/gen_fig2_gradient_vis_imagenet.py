#!/usr/bin/env python
"""
Generate Fig 2 right panel: Gradient Visualization (ImageNet) with LGC annotations.

ImageNet images (224x224) provide much clearer visualization than CIFAR-10 (32x32).
Selects images with LGC near the model-level mean (not outliers) to provide
representative visualization of gradient patterns.

Output: results/paper_fig2_gradient_vis_imagenet/fig2_gradient_analysis_imagenet.{png,jpg}
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.sap import estimate_p_lgc
from utils import seed_everything, get_device


def compute_saliency(model, x, y, device):
    """Compute input gradient magnitude (saliency map) for a single image."""
    x_input = x.unsqueeze(0).to(device).requires_grad_(True)
    y_input = y.unsqueeze(0).to(device)

    output = model(x_input)
    loss = F.cross_entropy(output, y_input)
    loss.backward()

    # Gradient magnitude across channels
    grad = x_input.grad.detach().cpu().squeeze(0)
    saliency = grad.abs().mean(dim=0)  # (H, W)
    saliency = saliency.numpy()

    # Percentile clipping + normalization for clear visualization
    vmin = np.percentile(saliency, 2)
    vmax = np.percentile(saliency, 98)
    if vmax > vmin:
        saliency = np.clip(saliency, vmin, vmax)
        saliency = (saliency - vmin) / (vmax - vmin)
    return saliency


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2 gradient visualization (ImageNet)")
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--n_images", type=int, default=3, help="Number of example images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default=r"results/paper_fig2_gradient_vis_imagenet")
    args = parser.parse_args()

    device = get_device()
    seed_everything(args.seed)

    # Load ImageNet data
    x_test, y_test = load_dataset("imagenet", n_examples=args.n_examples)

    # Load ImageNet models
    std_model = get_model("ResNet50", "imagenet", "Linf", device)
    std_model.eval()
    rob_model = get_model("Engstrom2019Robustness_ImageNet", "imagenet", "Linf", device)
    rob_model.eval()

    # Compute per-image LGC for both models
    print("Computing per-image LGC for Standard model...")
    batch_size = 20  # Smaller batch for 224x224 images
    std_lgc_all = []
    rob_lgc_all = []

    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i+batch_size].to(device)
        yb = y_test[i:i+batch_size].to(device)

        _, std_lgc = estimate_p_lgc(std_model, xb, yb, device, K=5, sigma=1/255)
        std_lgc_all.append(std_lgc.cpu())

        _, rob_lgc = estimate_p_lgc(rob_model, xb, yb, device, K=5, sigma=1/255)
        rob_lgc_all.append(rob_lgc.cpu())

    std_lgc_all = torch.cat(std_lgc_all).numpy()
    rob_lgc_all = torch.cat(rob_lgc_all).numpy()

    std_mean = np.mean(std_lgc_all)
    rob_mean = np.mean(rob_lgc_all)
    print(f"Standard LGC: mean={std_mean:.3f}, std={np.std(std_lgc_all):.3f}")
    print(f"Robust LGC:   mean={rob_mean:.3f}, std={np.std(rob_lgc_all):.3f}")

    # Select images with Standard LGC near the mean
    distances = np.abs(std_lgc_all - std_mean)
    selected_indices = np.argsort(distances)[:args.n_images]

    print(f"\nSelected images (indices): {selected_indices}")
    for idx in selected_indices:
        print(f"  Image {idx}: Standard LGC={std_lgc_all[idx]:.2f}, Robust LGC={rob_lgc_all[idx]:.2f}")

    # Generate figure
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    fig, axes = plt.subplots(args.n_images, 3, figsize=(7, 2.2 * args.n_images + 0.5))

    # Column titles
    axes[0, 0].set_title("Original Image", fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f"Standard (ResNet50)\nLGC: {std_mean:.2f}", fontsize=11, fontweight='bold')
    axes[0, 2].set_title(f"Robust (Engstrom)\nLGC: {rob_mean:.2f}", fontsize=11, fontweight='bold')

    for row, idx in enumerate(selected_indices):
        x_img = x_test[idx]
        y_label = y_test[idx]

        # Original image
        img_np = x_img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[row, 0].imshow(img_np)
        axes[row, 0].axis('off')

        # Standard model saliency
        std_saliency = compute_saliency(std_model, x_img, y_label, device)
        axes[row, 1].imshow(std_saliency, cmap='magma')
        axes[row, 1].axis('off')

        # Robust model saliency
        rob_saliency = compute_saliency(rob_model, x_img, y_label, device)
        axes[row, 2].imshow(rob_saliency, cmap='magma')
        axes[row, 2].axis('off')

    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    for ext in ['png', 'jpg']:
        out_path = os.path.join(args.output_dir, f"fig2_gradient_analysis_imagenet.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path}")

    plt.close()


if __name__ == "__main__":
    main()
