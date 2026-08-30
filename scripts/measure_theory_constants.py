#!/usr/bin/env python
"""
Clean diagnostic: estimate the bias-variance theorem's constants (c, g_r, kappa)
on REAL surrogates using the theorem's *actual* operator R = resize-only low-pass
(deterministic downsample -> upsample), with NO random translation/padding.

Why resize-only
---------------
The theorem (App. B) models DI's resize as a symmetric low-pass contraction R
(0 <= R <= I) and models translation SEPARATELY as R ~ I (Remark 1). Earlier
attempts that used the full DI transform (resize + random pad) were wrong for this
purpose: averaging over random pad positions collapses the signal (g_r -> 0). Here
R is the deterministic resize-restore operator only, so c, g_r, kappa all refer to
the same low-pass R and g_r <= 1 by construction.

Quantities (per surrogate)
--------------------------
  mu     = mean_k grad_x L(f(x + xi_k), y)        # signal, denoised over K probes
  R mu   = upsample(downsample(mu, r))            # resize-only low-pass, deterministic
  c      = cos(R mu, mu)                           # bias direction factor in (0,1]
  g_r    = ||R mu||^2 / ||mu||^2                   # signal retention (<= 1)
  kappa  = E_z[ ||R z||^2 / ||z||^2 ] / m          # ~ tr(R^2)/(n m), Hutchinson, m=n_eot
  rho*   = (c^2 - kappa/g_r) / (1 - c^2)
  tau*   = rho*/(1+rho*)   (only when rho* > 0; crossover exists iff kappa/g_r < c^2)

We also report LGC (mean cosine of probe gradients) and whether the crossover
condition kappa/g_r < c^2 holds. This tests only whether the idealized model can
be instantiated on real surrogates; the proof holds either way.

Run on the GPU box (needs torch + project loader). Example:
  python scripts/measure_theory_constants.py --n_examples 500 --K 5 --n_eot 10
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device

DEFAULT_SOURCES = [
    # standard
    "ResNet50_Standard", "ViT_B_16_ImageNet", "Swin_B_ImageNet",
    "ConvNeXt_B_ImageNet", "DenseNet121_Standard", "InceptionV3",
    # robust (PGD-AT ResNet50 epsilon spectrum + Engstrom + Mo2022 ViT-B)
    "Engstrom2019Robustness_ImageNet", "Salman_eps0.5", "Salman_eps1.0",
    "Salman_eps2.0", "Salman_eps4.0", "Salman_eps8.0", "Mo2022When_ViT-B",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=str, nargs="+", default=DEFAULT_SOURCES,
                   help="Model registry keys (see models/loader.py).")
    p.add_argument("--dataset", type=str, default="imagenet",
                   choices=["imagenet", "cifar10"])
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_eot", type=int, default=10,
                   help="DI averaging count m used in kappa = tr(R^2)/(n m).")
    p.add_argument("--K", type=int, default=5,
                   help="Noise probes for denoising the signal (LGC-style).")
    p.add_argument("--sigma", type=float, default=1.0 / 255)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_hutchinson", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def resize_only(g, r):
    """Deterministic resize-restore low-pass: downsample by r, upsample back.
    No padding, no translation. A genuine low-pass contraction on g."""
    H, W = g.shape[-2:]
    h, w = max(1, int(round(H * r))), max(1, int(round(W * r)))
    down = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
    return up


def clean_grad(model, x, y):
    xi = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xi), y, reduction="sum").backward()
    return xi.grad.detach().clone()


def estimate_kappa(args, device, shape):
    """kappa = E_z[ ||R z||^2 / ||z||^2 ] / m,  R = resize-only."""
    ratios = []
    done = 0
    while done < args.n_hutchinson:
        b = min(args.batch_size, args.n_hutchinson - done)
        z = torch.randn((b,) + shape, device=device)
        rz = resize_only(z, args.resize_rate)
        ratios.extend((rz.flatten(1).pow(2).sum(1)
                       / z.flatten(1).pow(2).sum(1)).cpu().tolist())
        done += b
    return float(np.mean(ratios)) / args.n_eot


def main():
    args = parse_args()
    device = get_device(args.device)
    seed_everything(args.seed)
    x_test, y_test = load_dataset(args.dataset, args.n_examples)

    hdr = (f"{'source':<34}{'c(±sd)':>13}{'g_r(±sd)':>12}"
           f"{'kappa':>8}{'k/gr':>7}{'cross?':>7}{'tau*':>8}{'LGC':>7}")
    print(hdr)
    print("-" * len(hdr))
    print("(LGC here = Eq.1 clean-vs-perturbed; tau* uses resize-only R)")

    for key in args.sources:
        try:
            model = get_model(key, args.dataset, "Linf", device)
            model.eval()
        except Exception as e:
            print(f"{key:<34}  SKIP ({type(e).__name__}: {str(e)[:40]})")
            continue
        shape = tuple(x_test.shape[1:])
        kappa = estimate_kappa(args, device, shape)

        cs, grs, lgcs = [], [], []
        for i in range(0, len(x_test), args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            # clean gradient g(x), and the denoised signal mu over K perturbed probes
            g_clean = clean_grad(model, xb, yb)
            mu = torch.zeros_like(xb)
            for _ in range(args.K):
                xn = xb + args.sigma * torch.randn_like(xb)
                gk = clean_grad(model, xn, yb)
                mu += gk
                # LGC per Eq. 1: cos(grad at x, grad at x+xi)  -- matches the paper
                lgcs.extend(F.cosine_similarity(
                    g_clean.flatten(1), gk.flatten(1)).cpu().tolist())
            mu /= args.K
            Rmu = resize_only(mu, args.resize_rate)
            cs.extend(F.cosine_similarity(Rmu.flatten(1), mu.flatten(1)).cpu().tolist())
            grs.extend((Rmu.flatten(1).pow(2).sum(1)
                        / mu.flatten(1).pow(2).sum(1)).cpu().tolist())

        c, gr = float(np.mean(cs)), float(np.mean(grs))
        c_sd, gr_sd = float(np.std(cs)), float(np.std(grs))   # per-image spread (error bars)
        lgc = float(np.mean(lgcs)) if lgcs else float("nan")
        kg = kappa / gr
        cross = kg < c * c                          # crossover exists iff k/gr < c^2
        denom = 1.0 - c * c
        if cross and denom > 1e-9:
            rho_star = (c * c - kg) / denom
            tau_star = rho_star / (1 + rho_star)
        else:
            rho_star, tau_star = float("nan"), float("nan")
        print(f"{key:<34}{c:>7.3f}±{c_sd:<5.2f}{gr:>6.3f}±{gr_sd:<5.2f}"
              f"{kappa:>8.4f}{kg:>7.3f}{('yes' if cross else 'no'):>7}"
              f"{tau_star:>8.3f}{lgc:>7.3f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
