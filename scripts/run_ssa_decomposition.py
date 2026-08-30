#!/usr/bin/env python
"""
SSA × transform decomposition.

Tests the frequency-interference hypothesis behind SSA's anomalous behavior in
paper Tab. 2 (SSA is the only frequency-domain attack and shows the largest DI
harm on robust source: -15.6%).

Hypothesis: SSA's DCT-domain spectrum scaling and DI's bilinear-resize low-pass
filter both perturb the LOW-FREQUENCY band where robust gradients concentrate.
Stacking them creates frequency-domain DOUBLE PENALTY. Spatial-only DI
components (translation) should NOT compound this way.

Predictions to test (analogous to paper Tab. 5 for non-SSA attacks):
  P1: SSA + resize-only ASR ≈ SSA + full-DI ASR  (resize is the harmful part)
  P2: SSA + translation-only ASR ≈ SSA-only ASR  (translation is neutral)
  P3: For Robust source, the 'resize on top of SSA' increment is large;
      for Standard source, it's small (matching the Scissors asymmetry).

Conditions (4 per source):
  C0: SSA-only       (DCT spectrum scaling, NO spatial transform)
  C1: SSA + resize   (DCT scaling + resize-then-pad-to-original)
  C2: SSA + transl   (DCT scaling + translation only, no resize)
  C3: SSA + full DI  (DCT scaling + full DI = resize + translation)

Sources: ResNet50 (Standard), Engstrom (Robust ε=4/255)
Target:  Swin-B  (matches paper Tab. 2)

                    R1/R2 minor weaknesses on SSA discussion.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device


# ============================================================================
# DCT (2D) helpers — follow SSA paper / TransferAttack convention
# ============================================================================
try:
    import torch_dct  # noqa: F401
    HAS_TORCH_DCT = True
except ImportError:
    HAS_TORCH_DCT = False


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """2D DCT-II with ortho normalization. Operates on last 2 dims."""
    if HAS_TORCH_DCT:
        import torch_dct as dct
        return dct.dct_2d(x, norm="ortho")
    # Fallback via FFT (slower; same math)
    return _dct_2d_fft(x)


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Inverse of dct_2d."""
    if HAS_TORCH_DCT:
        import torch_dct as dct
        return dct.idct_2d(x, norm="ortho")
    return _idct_2d_fft(x)


def _dct_1d_fft(x: torch.Tensor) -> torch.Tensor:
    """1D DCT-II ortho via FFT. Last dim is transformed."""
    N = x.shape[-1]
    # Symmetric extend: x_ext = concat(x, flip(x))
    x_ext = torch.cat([x, x.flip(-1)], dim=-1)
    X = torch.fft.fft(x_ext, dim=-1)[..., :N]
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * np.pi * k / (2 * N))
    out = (X * phase).real
    # Ortho normalization
    norm = torch.full((N,), np.sqrt(1.0 / (2 * N)),
                      device=x.device, dtype=x.dtype)
    norm[0] = np.sqrt(1.0 / (4 * N))
    return out * (2 * norm)


def _idct_1d_fft(x: torch.Tensor) -> torch.Tensor:
    """1D IDCT-II ortho via FFT (inverse of _dct_1d_fft)."""
    N = x.shape[-1]
    norm = torch.full((N,), np.sqrt(1.0 / (2 * N)),
                      device=x.device, dtype=x.dtype)
    norm[0] = np.sqrt(1.0 / (4 * N))
    x = x / (2 * norm)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    phase = torch.exp(1j * np.pi * k / (2 * N))
    X_full = torch.zeros(*x.shape[:-1], 2 * N,
                         dtype=torch.complex64, device=x.device)
    X_full[..., :N] = (x.to(torch.complex64) * phase)
    X_full[..., N+1:2*N] = -X_full[..., 1:N].flip(-1).conj()
    out = torch.fft.ifft(X_full, dim=-1).real[..., :N]
    return out


def _dct_2d_fft(x: torch.Tensor) -> torch.Tensor:
    return _dct_1d_fft(_dct_1d_fft(x).transpose(-1, -2)).transpose(-1, -2)


def _idct_2d_fft(x: torch.Tensor) -> torch.Tensor:
    return _idct_1d_fft(_idct_1d_fft(x).transpose(-1, -2)).transpose(-1, -2)


# ============================================================================
# Transform components
# ============================================================================
def ssa_transform(x: torch.Tensor, sigma_ssa: float = 16.0,
                  rho_ssa: float = 0.5) -> torch.Tensor:
    """
    SSA (Spectrum Simulation Attack, Long et al. ECCV'22) augmentation:
        1. Add additive gaussian noise: x' = x + N(0, (sigma/255)^2)
        2. DCT(x') -> multiply by random scaling mask m ~ U(1-rho, 1+rho)
        3. IDCT to get augmented input

    Note: SSA paper uses sigma=16 (in pixel space [0,255]) and rho=0.5.
    We assume input x in [0,1], so add noise scaled by sigma/255.
    """
    # Step 1: additive Gaussian noise (in pixel/255 scale)
    noise = torch.randn_like(x) * (sigma_ssa / 255.0)
    x_noisy = x + noise

    # Step 2: DCT then scale spectrum
    x_freq = dct_2d(x_noisy)
    mask = torch.empty_like(x_freq).uniform_(1 - rho_ssa, 1 + rho_ssa)
    x_freq_scaled = x_freq * mask

    # Step 3: inverse DCT
    x_aug = idct_2d(x_freq_scaled)
    # Clamp to valid pixel range
    x_aug = torch.clamp(x_aug, 0.0, 1.0)
    return x_aug


def resize_only(x: torch.Tensor, resize_rate: float = 0.9) -> torch.Tensor:
    """DI's resize component (no random translation pad — center pad instead).

    To isolate the LOW-PASS effect of resize from translation random shift,
    we resize and then center-pad back to original size. This matches the
    paper Tab. 5 'resize-only' decomposition.
    """
    img_size = x.shape[-1]
    if resize_rate >= 1.0:
        return x
    img_resize = random.randint(int(img_size * resize_rate), img_size)
    x_r = F.interpolate(x, size=(img_resize, img_resize),
                        mode="bilinear", align_corners=False)
    pad_total = img_size - img_resize
    pad_top = pad_total // 2
    pad_left = pad_total // 2
    pad_bottom = pad_total - pad_top
    pad_right = pad_total - pad_left
    return F.pad(x_r, (pad_left, pad_right, pad_top, pad_bottom),
                 mode="constant", value=0)


def translation_only(x: torch.Tensor, max_shift_ratio: float = 0.1
                     ) -> torch.Tensor:
    """DI's translation component: random pad on each side without resize.

    Matches paper Tab. 5 'translation-only' (random spatial shift, no
    frequency-domain perturbation from resize).
    """
    img_size = x.shape[-1]
    max_pad = int(img_size * max_shift_ratio)
    if max_pad < 1:
        return x
    pad_top = random.randint(0, max_pad)
    pad_left = random.randint(0, max_pad)
    pad_bottom = random.randint(0, max_pad)
    pad_right = random.randint(0, max_pad)
    x_p = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant", value=0)
    # Crop back to img_size from a random position
    h, w = x_p.shape[-2:]
    crop_y = random.randint(0, h - img_size)
    crop_x = random.randint(0, w - img_size)
    return x_p[..., crop_y:crop_y + img_size, crop_x:crop_x + img_size]


def full_di(x: torch.Tensor, resize_rate: float = 0.9) -> torch.Tensor:
    """Standard DI: random square resize THEN random-position pad to original.
    Matches attacks/sap.py:di_transform exactly (paper convention)."""
    img_size = x.shape[-1]
    if resize_rate >= 1.0:
        return x
    img_resize = random.randint(int(img_size * resize_rate), img_size)
    x_r = F.interpolate(x, size=(img_resize, img_resize),
                        mode="bilinear", align_corners=False)
    pad_top = random.randint(0, img_size - img_resize)
    pad_left = random.randint(0, img_size - img_resize)
    pad_bottom = img_size - img_resize - pad_top
    pad_right = img_size - img_resize - pad_left
    return F.pad(x_r, (pad_left, pad_right, pad_top, pad_bottom),
                 mode="constant", value=0)


# ============================================================================
# Combined transform: SSA + optional spatial component
# ============================================================================
SPATIAL_FNS = {
    "none":        lambda x: x,
    "resize":      resize_only,
    "translation": translation_only,
    "di":          full_di,
}


def make_combined_transform(spatial_mode: str, sigma_ssa: float,
                            rho_ssa: float, resize_rate: float):
    """Returns a function x -> SSA(x) -> spatial(x)."""
    spatial_fn = SPATIAL_FNS[spatial_mode]

    def transform(x: torch.Tensor) -> torch.Tensor:
        x = ssa_transform(x, sigma_ssa=sigma_ssa, rho_ssa=rho_ssa)
        if spatial_mode == "resize":
            x = spatial_fn(x, resize_rate=resize_rate)
        elif spatial_mode == "di":
            x = spatial_fn(x, resize_rate=resize_rate)
        else:
            x = spatial_fn(x)
        return x

    return transform


# ============================================================================
# Attack runner: MI-FGSM with custom transform + ensemble averaging
# ============================================================================
def run_attack(model, x, y, transform_fn, n_ensemble, eps, alpha, steps,
               decay=1.0):
    """
    SSA-style MI-FGSM:
      For each step t:
        For each ensemble sample n=1..N:
          x_n = transform_fn(x_adv)              # SSA + spatial transform
          g_n = grad_x L(model(x_n), y) wrt x_adv
        g_avg = mean(g_n)
        m_t   = decay * m_{t-1} + g_avg / |g_avg|_1
        x_adv = clip(x_adv + alpha * sign(m_t), 0, 1) within eps

    Returns adversarial examples [B, C, H, W].
    """
    images = x.clone().detach()
    labels = y.clone().detach()

    x_adv = images.clone().detach()
    m = torch.zeros_like(images)

    for t in range(steps):
        grad_sum = torch.zeros_like(images)
        for _ in range(n_ensemble):
            x_in = x_adv.clone().detach().requires_grad_(True)
            x_t = transform_fn(x_in)
            logits = model(x_t)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            model.zero_grad()
            if x_in.grad is not None:
                x_in.grad = None
            loss.backward()
            grad_sum += x_in.grad.detach()
        g_avg = grad_sum / n_ensemble

        # L1 normalize and momentum
        g_norm = g_avg.abs().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-12)
        g_hat = g_avg / g_norm
        m = decay * m + g_hat

        # Update
        x_adv = x_adv.detach() + alpha * m.sign()
        delta = torch.clamp(x_adv - images, min=-eps, max=eps)
        x_adv = torch.clamp(images + delta, min=0.0, max=1.0).detach()

    return x_adv


# ============================================================================
# Main experiment
# ============================================================================
DEFAULT_SOURCES = [
    ("ResNet50",                        "Standard ResNet50 (eps=0)"),
    ("Engstrom2019Robustness_ImageNet", "Engstrom Robust (eps=4/255)"),
]

CONDITIONS = ["none", "resize", "translation", "di"]
CONDITION_LABEL = {
    "none":        "SSA-only",
    "resize":      "SSA + resize",
    "translation": "SSA + translation",
    "di":          "SSA + full DI",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps_attack", type=float, default=16/255)
    p.add_argument("--alpha", type=float, default=2/255)
    p.add_argument("--sigma_ssa", type=float, default=16.0,
                   help="SSA additive noise (in pixel scale 0-255)")
    p.add_argument("--rho_ssa", type=float, default=0.5,
                   help="SSA spectrum scaling range U(1-rho, 1+rho)")
    p.add_argument("--n_ensemble", type=int, default=20,
                   help="SSA gradient ensemble size (paper: 20)")
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--target", type=str, default="Swin_B_ImageNet")
    p.add_argument("--results_dir", type=str,
                   default="results/ssa_decomposition")
    p.add_argument("--sources", type=str, nargs="+", default=None)
    p.add_argument("--conditions", type=str, nargs="+", default=None,
                   help="Subset of conditions to run")
    return p.parse_args()


def eval_asr_on_target(tgt_model, x_adv, y, batch_size, device):
    """Compute ASR = 1 - target accuracy on adv examples."""
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for i in range(0, x_adv.size(0), batch_size):
            xb = x_adv[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            preds = tgt_model(xb).argmax(dim=1)
            n_correct += (preds == yb).sum().item()
            n_total += xb.size(0)
    return 1.0 - n_correct / n_total


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78)
    print("Exp D: SSA × Transform Decomposition")
    print("=" * 78)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, "
          f"n_ensemble={args.n_ensemble}")
    print(f"  steps={args.steps}, eps={args.eps_attack:.4f}, "
          f"alpha={args.alpha:.4f}")
    print(f"  SSA: sigma={args.sigma_ssa}/255, rho={args.rho_ssa}")
    print(f"  Resize rate: {args.resize_rate}")
    print(f"  Target: {args.target}")
    print(f"  torch_dct available: {HAS_TORCH_DCT}")
    print()

    sources = DEFAULT_SOURCES
    if args.sources:
        all_sources = {n: lbl for n, lbl in DEFAULT_SOURCES}
        sources = [(s, all_sources.get(s, s)) for s in args.sources]

    conditions = args.conditions if args.conditions else CONDITIONS

    print(f"  Sources ({len(sources)}):")
    for n, lbl in sources:
        print(f"    - {n}  [{lbl}]")
    print(f"  Conditions ({len(conditions)}): {conditions}")
    print()

    # Load target once
    print(f"[Target] Loading {args.target}...")
    tgt_model = get_model(args.target, dataset="imagenet", device=device)
    tgt_model.eval()

    # Load data once
    seed_everything(42)
    print(f"[Data] Loading ImageNet ({args.n_examples} images)...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)

    # Target clean acc sanity
    n_correct = 0
    with torch.no_grad():
        for i in range(0, x_test.size(0), args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            n_correct += (tgt_model(xb).argmax(dim=1) == yb).sum().item()
    target_clean_acc = n_correct / x_test.size(0)
    print(f"  Target clean acc: {target_clean_acc*100:.2f}%")

    results = []
    overall_t0 = time.time()

    for src_name, src_label in sources:
        print(f"\n[Source] {src_name} -- {src_label}")
        try:
            src_model = get_model(src_name, dataset="imagenet", device=device)
            src_model.eval()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        for cond in conditions:
            transform_fn = make_combined_transform(
                spatial_mode=cond,
                sigma_ssa=args.sigma_ssa,
                rho_ssa=args.rho_ssa,
                resize_rate=args.resize_rate,
            )

            asr_seeds = []
            for seed_idx in range(args.n_seeds):
                seed_everything(42 + seed_idx * 100)
                t_start = time.time()
                x_adv_chunks = []
                pbar = tqdm(range(0, x_test.size(0), args.batch_size),
                            desc=f"  {cond:<12} seed={42+seed_idx*100}",
                            leave=False)
                for i in pbar:
                    xb = x_test[i:i + args.batch_size].to(device)
                    yb = y_test[i:i + args.batch_size].to(device)
                    x_adv = run_attack(
                        src_model, xb, yb, transform_fn,
                        n_ensemble=args.n_ensemble,
                        eps=args.eps_attack, alpha=args.alpha,
                        steps=args.steps,
                    )
                    x_adv_chunks.append(x_adv.cpu())
                x_adv_all = torch.cat(x_adv_chunks, dim=0)
                asr = eval_asr_on_target(
                    tgt_model, x_adv_all, y_test, args.batch_size, device
                )
                elapsed = time.time() - t_start
                asr_seeds.append(asr)
                print(f"  {CONDITION_LABEL[cond]:<22}  "
                      f"seed={42+seed_idx*100:<5}  ASR={asr*100:6.2f}%  "
                      f"t={elapsed:.1f}s")

            mean_asr = float(np.mean(asr_seeds))
            std_asr = float(np.std(asr_seeds, ddof=1)) if len(asr_seeds) > 1 else 0.0

            results.append({
                "source": src_name,
                "source_label": src_label,
                "condition": cond,
                "condition_label": CONDITION_LABEL[cond],
                "asr_mean": mean_asr,
                "asr_std": std_asr,
                "n_seeds": len(asr_seeds),
                "n_examples": args.n_examples,
                "target": args.target,
                "target_clean_acc": target_clean_acc,
            })

        del src_model
        torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_t0

    if not results:
        print("[Error] No results.")
        return

    # Save
    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.results_dir, "ssa_decomp_raw.csv")
    df.to_csv(raw_csv, index=False)

    # Pivot: source × condition table (single-level index for easy lookup)
    pivot = df.pivot_table(
        index="source",
        columns="condition",
        values="asr_mean",
    )
    pivot = pivot[CONDITIONS] if all(c in pivot.columns for c in CONDITIONS) \
            else pivot
    pivot_csv = os.path.join(args.results_dir, "ssa_decomp_pivot.csv")
    pivot.to_csv(pivot_csv)

    # Pretty summary
    print("\n" + "=" * 90)
    print(f"SUMMARY  (target={args.target}, ASR in %)")
    print("=" * 90)
    print(f"{'Source':<32}", end="")
    for cond in CONDITIONS:
        print(f"{CONDITION_LABEL[cond]:>20}", end="")
    print()
    print("-" * 90)
    for src_name, src_label in sources:
        if src_name not in df["source"].values:
            continue
        row = pivot.loc[src_name]
        print(f"{src_label:<32}", end="")
        for cond in CONDITIONS:
            v = row.get(cond, np.nan)
            print(f"{v*100:>+19.2f}%", end="")
        print()

    # Hypothesis tests for each source
    print("\n" + "=" * 90)
    print("HYPOTHESIS TESTS (per source)")
    print("=" * 90)
    for src_name, src_label in sources:
        if src_name not in pivot.index:
            continue
        row = pivot.loc[src_name]
        ssa_only = row.get("none", np.nan) * 100
        ssa_resize = row.get("resize", np.nan) * 100
        ssa_trans = row.get("translation", np.nan) * 100
        ssa_di = row.get("di", np.nan) * 100

        print(f"\n  [{src_label}]")
        print(f"    SSA-only:           {ssa_only:6.2f}%")
        print(f"    SSA + resize:       {ssa_resize:6.2f}%   "
              f"(Δ vs SSA-only = {ssa_resize - ssa_only:+.2f}%)")
        print(f"    SSA + translation:  {ssa_trans:6.2f}%   "
              f"(Δ vs SSA-only = {ssa_trans - ssa_only:+.2f}%)")
        print(f"    SSA + full DI:      {ssa_di:6.2f}%   "
              f"(Δ vs SSA-only = {ssa_di - ssa_only:+.2f}%)")

        # P1: resize ≈ full DI (resize is the main harm)
        delta_p1 = abs(ssa_resize - ssa_di)
        p1_ok = delta_p1 < 3.0  # within 3% absolute
        print(f"    P1 (resize ≈ full DI): |{ssa_resize:.1f} - {ssa_di:.1f}| "
              f"= {delta_p1:.2f}%  {'PASS' if p1_ok else 'FAIL'}")

        # P2: translation ≈ SSA-only (translation is neutral)
        delta_p2 = abs(ssa_trans - ssa_only)
        p2_ok = delta_p2 < 3.0
        print(f"    P2 (trans ≈ SSA-only): |{ssa_trans:.1f} - {ssa_only:.1f}| "
              f"= {delta_p2:.2f}%  {'PASS' if p2_ok else 'FAIL'}")

    print(f"\n  Total runtime: {overall_elapsed:.1f}s "
          f"({overall_elapsed/60:.1f} min)")
    print(f"\n[Saved] {raw_csv}")
    print(f"[Saved] {pivot_csv}")
    print("[Exp D] Done.")


if __name__ == "__main__":
    main()
