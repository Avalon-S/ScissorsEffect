"""Image-paired bootstrap intervals for the bias-variance table.

Both quantities are ratios of expectations over images, so the interval has to
resample images and recompute the ratio, not average per-image ratios. Prints
the LaTeX rows for tab:biasvar.

    python scripts/bootstrap_biasvar.py [--boot 10000] [--mode di09]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "results", "moment_decomposition_canonical")
M_EOT = 20
N_DIM = 3 * 224 * 224
SEED = 20260801

ORDER = [
    ("standard", "Swin-B", "Swin_B_ImageNet"),
    ("standard", "ResNet-50", "ResNet50"),
    ("standard", "DenseNet-121", "DenseNet121_Standard"),
    ("standard", "InceptionV3", "InceptionV3"),
    ("standard", "ConvNeXt-B", "ConvNeXt_B_ImageNet"),
    ("standard", "ViT-B/16", "ViT_B_16_ImageNet"),
    ("robust", r"Salman $\epsilon{=}8$", "Salman_eps8.0"),
    ("robust", "Engstrom", "Engstrom2019Robustness_ImageNet"),
    ("robust", r"Salman $\epsilon{=}4$", "Salman_eps4.0"),
    ("robust", r"Salman $\epsilon{=}2$", "Salman_eps2.0"),
    ("robust", r"Salman $\epsilon{=}1$", "Salman_eps1.0"),
    ("robust", r"Salman $\epsilon{=}0.5$", "Salman_eps0.5"),
    ("robust", "ARES ConvNeXt-AT", "ARES_ConvNeXt_B"),
    ("robust", "Singh ConvStem-AT", "Singh2023Revisiting_ViT-B-ConvStem"),
    ("robust", "Mo2022", "Mo2022When_ViT-B"),
]


def stats(g):
    """The three ratio-of-expectation constants on one image set."""
    rho = g["mu_sq"].mean() / (N_DIM * g["sig2"].mean())
    vr = g["V"].mean() / (M_EOT * N_DIM * g["sig2"].mean())
    bias = np.sqrt(g["sq_diff_clean"].mean() / g["sq_gclean"].mean())
    return rho, vr, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--mode", default="di09")
    args = ap.parse_args()

    per = pd.read_csv(os.path.join(CANON, "moments_per_image.csv"))
    per = per[per["mode"] == args.mode]
    rng = np.random.default_rng(SEED)

    print(f"mode={args.mode}  B={args.boot}  seed={SEED}\n")
    print(f"{'surrogate':22s} {'n':>4s} {'rho':>9s} "
          f"{'var ratio':>10s} {'95% CI':>18s} {'bias':>6s} {'95% CI':>14s}")
    rows = []
    for group, label, src in ORDER:
        g = per[per["source"] == src]
        arr = g[["mu_sq", "sig2", "V", "sq_diff_clean", "sq_gclean"]].to_numpy()
        cols = {c: i for i, c in enumerate(
            ["mu_sq", "sig2", "V", "sq_diff_clean", "sq_gclean"])}
        n = len(arr)
        rho, vr, bias = stats(g)

        idx = rng.integers(0, n, size=(args.boot, n))
        s = arr[idx].mean(axis=1)
        b_vr = s[:, cols["V"]] / (M_EOT * N_DIM * s[:, cols["sig2"]])
        b_bi = np.sqrt(s[:, cols["sq_diff_clean"]] / s[:, cols["sq_gclean"]])
        vlo, vhi = np.percentile(b_vr, [2.5, 97.5])
        blo, bhi = np.percentile(b_bi, [2.5, 97.5])

        print(f"{label:22s} {n:4d} {rho:9.1f} {vr:10.3f} "
              f"[{vlo:7.3f},{vhi:8.3f}] {bias:6.2f} [{blo:5.2f},{bhi:5.2f}]")
        rows.append((group, label, rho, vr, vlo, vhi, bias, blo, bhi))

    print("\n% --- LaTeX rows for tab:biasvar ---")
    cur = None
    for group, label, rho, vr, vlo, vhi, bias, blo, bhi in rows:
        if group != cur:
            if cur is not None:
                print(r"\midrule")
            print(r"\multicolumn{4}{l}{\textit{" + group + r"}} \\")
            cur = group
        best = r"\mathbf{" + f"{vr:.3f}" + "}" if label == r"Salman $\epsilon{=}8$" \
            else f"{vr:.3f}"
        print(f"{label} & ${rho:.1f}$ & ${best}$ "
              f"$[{vlo:.3f},{vhi:.3f}]$ & ${bias:.2f}$ $[{blo:.2f},{bhi:.2f}]$ "
              + chr(92) + chr(92))

    below = [r for r in rows if r[0] == "standard" and r[5] < 1.0]
    above = [r for r in rows if r[0] == "robust" and r[4] > 1.0]
    print(f"\nstandard with CI entirely below 1: {len(below)} of 6")
    print(f"robust   with CI entirely above 1: {len(above)} of 9  "
          f"({[r[1] for r in rows if r[0]=='robust' and r[4] <= 1.0]} not resolved)")


if __name__ == "__main__":
    main()
