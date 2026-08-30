"""Rebuild Figure 1 (the Scissors Effect) from the per-image attack outcomes.

Reads the two NPZ files written by run_heldout_cgdi.py, aggregates on the
all-images basis, and writes the two-panel figure. Keys in the NPZ are
"{source}|{arm}|{seed}|{target}", each a 1000-length 0/1 success vector.

    python scripts/plot_fig1_scissors.py [--out FIG.pdf] [--print-only]
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SEEDS = [100, 200, 300, 400, 500]
PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_BOOT = 10000
SEED = 20260801

PANELS = [
    dict(title="ImageNet   (4 targets)",
         npz="fig1_imagenet_n1000/per_image_imagenet.npz",
         targets=["ConvNeXt_B_ImageNet", "InceptionV3",
                  "Swin_B_ImageNet", "ViT_B_16_ImageNet"],
         sources=[("ResNet50", "ResNet-50 (standard)"),
                  ("Engstrom2019Robustness_ImageNet", "Engstrom (robust)"),
                  ("Salman_eps2.0", r"Salman $\epsilon$=2 (robust)")]),
    dict(title="CIFAR-10   (2 targets)",
         npz="fig1_cifar10_n1000/per_image_cifar10.npz",
         targets=["C10_VGG16", "C10_DenseNet121"],
         sources=[("C10_ResNet18", "ResNet-18 (standard)"),
                  ("Engstrom2019Robustness", "Engstrom (robust)"),
                  ("Rice2020Overfitting", "Rice (robust)")]),
]


def arm(p):
    return "p0" if p == 0 else ("p1" if p == 1 else f"p{p:g}")


def per_image(d, src, p, targets):
    """Per-image success averaged over seeds and targets."""
    return np.mean([np.mean([d[f"{src}|{arm(p)}|{s}|{t}"] for s in SEEDS], axis=0)
                    for t in targets], axis=0)


def curve(d, src, targets, rng):
    base = per_image(d, src, 0.0, targets)
    out = []
    for p in PS:
        cur = per_image(d, src, p, targets)
        diff = 100 * (cur - base)
        idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
        b = diff[idx].mean(axis=1)
        out.append((diff.mean(), *np.percentile(b, [2.5, 97.5])))
    return 100 * base.mean(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        ROOT, "figures", "fig1_scissors_v3.pdf"))
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    data = []
    for panel in PANELS:
        d = np.load(os.path.join(RES, panel["npz"]), allow_pickle=True)
        rows = []
        for src, label in panel["sources"]:
            b, c = curve(d, src, panel["targets"], rng)
            rows.append((label, b, c))
            print(f"{panel['title'][:8]:9s} {label:26s} base {b:6.3f}%   "
                  f"D(1) {c[-1][0]:+7.3f} [{c[-1][1]:+.2f}, {c[-1][2]:+.2f}]")
        data.append((panel, rows))

    if args.print_only:
        return

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ["tab:blue", "tab:red", "tab:orange"]
    marks = ["o", "s", "^"]
    for ax, (panel, rows) in zip(axes, data):
        for (label, b, c), col, mk in zip(rows, colors, marks):
            m = np.array([x[0] for x in c])
            lo = np.array([x[1] for x in c])
            hi = np.array([x[2] for x in c])
            ax.plot(PS, m, marker=mk, ms=4, color=col, mfc="white",
                    label=f"{label}, {b:.1f}%")
            ax.fill_between(PS, lo, hi, color=col, alpha=0.18, lw=0)
            ax.annotate(f"{m[-1]:+.1f}", (PS[-1], m[-1]), textcoords="offset points",
                        xytext=(6, 0), color=col, fontsize=8, va="center")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlabel("diversity probability $p$")
        ax.set_ylabel(r"$\Delta$ASR vs. $p{=}0$  (pp)")
        ax.legend(fontsize=7, loc="lower left", frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
