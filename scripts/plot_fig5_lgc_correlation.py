"""Figure 5: LGC against the DI effect, two panels.

Left  : CIFAR-10, 13 surrogates at r=0.9, transferred to a naturally trained
        WRN-28-10 outside the pool.
Right : ImageNet, the 7 held-out surrogates, transferred to four cross-family
        targets with self-pairs excluded.

Both panels use the all-images basis and DI at p=0.8, so they are directly
comparable. The right panel replaces the controlled eps-spectrum, whose
correlation was carried entirely by its single standard point: dropping that one
point moved Pearson from -0.865 to -0.331 (p=0.59). The held-out panel has five
standard points and no comparable leverage (leave-one-out Pearson stays within
-0.80 to -0.90).

Writes figures/fig5_lgc_correlation_v3.{pdf,png} and prints every number the
caption quotes.
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
TAU_OP = 0.92
SEEDS = ["100", "200", "300", "400", "500"]
IMAGENET_TARGETS = ["ConvNeXt_B_ImageNet", "InceptionV3",
                    "Swin_B_ImageNet", "ViT_B_16_ImageNet"]


def cifar10_points():
    """13 surrogates at r=0.9, all images, p=0.8, target outside the pool."""
    path = os.path.join(RESULTS, "cifar10_correlation_rates", "expanded_results.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if abs(float(r["resize_rate"]) - 0.9) < 1e-9]
    assert len(rows) == 13, len(rows)
    return [(r["display_name"], r["model_type"] == "standard",
             float(r["lgc"]), float(r["delta_asr_extreme"])) for r in rows]


def imagenet_points():
    """7 held-out surrogates, all images, p=0.8 (the blind_di arm)."""
    d = np.load(os.path.join(RESULTS, "heldout_cgdi", "per_image_imagenet.npz"),
                allow_pickle=True)
    with open(os.path.join(RESULTS, "heldout_cgdi", "heldout_imagenet.csv"),
              newline="") as f:
        raw = list(csv.DictReader(f))
    lgc = {r["source"]: float(r["lgc_mean"]) for r in raw}
    typ = {r["source"]: r["src_type"] for r in raw}
    srcs = sorted({k.split("|")[0] for k in d.keys() if "|" in k})
    assert len(srcs) == 7, srcs

    out = []
    for s in srcs:
        targets = [t for t in IMAGENET_TARGETS if t != s]   # never score a self-pair
        assert targets, s
        per = []
        for t in targets:
            a0 = np.mean([d[f"{s}|p0|{x}|{t}"].mean() for x in SEEDS])
            a1 = np.mean([d[f"{s}|blind_di|{x}|{t}"].mean() for x in SEEDS])
            per.append(100 * (a1 - a0))
        out.append((s, typ[s] == "Standard", lgc[s], float(np.mean(per))))
    return out


def stats_for(points):
    x = [p[2] for p in points]
    y = [p[3] for p in points]
    pr = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    loo = [stats.pearsonr([v for j, v in enumerate(x) if j != i],
                          [v for j, v in enumerate(y) if j != i])[0]
           for i in range(len(x))]
    return pr, sp, (min(loo), max(loo))


def fmt_p(p):
    """p to three decimals, but never as 'p=0.000'."""
    return r"p<0.001" if p < 0.001 else f"p={p:.3f}"


def panel(ax, points, title, pr, sp):
    for is_std, colour, marker, label in [(True, "#1f6fb4", "o", "standard"),
                                          (False, "#c0392b", "s", "robust")]:
        xs = [p[2] for p in points if p[1] == is_std]
        ys = [p[3] for p in points if p[1] == is_std]
        ax.scatter(xs, ys, c=colour, marker=marker, s=42, label=label,
                   edgecolors="none", zorder=3)
    ax.axhline(0, color="#444444", lw=1.0, zorder=1)
    ax.axvline(TAU_OP, color="#888888", ls="--", lw=1.0, zorder=1)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("LGC", fontsize=11)
    ax.set_ylabel(r"$\Delta$ASR (pp)", fontsize=11)
    ax.grid(axis="y", color="#dddddd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=9, loc="upper right", frameon=True,
                    framealpha=0.95, edgecolor="#999999", fancybox=False,
                    borderpad=0.5)
    leg.get_frame().set_linewidth(0.7)
    txt = (f"Pearson ${pr[0]:.2f}$ (${fmt_p(pr[1])}$)\n"
           f"Spearman ${sp[0]:.2f}$ (${fmt_p(sp[1])}$)\n"
           r"$\tau_{\rm op}=0.92$ dashed")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=8.5,
            va="bottom", ha="left", zorder=4,
            bbox=dict(boxstyle="square,pad=0.5", facecolor="white",
                      edgecolor="#999999", linewidth=0.7, alpha=0.95))


def main():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    c10 = cifar10_points()
    imn = imagenet_points()

    pr_c, sp_c, loo_c = stats_for(c10)
    pr_i, sp_i, loo_i = stats_for(imn)

    print("CIFAR-10 (r=0.9, p=0.8, all images, target outside pool)")
    for n, is_std, l, dd in sorted(c10, key=lambda p: p[2]):
        print(f"  {n:26s} {'std' if is_std else 'rob'}  LGC {l:.4f}  D {dd:+7.2f}")
    print(f"  Pearson {pr_c[0]:+.3f} (p={pr_c[1]:.4f})  "
          f"Spearman {sp_c[0]:+.3f} (p={sp_c[1]:.4f})  LOO {loo_c[0]:+.3f}..{loo_c[1]:+.3f}")

    print("\nImageNet held-out panel (p=0.8, all images, self-pairs excluded)")
    for n, is_std, l, dd in sorted(imn, key=lambda p: p[2]):
        print(f"  {n:36s} {'std' if is_std else 'rob'}  LGC {l:.4f}  D {dd:+7.2f}")
    print(f"  Pearson {pr_i[0]:+.3f} (p={pr_i[1]:.4f})  "
          f"Spearman {sp_i[0]:+.3f} (p={sp_i[1]:.4f})  LOO {loo_i[0]:+.3f}..{loo_i[1]:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    panel(axes[0], c10, f"CIFAR-10, $r=0.9$  ($n={len(c10)}$)", pr_c, sp_c)
    panel(axes[1], imn, f"ImageNet, held-out surrogates  ($n={len(imn)}$)", pr_i, sp_i)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig5_lgc_correlation_v3.{ext}"),
                    dpi=200, bbox_inches="tight")
    print("\nwrote", os.path.join(FIGDIR, "fig5_lgc_correlation_v3.pdf"))


if __name__ == "__main__":
    main()
