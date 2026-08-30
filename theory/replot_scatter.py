#!/usr/bin/env python
"""Draw the LGC against DI-effect scatter over the epsilon spectrum."""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(__file__)
csv_path = os.path.join(here, "..", "results", "lgc_eps_sweep",
                        "joined_with_exp_b.csv")

xs, ys, eps = [], [], []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        d = r["delta_di_minus_mi"]
        if d == "" or d is None:
            continue                       # Engstrom / Mo2022 have no sweep D_ASR
        xs.append(float(r["lgc_mean"]))
        ys.append(float(d) * 100)
        eps.append(float(r["eps_train_x255"]))

xs, ys, eps = np.array(xs), np.array(ys), np.array(eps)
tau = 0.92
coef = np.polyfit(xs, ys, 1)
r = np.corrcoef(xs, ys)[0, 1]

fig, ax = plt.subplots(figsize=(7, 5))
xf = np.linspace(xs.min(), xs.max(), 100)
ax.plot(xf, coef[0] * xf + coef[1], "k--", alpha=0.6, lw=1.5,
        label=f"linear fit ($r={r:.2f}$)")
ax.axvline(tau, color="red", ls=":", alpha=0.7, label=r"CG-DI $\tau=0.92$")
ax.axhline(0, color="gray", alpha=0.3, lw=0.8)

for x, y, e in zip(xs, ys, eps):
    marker = "o" if x > tau else "s"
    ax.scatter(x, y, s=100, c="tab:blue", marker=marker, edgecolors="black",
               linewidths=1.0, zorder=5)
    lab = fr"$\epsilon={e:.1f}/255$"
    if x > 0.90:                           # right cluster -> label to the LEFT
        ax.annotate(lab, (x, y), xytext=(x - 0.007, y), ha="right",
                    va="center", fontsize=9)
    else:                                  # left points -> label to the right
        ax.annotate(lab, (x, y), xytext=(x + 0.007, y + 0.6), ha="left",
                    fontsize=9)

ax.set_xlim(0.62, 1.01)
ax.set_xlabel("LGC (mean over clean-correct samples)", fontsize=12)
# A difference of two rates is in percentage points, not percent.
ax.set_ylabel(r"$\Delta_{\mathrm{ASR}}$ = ASR(DI) $-$ ASR(MI)  (pp)",
              fontsize=12)
ax.set_title(r"LGC vs DI sensitivity across the controlled $\epsilon$-spectrum"
             "\n" fr"target Swin-B, $N=500$, 3 seeds | Pearson $r={r:.2f}$, $n={len(xs)}$",
             fontsize=11)
ax.grid(True, ls="--", alpha=0.4)
from matplotlib.lines import Line2D
marker_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="tab:blue",
           markeredgecolor="black", markersize=9,
           label=r"LGC $\leq\tau$ (little-robustness)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
           markeredgecolor="black", markersize=9,
           label=r"LGC $>\tau$ (robust)"),
]
leg_fit = ax.legend(loc="lower left", fontsize=9)      # linear fit + CG-DI line
ax.add_artist(leg_fit)
ax.legend(handles=marker_handles, loc="upper right", fontsize=9)  # marker key
fig.tight_layout()
out = os.path.join(here, "..", "figures", "lgc_vs_dasr_scatter.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
print("saved", os.path.normpath(out))
