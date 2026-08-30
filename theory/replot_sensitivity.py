#!/usr/bin/env python
"""Draw transfer ASR against the LGC threshold tau, for both source types."""
import os
import csv
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

here = os.path.dirname(__file__)
csv_path = os.path.join(
    here, "..", "results", "paper_fig5_sensitivity_imagenet",
    "sensitivity_tau.csv")

rows = {}
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.setdefault(r["surrogate"], {"tau": [], "asr": []})
        rows[r["surrogate"]]["tau"].append(float(r["tau"]))
        rows[r["surrogate"]]["asr"].append(float(r["asr_mean"]))

robust = next(k for k in rows if k.startswith("Robust"))
standard = next(k for k in rows if k.startswith("Standard"))
# The plateau, not the mean over all tau: the tau=0.80 point sits below it and
# the text quotes the plateau level.
_plateau = [a for tau, a in zip(rows[standard]["tau"], rows[standard]["asr"])
            if tau >= 0.86]
std_mean = sum(_plateau) / len(_plateau)

fig, ax = plt.subplots(figsize=(7.0, 5.0))
ax.plot(rows[robust]["tau"], rows[robust]["asr"], "-o", color="red",
        lw=2, ms=8, label="Robust Source")
ax.plot(rows[standard]["tau"], rows[standard]["asr"], "-s", color="blue",
        lw=2, ms=7, label="Standard Source")
ax.axhline(std_mean, color="blue", ls=":", lw=1.6,
           label="standard plateau, " + r"$\tau\geq0.86$"
                 + f" ({std_mean:.1f}%)")
ax.axvspan(0.86, 0.96, color="grey", alpha=0.10, zorder=0)
ax.text(0.91, 41.2, r"both stable, $\tau\in[0.86,0.96]$", color="0.35",
        fontsize=10, ha="center")
ax.set_xlabel(r"LGC Threshold ($\tau$)", fontsize=14)
ax.set_ylabel("Transfer ASR (%)", fontsize=14)
ax.set_ylim(40, 80)
ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=12, loc="upper right")
fig.tight_layout()
out = os.path.join(here, "..", "figures", "fig3_sensitivity.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print("saved", os.path.normpath(out))
