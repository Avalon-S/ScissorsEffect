#!/usr/bin/env python
"""
Per-image association between trajectory statistics and transfer outcome.

Pairs, for each image, the DI-induced change in a trajectory statistic

    d_traj(i) = mean_t align_t^{DI}(i) - mean_t align_t^{noDI}(i)

(and likewise the area between the two target-loss and target-margin curves)
with whether DI flipped that image from transfer success to failure or the
reverse. Reports a Mann-Whitney comparison, a point-biserial correlation against
the signed outcome, and P(DI hurts) across quintiles of the statistic -- the
same three tests used for the clean-point control, so the two are comparable.

  python scripts/analyze_trajectory_mediation.py --npz results/trajectory/trajectories.npz
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str,
                   default="results/trajectory/trajectories.npz")
    p.add_argument("--out", type=str,
                   default="results/trajectory/mediation.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.npz):
        sys.exit(f"missing {args.npz}")
    z = np.load(args.npz)

    keys = [k for k in z.files if k.endswith("|align")]
    combos = sorted({(k.split("|")[0], k.split("|")[3]) for k in keys})
    seeds = sorted({int(k.split("|")[2]) for k in keys})

    rows = []
    for src, tgt in combos:
        mask = z[f"tgt_correct__{tgt}"].astype(bool)

        # Average the trajectory statistics over seeds before differencing, so
        # the per-image quantity is not dominated by single-seed transform noise.
        def avg(arm, field):
            return np.mean([z[f"{src}|{arm}|{s}|{tgt}|{field}"]
                            for s in seeds], axis=0)

        d_align = (avg("di", "align") - avg("base", "align")).mean(axis=1)
        # area between target-loss curves: the integral of the loss the attack
        # actually achieves on the target, which is what drives the outcome
        d_lossarea = (avg("di", "loss") - avg("base", "loss")).sum(axis=1)
        d_marginarea = (avg("di", "margin") - avg("base", "margin")).sum(axis=1)

        sb = np.mean([z[f"{src}|base|{s}|{tgt}|succ"] for s in seeds], axis=0)
        sd = np.mean([z[f"{src}|di|{s}|{tgt}|succ"] for s in seeds], axis=0)

        # An image counts as flipped only if the change is consistent across all
        # seeds (mean 0 -> 1 or 1 -> 0), which avoids labelling seed noise.
        hurt = (sb == 1.0) & (sd == 0.0)
        helped = (sb == 0.0) & (sd == 1.0)
        same = ~(hurt | helped)
        keep = mask
        for name, stat in (("d_align_traj", d_align),
                           ("d_loss_area", d_lossarea),
                           ("d_margin_area", d_marginarea)):
            a, b = stat[keep & (hurt | helped)], stat[keep & same]
            if len(a) < 10 or len(b) < 10:
                continue
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            # point-biserial against a signed outcome, matching the published test
            sgn = np.zeros(keep.sum())
            sub = np.where(keep)[0]
            sgn[helped[sub]] = 1.0
            sgn[hurt[sub]] = -1.0
            r, pr = stats.pearsonr(stat[keep], sgn)
            # monotonicity across quintiles of the statistic
            q = pd.qcut(pd.Series(stat[keep]), 5, labels=False, duplicates="drop")
            pf = pd.Series(hurt[sub].astype(float)).groupby(q).mean()
            rows.append(dict(
                source=src, target=tgt, statistic=name,
                n=int(keep.sum()), n_changed=int((hurt | helped)[keep].sum()),
                n_hurt=int(hurt[keep].sum()), n_helped=int(helped[keep].sum()),
                mean_changed=float(a.mean()), mean_same=float(b.mean()),
                mw_p=float(u.pvalue), pointbiserial_r=float(r),
                pointbiserial_p=float(pr),
                quintile_P_hurt_min=float(pf.min()),
                quintile_P_hurt_max=float(pf.max()),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(f"[Saved] {args.out}\n")
    for st in df["statistic"].unique():
        s = df[df["statistic"] == st]
        print(f"=== {st} ===")
        print(s[["source", "target", "n", "n_hurt", "n_helped", "mean_changed",
                 "mean_same", "mw_p", "pointbiserial_r", "pointbiserial_p",
                 "quintile_P_hurt_min", "quintile_P_hurt_max"]]
              .round(4).to_string(index=False))
        sig = (s["pointbiserial_p"] < 0.05).sum()
        print(f"  -> point-biserial significant on {sig}/{len(s)} pairs; "
              f"|r| median {s['pointbiserial_r'].abs().median():.3f}\n")

    print("Comparison point: the published CLEAN-POINT control reported "
          "Mann-Whitney p=0.43, r=0.02, and P(DI hurts) flat at 0.17-0.21 "
          "across quintiles.")


if __name__ == "__main__":
    main()
