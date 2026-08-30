#!/usr/bin/env python
"""
Identifiability of the continuous-p optimum.

From per-image indicators over a grid of diversity probabilities, computes for
each surrogate:

  1. ASR(p) with an image-paired bootstrap CI at every p, resampling images once
     and reusing that resample across the grid so the pairing is preserved.
  2. The identifiability margin ASR(p*) - max_{p != p*} ASR(p) with a paired
     bootstrap CI. p* counts as identifiable only if that CI excludes zero.
  3. The set of p values whose difference from ASR(p*) has a CI containing zero,
     i.e. those statistically indistinguishable from the argmax.
  4. Where the binary CG-DI choice falls relative to that set.

  python scripts/analyze_pstar_identifiability.py --npz results/continuous_p_perimage/per_image_imagenet.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

TAU_OP = 0.92
P_ON = 0.8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str,
                   default="results/continuous_p_perimage/per_image_imagenet.npz")
    p.add_argument("--lgc_csv", type=str,
                   default="results/continuous_p_perimage/heldout_imagenet.csv")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=str,
                   default="results/continuous_p_perimage/pstar_identifiability.csv")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.npz):
        sys.exit(f"missing {args.npz}")
    z = np.load(args.npz)
    keys = [k for k in z.files if "|" in k]
    sources = sorted({k.split("|")[0] for k in keys})
    targets = sorted({k.split("|")[3] for k in keys})
    arms = sorted({k.split("|")[1] for k in keys},
                  key=lambda a: float(a[1:]))
    pvals = [float(a[1:]) for a in arms]
    rng = np.random.default_rng(0)

    lgc = {}
    if os.path.exists(args.lgc_csv):
        d = pd.read_csv(args.lgc_csv)
        lgc = d.groupby("source")["lgc_mean"].mean().to_dict()

    rows = []
    for src in sources:
        for tgt in targets:
            mask = z.get(f"tgt_correct__{tgt}")
            mask = mask.astype(bool) if mask is not None else None
            # per-image success, seed-averaged, one column per p
            cols = []
            for a in arms:
                ks = [k for k in keys
                      if k.startswith(f"{src}|{a}|") and k.endswith(f"|{tgt}")]
                if not ks:
                    cols = []
                    break
                v = np.mean([z[k] for k in ks], axis=0)
                cols.append(v[mask] if mask is not None and len(mask) == len(v) else v)
            if not cols:
                continue
            M = np.stack(cols)                       # [n_p, n_img]
            n_img = M.shape[1]
            asr = M.mean(1) * 100

            # one shared image resample across all p, preserving pairing
            idx = rng.integers(0, n_img, size=(args.n_boot, n_img))
            boot = M[:, idx].mean(axis=2) * 100      # [n_p, B]
            lo = np.percentile(boot, 100 * args.alpha / 2, axis=1)
            hi = np.percentile(boot, 100 * (1 - args.alpha / 2), axis=1)

            j = int(np.argmax(asr))
            p_star = pvals[j]
            # margin against the best OTHER p, resampled jointly
            others = np.delete(np.arange(len(pvals)), j)
            margin_b = boot[j] - boot[others].max(axis=0)
            m_lo = np.percentile(margin_b, 100 * args.alpha / 2)
            m_hi = np.percentile(margin_b, 100 * (1 - args.alpha / 2))
            identifiable = m_lo > 0

            # statistically indistinguishable set: p whose gap to p* straddles 0
            indist = []
            for i in range(len(pvals)):
                if i == j:
                    indist.append(pvals[i])
                    continue
                g = boot[j] - boot[i]
                glo = np.percentile(g, 100 * args.alpha / 2)
                ghi = np.percentile(g, 100 * (1 - args.alpha / 2))
                if glo <= 0 <= ghi:
                    indist.append(pvals[i])

            l = lgc.get(src, np.nan)
            p_cgdi = 0.0 if l > TAU_OP else P_ON
            k_cgdi = int(np.argmin([abs(p - p_cgdi) for p in pvals]))

            rows.append(dict(
                source=src, target=tgt, n_img=n_img, lgc=l,
                p_star=p_star, asr_at_pstar=asr[j],
                margin=asr[j] - np.delete(asr, j).max(),
                margin_lo=m_lo, margin_hi=m_hi,
                identifiable=identifiable,
                n_indistinguishable=len(indist),
                indistinguishable_set=";".join(f"{p:g}" for p in indist),
                p_cgdi=p_cgdi, asr_at_cgdi=asr[k_cgdi],
                cgdi_gap=asr[k_cgdi] - asr[j],
                cgdi_in_indist_set=(pvals[k_cgdi] in indist),
            ))

            print(f"\n=== {src}  ->  {tgt}   (n_img={n_img}, LGC={l:.3f}) ===")
            print("     p    ASR      95% CI")
            for i, p in enumerate(pvals):
                star = " *" if i == j else ("  " if pvals[i] not in indist else " ~")
                print(f"  {p:4.1f}  {asr[i]:6.2f}  [{lo[i]:6.2f}, {hi[i]:6.2f}]{star}")
            print(f"  p* = {p_star:g}, margin over runner-up = "
                  f"{asr[j]-np.delete(asr,j).max():+.2f} pp "
                  f"[{m_lo:+.2f}, {m_hi:+.2f}]  -> "
                  f"{'IDENTIFIABLE' if identifiable else 'NOT identifiable'}")
            print(f"  statistically indistinguishable from p*: "
                  f"{{{', '.join(f'{p:g}' for p in indist)}}}  ({len(indist)} of {len(pvals)})")
            print(f"  CG-DI picks p={p_cgdi:g}: ASR {asr[k_cgdi]:.2f} "
                  f"({asr[k_cgdi]-asr[j]:+.2f} pp vs p*), "
                  f"{'inside' if pvals[k_cgdi] in indist else 'OUTSIDE'} the "
                  f"indistinguishable set")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[Saved] {args.out}")
    n_id = int(df["identifiable"].sum())
    print(f"\np* is identifiable on {n_id}/{len(df)} source-target pairs. "
          f"Near-optimality claims are restricted to those.")
    inside = int(df["cgdi_in_indist_set"].sum())
    print(f"CG-DI's binary choice is inside the statistically indistinguishable "
          f"set on {inside}/{len(df)} pairs.")


if __name__ == "__main__":
    main()
