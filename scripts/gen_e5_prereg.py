#!/usr/bin/env python
"""
Emit and hash sign predictions for the DI interaction on top of each attack
method, from the stage-1 regime measurements alone.

Two readings of the same measurement are registered, because they disagree and
the outcome should adjudicate rather than the author:

  perdraw           variance of a single method draw
  ensemble_averaged variance of the m_S-averaged gradient, which is what the
                    attack actually uses (rho_avg = m rho_perdraw)

For each, records whether the method shifts the surrogate rightward relative to
no method, whether that shift grows with the method's strength parameter, and
whether it is specific to the frequency-domain method or shared by the spatial
ones.

  python scripts/gen_e5_prereg.py --gcr results/ssa_regime/method_gcr.csv
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd


# Ensemble size each method's attack actually averages over per step. Corollary
# 2 defines the M-conditioned gradient WITH m_S-fold averaging
# (Cov = sigma^2 S^2 / m_S), so the per-draw variance is not the quantity the
# corollary talks about. Plain MI-FGSM does not average, hence m=1 for "none".
# Both readings are registered below and the outcome adjudicates between them,
# rather than one being chosen after the fact.
ENSEMBLE_M = {"none": 1, "admix": 15, "sia": 20, "bsr": 20}
SSA_M = 20


def m_of(method):
    if method.startswith("ssa_rho"):
        return SSA_M
    return ENSEMBLE_M.get(method, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcr", type=str, default="results/ssa_regime/method_gcr.csv")
    p.add_argument("--out", type=str, default="prereg")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    g = pd.read_csv(args.gcr)

    # Second reading: GCR of the m_S-AVERAGED gradient, which is what the
    # corollary actually defines. rho_avg = m * rho_perdraw, since averaging m
    # independent draws divides the variance by m while leaving the mean.
    g["m_ensemble"] = g["method"].map(m_of)
    rho_pd = g["GCR_eff"] / (1 - g["GCR_eff"]).clip(lower=1e-12)
    rho_avg = rho_pd * g["m_ensemble"]
    g["GCR_eff_avg"] = rho_avg / (1 + rho_avg)

    preds, checks = [], []
    for src, sub in g.groupby("source"):
        sub = sub.set_index("method")
        if "none" not in sub.index:
            continue
        base = float(sub.loc["none", "GCR_eff"])
        ssa = sub[sub.index.str.startswith("ssa_rho")].copy()
        ssa["rho"] = [float(i.replace("ssa_rho", "")) for i in ssa.index]
        ssa = ssa.sort_values("rho")
        spatial = [m for m in ("admix", "sia", "bsr") if m in sub.index]

        base_avg = float(sub.loc["none", "GCR_eff_avg"])
        rec = dict(source=src)
        for tag, col, b in (("perdraw", "GCR_eff", base),
                            ("ensemble_averaged", "GCR_eff_avg", base_avg)):
            sh = {m: float(sub.loc[m, col]) - b for m in sub.index if m != "none"}
            ss = [sh[i] for i in ssa.index]
            rec[tag] = dict(
                base_GCR=round(b, 4),
                check_i_ssa_rightward=bool(all(v > 0 for v in ss)),
                check_ii_monotone_in_rho=bool(
                    all(a <= v + 1e-9 for a, v in zip(ss, ss[1:]))),
                check_v_ssa_specific=bool(
                    max(ss) > max([sh[m] for m in spatial] or [-9e9])),
                ssa_shifts={i: round(sh[i], 4) for i in ssa.index},
                spatial_shifts={m: round(sh[m], 4) for m in spatial})
        shift = {m: float(sub.loc[m, "GCR_eff"]) - base for m in sub.index
                 if m != "none"}
        checks.append(rec)
        for m in sub.index:
            if m == "none":
                continue
            preds.append(dict(
                source=src, method=m,
                GCR_eff=round(float(sub.loc[m, "GCR_eff"]), 4),
                shift_vs_none=round(shift[m], 4),
                predicted_effect_on_D=(
                    "more negative than D(none)" if shift[m] > 0
                    else "less negative / toward positive"),
            ))

    doc = {
        "protocol": "E5 stage-2 pre-registration, derived from stage-1 "
                    "gradient-moment measurements only. No attack outcome was "
                    "available when this file was written.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": args.gcr,
        "prediction_rule": [
            "i.   GCR_eff(SSA) > GCR_eff(none)      (rightward regime shift)",
            "ii.  the shift increases with rho",
            "iii. robust surrogate: D(SSA) < D(none) < 0",
            "iv.  standard surrogate: D moves toward zero as the shift grows",
            "v.   the shift is specific to SSA, not shared by admix/sia/bsr",
        ],
        "falsification": "Any failed check is reported. If (i), (ii) or (v) "
                         "fails, Corollary 2 does not explain why SSA is the "
                         "exception and is reported as an unresolved method "
                         "interaction, moving out of the main text.",
        "two_readings": "perdraw = variance of a single method draw; ensemble_averaged = variance of the m_S-averaged gradient, which is what Corollary 2 defines (Cov = sigma^2 S^2 / m_S) and what the attack actually uses. BOTH are registered before stage 2 so that neither can be selected after seeing the outcome.",
        "stage1_checks": checks,
        "per_method_predictions": preds,
    }
    blob = json.dumps(doc, indent=2)
    doc["sha256_of_body"] = hashlib.sha256(blob.encode()).hexdigest()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(args.out, f"E5_predictions_{stamp}.json")
    existing = [f for f in os.listdir(args.out) if f.startswith("E5_predictions_")]
    if existing and not args.force:
        sys.exit(f"REFUSING: prereg already exists ({existing}).")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[Pre-registration written] {path}")
    print(f"[sha256] {doc['sha256_of_body']}\n")
    for c in checks:
        print(f"  {c['source']}")
        for tag in ("perdraw", "ensemble_averaged"):
            d = c[tag]
            print(f"    [{tag}] base={d['base_GCR']:.4f}  "
                  f"(i) rightward={d['check_i_ssa_rightward']}  "
                  f"(ii) monotone={d['check_ii_monotone_in_rho']}  "
                  f"(v) SSA-specific={d['check_v_ssa_specific']}")
            print(f"        SSA {d['ssa_shifts']}")
            print(f"        spatial {d['spatial_shifts']}")


if __name__ == "__main__":
    main()
