#!/usr/bin/env python
"""
Emit and hash the predictions of the CG-DI rule and of the per-surrogate
theoretical threshold, from gradient-geometry measurements alone.

Reads only quantities that involve no source-side attack outcome, and writes:

  (a) the operational decision p = 0 if LGC > tau_op else 0.8, with tau_op, the
      probe scale and K frozen at their development-set values;
  (b) the theoretical prediction "DI helps iff GCR < tau*(f)", where tau*(f)
      comes from that surrogate's own measured (c, g_r, a, b) via
      rho* = (c^2 - b)/(a - c^2) and tau* = rho*/(1+rho*).

The output is hashed, and the script refuses to overwrite an existing file so
that predictions cannot be regenerated once outcomes are known.

  python scripts/gen_e6_prereg.py --moments <moments_summary.csv> --lgc <lgc_summary.csv>
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --- Development-set values. FROZEN. Chosen on CIFAR-10 (full pool) plus the two
# ImageNet surrogates that the published paper already used as sources
# (ResNet50, Engstrom). Nothing below may be tuned on held-out outcomes.
DEV_SET = ["ResNet50", "Engstrom2019Robustness_ImageNet"]
TAU_OP = 0.92          # CG-DI threshold, on LGC_emp (Eq. 1)
EPS_CHK = 1.0 / 255    # probe scale
K_PROBE = 5            # probes per image
P_ON, P_OFF = 0.8, 0.0

# Surrogates that the published paper already reports source-side DI outcomes
# for. Everything else is held out.
ALREADY_RUN_AS_SOURCE = {
    "ResNet50", "Engstrom2019Robustness_ImageNet",
    "Salman_eps0.5", "Salman_eps1.0", "Salman_eps2.0",
    "Salman_eps4.0", "Salman_eps8.0", "Mo2022When_ViT-B",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--moments", type=str,
                   default="results/moment_decomposition/moments_summary.csv")
    p.add_argument("--lgc", type=str,
                   default="results/moment_decomposition/lgc_bridge_summary.csv")
    p.add_argument("--mode", type=str, default="di09",
                   help="Which transform's constants define tau*: di09 is the "
                        "torchattacks DI used by the main-paper results.")
    p.add_argument("--extra_moments", type=str, nargs="*", default=[],
                   help="Additional moments_summary.csv files (held-out models "
                        "measured in a separate E1 run).")
    p.add_argument("--extra_lgc", type=str, nargs="*", default=[])
    p.add_argument("--out", type=str, default="prereg")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing prereg file. Using this after "
                        "seeing outcomes invalidates the protocol.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Extras are concatenated after the main file with keep='last', so a
    # re-measured surrogate supersedes the earlier row.
    mom = pd.concat([pd.read_csv(f) for f in [args.moments] + args.extra_moments])
    lgc = pd.concat([pd.read_csv(f) for f in [args.lgc] + args.extra_lgc])
    mom = mom.drop_duplicates(subset=["source", "mode"], keep="last")
    lgc = lgc.drop_duplicates(subset=["source", "eps_chk"], keep="last")

    mom = mom[mom["mode"] == args.mode]
    if mom.empty:
        sys.exit(f"no rows with mode={args.mode} in {args.moments}")

    # LGC_emp at the frozen probe scale
    lgc_e = lgc[np.isclose(lgc["eps_chk"], EPS_CHK, atol=1e-6)]
    lgc_map = dict(zip(lgc_e["source"], lgc_e["lgc_emp"]))
    gcr_map = dict(zip(lgc_e["source"], lgc_e["gcr"]))

    preds = []
    for _, r in mom.iterrows():
        src = r["source"]
        c, gr, a, b = r["c"], r["g_r"], r["a"], r["b"]
        crossover = (b < c * c) and (a > c * c)
        if crossover:
            rho_s = (c * c - b) / (a - c * c)
            tau_star = rho_s / (1 + rho_s)
        else:
            tau_star = None

        lgc_v = lgc_map.get(src, np.nan)
        gcr_v = gcr_map.get(src, r.get("gcr", np.nan))

        # (a) CG-DI operational decision -- uses LGC_emp only
        cgdi_p = P_OFF if lgc_v > TAU_OP else P_ON
        cgdi_says = "DI off (predicts DI would hurt)" if cgdi_p == P_OFF \
            else "DI on (predicts DI would help)"

        # (b) theory per-surrogate prediction -- uses GCR vs tau*(f)
        if tau_star is None:
            # b >= c^2: phi < 1 for every rho, so the model says DI never helps
            theory = "hurt (no crossover: model predicts DI hurts at every GCR)"
        else:
            theory = "help" if gcr_v < tau_star else "hurt"

        preds.append(dict(
            source=src,
            held_out=src not in ALREADY_RUN_AS_SOURCE,
            c=round(float(c), 4), g_r=round(float(gr), 4),
            a=round(float(a), 4), b=round(float(b), 4),
            crossover_exists=bool(crossover),
            tau_star=None if tau_star is None else round(float(tau_star), 4),
            LGC_emp=round(float(lgc_v), 4) if np.isfinite(lgc_v) else None,
            GCR=round(float(gcr_v), 4) if np.isfinite(gcr_v) else None,
            cgdi_p=cgdi_p, cgdi_prediction=cgdi_says,
            theory_prediction=theory,
            agree=(("hurt" in theory) == (cgdi_p == P_OFF)),
        ))

    doc = {
        "protocol": "E6/E5 pre-registration. Predictions derived ONLY from "
                    "gradient-geometry measurements (E1); no source-side attack "
                    "outcome was available when this file was written.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_hyperparameters": {
            "tau_op": TAU_OP, "eps_chk": EPS_CHK, "K_probe": K_PROBE,
            "p_on": P_ON, "p_off": P_OFF,
            "development_set": DEV_SET,
            "note": "tau_op, eps_chk and K were fixed in the published version "
                    "of the paper; they are not re-tuned here.",
        },
        "transform_mode_for_tau_star": args.mode,
        "inputs": {
            "moments": [args.moments] + args.extra_moments,
            "lgc": [args.lgc] + args.extra_lgc,
        },
        "falsification_criteria": {
            "cgdi": "For each held-out surrogate, CG-DI's binary decision is "
                    "WRONG if the measured sign of D = ASR(DI) - ASR(no-DI) "
                    "disagrees with cgdi_prediction. Every disagreement is "
                    "reported, including the known Salman eps=8 failure.",
            "theory": "The per-surrogate rule 'DI helps iff GCR < tau*(f)' is "
                      "FALSIFIED on any surrogate whose measured sign of D "
                      "disagrees with theory_prediction. Standard-side "
                      "surrogates are the decisive test, since the rule "
                      "predicts harm for them.",
        },
        "predictions": preds,
    }

    blob = json.dumps(doc, indent=2, sort_keys=False)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    doc["sha256_of_body"] = sha

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(args.out, f"E6_predictions_{stamp}.json")
    existing = [f for f in os.listdir(args.out) if f.startswith("E6_predictions_")]
    if existing and not args.force:
        sys.exit(f"REFUSING: a prereg file already exists ({existing}). "
                 f"Regenerating after seeing outcomes would invalidate the "
                 f"protocol. Use --force only if no held-out attack has run.")

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    with open(os.path.join(args.out, "LATEST_SHA256.txt"), "w") as f:
        f.write(f"{sha}  {os.path.basename(path)}\n")

    print(f"[Pre-registration written] {path}")
    print(f"[sha256] {sha}\n")
    hdr = (f"{'surrogate':<34}{'held-out':>9}{'LGC':>7}{'GCR':>7}"
           f"{'tau*':>8}{'CG-DI':>8}  theory")
    print(hdr)
    print("-" * (len(hdr) + 10))
    for p in preds:
        print(f"{p['source']:<34}{str(p['held_out']):>9}"
              f"{p['LGC_emp'] if p['LGC_emp'] is not None else float('nan'):>7.3f}"
              f"{p['GCR'] if p['GCR'] is not None else float('nan'):>7.3f}"
              f"{(p['tau_star'] if p['tau_star'] is not None else float('nan')):>8.3f}"
              f"{('p=0' if p['cgdi_p'] == 0 else 'p=0.8'):>8}  "
              f"{p['theory_prediction']}"
              f"{'' if p['agree'] else '   <-- CG-DI and theory DISAGREE'}")

    n_ho = sum(p["held_out"] for p in preds)
    n_dis = sum(1 for p in preds if not p["agree"])
    print(f"\n{n_ho} held-out surrogates; {n_dis} where CG-DI and the "
          f"per-surrogate theory make different predictions.")
    print("These disagreements are the informative cases: the held-out run "
          "adjudicates between the operational rule and the theory.")


if __name__ == "__main__":
    main()
