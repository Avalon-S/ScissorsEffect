"""Blocking validation of the canonical gradient-moment release.

Exits non-zero on any failure. Run it after build_canonical_moments.py, and
again after any change to the export.

The last check is the decisive one: if the export cannot reproduce the seven
frozen E6 constants, it is not the data the pre-registration was computed from.

    python scripts/validate_canonical_moments.py [--dir PATH] [--prereg PATH]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M_EOT = 20
N_DIM = 3 * 224 * 224

EXPECTED_SOURCES = {
    "ARES_ConvNeXt_B", "ConvNeXt_B_ImageNet", "DenseNet121_Standard",
    "Engstrom2019Robustness_ImageNet", "InceptionV3", "Mo2022When_ViT-B",
    "ResNet50", "Salman_eps0.5", "Salman_eps1.0", "Salman_eps2.0",
    "Salman_eps4.0", "Salman_eps8.0", "Singh2023Revisiting_ViT-B-ConvStem",
    "Swin_B_ImageNet", "ViT_B_16_ImageNet",
}
EXPECTED_ROWS = {
    "moments_per_image.csv": 22128, "moments_summary.csv": 60,
    "lgc_bridge_per_image.csv": 27660, "lgc_bridge_summary.csv": 75,
}
EXPECTED_N_IMG = {"DenseNet121_Standard": 370, "ARES_ConvNeXt_B": 379,
                  "Singh2023Revisiting_ViT-B-ConvStem": 380}

fails = []


def ok(cond, label, detail=""):
    if not cond:
        fails.append(label)
    print(f"  [{'OK ' if cond else 'BAD'}] {label}{'  ' + detail if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        ROOT, "results", "moment_decomposition_canonical"))
    ap.add_argument("--prereg", default=os.path.join(
        ROOT, "prereg",
        "E6_predictions_20260801T000247Z.json"))
    args = ap.parse_args()
    D = args.dir

    print(f"canonical dir: {D}\n")
    print("== 1. files, row counts, and full-file hashes ==")
    sums = {}
    sp = os.path.join(D, "SHA256SUMS")
    if os.path.exists(sp):
        for line in open(sp, encoding="utf-8"):
            h, n = line.split()
            sums[n] = h
    for name, want in EXPECTED_ROWS.items():
        p = os.path.join(D, name)
        if not os.path.exists(p):
            ok(False, f"{name} present")
            continue
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        ok(n == want, f"{name} has {want} rows", f"got {n}")
        if name in sums:
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            ok(h == sums[name], f"{name} matches its recorded sha256")
    if fails:
        print("\nfiles are wrong; stopping")
        sys.exit(1)

    mom = pd.read_csv(os.path.join(D, "moments_per_image.csv"))
    msum = pd.read_csv(os.path.join(D, "moments_summary.csv"))
    brg = pd.read_csv(os.path.join(D, "lgc_bridge_per_image.csv"))

    print("\n== 2. source set and per-source image counts ==")
    got = set(mom["source"])
    ok(got == EXPECTED_SOURCES, "exactly the 15 expected sources",
       f"extra={sorted(got - EXPECTED_SOURCES)} missing={sorted(EXPECTED_SOURCES - got)}")
    ok(len(msum) == 60 and not msum.duplicated(["source", "mode"]).any(),
       "summary is 15 sources x 4 modes with no duplicate key")
    for src, want in EXPECTED_N_IMG.items():
        n = sorted(msum.loc[msum["source"] == src, "n_img"].unique())
        ok(n == [want], f"{src} has n_img={want} in every mode", f"got {n}")
    ok(not (msum["n_img"] == 1).any(), "no n_img=1 row survives")
    ok(not mom["source"].str.contains("anomaly", case=False).any(),
       "no 'anomaly' source in the release")

    print("\n== 3. summary is exactly recomputable from the per-image rows ==")
    num = [c for c in mom.columns if c not in ("source", "mode", "img")]
    worst, worst_col, n_cols = 0.0, "", 0
    cross_ok = True
    for _, r in msum.iterrows():
        g = mom[(mom["source"] == r["source"]) & (mom["mode"] == r["mode"])]
        m = g[num].mean()
        E_mu_sq, E_noise = m["mu_sq"], N_DIM * m["sig2"]
        rho = E_mu_sq / E_noise
        gr = m["sq_gbar"] / E_mu_sq
        v_r = max(m["V"] - m["Vn"], 0.0) / E_mu_sq
        c = m["c_mu"]
        a = 1 + v_r / (M_EOT * gr)
        b = (m["Vn"] / E_noise) / (M_EOT * gr)
        cross = bool(np.isfinite(a) and (b < c * c) and (a > c * c))
        rho_s = (c * c - b) / (a - c * c) if cross else np.nan
        rec = {
            "n_img": len(g), "c": c, "c_raw": m["c_raw"],
            "cos_lin": m["cos_lin"], "g_r": gr, "rho": rho,
            "gcr": rho / (1 + rho),
            "kappa_eff": m["V"] / (M_EOT * E_noise), "a": a, "b": b,
            "tau_star": rho_s / (1 + rho_s) if cross else np.nan,
            "V_over_m": m["V"] / M_EOT,
            "norm_ratio": np.sqrt(m["sq_gbar"] / m["sq_gclean"]),
            "bias_disp": np.sqrt(m["sq_diff_clean"] / m["sq_gclean"]),
            "res_draw": np.sqrt(m["msq_diff_lin"] / m["msq_draw"]),
            "res_mean": np.sqrt(m["sq_diff_mean_lin"] / m["sq_gbar_lin"]),
        }
        if (r["crossover"] == "yes") != cross:
            cross_ok = False
        for k, v in rec.items():
            n_cols += 1
            if np.isnan(v):
                if not pd.isna(r[k]):
                    cross_ok = False
                continue
            dev = abs(float(r[k]) - v) / max(abs(v), 1e-12)
            if dev > worst:
                worst, worst_col = dev, k
    missing = sorted(set(msum.columns) - set(rec) - {"source", "mode", "crossover"})
    ok(not missing, "every summary column is covered by this check",
       f"unchecked: {missing}")
    ok(worst < 1e-9,
       f"all {len(rec)} summary columns re-derive from per-image data "
       f"({n_cols} values)", f"max rel. dev {worst:.2e} in '{worst_col}'")
    ok(cross_ok, "crossover flag and tau_star NaN pattern re-derive")

    print("\n== 4. the frozen E6 constants re-derive from this export ==")
    pre = json.load(open(args.prereg, encoding="utf-8"))
    mode = pre.get("transform_mode_for_tau_star", "di09")
    held = [p for p in pre["predictions"] if p.get("held_out")]
    ok(len(held) == 7, "E6 records seven held-out surrogates", f"got {len(held)}")
    idx = msum.set_index(["source", "mode"])
    worst, n_finite, pred_ok = 0.0, 0, True
    for p in held:
        row = idx.loc[(p["source"], mode)]
        for k in ("c", "g_r", "a", "b"):
            worst = max(worst, abs(float(row[k]) - p[k]))
        cross_now = row["crossover"] == "yes"
        if p["crossover_exists"] != cross_now:
            pred_ok = False
        if p["crossover_exists"]:
            n_finite += 1
            if abs(float(row["tau_star"]) - p["tau_star"]) > 5e-4:
                pred_ok = False
    ok(worst < 5e-4, "frozen c, g_r, a, b reproduce", f"max abs dev {worst:.2e}")
    ok(pred_ok, "no frozen crossover decision or tau* changes")
    ok(n_finite == 6, "six of seven held-out have a finite crossover",
       f"got {n_finite}")

    print("\n== 5. the bridge summary re-derives from its per-image rows ==")
    ok(set(brg["source"]) == EXPECTED_SOURCES, "bridge has the same 15 sources")
    bsum = pd.read_csv(os.path.join(D, "lgc_bridge_summary.csv"))
    lg = brg.groupby(["source", "eps_chk"])[
        ["lgc_emp", "dot12", "sq_a", "sq_b", "mu_sq", "sig2"]].mean()
    lg["lgc2"] = lg["dot12"] / np.sqrt(lg["sq_a"] * lg["sq_b"])
    lg["rho"] = lg["mu_sq"] / (N_DIM * lg["sig2"])
    lg["gcr"] = lg["rho"] / (1 + lg["rho"])
    lg["consistency_check"] = lg["lgc2"] - lg["gcr"]
    lg["gap_eq1_vs_lgc2"] = lg["lgc_emp"] - lg["lgc2"]
    lg["lgc_emp_sq"] = lg["lgc_emp"] ** 2
    got = bsum.set_index(["source", "eps_chk"])
    cols = ["lgc_emp", "lgc2", "gcr", "consistency_check",
            "gap_eq1_vs_lgc2", "lgc_emp_sq", "rho"]
    ok(sorted(got.columns) == sorted(cols), "bridge summary has the expected columns",
       f"got {sorted(got.columns)}")
    ok(len(got) == 75 and got.index.is_unique,
       "bridge summary is 15 sources x 5 probe scales with a unique key")
    worst_b, worst_bc = 0.0, ""
    for key, row in got.iterrows():
        for k in cols:
            v = float(lg.loc[key, k])
            dev = abs(float(row[k]) - v) / max(abs(v), 1e-12)
            if dev > worst_b:
                worst_b, worst_bc = dev, k
    ok(worst_b < 1e-9, "every bridge summary column re-derives from per-image data",
       f"max rel. dev {worst_b:.2e} in '{worst_bc}'")

    print("\n" + "=" * 60)
    if fails:
        print(f"{len(fails)} FAILURE(S): {fails}")
        sys.exit(1)
    print("CANONICAL EXPORT VALID")


if __name__ == "__main__":
    main()
