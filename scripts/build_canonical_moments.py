"""Build the canonical gradient-moment release from the three measurement runs.

The paper's moment tables draw on three measurement directories: the main run,
the DenseNet-121 re-measurement, and the two held-out surrogates. This script
merges their per-image files, keeps the re-measured DenseNet rows, and recomputes
the summaries from the merged data; concatenating the three existing summaries
would carry the superseded aggregates through. The aggregation is the same as in
run_moment_decomposition.py, and validate_canonical_moments.py re-derives it.

Output in results/moment_decomposition_canonical/: the two per-image files
(22,128 and 27,660 rows), their summaries (60 and 75 rows), and SHA256SUMS.
The pre-registration records under prereg/ are left untouched.
"""
import hashlib
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "moment_decomposition_canonical")

BASE = "moment_decomposition"
DENSENET_FIXED = "moment_decomposition_densenet_fixed"
HELDOUT = "moment_decomposition_heldout"
DEFECTIVE_SOURCE = "DenseNet121_Standard"   # defective only in BASE

M_EOT = 20        # draws used to form the moment estimates, as in the E6 run
N_DIM = 3 * 224 * 224


def load(d, name):
    return pd.read_csv(os.path.join(RES, d, name))


def merge_per_image(name):
    base = load(BASE, name)
    n_drop = int((base["source"] == DEFECTIVE_SOURCE).sum())
    base = base[base["source"] != DEFECTIVE_SOURCE]
    out = pd.concat([base, load(DENSENET_FIXED, name), load(HELDOUT, name)],
                    ignore_index=True)
    key = ["source", "mode", "img"] if "mode" in out.columns else \
          ["source", "img", "eps_chk"]
    assert not out.duplicated(key).any(), f"{name}: duplicate primary keys"
    out = out.sort_values(key).reset_index(drop=True)
    print(f"  {name}: dropped {n_drop} defective rows, merged to {len(out)}")
    return out


def moments_summary(df):
    num = [c for c in df.columns if c not in ("source", "mode", "img")]
    recs = []
    for (src, mode), g in df.groupby(["source", "mode"], sort=True):
        m = g[num].mean()
        E_mu_sq = m["mu_sq"]
        E_noise = N_DIM * m["sig2"]
        rho = E_mu_sq / E_noise
        gr = m["sq_gbar"] / E_mu_sq
        kappa_eff = m["V"] / (M_EOT * E_noise)
        v_r = max(m["V"] - m["Vn"], 0.0) / E_mu_sq
        t_over_n = m["Vn"] / E_noise
        c = m["c_mu"]
        a = 1 + v_r / (M_EOT * gr) if gr > 0 else np.nan
        b = t_over_n / (M_EOT * gr) if gr > 0 else np.nan
        cross = bool(np.isfinite(a) and (b < c * c) and (a > c * c))
        rho_s = (c * c - b) / (a - c * c) if cross else np.nan
        recs.append({
            "source": src, "mode": mode, "n_img": len(g),
            "c": c, "c_raw": m["c_raw"], "cos_lin": m["cos_lin"],
            "g_r": gr, "kappa_eff": kappa_eff, "a": a, "b": b,
            "rho": rho, "gcr": rho / (1 + rho),
            "crossover": "yes" if cross else "no",
            "tau_star": rho_s / (1 + rho_s) if cross else np.nan,
            "V_over_m": m["V"] / M_EOT,
            "norm_ratio": np.sqrt(m["sq_gbar"] / m["sq_gclean"]),
            "bias_disp": np.sqrt(m["sq_diff_clean"] / m["sq_gclean"]),
            "res_draw": np.sqrt(m["msq_diff_lin"] / m["msq_draw"]),
            "res_mean": np.sqrt(m["sq_diff_mean_lin"] / m["sq_gbar_lin"]),
        })
    return pd.DataFrame(recs).set_index(["source", "mode"])


def bridge_summary(dl):
    lg = dl.groupby(["source", "eps_chk"])[
        ["lgc_emp", "dot12", "sq_a", "sq_b", "mu_sq", "sig2"]].mean()
    lg["lgc2"] = lg["dot12"] / np.sqrt(lg["sq_a"] * lg["sq_b"])
    lg["rho"] = lg["mu_sq"] / (N_DIM * lg["sig2"])
    lg["gcr"] = lg["rho"] / (1 + lg["rho"])
    lg["consistency_check"] = lg["lgc2"] - lg["gcr"]
    lg["gap_eq1_vs_lgc2"] = lg["lgc_emp"] - lg["lgc2"]
    lg["lgc_emp_sq"] = lg["lgc_emp"] ** 2
    return lg[["lgc_emp", "lgc2", "gcr", "consistency_check",
               "gap_eq1_vs_lgc2", "lgc_emp_sq", "rho"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("merging per-image files")
    mom = merge_per_image("moments_per_image.csv")
    brg = merge_per_image("lgc_bridge_per_image.csv")

    print("recomputing summaries from the merged per-image data")
    msum = moments_summary(mom)
    bsum = bridge_summary(brg)

    paths = {}
    for name, obj, idx in [("moments_per_image.csv", mom, False),
                           ("moments_summary.csv", msum, True),
                           ("lgc_bridge_per_image.csv", brg, False),
                           ("lgc_bridge_summary.csv", bsum, True)]:
        p = os.path.join(OUT, name)
        obj.to_csv(p, index=idx)
        paths[name] = p

    lines = []
    for name in sorted(paths):
        h = hashlib.sha256(open(paths[name], "rb").read()).hexdigest()
        n = sum(1 for _ in open(paths[name], encoding="utf-8")) - 1
        lines.append(f"{h}  {name}")
        print(f"  {name:28s} {n:>6} rows  sha256 {h[:16]}...")
    open(os.path.join(OUT, "SHA256SUMS"), "w", encoding="utf-8",
         newline="\n").write("\n".join(lines) + "\n")

    print(f"\nsources: {mom['source'].nunique()}   "
          f"modes: {sorted(mom['mode'].unique())}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
