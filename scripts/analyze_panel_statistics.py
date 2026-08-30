#!/usr/bin/env python
"""
Dependent-panel statistics for the ten-attack study.

The ten attacks share images, source and target models, software infrastructure
and algorithmic ancestry, so a sign test that treats them as independent has no
calibrated interpretation. From the per-image shards this script computes:

(1) Per-attack image-paired bootstrap CIs. One image resample is shared across
    attacks, preserving the dependence induced by the shared image set. An
    attack counts as individually resolved only if its CI excludes zero.

(2a) The panel mean under that same shared resample, which handles the shared
     image set exactly.
(2b) The mean pairwise correlation of the per-image effect vectors and the
     implied effective number of independent attacks, k/(1+(k-1)rho_bar).
(2c) DerSimonian-Laird random-effects pooling across attacks, reporting tau and
     I^2 so between-attack heterogeneity is visible.

(3) An image-block sign-flip permutation test: one sign per image, shared across
    attacks, preserving the cross-attack dependence under the null.

  python scripts/analyze_panel_statistics.py --results_dir results/panel_perimage --target Swin_B_ImageNet
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(12345)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/panel_perimage")
    p.add_argument("--target", type=str, default="Swin_B_ImageNet")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out_prefix", type=str, default=None,
                   help="Output basename. Defaults to panel_stats__<target>, so "
                        "analysing a second target cannot overwrite the first.")
    return p.parse_args()


def load_shards(shard_dir, target):
    """shards/{source}__{attack_key}__{arm}__{seed}.npz -> success[target]"""
    pat = re.compile(r"^(.+?)__(.+?)__(base|di)__(\d+)\.npz$")
    out = {}
    for f in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        src, atk, arm, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        z = np.load(f)
        if target not in z:
            continue
        out[(src, atk, arm, seed)] = z[target].astype(np.float64)
    return out


def dersimonian_laird(deltas, ses):
    """Random-effects pooling. Returns theta, se_theta, tau2, I2, n_eff."""
    d = np.asarray(deltas, float)
    v = np.asarray(ses, float) ** 2
    w = 1.0 / v
    theta_fe = (w * d).sum() / w.sum()
    Q = (w * (d - theta_fe) ** 2).sum()
    k = len(d)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    theta = (w_re * d).sum() / w_re.sum()
    se = np.sqrt(1.0 / w_re.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return theta, se, tau2, I2, Q, k


def main():
    args = parse_args()
    shard_dir = os.path.join(args.results_dir, "shards")
    if not os.path.isdir(shard_dir):
        sys.exit(f"no shards at {shard_dir}; run run_modern_attacks_perimage.py first")

    shards = load_shards(shard_dir, args.target)
    if not shards:
        sys.exit(f"no shards contain target {args.target}")

    masks_p = os.path.join(args.results_dir, "masks.npz")
    mask = None
    if os.path.exists(masks_p):
        mz = np.load(masks_p)
        k = f"tgt_correct__{args.target}"
        if k in mz:
            mask = mz[k].astype(bool)

    sources = sorted({k[0] for k in shards})
    z_crit = 1.959963984540054 if abs(args.alpha - 0.05) < 1e-9 else None

    all_rows = []
    for src in sources:
        atks = sorted({k[1] for k in shards if k[0] == src})
        print("\n" + "=" * 78)
        print(f"SOURCE: {src}   TARGET: {args.target}")
        print("=" * 78)

        # ---- per-image, seed-averaged paired difference, per attack ---------
        per_attack_diff = {}   # attack -> array over images of (di - base)
        for a in atks:
            seeds_b = sorted({k[3] for k in shards if k[:3] == (src, a, "base")})
            seeds_d = sorted({k[3] for k in shards if k[:3] == (src, a, "di")})
            seeds = sorted(set(seeds_b) & set(seeds_d))
            if not seeds:
                continue
            B = np.mean([shards[(src, a, "base", s)] for s in seeds], axis=0)
            D = np.mean([shards[(src, a, "di", s)] for s in seeds], axis=0)
            diff = D - B
            if mask is not None and len(mask) == len(diff):
                diff = diff[mask]
                B, D = B[mask], D[mask]
            per_attack_diff[a] = diff

        if not per_attack_diff:
            continue
        n_img = len(next(iter(per_attack_diff.values())))
        atk_list = list(per_attack_diff)
        M = np.stack([per_attack_diff[a] for a in atk_list])   # [A, n_img]

        # ---- (1) image-paired bootstrap, shared resample across attacks -----
        # The SAME image resample is used for every attack, which preserves the
        # cross-attack dependence induced by the shared image set.
        idx = rng.integers(0, n_img, size=(args.n_boot, n_img))
        boot = M[:, idx].mean(axis=2)                          # [A, n_boot]
        lo = np.percentile(boot, 100 * args.alpha / 2, axis=1) * 100
        hi = np.percentile(boot, 100 * (1 - args.alpha / 2), axis=1) * 100
        est = M.mean(axis=1) * 100
        se = boot.std(axis=1, ddof=1) * 100

        print(f"\n(1) Per-attack image-paired bootstrap "
              f"(n_img={n_img}, B={args.n_boot})")
        print(f"    {'attack':<12}{'Delta(pp)':>11}{'SE':>7}"
              f"{'95% CI':>20}   resolved")
        for j, a in enumerate(atk_list):
            res = "yes" if (lo[j] > 0) or (hi[j] < 0) else "NO"
            print(f"    {a:<12}{est[j]:>+11.2f}{se[j]:>7.2f}"
                  f"   [{lo[j]:+6.2f}, {hi[j]:+6.2f}]   {res}")
            all_rows.append(dict(source=src, target=args.target, attack=a,
                                 delta_pp=est[j], se_pp=se[j],
                                 ci_lo=lo[j], ci_hi=hi[j],
                                 resolved=(res == "yes")))
        n_res = sum(1 for j in range(len(atk_list)) if lo[j] > 0 or hi[j] < 0)
        print(f"    -> {n_res}/{len(atk_list)} individually resolved; "
              f"{len(atk_list)-n_res} contribute directional consistency only.")

        # ---- (2a) panel mean under the SHARED image resample ----------------
        # The bootstrap above reuses one image resample for every attack, so the
        # distribution of the panel mean already carries the dependence induced
        # by the shared image set -- no modelling assumption required.
        panel_boot = boot.mean(axis=0) * 100
        p_lo = np.percentile(panel_boot, 100 * args.alpha / 2)
        p_hi = np.percentile(panel_boot, 100 * (1 - args.alpha / 2))
        print(f"\n(2a) Panel mean, shared image-paired bootstrap")
        print(f"    mean Delta = {est.mean():+.2f} pp   "
              f"95% CI [{p_lo:+.2f}, {p_hi:+.2f}]")
        print(f"    (this interval accounts for the shared image set exactly; it "
              f"does NOT account for shared algorithmic ancestry)")

        # ---- (2b) how dependent is the panel, really? -----------------------
        # Cov(Delta_a, Delta_a') = Cov(d_a, d_a')/n_img, so the correlation of the
        # per-image effect vectors IS the correlation of the estimators. Under an
        # exchangeable correlation rho_bar, k correlated observations carry the
        # information of  k / (1 + (k-1) rho_bar)  independent ones.
        Cm = np.corrcoef(M)
        k_a = len(atk_list)
        iu = np.triu_indices(k_a, 1)
        rho_bar = float(np.mean(Cm[iu])) if k_a > 1 else 0.0
        n_ind = k_a / (1 + (k_a - 1) * max(rho_bar, 0.0))
        print(f"\n(2b) Dependence across attacks (shared images and ancestry)")
        print(f"    mean pairwise correlation of per-image effects "
              f"rho_bar = {rho_bar:+.3f}")
        print(f"    effective number of independent attacks "
              f"= k/(1+(k-1)rho_bar) = {n_ind:.1f}  (nominal k = {k_a})")

        # ---- (2c) random-effects pooling over attacks -----------------------
        theta, se_t, tau2, I2, Q, k = dersimonian_laird(est, se)
        ci = (theta - 1.96 * se_t, theta + 1.96 * se_t)
        print(f"\n(2c) Random-effects pooling over attacks (DerSimonian-Laird)")
        print(f"    theta = {theta:+.2f} pp   95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        print(f"    between-attack SD tau = {np.sqrt(tau2):.2f} pp    "
              f"I^2 = {100*I2:.0f}%    Q = {Q:.1f} (df={k-1})")
        print(f"    tau >> per-attack SE means the ten attacks are NOT ten "
              f"replicates of one effect; the panel documents a consistent sign "
              f"across heterogeneous magnitudes, not a single pooled magnitude.")

        # ---- (3) image-block permutation test -------------------------------
        # Null: arm labels exchangeable within an image. One sign per image,
        # shared across attacks -> preserves cross-attack dependence exactly.
        obs = M.mean(axis=1).mean() * 100
        signs = rng.choice([-1.0, 1.0], size=(args.n_perm, n_img))
        null = (signs @ M.T / n_img).mean(axis=1) * 100
        # Add-one correction: a permutation p-value can never legitimately be 0.
        P = args.n_perm
        p_two = (1 + (np.abs(null) >= abs(obs)).sum()) / (P + 1)
        p_one = (1 + ((null <= obs) if obs < 0 else (null >= obs)).sum()) / (P + 1)
        fmt = lambda p: f"< {1/(P+1):.1e}" if p <= 1.5 / (P + 1) else f"{p:.4g}"
        print(f"\n(3) Image-block sign-flip permutation test (P={P})")
        print(f"    observed panel mean = {obs:+.2f} pp")
        print(f"    two-sided p = {fmt(p_two)}   one-sided p = {fmt(p_one)}")
        print(f"    (attacks share one permutation per image, so the shared "
              f"experimental structure is preserved under the null)")

        all_rows.append(dict(source=src, target=args.target, attack="__PANEL__",
                             delta_pp=est.mean(), se_pp=panel_boot.std(ddof=1),
                             ci_lo=p_lo, ci_hi=p_hi, resolved=None,
                             re_theta=theta, re_ci_lo=ci[0], re_ci_hi=ci[1],
                             tau_pp=np.sqrt(tau2), I2=I2,
                             rho_bar=rho_bar, n_independent=n_ind,
                             perm_p_two=p_two, perm_p_one=p_one))

    prefix = args.out_prefix or f"panel_stats__{args.target}"
    out = os.path.join(args.results_dir, prefix + ".csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n[Saved] {out}")
    print("\nNo binomial p-values are computed anywhere in this analysis.")


if __name__ == "__main__":
    main()
