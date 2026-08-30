"""Paired confidence intervals for the 2x2 factorial.

The resampling unit is the image, paired across the four cells, and the effects
are averaged over source-target pairs within a source type. Both identities that
define the design are re-asserted on the point estimates before any interval is
formed, so a mis-specified cell cannot pass silently.
"""
import os
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "factorial_decomposition")
CELLS = {"none": "00_none", "res": "10_resize", "tr": "01_translation",
         "both": "11_both"}
BOOT = 10000
ROBUST = ("Engstrom", "Salman", "Mo2022", "ARES", "Singh")


def main():
    d = np.load(os.path.join(R, "per_image_success.npz"), allow_pickle=True)
    keys = [k for k in d.keys() if "|" in k]
    srcs = sorted({k.split("|")[0] for k in keys})
    seeds = sorted({k.split("|")[2] for k in keys})
    tgts = sorted({k.split("|")[3] for k in keys})
    n = len(d["labels"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(BOOT, n))

    def cell(s, c, t):
        """per-image success, averaged over seeds"""
        return np.mean([d[f"{s}|{CELLS[c]}|{sd}|{t}"] for sd in seeds], axis=0)

    groups = {"Robust": [], "Standard": []}
    for s in srcs:
        g = "Robust" if any(s.startswith(p) for p in ROBUST) else "Standard"
        for t in tgts:
            if f"{s}|{CELLS['none']}|{seeds[0]}|{t}" not in d:
                continue
            groups[g].append((s, t))

    for g, pairs in groups.items():
        if not pairs:
            continue
        # per-image effect vectors, averaged over the pairs in this group
        def eff(fn):
            return np.mean([fn(s, t) for s, t in pairs], axis=0)

        me_res = eff(lambda s, t: 0.5 * ((cell(s, "res", t) - cell(s, "none", t))
                                         + (cell(s, "both", t) - cell(s, "tr", t))))
        me_tr = eff(lambda s, t: 0.5 * ((cell(s, "tr", t) - cell(s, "none", t))
                                        + (cell(s, "both", t) - cell(s, "res", t))))
        se_res = eff(lambda s, t: cell(s, "res", t) - cell(s, "none", t))
        se_tr = eff(lambda s, t: cell(s, "tr", t) - cell(s, "none", t))
        inter = eff(lambda s, t: (cell(s, "both", t) - cell(s, "res", t))
                    - (cell(s, "tr", t) - cell(s, "none", t)))
        both = eff(lambda s, t: cell(s, "both", t) - cell(s, "none", t))

        # the two identities, asserted on the point estimates
        a = 100 * (me_res.mean() + me_tr.mean())
        b = 100 * (se_res.mean() + se_tr.mean() + inter.mean())
        c_ = 100 * both.mean()
        assert abs(a - c_) < 1e-9, (a, c_)
        assert abs(b - c_) < 1e-9, (b, c_)

        print(f"\n{g}  ({len(pairs)} source-target pairs, identities hold to "
              f"{max(abs(a - c_), abs(b - c_)):.2e} pp)")
        print(f"  {'effect':16s}{'estimate':>10s}{'95% CI':>22s}")
        for name, v in [("ME resize", me_res), ("ME translation", me_tr),
                        ("SE resize", se_res), ("SE translation", se_tr),
                        ("interaction", inter), ("both", both)]:
            pt = 100 * v.mean()
            bs = 100 * v[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"  {name:16s}{pt:+10.2f}   [{lo:+7.2f}, {hi:+7.2f}]")


if __name__ == "__main__":
    main()
