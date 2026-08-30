"""The full 2x2 factorial on the all-images basis, with paired CIs, ready to
paste into tab:transform."""
import numpy as np, os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "factorial_decomposition")
C = {"none": "00_none", "res": "10_resize", "tr": "01_translation",
     "both": "11_both", "di": "di_full"}
ROB = ("Engstrom", "Salman", "Mo2022", "ARES", "Singh")
BOOT = 10000

d = np.load(os.path.join(R, "per_image_success.npz"), allow_pickle=True)
ks = [k for k in d.keys() if "|" in k]
srcs = sorted({k.split("|")[0] for k in ks})
seeds = sorted({k.split("|")[2] for k in ks})
tg = sorted({k.split("|")[3] for k in ks})
n = len(d["labels"])
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(BOOT, n))

def cell(s, c, t):
    return np.mean([d[f"{s}|{C[c]}|{sd}|{t}"] for sd in seeds], axis=0)

def is_rob(s):
    return any(s.startswith(p) for p in ROB)

EFF = {
 "ME_res": lambda s, t: 0.5*((cell(s,"res",t)-cell(s,"none",t)) + (cell(s,"both",t)-cell(s,"tr",t))),
 "ME_tr":  lambda s, t: 0.5*((cell(s,"tr",t)-cell(s,"none",t)) + (cell(s,"both",t)-cell(s,"res",t))),
 "SE_res": lambda s, t: cell(s,"res",t)-cell(s,"none",t),
 "SE_tr":  lambda s, t: cell(s,"tr",t)-cell(s,"none",t),
 "INT":    lambda s, t: (cell(s,"both",t)-cell(s,"res",t)) - (cell(s,"tr",t)-cell(s,"none",t)),
 "both":   lambda s, t: cell(s,"both",t)-cell(s,"none",t),
 "di":     lambda s, t: cell(s,"di",t)-cell(s,"none",t),
}

for g in ["Robust", "Standard"]:
    pairs = [(s, t) for s in srcs for t in tg
             if (is_rob(s) == (g == "Robust")) and f"{s}|{C['none']}|{seeds[0]}|{t}" in d]
    vals, cis = {}, {}
    for name, fn in EFF.items():
        v = np.mean([fn(s, t) for s, t in pairs], axis=0)
        vals[name] = 100 * v.mean()
        bs = 100 * v[idx].mean(axis=1)
        cis[name] = np.percentile(bs, [2.5, 97.5])
    # identities
    assert abs(vals["ME_res"] + vals["ME_tr"] - vals["both"]) < 1e-9
    assert abs(vals["SE_res"] + vals["SE_tr"] + vals["INT"] - vals["both"]) < 1e-9
    n_small = sum(1 for s, t in pairs if abs(100 * EFF["INT"](s, t).mean()) < 2)
    print(f"\n{g}  ({len(pairs)} pairs; |INT|<2 on {n_small} of {len(pairs)})")
    for name in EFF:
        lo, hi = cis[name]
        print(f"  {name:8s} {vals[name]:+7.2f}  [{lo:+7.2f}, {hi:+7.2f}]")
    print(f"  resize share: {abs(vals['ME_res'])/abs(vals['both'])*100:.0f}% of the "
          f"composition, {abs(vals['ME_res'])/abs(vals['di'])*100:.0f}% of the operator")
