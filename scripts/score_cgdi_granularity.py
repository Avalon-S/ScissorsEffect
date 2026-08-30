"""Model-level CG-DI on CIFAR-10, built from the arms already run.

Per-image CG-DI picks p per image. The model-level rule computes one mean LGC per
surrogate and picks a single p for the whole attack, so its result is exactly the
p0 arm or the blind_di arm. Both were run, so no new experiment is needed.
"""
import csv, os
from collections import defaultdict

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "cifar10_cgdi_disjoint")
TAU = 0.92

rows = list(csv.DictReader(open(os.path.join(R, "heldout_cifar10.csv"), newline="")))
lgc = {r["source"]: float(r["lgc_mean"]) for r in rows}
frac = {r["source"]: float(r["cgdi_frac_off"]) for r in rows}
typ = {r["source"]: r["src_type"] for r in rows}

# mean ASR over seeds, per (source, arm, target)
acc = defaultdict(list)
for r in rows:
    acc[(r["source"], r["arm"], r["target"])].append(float(r["asr"]))
asr = {k: sum(v) / len(v) for k, v in acc.items()}
targets = sorted({r["target"] for r in rows})
srcs = sorted({r["source"] for r in rows}, key=lambda s: (typ[s] != "Robust", lgc[s]))

print(f"targets: {targets}\n")
hdr = f"{'source':24s}{'LGC':>7s}{'off%':>6s}{'rule':>7s}"
for t in targets:
    hdr += f"{'  MI':>8s}{'  DI':>8s}{' perIm':>8s}{' model':>8s}"
print(hdr)

per_gap, mod_gap = {}, {}
for s in srcs:
    line = f"{s:24s}{lgc[s]:7.4f}{100*frac[s]:6.0f}"
    decision = "p=0" if lgc[s] > TAU else "p=0.8"
    line += f"{decision:>7s}"
    pg, mg = [], []
    for t in targets:
        mi = asr[(s, "p0", t)]
        di = asr[(s, "blind_di", t)]
        pi = asr[(s, "cgdi", t)]
        ml = mi if lgc[s] > TAU else di
        best = max(mi, di)
        pg.append(pi - best)
        mg.append(ml - best)
        line += f"{mi:8.2f}{di:8.2f}{pi:8.2f}{ml:8.2f}"
    per_gap[s] = sum(pg) / len(pg)
    mod_gap[s] = sum(mg) / len(mg)
    print(line)

print(f"\n{'source':24s}{'vs best (per-image)':>22s}{'vs best (model-level)':>24s}")
for s in srcs:
    print(f"{s:24s}{per_gap[s]:22.2f}{mod_gap[s]:24.2f}")
print(f"\n{'mean':24s}{sum(per_gap.values())/len(per_gap):22.2f}"
      f"{sum(mod_gap.values())/len(mod_gap):24.2f}")
print(f"{'worst':24s}{min(per_gap.values()):22.2f}{min(mod_gap.values()):24.2f}")
