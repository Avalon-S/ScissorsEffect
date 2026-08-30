"""Score the CG-DI probe against a training-metadata rule over all
model-dataset cases, and print the tally.

Both rules predict the SIGN of D = ASR(DI) - ASR(no DI):

  metadata rule : DI helps iff the surrogate was trained normally
  CG-DI  rule   : DI helps iff LGC < tau_op

A case is counted correct when the predicted sign matches the measured one.

The tally depends on the resize rate each dataset is scored at, so the rate is
an explicit argument and is printed with the result.

Usage:
    python scripts/score_cgdi_vs_metadata.py            # all datasets at r=0.9
    python scripts/score_cgdi_vs_metadata.py --c100_rate 0.6
"""
import argparse
import csv
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
TAU_OP = 0.92


def read(path):
    with open(os.path.join(RESULTS, path), newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def imagenet_cases():
    """Seven held-out ImageNet surrogates, D and LGC pooled over targets."""
    eff = read("heldout_cgdi/heldout_imagenet_effects.csv")
    raw = read("heldout_cgdi/heldout_imagenet.csv")
    lgc = {r["source"]: float(r["lgc_mean"]) for r in raw}
    per = defaultdict(list)
    typ = {}
    for r in eff:
        per[r["source"]].append(float(r["D"]))
        typ[r["source"]] = r["src_type"]
    return [("ImageNet", s, typ[s], lgc[s], mean(d)) for s, d in per.items()]


def cifar10_cases(rate):
    """Thirteen CIFAR-10 surrogates against a target outside the pool."""
    rows = read("cifar10_correlation_rates/expanded_results.csv")
    out = []
    for r in rows:
        if abs(float(r["resize_rate"]) - rate) > 1e-9:
            continue
        out.append(("CIFAR-10", r["display_name"],
                    "Standard" if r["model_type"] == "standard" else "Robust",
                    float(r["lgc"]), float(r["delta_asr_extreme"])))
    return out


def cifar100_cases(rate):
    """Seven CIFAR-100 surrogates; the run refuses self-pairs."""
    tag = "r09" if abs(rate - 0.9) < 1e-9 else "r06"
    eff = read(f"cifar100_complete_{tag}/heldout_cifar100_effects.csv")
    raw = read(f"cifar100_complete_{tag}/heldout_cifar100.csv")
    lgc = {r["source"]: float(r["lgc_mean"]) for r in raw}
    per = defaultdict(list)
    typ = {}
    for r in eff:
        per[r["source"]].append(float(r["D"]))
        typ[r["source"]] = r["src_type"]
    return [("CIFAR-100", s, typ[s], lgc[s], mean(d)) for s, d in per.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c10_rate", type=float, default=0.9)
    ap.add_argument("--c100_rate", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=TAU_OP)
    args = ap.parse_args()

    cases = (imagenet_cases()
             + cifar10_cases(args.c10_rate)
             + cifar100_cases(args.c100_rate))

    print(f"tau_op = {args.tau}   CIFAR-10 r = {args.c10_rate}   "
          f"CIFAR-100 r = {args.c100_rate}")
    print(f"{'dataset':10s} {'surrogate':38s} {'type':9s} {'LGC':>7s} "
          f"{'D':>7s}  meta cgdi")
    print("-" * 88)

    meta_ok = cgdi_ok = 0
    disagree = []
    for ds, name, typ, lgc, d in sorted(cases):
        helps = d > 0
        meta_pred = (typ == "Standard")
        cgdi_pred = lgc < args.tau
        m = meta_pred == helps
        c = cgdi_pred == helps
        meta_ok += m
        cgdi_ok += c
        if m != c:
            disagree.append((ds, name, typ, lgc, d, m, c))
        print(f"{ds:10s} {name:38s} {typ:9s} {lgc:7.4f} {d:+7.2f}  "
              f"{'Y' if m else 'n':>4s} {'Y' if c else 'n':>4s}")

    n = len(cases)
    print("-" * 88)
    print(f"n = {n} cases: metadata correct on {meta_ok}, CG-DI correct on {cgdi_ok}")
    if disagree:
        print(f"\n{len(disagree)} disagreement(s):")
        for ds, name, typ, lgc, d, m, c in disagree:
            winner = "metadata" if m else "CG-DI"
            print(f"  {ds} {name} ({typ}, LGC {lgc:.4f}, D {d:+.2f}) -> {winner}")
    else:
        print("\nthe two rules agree on every case")


if __name__ == "__main__":
    main()
