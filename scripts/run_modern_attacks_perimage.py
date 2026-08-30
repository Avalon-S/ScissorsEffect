#!/usr/bin/env python
"""
Ten-attack panel with per-image success logging.

Reproduces the protocol of scripts/run_modern_attacks.py -- same attacks, same
wrapper (imported rather than reimplemented), same seeds, same epsilon and
steps, di_prob = 1.0 and resize_rate = 0.9 -- and additionally writes, for every
(source, attack, arm, seed, target), the boolean success vector over images. The
seed-level aggregates should therefore reproduce those of the original run,
which is a check that the added logging does not perturb the experiment.

A second target is evaluated at no extra attack cost. Shards are written
incrementally and --resume skips completed combinations.

Downstream: scripts/analyze_panel_statistics.py.

Example
-------
  python scripts/run_modern_attacks_perimage.py --n_examples 1000 --n_seeds 5
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device

# Reuse the published experiment's attack machinery verbatim: importing keeps
# this script from silently drifting from the protocol it is meant to reproduce.
from run_modern_attacks import (
    TransferAttackWrapper,
    get_attack_batch_size,
    run_attack_batch,
)

ALL_ATTACKS = [
    ("MI-FGSM", "mifgsm", "Classical", "CVPR'18"),
    ("NI-FGSM", "nifgsm", "Classical", "ICLR'20"),
    ("VMI-FGSM", "vmifgsm", "Classical", "CVPR'21"),
    ("Admix", "admix", "Modern", "ICCV'21"),
    ("SSA", "ssm", "Modern", "ECCV'22"),
    ("SIA", "sia", "Modern", "ICCV'23"),
    ("GRA", "gra", "Modern", "ICCV'23"),
    ("PGN", "pgn", "Modern", "NeurIPS'23"),
    ("BSR", "bsr", "Modern", "CVPR'24"),
    ("AdaMSI", "adamsi_fgm", "Modern", "AAAI'24"),
]

SOURCES = [
    ("Engstrom2019Robustness_ImageNet", "Robust", "Engstrom"),
    ("ResNet50", "Standard", "ResNet50"),
]

DEFAULT_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attacks", type=str, nargs="+", default=None,
                   help="Subset of TransferAttack keys (default: all ten).")
    p.add_argument("--sources", type=str, nargs="+", default=None)
    p.add_argument("--targets", type=str, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--n_examples", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps", type=float, default=16.0 / 255)
    # These two MUST match the published run (run_modern_attacks.py passes them
    # explicitly, overriding the wrapper defaults of 0.7/0.85). Getting them wrong
    # would silently break the reproduction check against Table 3.
    p.add_argument("--di_prob", type=float, default=1.0)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="results/panel_perimage")
    p.add_argument("--resume", action="store_true",
                   help="Skip (source, attack, arm, seed) combinations already "
                        "present in the on-disk npz shards.")
    return p.parse_args()


def predict_batched(model, x, device, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    shard_dir = os.path.join(args.results_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    attacks = ALL_ATTACKS if args.attacks is None else \
        [a for a in ALL_ATTACKS if a[1] in args.attacks]
    sources = SOURCES if args.sources is None else \
        [s for s in SOURCES if s[0] in args.sources]
    seeds = [100 * (i + 1) for i in range(args.n_seeds)]

    print("=" * 78)
    print("E4: ten-attack panel with per-image success logging")
    print("=" * 78)
    print(f"  N={args.n_examples}  seeds={seeds}  eps={args.eps*255:.0f}/255  "
          f"steps={args.steps}  di_prob={args.di_prob}  "
          f"resize_rate={args.resize_rate}")
    print(f"  attacks={[a[0] for a in attacks]}")
    print(f"  sources={[s[2] for s in sources]}  targets={args.targets}\n")

    # Data is reloaded per seed inside the loop (matching the published script),
    # but load_imagenet is deterministic given n_examples, so the image set is
    # identical across seeds and arms. That is what makes the pairing valid; we
    # assert it once here rather than assuming it.
    seed_everything(seeds[0])
    x_ref, y_ref = load_dataset("imagenet", args.n_examples)
    seed_everything(seeds[-1])
    x_chk, y_chk = load_dataset("imagenet", args.n_examples)
    assert torch.equal(y_ref, y_chk) and torch.allclose(x_ref, x_chk), \
        "image set differs across seeds -- image-level pairing would be invalid"
    print("[Check] image set is seed-independent: pairing across arms is valid.")

    targets, tgt_correct = {}, {}
    for t in args.targets:
        tm = get_model(t, "imagenet", "Linf", device)
        tm.eval()
        targets[t] = tm
        tgt_correct[t] = (predict_batched(tm, x_ref, device) == y_ref).numpy()
        print(f"[Target] {t}: clean-correct {tgt_correct[t].sum()}/{len(y_ref)}")

    rows = []
    y_np = y_ref.numpy()

    for src_key, src_type, src_disp in sources:
        sm = get_model(src_key, "imagenet", "Linf", device)
        sm.eval()
        src_correct = (predict_batched(sm, x_ref, device) == y_ref).numpy()
        print(f"\n{'='*70}\n[Source] {src_disp} ({src_type}) "
              f"clean-correct {src_correct.sum()}/{len(y_np)}\n{'='*70}")

        for name, key, atype, venue in attacks:
            bs = get_attack_batch_size(key, args.batch_size)
            print(f"\n  [{name}] batch_size={bs}")
            for arm, add_di in (("base", False), ("di", True)):
                for seed in seeds:
                    tag = f"{src_disp}__{key}__{arm}__{seed}"
                    shard = os.path.join(shard_dir, tag + ".npz")
                    if args.resume and os.path.exists(shard):
                        print(f"    {arm:<4} seed={seed}  [cached]")
                        z = np.load(shard)
                        for t in targets:
                            s = z[t]
                            mask = src_correct & tgt_correct[t]
                            rows.append(dict(
                                source=src_disp, src_type=src_type, attack=name,
                                attack_key=key, attack_type=atype, venue=venue,
                                arm=arm, seed=seed, target=t,
                                n_eval=int(mask.sum()),
                                asr=float(100 * s[mask].mean())))
                        continue

                    t0 = time.time()
                    seed_everything(seed)
                    x_test, y_test = load_dataset("imagenet", args.n_examples)
                    try:
                        atk = TransferAttackWrapper(
                            key, sm, epsilon=args.eps, steps=args.steps,
                            add_di=add_di, di_prob=args.di_prob,
                            resize_rate=args.resize_rate, device=device)
                        x_adv = run_attack_batch(atk, x_test, y_test, bs, device,
                                                 desc=f"{name} {arm} s={seed}")
                    except Exception as e:
                        print(f"    {arm:<4} seed={seed}  FAILED: "
                              f"{type(e).__name__}: {str(e)[:70]}")
                        continue

                    per_t = {}
                    for t, tm in targets.items():
                        pred = predict_batched(tm, x_adv, device).numpy()
                        s = (pred != y_np)
                        per_t[t] = s
                        mask = src_correct & tgt_correct[t]
                        rows.append(dict(
                            source=src_disp, src_type=src_type, attack=name,
                            attack_key=key, attack_type=atype, venue=venue,
                            arm=arm, seed=seed, target=t,
                            n_eval=int(mask.sum()),
                            asr=float(100 * s[mask].mean())))
                    np.savez_compressed(shard, **per_t)
                    print(f"    {arm:<4} seed={seed}  " + "  ".join(
                        f"{t.split('_')[0]}={100*per_t[t][src_correct & tgt_correct[t]].mean():.1f}"
                        for t in targets) + f"   ({time.time()-t0:.0f}s)",
                        flush=True)

                    # Persist the running table after every generation: this job
                    # runs for ~a day and must survive an interruption.
                    pd.DataFrame(rows).to_csv(
                        os.path.join(args.results_dir, "panel_asr.csv"),
                        index=False)

        del sm
        torch.cuda.empty_cache()

    np.savez_compressed(
        os.path.join(args.results_dir, "masks.npz"),
        labels=y_np,
        **{f"tgt_correct__{t}": tgt_correct[t] for t in targets},
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "panel_asr.csv"), index=False)
    print(f"\n[Saved] {os.path.join(args.results_dir, 'panel_asr.csv')}")
    print(f"[Saved] per-image shards in {shard_dir}")

    # Console read-out mirroring Table 3 (seed-averaged), for the reproduction
    # check against the published numbers.
    piv = (df.groupby(["source", "attack", "target", "arm"])["asr"].mean()
             .unstack("arm"))
    piv["delta"] = piv.get("di", np.nan) - piv.get("base", np.nan)
    with pd.option_context("display.width", 200):
        print("\n" + piv.round(2).to_string())


if __name__ == "__main__":
    main()
