#!/usr/bin/env python
"""
Attack-trajectory instrumentation.

Runs MI-FGSM and DI-FGSM on identical images and seeds and records, at every
attack step t and for every image:

    align_t   = cos( sign(g_src^t), sign(g_tgt^t) )
    loss_t    = target cross-entropy at the current iterate
    margin_t  = target logit margin (true class minus best other)

together with the per-image transfer outcome. Supports the population question
(whether the DI-vs-no-DI gap accumulates over t, and in opposite directions for
standard and robust surrogates) and the per-image question (whether a
trajectory-integrated statistic predicts which images DI flips), the latter
handled by scripts/analyze_trajectory_mediation.py.

Example
-------
  python scripts/run_trajectory_analysis.py --n_examples 500 --n_seeds 3
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device


def di_transform(x, resize_rate=0.9):
    S = x.shape[-1]
    lo = int(S * resize_rate)
    rnd = torch.randint(lo, S + 1, (1,)).item()
    xr = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pt = torch.randint(0, S - rnd + 1, (1,)).item()
    pl = torch.randint(0, S - rnd + 1, (1,)).item()
    return F.pad(xr, (pl, S - rnd - pl, pt, S - rnd - pt), value=0)


def sign_cos(a, b):
    return F.cosine_similarity(torch.sign(a.flatten(1)),
                               torch.sign(b.flatten(1)), dim=1)


def target_probe(tm, x, y):
    """Target-side gradient, loss and logit margin at the current iterate."""
    xi = x.clone().detach().requires_grad_(True)
    logits = tm(xi)
    loss_vec = F.cross_entropy(logits, y, reduction="none")
    loss_vec.sum().backward()
    g = xi.grad.detach().clone()
    tm.zero_grad(set_to_none=True)
    with torch.no_grad():
        true = logits.gather(1, y.view(-1, 1)).squeeze(1)
        tmp = logits.clone()
        tmp.scatter_(1, y.view(-1, 1), -float("inf"))
        margin = true - tmp.max(1).values
    return g, loss_vec.detach(), margin


def run_arm(sm, tgts, x, y, use_di, eps, alpha, steps, resize_rate, decay=1.0):
    """One MI/DI-FGSM trajectory, instrumented at every step."""
    x = x.detach()
    B = x.size(0)
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    rec = {t: {"align": [], "loss": [], "margin": []} for t in tgts}
    for _ in range(steps):
        xin = x + delta
        if use_di:
            xin = di_transform(xin, resize_rate)
        loss = F.cross_entropy(sm(xin), y)
        g_src = torch.autograd.grad(loss, delta)[0]

        with torch.enable_grad():
            for tname, tm in tgts.items():
                g_t, l_t, m_t = target_probe(tm, (x + delta).detach(), y)
                rec[tname]["align"].append(sign_cos(g_src, g_t).cpu().numpy())
                rec[tname]["loss"].append(l_t.cpu().numpy())
                rec[tname]["margin"].append(m_t.cpu().numpy())

        gn = g_src / (g_src.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-10)
        momentum = decay * momentum + gn
        delta = (delta.detach() + alpha * momentum.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)

    xadv = (x + delta).detach()
    succ = {}
    with torch.no_grad():
        for tname, tm in tgts.items():
            succ[tname] = (tm(xadv).argmax(1) != y).cpu().numpy()
    out = {t: {k: np.stack(v, axis=1) for k, v in rec[t].items()}
           for t in tgts}     # each [B, steps]
    return out, succ


DEFAULT_SOURCES = [
    ("ResNet50", "Standard"),
    ("Engstrom2019Robustness_ImageNet", "Robust"),
    ("Salman_eps2.0", "Robust"),
    ("Mo2022When_ViT-B", "Robust"),
]
DEFAULT_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=str, nargs="+", default=None)
    p.add_argument("--targets", type=str, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--eps", type=float, default=16.0 / 255)
    p.add_argument("--alpha", type=float, default=2.0 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="results/trajectory")
    return p.parse_args()


def predict(model, x, device, bs):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    sources = DEFAULT_SOURCES if not args.sources else \
        [(s.split(":")[0], s.split(":")[1] if ":" in s else "?")
         for s in args.sources]
    seeds = [100 * (i + 1) for i in range(args.n_seeds)]

    print("=" * 78)
    print("E2: trajectory-level alignment and loss evolution")
    print("=" * 78)
    print(f"  N={args.n_examples} seeds={seeds} steps={args.steps} "
          f"sources={[s for s, _ in sources]} targets={args.targets}\n")

    seed_everything(seeds[0])
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    N = x_test.size(0)

    tgts, tcorr = {}, {}
    for t in args.targets:
        tm = get_model(t, dataset="imagenet", device=device)
        tm.eval()
        tgts[t] = tm
        tcorr[t] = (predict(tm, x_test, device, args.batch_size)
                    == y_test).numpy()
        print(f"[Target] {t}: clean-correct {tcorr[t].sum()}/{N}")

    store, rows = {}, []
    for src_key, src_type in sources:
        try:
            sm = get_model(src_key, dataset="imagenet", device=device)
            sm.eval()
        except Exception as e:
            print(f"[SKIP] {src_key}: {type(e).__name__}: {str(e)[:60]}")
            continue
        scorr = (predict(sm, x_test, device, args.batch_size) == y_test).numpy()
        print(f"\n[Source] {src_key} ({src_type}) clean-correct {scorr.sum()}/{N}")

        for seed in seeds:
            for use_di in (False, True):
                t0 = time.time()
                seed_everything(seed)
                arm = "di" if use_di else "base"
                acc = {t: {k: [] for k in ("align", "loss", "margin")}
                       for t in tgts}
                succ_all = {t: [] for t in tgts}
                for i in range(0, N, args.batch_size):
                    xb = x_test[i:i + args.batch_size].to(device)
                    yb = y_test[i:i + args.batch_size].to(device)
                    out, succ = run_arm(sm, tgts, xb, yb, use_di, args.eps,
                                        args.alpha, args.steps,
                                        args.resize_rate)
                    for t in tgts:
                        for k in ("align", "loss", "margin"):
                            acc[t][k].append(out[t][k])
                        succ_all[t].append(succ[t])
                for t in tgts:
                    for k in ("align", "loss", "margin"):
                        store[f"{src_key}|{arm}|{seed}|{t}|{k}"] = \
                            np.concatenate(acc[t][k], axis=0).astype(np.float32)
                    s = np.concatenate(succ_all[t], axis=0)
                    store[f"{src_key}|{arm}|{seed}|{t}|succ"] = s
                    mask = scorr & tcorr[t]
                    rows.append(dict(
                        source=src_key, src_type=src_type, arm=arm, seed=seed,
                        target=t, n_eval=int(mask.sum()),
                        asr=float(100 * s[mask].mean()),
                        align_final=float(
                            store[f"{src_key}|{arm}|{seed}|{t}|align"][mask, -1].mean()),
                        align_mean=float(
                            store[f"{src_key}|{arm}|{seed}|{t}|align"][mask].mean()),
                        loss_final=float(
                            store[f"{src_key}|{arm}|{seed}|{t}|loss"][mask, -1].mean()),
                    ))
                print(f"  {arm:<5} seed={seed}  " + "  ".join(
                    f"{t.split('_')[0][:7]}: ASR={rows[-len(tgts)+k]['asr']:.1f} "
                    f"align_mean={rows[-len(tgts)+k]['align_mean']:+.5f}"
                    for k, t in enumerate(tgts)) +
                    f"   ({time.time()-t0:.0f}s)", flush=True)
        del sm
        torch.cuda.empty_cache()

    np.savez_compressed(
        os.path.join(args.results_dir, "trajectories.npz"),
        labels=y_test.numpy(),
        **{f"tgt_correct__{t}": tcorr[t] for t in tgts},
        **store)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "trajectory_summary.csv"),
              index=False)
    print(f"\n[Saved] {os.path.join(args.results_dir, 'trajectories.npz')}")
    with pd.option_context("display.width", 200):
        piv = df.groupby(["source", "src_type", "target", "arm"])[
            ["asr", "align_mean", "align_final", "loss_final"]].mean()
        print("\n" + piv.round(4).to_string())


if __name__ == "__main__":
    main()
