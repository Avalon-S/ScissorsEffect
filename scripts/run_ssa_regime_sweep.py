#!/usr/bin/env python
"""
Effective gradient regime under transformation-heavy attack methods, and the
resulting DI interaction.

Stage 1 (--stage measure, no attack). For each method M in {none, SSA(rho), SIA,
BSR, Admix}, measures the moments of the M-conditioned gradient
g_M = (1/m) sum_i grad_x L(f(T_M^i(x))) and reports

    GCR_eff = ||E g_M||^2 / ( ||E g_M||^2 + V_M ).

Every draw perturbs the input as well as drawing the method transform, so that
"none" is not degenerate: without the input probe the identity transform has
zero draw-to-draw variance and GCR_eff(none) = 1 by construction, against which
any random method would appear to shift the regime leftward. With the probe in
place "none" reduces to the input-noise SNR and all arms share one noise source.
Sweeping rho gives a monotone strength axis.

Stage 2 (--stage attack). Measures D(strength) = ASR(M + DI) - ASR(M) on the
same grid. Its DI arm averages m draws of the DI transform per step, so it is
not the standard single-draw DI operator.

  python scripts/run_ssa_regime_sweep.py --stage measure
  python scripts/run_ssa_regime_sweep.py --stage attack
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "TransferAttack"))

from data import load_dataset
from models import get_model
from utils import seed_everything, get_device

# DCT helpers: bind SSM's methods to a bare object rather than constructing an
# attack, so the transform is byte-identical to TransferAttack's without pulling
# in its model-loading machinery.
from transferattack.input_transformation.ssm import SSM


class _Shim:
    pass


_dct = _Shim()
for _n in ("dct", "idct", "dct_2d", "idct_2d"):
    setattr(_dct, _n, types.MethodType(getattr(SSM, _n), _dct))


def t_none(x):
    return x


def make_ssa(rho, noise_eps=16.0 / 255):
    """SSA's transform, exactly as in TransferAttack ssm.py: additive Gaussian
    noise, then a uniform multiplicative mask in [1-rho, 1+rho] on the DCT."""
    def f(x):
        gauss = torch.randn_like(x) * noise_eps
        x_dct = _dct.dct_2d(x + gauss)
        mask = torch.rand_like(x) * 2 * rho + 1 - rho
        return _dct.idct_2d(x_dct * mask)
    return f


def t_admix(x, portion=0.2):
    """Admix: blend with a shuffled other image, single scale (the per-step
    transform; the attack's scale ensemble is the m-fold averaging we already do)."""
    idx = torch.randperm(x.size(0), device=x.device)
    return x + portion * x[idx]


def t_sia(x, splits=3):
    """SIA (structure-invariant): random block-wise transform on a splits x splits
    grid. Simplified to the three dominant block ops (identity / scale / flip)."""
    B, C, H, W = x.shape
    out = x.clone()
    hs, ws = H // splits, W // splits
    for a in range(splits):
        for b in range(splits):
            blk = out[:, :, a * hs:(a + 1) * hs, b * ws:(b + 1) * ws]
            op = torch.randint(0, 3, (1,)).item()
            if op == 1:
                blk = blk * (torch.rand(1, device=x.device) * 1.5 + 0.25)
            elif op == 2:
                blk = torch.flip(blk, dims=[3])
            out[:, :, a * hs:(a + 1) * hs, b * ws:(b + 1) * ws] = blk
    return out


def t_bsr(x, splits=3):
    """BSR (block shuffle and rotation): shuffle the blocks of a splits x splits
    grid and randomly rotate the whole image by a small angle (approximated by
    a 90-degree-free random block permutation, which is the dominant effect)."""
    B, C, H, W = x.shape
    hs, ws = H // splits, W // splits
    blocks = [x[:, :, a * hs:(a + 1) * hs, b * ws:(b + 1) * ws]
              for a in range(splits) for b in range(splits)]
    perm = torch.randperm(len(blocks)).tolist()
    rows = []
    for a in range(splits):
        rows.append(torch.cat(
            [blocks[perm[a * splits + b]] for b in range(splits)], dim=3))
    out = torch.cat(rows, dim=2)
    if out.shape[-2:] != (H, W):
        out = F.interpolate(out, size=(H, W), mode="bilinear",
                            align_corners=False)
    return out


def di_transform(x, resize_rate=0.9):
    S = x.shape[-1]
    lo = int(S * resize_rate)
    rnd = torch.randint(lo, S + 1, (1,)).item()
    xr = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pt = torch.randint(0, S - rnd + 1, (1,)).item()
    pl = torch.randint(0, S - rnd + 1, (1,)).item()
    return F.pad(xr, (pl, S - rnd - pl, pt, S - rnd - pt), value=0)


RHO_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]      # 0.5 is SSA's official value


def methods(rho_grid):
    m = {"none": t_none}
    for r in rho_grid:
        m[f"ssa_rho{r}"] = make_ssa(r)
    m["admix"] = t_admix
    m["sia"] = t_sia
    m["bsr"] = t_bsr
    return m


DEFAULT_SOURCES = ["ResNet50", "Engstrom2019Robustness_ImageNet", "Salman_eps2.0"]
DEFAULT_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["measure", "attack"], required=True)
    p.add_argument("--sources", type=str, nargs="+", default=DEFAULT_SOURCES)
    p.add_argument("--targets", type=str, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--rho_grid", type=float, nargs="+", default=RHO_GRID)
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--probe_sigma", type=float, default=1.0/255,
                   help="Input-probe scale shared by every method arm, so "
                        "GCR_eff is measured against a common noise source "
                        "(matches E1's eps_chk).")
    p.add_argument("--m_eot", type=int, default=20,
                   help="SSA's official num_spectrum is 20; used as m for the "
                        "moment measurement and as the ensemble in the attack.")
    p.add_argument("--eps", type=float, default=16.0 / 255)
    p.add_argument("--alpha", type=float, default=2.0 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str, default="results/ssa_regime")
    return p.parse_args()


def grad_through(model, x, y, tfm):
    xi = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(tfm(xi)), y, reduction="sum").backward()
    g = xi.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return g


def sqn(g):
    return g.flatten(1).pow(2).sum(1)


def predict(model, x, device, bs):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs].to(device)).argmax(1).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------- stage 1 ---
def stage_measure(args, device, x_test, y_test):
    M = methods(args.rho_grid)
    rows = []
    for key in args.sources:
        model = get_model(key, dataset="imagenet", device=device)
        model.eval()
        correct = (predict(model, x_test, device, args.batch_size)
                   == y_test).numpy()
        print(f"\n[{key}] clean-correct {correct.sum()}/{len(y_test)}")
        for name, tfm in M.items():
            t0 = time.time()
            seed_everything(42)
            for i in range(0, args.n_examples, args.batch_size):
                xb = x_test[i:i + args.batch_size].to(device)
                yb = y_test[i:i + args.batch_size].to(device)
                B = xb.size(0)
                S1 = torch.zeros_like(xb)
                S2 = torch.zeros(B, device=device)
                for _ in range(args.m_eot):
                    # Each draw perturbs the INPUT as well as drawing the method
                    # transform. Without the input probe the baseline is
                    # degenerate: method="none" has zero draw-to-draw variance by
                    # construction, so GCR_eff(none) = 1 exactly and EVERY random
                    # method appears to shift the surrogate leftward. The
                    # theorem's rho is an input-noise SNR, so the two arms have to
                    # share that noise source to be comparable; with the probe in
                    # place, "none" reduces to E1's rho.
                    xn = (xb + (torch.rand_like(xb) * 2 - 1)
                          * args.probe_sigma).clamp(0, 1)
                    g = grad_through(model, xn, yb, tfm)
                    S1 += g
                    S2 += sqn(g)
                gbar = S1 / args.m_eot
                V = (S2 - args.m_eot * sqn(gbar)) / (args.m_eot - 1)
                sq_gbar = sqn(gbar).cpu().numpy()
                Vv = V.cpu().numpy()
                for b in range(B):
                    if not correct[i + b]:
                        continue
                    rows.append(dict(source=key, method=name, img=i + b,
                                     sq_gbar=float(sq_gbar[b]), V=float(Vv[b])))
            print(f"  {name:<14} ({time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    os.makedirs(args.results_dir, exist_ok=True)
    df.to_csv(os.path.join(args.results_dir, "method_moments_per_image.csv"),
              index=False)
    # GCR_eff as a ratio of expectations, never a mean of per-image ratios.
    g = df.groupby(["source", "method"])[["sq_gbar", "V"]].mean()
    g["rho_eff"] = g["sq_gbar"] / g["V"]
    g["GCR_eff"] = g["rho_eff"] / (1 + g["rho_eff"])
    base = g.xs("none", level="method")["GCR_eff"]
    g["shift_vs_none"] = g.index.get_level_values(0).map(base)
    g["shift_vs_none"] = g["GCR_eff"] - g["shift_vs_none"]
    out = os.path.join(args.results_dir, "method_gcr.csv")
    g.to_csv(out)
    print(f"\n[Saved] {out}")
    with pd.option_context("display.width", 200):
        print("\n" + g.round(4).to_string())
    print("\nCorollary 2 predicts shift_vs_none > 0 for SSA (rightward regime "
          "shift). It makes no such prediction for the spatial methods; if they "
          "shift too, the corollary does not single SSA out and must be "
          "restated. Commit these numbers before running --stage attack.")


# ---------------------------------------------------------------- stage 2 ---
def mi_fgsm_ens(model, x, y, tfm, add_di, eps, alpha, steps, m_ens, decay=1.0):
    """MI-FGSM whose per-step gradient is averaged over m_ens draws of the
    method transform, optionally with DI composed on top (DI o M)."""
    x = x.detach()
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        acc = torch.zeros_like(x)
        for _ in range(m_ens):
            xin = x + delta
            xin = tfm(xin)
            if add_di:
                xin = di_transform(xin)
            loss = F.cross_entropy(model(xin), y)
            acc = acc + torch.autograd.grad(loss, delta, retain_graph=False)[0]
        grad = acc / m_ens
        grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-10)
        momentum = decay * momentum + grad
        delta = (delta.detach() + alpha * momentum.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)
    return (x + delta).detach()


def stage_attack(args, device, x_test, y_test):
    M = methods(args.rho_grid)
    N = x_test.size(0)
    y_np = y_test.numpy()
    tgts, tcorr = {}, {}
    for t in args.targets:
        tm = get_model(t, dataset="imagenet", device=device)
        tm.eval()
        tgts[t] = tm
        tcorr[t] = (predict(tm, x_test, device, args.batch_size) == y_test).numpy()
        print(f"[Target] {t}: clean-correct {tcorr[t].sum()}/{N}")

    seeds = [100 * (i + 1) for i in range(args.n_seeds)]
    rows = []
    for key in args.sources:
        sm = get_model(key, dataset="imagenet", device=device)
        sm.eval()
        scorr = (predict(sm, x_test, device, args.batch_size) == y_test).numpy()
        print(f"\n[Source] {key} clean-correct {scorr.sum()}/{N}")
        for name, tfm in M.items():
            for add_di in (False, True):
                for seed in seeds:
                    t0 = time.time()
                    seed_everything(seed)
                    pred = {t: np.zeros(N, dtype=np.int64) for t in tgts}
                    for i in range(0, N, args.batch_size):
                        xb = x_test[i:i + args.batch_size].to(device)
                        yb = y_test[i:i + args.batch_size].to(device)
                        xadv = mi_fgsm_ens(sm, xb, yb, tfm, add_di, args.eps,
                                           args.alpha, args.steps, args.m_eot)
                        with torch.no_grad():
                            for t, tm in tgts.items():
                                pred[t][i:i + xb.size(0)] = \
                                    tm(xadv).argmax(1).cpu().numpy()
                    for t in tgts:
                        mask = scorr & tcorr[t]
                        rows.append(dict(
                            source=key, method=name, di=add_di, seed=seed,
                            target=t,
                            asr=float(100 * (pred[t] != y_np)[mask].mean())))
                    print(f"  {name:<14} di={int(add_di)} seed={seed} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(
                        os.path.join(args.results_dir, "ssa_sweep_asr.csv"),
                        index=False)
        del sm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "ssa_sweep_asr.csv"), index=False)
    piv = df.groupby(["source", "method", "target", "di"])["asr"].mean() \
            .unstack("di")
    piv.columns = ["no_di", "di"]
    piv["D"] = piv["di"] - piv["no_di"]
    out = os.path.join(args.results_dir, "ssa_sweep_effects.csv")
    piv.to_csv(out)
    print(f"\n[Saved] {out}")
    with pd.option_context("display.width", 200):
        print("\n" + piv.round(2).to_string())
    print("\nCompare the sign of D across the rho grid against the flip location "
          "predicted in stage 1. A flip in the wrong place, or no flip, "
          "falsifies the pre-registered prediction and Corollary 2 is reported "
          "as an unresolved method interaction.")


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    seed_everything(42)
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    print(f"[Data] {tuple(x_test.shape)}  stage={args.stage}")
    if args.stage == "measure":
        stage_measure(args, device, x_test, y_test)
    else:
        stage_attack(args, device, x_test, y_test)


if __name__ == "__main__":
    main()
