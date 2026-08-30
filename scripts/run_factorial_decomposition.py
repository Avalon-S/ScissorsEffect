#!/usr/bin/env python
"""
2x2 factorial decomposition of input diversity into resize and translation.

Cells:
    00_none          identity
    10_resize        random rescale then resize back, no translation
    01_translation   random integer shift with zero padding, no resize
    11_both          translation o resize
    di_full          the full DI operator on the same (r > 1) canvas
    di09             shrink to a random size in [0.9 S, S] and zero-pad back,
                     reported as a bridge to the main-paper operator

The factorial uses the r > 1 canvas: with r < 1 the pad offset exists only as a
consequence of the shrink, so translation cannot be varied independently and a
2x2 design is undefined. Reports, per source-target pair,

    ME_resize      = 1/2 [ (10 - 00) + (11 - 01) ]
    ME_translation = 1/2 [ (01 - 00) + (11 - 10) ]
    INT            = 11 - 10 - 01 + 00

Two identities are asserted: ME_resize + ME_translation = a11 - a00, and
simple_resize + simple_translation + INT = a11 - a00. Per-image success
indicators are saved so intervals can be computed as image-level bootstraps.

Example
-------
  python scripts/run_factorial_decomposition.py --n_examples 1000 --n_seeds 5
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


# ---------------------------------------------------------------------------
# Factors. Identical parameterization to scripts/run_transform_decomposition.py
# and scripts/run_moment_decomposition.py, so E1's measured constants describe
# exactly these operators.
# ---------------------------------------------------------------------------

def f_identity(x):
    return x


def f_resize(x, resize_rate=1.1):
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=min(img_size, img_resize),
                        high=max(img_size, img_resize), size=(1,)).item()
    r = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    return F.interpolate(r, size=[img_size, img_size],
                         mode='bilinear', align_corners=False)


def f_translation(x, max_shift=25):
    img_size = x.shape[-1]
    sh = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    sw = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    pad_top, pad_bottom = (sh, 0) if sh >= 0 else (0, -sh)
    pad_left, pad_right = (sw, 0) if sw >= 0 else (0, -sw)
    padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], value=0)
    h0 = pad_bottom if sh < 0 else 0
    w0 = pad_right if sw < 0 else 0
    return padded[:, :, h0:h0 + img_size, w0:w0 + img_size]


def f_both(x):
    """The factorial (1,1) cell: translation composed with resize."""
    return f_translation(f_resize(x))


def f_full_di(x, resize_rate=1.1):
    """The real DI operator (resize + random pad), reported alongside the
    factorial so the composition can be compared against what DI actually does."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=min(img_size, img_resize),
                        high=max(img_size, img_resize), size=(1,)).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear',
                             align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=max(1, h_rem), size=(1,)).item()
    pad_left = torch.randint(low=0, high=max(1, w_rem), size=(1,)).item()
    padded = F.pad(rescaled, [pad_left, w_rem - pad_left,
                              pad_top, h_rem - pad_top], value=0)
    return F.interpolate(padded, size=[img_size, img_size],
                         mode='bilinear', align_corners=False)


def f_di09(x, resize_rate=0.9):
    """The DI operator used by the MAIN-PAPER results (torchattacks style):
    shrink to a random size in [0.9*S, S] and zero-pad back to S.

    It is reported alongside the factorial, but it cannot itself be factorised:
    with r < 1 the pad offset exists ONLY because the image was shrunk, so
    translation is not manipulable independently of resize. That is precisely
    why the factorial uses the r > 1 canvas (f_resize / f_translation / f_both),
    where the two factors are separable. Including this cell makes the
    relationship between the two operators explicit instead of leaving it as an
    unexplained gap between Table 6 and the headline number."""
    S = x.shape[-1]
    lo = int(S * resize_rate)
    rnd = torch.randint(lo, S + 1, (1,)).item()
    xr = F.interpolate(x, size=(rnd, rnd), mode="bilinear", align_corners=False)
    pt = torch.randint(0, S - rnd + 1, (1,)).item()
    pl = torch.randint(0, S - rnd + 1, (1,)).item()
    return F.pad(xr, (pl, S - rnd - pl, pt, S - rnd - pt), value=0)


CELLS = {
    "00_none": f_identity,
    "10_resize": f_resize,
    "01_translation": f_translation,
    "11_both": f_both,
    "di_full": f_full_di,
    "di09": f_di09,
}


# ---------------------------------------------------------------------------

def mi_fgsm(model, x, y, transform, eps, alpha, steps, decay=1.0):
    """MI-FGSM with a configurable input transform (applied every iteration,
    diversity probability 1, matching the Table 6 protocol)."""
    x = x.detach()
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        logits = model(transform(x + delta))
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, delta)[0]
        grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-10)
        momentum = decay * momentum + grad
        delta = (delta.detach() + alpha * momentum.sign())
        delta = delta.clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
        delta.requires_grad_(True)
    return (x + delta).detach()


ROBUST_SOURCES = [
    ("Engstrom2019Robustness_ImageNet", "Robust"),
    ("Salman_eps2.0", "Robust"),
    ("Salman_eps4.0", "Robust"),
    ("Mo2022When_ViT-B", "Robust"),
]
STANDARD_SOURCES = [
    ("ResNet50", "Standard"),
    ("DenseNet121_Standard", "Standard"),
]
DEFAULT_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet", "ViT_B_16_ImageNet"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help="'name:type' pairs; default = 4 robust + 2 standard.")
    p.add_argument("--targets", type=str, nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--cells", type=str, nargs="+", default=list(CELLS.keys()))
    p.add_argument("--n_examples", type=int, default=1000)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=100)
    p.add_argument("--seed_step", type=int, default=100)
    p.add_argument("--eps", type=float, default=16.0 / 255)
    p.add_argument("--alpha", type=float, default=2.0 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--results_dir", type=str,
                   default="results/factorial_decomposition")
    return p.parse_args()


def predict(model, x, bs):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs]).argmax(1).cpu())
    return torch.cat(out)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.sources:
        sources = [(s.split(":")[0], s.split(":")[1] if ":" in s else "?")
                   for s in args.sources]
    else:
        sources = ROBUST_SOURCES + STANDARD_SOURCES
    seeds = [args.seed0 + i * args.seed_step for i in range(args.n_seeds)]

    print("=" * 78)
    print("E3: 2x2 factorial decomposition (resize x translation)")
    print("=" * 78)
    print(f"  N={args.n_examples}  seeds={seeds}  cells={args.cells}")
    print(f"  sources={[s for s, _ in sources]}")
    print(f"  targets={args.targets}")
    print(f"  eps={args.eps*255:.0f}/255 alpha={args.alpha*255:.0f}/255 "
          f"steps={args.steps} batch={args.batch_size}\n")

    seed_everything(args.seed0)
    x_test, y_test = load_dataset("imagenet", args.n_examples)
    N = x_test.size(0)

    # Targets stay resident (3 ImageNet models ~ 1 GB of parameters).
    targets, tgt_correct = {}, {}
    for t in args.targets:
        tm = get_model(t, dataset="imagenet", device=device)
        tm.eval()
        targets[t] = tm
        pred = predict(tm, x_test.to(device), args.batch_size)
        tgt_correct[t] = (pred == y_test).numpy()
        print(f"[Target] {t}: clean-correct {tgt_correct[t].sum()}/{N}")

    # success[source][cell][seed][target] = bool array over images
    succ = {}
    rows = []

    for src_name, src_type in sources:
        try:
            sm = get_model(src_name, dataset="imagenet", device=device)
            sm.eval()
        except Exception as e:
            print(f"[SKIP] {src_name}: {type(e).__name__}: {str(e)[:60]}")
            continue
        src_pred = predict(sm, x_test.to(device), args.batch_size)
        src_correct = (src_pred == y_test).numpy()
        print(f"\n[Source] {src_name} ({src_type}) "
              f"clean-correct {src_correct.sum()}/{N}")

        for cell in args.cells:
            tfm = CELLS[cell]
            for seed in seeds:
                t0 = time.time()
                seed_everything(seed)
                adv_pred = {t: np.zeros(N, dtype=np.int64) for t in targets}
                for i in range(0, N, args.batch_size):
                    xb = x_test[i:i + args.batch_size].to(device)
                    yb = y_test[i:i + args.batch_size].to(device)
                    xadv = mi_fgsm(sm, xb, yb, tfm, args.eps, args.alpha,
                                   args.steps)
                    with torch.no_grad():
                        for t, tm in targets.items():
                            adv_pred[t][i:i + xb.size(0)] = \
                                tm(xadv).argmax(1).cpu().numpy()
                for t in targets:
                    mask = src_correct & tgt_correct[t]
                    s = (adv_pred[t] != y_test.numpy())
                    succ[(src_name, cell, seed, t)] = s
                    rows.append(dict(
                        source=src_name, src_type=src_type, cell=cell,
                        seed=seed, target=t, n_eval=int(mask.sum()),
                        asr=float(100.0 * s[mask].mean()),
                    ))
                print(f"  {cell:<16} seed={seed}  " + "  ".join(
                    f"{t.split('_')[0]}={rows[-len(targets)+k]['asr']:.1f}"
                    for k, t in enumerate(targets)) +
                    f"   ({time.time()-t0:.0f}s)", flush=True)

        del sm
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    p_csv = os.path.join(args.results_dir, "factorial_asr.csv")
    df.to_csv(p_csv, index=False)

    # Per-image indicators: compact npz, keyed "source|cell|seed|target".
    np.savez_compressed(
        os.path.join(args.results_dir, "per_image_success.npz"),
        labels=y_test.numpy(),
        **{f"correct__{t}": tgt_correct[t] for t in targets},
        **{"|".join(map(str, k)): v for k, v in succ.items()},
    )
    print(f"\n[Saved] {p_csv}")
    print(f"[Saved] {os.path.join(args.results_dir, 'per_image_success.npz')}")

    # ---- factorial effects, seed-averaged (CIs are computed by the analysis
    # script from the per-image indicators; this is a quick console read-out) --
    piv = df.groupby(["source", "src_type", "target", "cell"])["asr"].mean()
    recs = []
    for (src, styp, tgt), g in piv.groupby(level=[0, 1, 2]):
        d = {c: g.loc[(src, styp, tgt, c)] for c in
             g.index.get_level_values(3)}
        if not {"00_none", "10_resize", "01_translation", "11_both"} <= set(d):
            continue
        a00, a10, a01, a11 = (d["00_none"], d["10_resize"],
                              d["01_translation"], d["11_both"])
        recs.append(dict(
            source=src, src_type=styp, target=tgt,
            base=a00, resize=a10, translation=a01, both=a11,
            di_full=d.get("di_full", np.nan),
            ME_resize=0.5 * ((a10 - a00) + (a11 - a01)),
            ME_translation=0.5 * ((a01 - a00) + (a11 - a10)),
            INT=a11 - a10 - a01 + a00,
            simple_resize=a10 - a00,
            simple_translation=a01 - a00,
            total_both=a11 - a00,
            total_di=d.get("di_full", np.nan) - a00,
        ))
    eff = pd.DataFrame(recs)
    p_eff = os.path.join(args.results_dir, "factorial_effects.csv")
    eff.to_csv(p_eff, index=False)
    print(f"[Saved] {p_eff}\n")
    if eff.empty:
        # Legitimate when --cells selects a subset that does not contain the
        # four factorial cells (e.g. running only the di09 bridge cell). The
        # per-cell ASR and the per-image indicators are already saved above.
        print("No factorial effects: the run did not include all four cells "
              f"(got {sorted(set(df['cell']))}). Per-cell ASR is in "
              f"{p_csv}; combine with the main factorial run for the effects.")
        return
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(eff.round(2).to_string(index=False))

    # Two identities that must hold exactly:
    #   (i)  ME_resize + ME_translation = a11 - a00.  Main effects are averaged
    #        over the other factor, so they already absorb the interaction.
    #   (ii) simple_resize + simple_translation + INT = a11 - a00.  Dropping
    #        INT from this one turns it into an additive attribution, which it
    #        is not.
    r1 = (eff["ME_resize"] + eff["ME_translation"] - eff["total_both"]).abs().max()
    r2 = (eff["simple_resize"] + eff["simple_translation"] + eff["INT"]
          - eff["total_both"]).abs().max()
    print(f"\nIdentity checks (both must be ~0; a residual is a bug):")
    print(f"  ME_resize + ME_translation - (a11-a00)              max|res| = {r1:.2e}")
    print(f"  simple_resize + simple_translation + INT - (a11-a00) max|res| = {r2:.2e}")
    assert r1 < 1e-6 and r2 < 1e-6, "factorial identities violated"
    print("\nThe published '67%' figure is simple_resize/total_both while taking "
          "INT = 0. The INT column is exactly the size of that assumption.")
    print("Also compare total_both (factorial composition) against total_di (the "
          "real DI operator): they need not agree, and the gap is why the old "
          "table's (1,1) cell could not serve as a factorial cell.")


if __name__ == "__main__":
    main()
