#!/usr/bin/env python
"""
Moment decomposition of the transformed input gradient.

For each surrogate and each input transform T, estimates over m EOT draws:

    g_i    = grad_x L(f(T_i(x)), y)      gradient wrt the original input
    gbar   = (1/m) sum_i g_i             the averaged direction
    V      = (1/(m-1))[ sum_i ||g_i||^2 - m||gbar||^2 ]   variance across draws
    c      = cos(gbar, mu)               directional bias
    g_r    = ||gbar||^2 / ||mu||^2       signal retention
    kappa  = V / (m n sigma^2)           normalised variance of the average

with mu and sigma^2 estimated from K small-noise probes. V is split into
transform variance and input-noise pass-through by repeating the probe at a
fixed transform draw.

Also computed per image:
  - the linearization residual ||gbar - J_T^T g_clean|| / ||g_clean||, testing
    whether the transformed gradient is described by a linear operator applied
    to the clean gradient;
  - LGC (clean-vs-probe cosine) and LGC_2 (two independent probes, i.e. the
    moment ratio rho/(1+rho)) at several probe scales;
  - for each target model, gamma_mu = |cos(mu, u)| and gamma_R = |cos(gbar, u)|.

Constants are aggregated as ratios of expectations over images; raw second
moments are written out so other aggregations can be checked.

Example
-------
  python scripts/run_moment_decomposition.py --n_examples 500 --m_eot 20
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

# Some held-out robust checkpoints (e.g. ARES ConvNeXt-B) are local files with a
# timm architecture and their own normalization, not RobustBench ids. Reuse the
# dispatch that run_recipe_panel already defines rather than duplicating it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from run_recipe_panel import get_source_model as _panel_loader
except Exception:
    _panel_loader = None


def load_surrogate(key, dataset, device):
    if _panel_loader is not None and dataset == "imagenet":
        try:
            return _panel_loader(key, device)
        except Exception:
            pass
    return get_model(key, dataset=dataset, device=device)


# ---------------------------------------------------------------------------
# Transform components.  These are byte-identical in behaviour to the ones used
# by the attack scripts (scripts/run_transform_decomposition.py, attacks/sap.py),
# so the constants measured here describe the transforms the attacks actually use.
# ---------------------------------------------------------------------------

def apply_resize_only(x, resize_rate=1.1):
    """Random rescale then resize back. No translation/padding."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=min(img_size, img_resize),
                        high=max(img_size, img_resize), size=(1,)).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    return F.interpolate(rescaled, size=[img_size, img_size],
                         mode='bilinear', align_corners=False)


def apply_translation_only(x, max_shift=25):
    """Random integer shift with zero padding. No resize."""
    img_size = x.shape[-1]
    sh = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    sw = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    pad_top, pad_bottom = (sh, 0) if sh >= 0 else (0, -sh)
    pad_left, pad_right = (sw, 0) if sw >= 0 else (0, -sw)
    padded = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], value=0)
    h0 = pad_bottom if sh < 0 else 0
    w0 = pad_right if sw < 0 else 0
    return padded[:, :, h0:h0 + img_size, w0:w0 + img_size]


def apply_full_di(x, resize_rate=1.1):
    """Full DI (resize + random pad), the run_transform_decomposition variant."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=min(img_size, img_resize),
                        high=max(img_size, img_resize), size=(1,)).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=max(1, h_rem), size=(1,)).item()
    pad_left = torch.randint(low=0, high=max(1, w_rem), size=(1,)).item()
    padded = F.pad(rescaled, [pad_left, w_rem - pad_left, pad_top, h_rem - pad_top],
                   value=0)
    return F.interpolate(padded, size=[img_size, img_size],
                         mode='bilinear', align_corners=False)


def apply_di09(x, resize_rate=0.9):
    """torchattacks-style DI (shrink to a random size in [r*S, S], then zero-pad
    back). This is the transform used by Table 5 / the main-paper DI results."""
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_resize, high=img_size + 1, size=(1,)).item()
    xr = F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False)
    pad_top = torch.randint(0, img_size - rnd + 1, (1,)).item()
    pad_left = torch.randint(0, img_size - rnd + 1, (1,)).item()
    return F.pad(xr, (pad_left, img_size - rnd - pad_left,
                      pad_top, img_size - rnd - pad_top), value=0)


TRANSFORMS = {
    "resize": apply_resize_only,
    "translation": apply_translation_only,
    "full_di": apply_full_di,
    "di09": apply_di09,
}


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def grad_clean(model, x, y):
    xi = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(xi), y, reduction="sum").backward()
    g = xi.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return g


def grad_through_transform(model, x, y, tfm):
    """grad_x L(f(T(x))) -- gradient wrt the ORIGINAL input, flowing through the
    (differentiable) transform. This is exactly the DI-FGSM update direction."""
    xi = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(tfm(xi)), y, reduction="sum").backward()
    g = xi.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    return g


def transform_adjoint(x, v, tfm):
    """J_T^T v, the adjoint of the transform applied to v. Model-free: we only
    differentiate the transform itself. Used for the linearization-residual test."""
    xi = x.clone().detach().requires_grad_(True)
    out = tfm(xi)
    out.backward(v)
    return xi.grad.detach().clone()


def sqnorm(g):
    return g.flatten(1).pow(2).sum(1)


def dot(a, b):
    return (a.flatten(1) * b.flatten(1)).sum(1)


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


# ---------------------------------------------------------------------------

DEFAULT_SOURCES = [
    "ResNet50", "ViT_B_16_ImageNet", "Swin_B_ImageNet", "ConvNeXt_B_ImageNet",
    "DenseNet121_Standard", "InceptionV3",
    "Engstrom2019Robustness_ImageNet", "Salman_eps0.5", "Salman_eps1.0",
    "Salman_eps2.0", "Salman_eps4.0", "Salman_eps8.0", "Mo2022When_ViT-B",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=str, nargs="+", default=DEFAULT_SOURCES)
    p.add_argument("--targets", type=str, nargs="+",
                   default=["ViT_B_16_ImageNet", "Swin_B_ImageNet",
                            "ConvNeXt_B_ImageNet"],
                   help="Targets for the general-target test of Prop. 2: "
                        "Gamma = gamma_R^2/gamma_mu^2 with u = the target gradient. "
                        "Pass 'none' to skip.")
    p.add_argument("--dataset", type=str, default="imagenet",
                   choices=["imagenet", "cifar10", "cifar100"])
    p.add_argument("--modes", type=str, nargs="+", default=list(TRANSFORMS.keys()))
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--m_eot", type=int, default=20,
                   help="EOT draws of the transform (m in the theorem).")
    p.add_argument("--k_mu", type=int, default=10,
                   help="Probe set A size, for mu_hat / sigma^2 / rho.")
    p.add_argument("--k_lgc", type=int, default=5,
                   help="Probe set B size per eps_chk, for LGC_emp / omega.")
    p.add_argument("--k_noise", type=int, default=5,
                   help="Probes at a FIXED transform draw, to split V into "
                        "transform variance vs noise pass-through.")
    p.add_argument("--sigma_ref", type=float, default=1.0 / 255,
                   help="Reference probe scale (paper's eps_chk).")
    p.add_argument("--eps_chk_list", type=float, nargs="+",
                   default=[0.5 / 255, 1.0 / 255, 2.0 / 255, 4.0 / 255, 8.0 / 255])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--filter_correct", action="store_true", default=True)
    p.add_argument("--no_filter_correct", dest="filter_correct", action="store_false")
    p.add_argument("--results_dir", type=str, default="results/moment_decomposition")
    return p.parse_args()


def predict(model, x, bs=8):
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            out.append(model(x[i:i + bs]).argmax(1))
    return torch.cat(out)


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78)
    print("E1: moment decomposition of the transformed gradient")
    print("=" * 78)
    print(f"  dataset={args.dataset}  N={args.n_examples}  m_eot={args.m_eot}")
    print(f"  K_mu={args.k_mu}  K_lgc={args.k_lgc}  K_noise={args.k_noise}")
    print(f"  sigma_ref={args.sigma_ref*255:.2f}/255  modes={args.modes}")
    print(f"  eps_chk x255 = {[round(e*255,2) for e in args.eps_chk_list]}")
    print()

    seed_everything(args.seed)
    x_test, y_test = load_dataset(args.dataset, args.n_examples)
    n_dim = int(np.prod(x_test.shape[1:]))
    print(f"[Data] {x_test.shape}, n = {n_dim}")

    # ---- target gradients, computed once and cached (they do not depend on the
    # source). Used for the general-target form of the theorem (Prop. 2):
    #   gamma_mu = |cos(mu, u)| , gamma_R = |cos(gbar, u)| , Gamma = gamma_R^2/gamma_mu^2
    # DI necessarily hurts as GCR->1 iff Gamma < 1. This is the condition the
    # theorem actually needs, and it is measurable rather than assumed.
    targets = [] if (len(args.targets) == 1 and args.targets[0] == "none") \
        else args.targets
    tgt_grads, tgt_correct = {}, {}
    for tname in targets:
        try:
            tm = get_model(tname, dataset=args.dataset, device=device)
            tm.eval()
        except Exception as e:
            print(f"[SKIP target] {tname}: {type(e).__name__}: {str(e)[:60]}")
            continue
        pred = predict(tm, x_test.to(device), args.batch_size)
        tgt_correct[tname] = (pred.cpu() == y_test).numpy()
        chunks = []
        for i in range(0, args.n_examples, args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            chunks.append(grad_clean(tm, xb, yb).cpu())
        tgt_grads[tname] = chunks
        print(f"[Target] {tname}: clean-correct "
              f"{tgt_correct[tname].sum()}/{len(tgt_correct[tname])}")
        del tm
        torch.cuda.empty_cache()

    rows_img = []     # per-image, per-mode
    rows_lgc = []     # per-image, per-eps_chk
    rows_tgt = []     # per-image, per-mode, per-target

    for key in args.sources:
        t0 = time.time()
        try:
            model = load_surrogate(key, args.dataset, device)
            model.eval()
        except Exception as e:
            print(f"[SKIP] {key}: {type(e).__name__}: {str(e)[:70]}")
            continue

        if args.filter_correct:
            pred = predict(model, x_test.to(device), args.batch_size)
            correct = (pred.cpu() == y_test).numpy()
        else:
            correct = np.ones(len(x_test), dtype=bool)
        print(f"\n[{key}] clean-correct {correct.sum()}/{len(correct)}")

        seed_everything(args.seed)  # identical transform draws across models
        # Dedicated RNG for input noise, so that reseeding the GLOBAL rng (which we
        # do to pin transform draws) never desynchronizes or duplicates the probes.
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + 7919)

        def unif(shape, scale):
            return (torch.rand(shape, generator=gen, device=device) * 2 - 1) * scale

        for i in range(0, args.n_examples, args.batch_size):
            xb = x_test[i:i + args.batch_size].to(device)
            yb = y_test[i:i + args.batch_size].to(device)
            B = xb.size(0)
            idx = np.arange(i, i + B)
            keep = correct[i:i + B]

            g_clean = grad_clean(model, xb, yb)

            # ---- probe set A: mu_hat, sigma^2, rho -------------------------
            mu = torch.zeros_like(xb)
            gA = []
            for _ in range(args.k_mu):
                xi = (xb + unif(xb.shape, args.sigma_ref)).clamp(0, 1)
                gk = grad_clean(model, xi, yb)
                gA.append(gk)
                mu += gk
            mu /= args.k_mu
            sse = torch.zeros(B, device=device)
            for gk in gA:
                sse += sqnorm(gk - mu)
            sig2 = sse / (n_dim * (args.k_mu - 1))                 # sigma^2 per coordinate
            mu_sq_raw = sqnorm(mu)
            # Unbiased estimate of ||mu||^2. Left UNCLAMPED on purpose: for
            # low-SNR images it can come out negative, and clamping it to a tiny
            # positive number is what makes per-image ratios like g_r explode.
            # Every constant below is aggregated as a ratio of means over images
            # (the theorem's constants are ratios of expectations), so individual
            # negative draws are harmless and clamping is not needed.
            mu_sq = mu_sq_raw - n_dim * sig2 / args.k_mu
            rho = mu_sq / (n_dim * sig2).clamp(min=1e-30)
            del gA

            # ---- probe set B: LGC_emp(eps) and omega(eps) ------------------
            for eps in args.eps_chk_list:
                # Two quantities, deliberately kept apart:
                #   LGC_emp : Eq. 1 of the paper -- cos(clean gradient, probe
                #             gradient). This is what CG-DI computes.
                #   LGC_2   : cos over TWO INDEPENDENT probes. This is the
                #             theorem's quantity: for g = mu + eta with iid eta,
                #             E<g,g'>/E||g||^2 = rho/(1+rho) = GCR by construction.
                # The clean point is not a draw from any distribution (there is no
                # randomness at a fixed x), so Eq. 1 cannot be read as the
                # theorem's moment ratio. Measuring both, at several probe scales,
                # is what lets us state the relation between them rather than
                # assume one.
                lgc_acc = torch.zeros(B, device=device)
                d12 = torch.zeros(B, device=device)      # <g_a, g_b>
                s1 = torch.zeros(B, device=device)       # ||g_a||^2
                s2b = torch.zeros(B, device=device)      # ||g_b||^2
                for _ in range(args.k_lgc):
                    xa = (xb + unif(xb.shape, eps)).clamp(0, 1)
                    xbb = (xb + unif(xb.shape, eps)).clamp(0, 1)
                    ga = grad_clean(model, xa, yb)
                    gb = grad_clean(model, xbb, yb)
                    lgc_acc += 0.5 * (cos(g_clean, ga) + cos(g_clean, gb))
                    d12 += dot(ga, gb)
                    s1 += sqnorm(ga)
                    s2b += sqnorm(gb)
                lgc_v = (lgc_acc / args.k_lgc).cpu().numpy()
                d12_v = (d12 / args.k_lgc).cpu().numpy()
                s1_v = (s1 / args.k_lgc).cpu().numpy()
                s2_v = (s2b / args.k_lgc).cpu().numpy()
                for b in range(B):
                    if not keep[b]:
                        continue
                    # Raw moments only: LGC_2 and rho are ratios and are formed
                    # in the summary from separately averaged numerators and
                    # denominators.
                    rows_lgc.append(dict(
                        source=key, img=int(idx[b]), eps_chk=float(eps),
                        lgc_emp=float(lgc_v[b]),
                        dot12=float(d12_v[b]), sq_a=float(s1_v[b]),
                        sq_b=float(s2_v[b]),
                        mu_sq=float(mu_sq[b]), sig2=float(sig2[b]),
                    ))

            # ---- transformed-gradient moments, per mode --------------------
            for mode in args.modes:
                tfm = TRANSFORMS[mode]
                S1 = torch.zeros_like(xb)
                S2 = torch.zeros(B, device=device)
                S1L = torch.zeros_like(xb)
                SDL = torch.zeros(B, device=device)   # sum ||g_i - g_lin_i||^2
                for j in range(args.m_eot):
                    torch.manual_seed(args.seed * 100003 + i * 97 + j)  # shared draw
                    g_i = grad_through_transform(model, xb, yb, tfm)
                    torch.manual_seed(args.seed * 100003 + i * 97 + j)  # SAME draw
                    g_lin = transform_adjoint(xb, g_clean, tfm)
                    S1 += g_i
                    S2 += sqnorm(g_i)
                    S1L += g_lin
                    SDL += sqnorm(g_i - g_lin)
                gbar = S1 / args.m_eot
                gbar_lin = S1L / args.m_eot
                V = (S2 - args.m_eot * sqnorm(gbar)) / (args.m_eot - 1)

                # noise pass-through at a FIXED transform draw T0: the spread of
                # grad L(f(T0(x+xi_k))) over k estimates sigma^2 tr(T^T T), the part
                # of V that is input noise rather than transform randomness.
                gn_sum = torch.zeros_like(xb)
                gn_list = []
                for _ in range(args.k_noise):
                    xi = (xb + unif(xb.shape, args.sigma_ref)).clamp(0, 1)
                    torch.manual_seed(args.seed * 100003 + i * 97 + 0)  # fixed T0
                    gk = grad_through_transform(model, xi, yb, tfm)
                    gn_list.append(gk)
                    gn_sum += gk
                gn_bar = gn_sum / args.k_noise
                Vn = torch.zeros(B, device=device)
                for gk in gn_list:
                    Vn += sqnorm(gk - gn_bar)
                Vn = Vn / (args.k_noise - 1)
                del gn_list

                # Only bounded quantities (cosines) are stored as per-image
                # ratios. Everything else is stored as a RAW second moment, and
                # the constants are formed downstream as ratios of means. Storing
                # ratios per image and averaging them is what produced g_r ~ 1e18
                # in the first run: for high-SNR robust surrogates the estimated
                # ||mu||^2 of an individual image can be ~0, and a handful of such
                # images destroy the mean.
                pack = dict(
                    c_mu=cos(gbar, mu), c_raw=cos(gbar, g_clean),
                    cos_lin=cos(gbar, gbar_lin),
                    V=V, Vn=Vn, sig2=sig2,
                    mu_sq=mu_sq, mu_sq_raw=mu_sq_raw,
                    sq_gbar=sqnorm(gbar),
                    sq_gclean=sqnorm(g_clean),
                    sq_gbar_lin=sqnorm(gbar_lin),
                    msq_draw=S2 / args.m_eot,          # mean_i ||g_i||^2
                    msq_diff_lin=SDL / args.m_eot,     # mean_i ||g_i - g_lin_i||^2
                    sq_diff_mean_lin=sqnorm(gbar - gbar_lin),
                    sq_diff_clean=sqnorm(gbar - g_clean),
                )
                pack = {k: v.detach().cpu().numpy() for k, v in pack.items()}
                for b in range(B):
                    if not keep[b]:
                        continue
                    r = dict(source=key, mode=mode, img=int(idx[b]))
                    r.update({k: float(v[b]) for k, v in pack.items()})
                    rows_img.append(r)

                # ---- general-target constants (Prop. 2), per target ----------
                bi = i // args.batch_size
                for tname in tgt_grads:
                    u = tgt_grads[tname][bi].to(device)
                    gam_mu = cos(mu, u).abs()
                    gam_R = cos(gbar, u).abs()
                    Gamma = (gam_R / gam_mu.clamp(min=1e-12)).pow(2)
                    s_mu = cos(torch.sign(mu), torch.sign(u))
                    s_R = cos(torch.sign(gbar), torch.sign(u))
                    tk = tgt_correct[tname][i:i + B]
                    vals = {k: v.detach().cpu().numpy() for k, v in dict(
                        gamma_mu=gam_mu, gamma_R=gam_R, Gamma=Gamma,
                        sign_mu=s_mu, sign_R=s_R, d_sign=s_R - s_mu).items()}
                    for b in range(B):
                        if not (keep[b] and tk[b]):
                            continue
                        rr = dict(source=key, mode=mode, target=tname,
                                  img=int(idx[b]))
                        rr.update({k: float(v[b]) for k, v in vals.items()})
                        rows_tgt.append(rr)

            if (i // args.batch_size) % 10 == 0:
                done = min(i + B, args.n_examples)
                print(f"    {done}/{args.n_examples}  ({time.time()-t0:.0f}s)", flush=True)

        print(f"  [{key}] done in {time.time()-t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    if not rows_img:
        print("[Error] nothing measured.")
        return

    df = pd.DataFrame(rows_img)
    dl = pd.DataFrame(rows_lgc)
    p_img = os.path.join(args.results_dir, "moments_per_image.csv")
    p_lgc = os.path.join(args.results_dir, "lgc_bridge_per_image.csv")
    df.to_csv(p_img, index=False)
    dl.to_csv(p_lgc, index=False)
    print(f"\n[Saved] {p_img}  ({len(df)} rows)")
    print(f"[Saved] {p_lgc}  ({len(dl)} rows)")

    # ---- model-level summary (image means; CIs are computed downstream) -----
    num_cols = [c for c in df.columns if c not in ("source", "mode", "img")]
    m_eot = args.m_eot
    recs = []
    for (src, mode), g in df.groupby(["source", "mode"], sort=False):
        m = g[num_cols].mean()
        # Every constant is a ratio of expectations, formed here from separately
        # averaged numerators and denominators.
        E_mu_sq = m["mu_sq"]
        E_noise = n_dim * m["sig2"]                    # n sigma^2
        rho = E_mu_sq / E_noise
        gr = m["sq_gbar"] / E_mu_sq
        kappa_eff = m["V"] / (m_eot * E_noise)
        v_T = max(m["V"] - m["Vn"], 0.0)
        v_r = v_T / E_mu_sq
        t_over_n = m["Vn"] / E_noise
        c = m["c_mu"]                                   # bounded: safe to average
        # Prop. B:  phi(rho) = c^2 (1+rho) / (a rho + b)
        a = 1 + v_r / (m_eot * gr) if gr > 0 else np.nan
        b = t_over_n / (m_eot * gr) if gr > 0 else np.nan
        cross = np.isfinite(a) and (b < c * c) and (a > c * c)
        rho_s = (c * c - b) / (a - c * c) if cross else np.nan
        recs.append({
            "source": src, "mode": mode, "n_img": len(g),
            "c": c, "c_raw": m["c_raw"], "cos_lin": m["cos_lin"],
            "g_r": gr, "kappa_eff": kappa_eff, "a": a, "b": b,
            "rho": rho, "gcr": rho / (1 + rho),
            "crossover": "yes" if cross else "no",
            "tau_star": rho_s / (1 + rho_s) if cross else np.nan,
            "V_over_m": m["V"] / m_eot,
            # sqrt of a ratio of mean squares, not a mean of ratios
            "norm_ratio": np.sqrt(m["sq_gbar"] / m["sq_gclean"]),
            "bias_disp": np.sqrt(m["sq_diff_clean"] / m["sq_gclean"]),
            "res_draw": np.sqrt(m["msq_diff_lin"] / m["msq_draw"]),
            "res_mean": np.sqrt(m["sq_diff_mean_lin"] / m["sq_gbar_lin"]),
        })
    summ = pd.DataFrame(recs).set_index(["source", "mode"])
    p_sum = os.path.join(args.results_dir, "moments_summary.csv")
    summ.to_csv(p_sum)
    print(f"[Saved] {p_sum}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summ.round(4).to_string())

    lg = dl.groupby(["source", "eps_chk"])[
        ["lgc_emp", "dot12", "sq_a", "sq_b", "mu_sq", "sig2"]].mean()
    # LGC_2: the theorem's moment ratio, measured directly on independent draws.
    lg["lgc2"] = lg["dot12"] / np.sqrt(lg["sq_a"] * lg["sq_b"])
    # GCR from the independent moment estimator (probe set A). If the additive
    # model holds, lgc2 and gcr must agree to O(1/n); any disagreement is a
    # failure of the model, not of either estimator.
    lg["rho"] = lg["mu_sq"] / (n_dim * lg["sig2"])
    lg["gcr"] = lg["rho"] / (1 + lg["rho"])
    lg["consistency_check"] = lg["lgc2"] - lg["gcr"]
    # The measured relation between Eq. 1 and the theorem's quantity.
    lg["gap_eq1_vs_lgc2"] = lg["lgc_emp"] - lg["lgc2"]
    lg["lgc_emp_sq"] = lg["lgc_emp"] ** 2   # candidate relation: LGC_emp^2 = LGC_2
    lgc_sum = lg[["lgc_emp", "lgc2", "gcr", "consistency_check",
                  "gap_eq1_vs_lgc2", "lgc_emp_sq", "rho"]]
    p_lsum = os.path.join(args.results_dir, "lgc_bridge_summary.csv")
    lgc_sum.to_csv(p_lsum)
    print(f"\n[Saved] {p_lsum}")
    print("\nLGC bridge (Prop. A):  LGC_emp  vs  predicted (rho+omega)/(1+rho)  vs  GCR=rho/(1+rho)")
    with pd.option_context("display.width", 200):
        print(lgc_sum.round(4).to_string())

    if rows_tgt:
        dt = pd.DataFrame(rows_tgt)
        p_tgt = os.path.join(args.results_dir, "general_target_per_image.csv")
        dt.to_csv(p_tgt, index=False)
        # Gamma must be a ratio of expectations (the theorem's constants are
        # moments); the per-image ratio is degenerate because gamma_mu can be
        # ~0. Aggregate as mean(gamma_R^2)/mean(gamma_mu^2).
        dt["gamma_mu2"] = dt["gamma_mu"] ** 2
        dt["gamma_R2"] = dt["gamma_R"] ** 2
        tsum = dt.groupby(["source", "mode", "target"])[
            ["gamma_mu", "gamma_R", "gamma_mu2", "gamma_R2",
             "sign_mu", "sign_R", "d_sign"]].mean()
        tsum["Gamma"] = tsum["gamma_R2"] / tsum["gamma_mu2"]
        tsum = tsum[["gamma_mu", "gamma_R", "Gamma", "sign_mu", "sign_R", "d_sign"]]
        p_tsum = os.path.join(args.results_dir, "general_target_summary.csv")
        tsum.to_csv(p_tsum)
        print(f"\n[Saved] {p_tgt}  ({len(dt)} rows)")
        print(f"[Saved] {p_tsum}")
        print("\nGeneral-target test (Prop. 2): DI necessarily hurts as GCR->1 iff Gamma<1")
        with pd.option_context("display.width", 220):
            print(tsum.round(5).to_string())


if __name__ == "__main__":
    main()
