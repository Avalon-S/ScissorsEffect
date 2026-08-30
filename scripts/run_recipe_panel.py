#!/usr/bin/env python
"""
Cross-recipe robustness panel: is the Scissors Effect specific to the
Salman / ResNet-50 / PGD-AT family, or does it replicate across adversarial
training recipes and architectures, and across more than one target?

Why this experiment
-------------------
The controlled epsilon-sweep (run_eps_sweep.py) isolates robustness STRENGTH but
holds the recipe and backbone fixed (Salman ResNet-50, PGD-AT) and uses a single
target (Swin-B). A fair criticism is that this leaves open whether DI harm is
specific to that one family/target. There is no public second epsilon-spectrum at
a fixed non-ResNet-50 backbone, so we do NOT fake a second strength sweep. Instead
we test REPLICATION: across a panel of robust ImageNet surrogates spanning
genuinely different AT recipes and architectures (fast-AT, PGD-AT, ViT-AT, and
transformer/ConvNeXt/XCiT robust models), against TWO targets, we ask whether
  (a) DI still harms every robust surrogate (D_ASR = ASR(DI) - ASR(MI) < 0), and
  (b) the sign tracks the LGC regime (LGC high) rather than the recipe/architecture.

What this supports (and what it does NOT)
-----------------------------------------
Supports: the Scissors direction is not a Salman-RN50-PGD-AT artifact; it is a
property of the resulting gradient regime (LGC), reproduced across recipes and a
second target. Does NOT claim: a second *controlled strength sweep* (checkpoint
availability); we are explicit that strength is controlled only on the Salman
spectrum, and cross-recipe evidence is about replication of the sign + LGC-tracking.

Sources/targets are RobustBench ImageNet Linf model ids (loaded straight through
get_model, which forwards unknown names to robustbench.load_model). Edit the panel
to whichever checkpoints are available; a source that will not load is skipped
with a message rather than aborting the panel.

Run (RTX 4090, ImageNet):
  python scripts/run_recipe_panel.py \
      --n_examples 500 --n_seeds 3 --batch_size 8 \
      --targets Swin_B_ImageNet ConvNeXt_B_ImageNet \
      --results_dir results/recipe_panel
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset
from models import get_model
from attacks.wrappers import get_attack
from attacks.sap import estimate_p_lgc
from utils import seed_everything, get_device


# Panel: (model_id, label, recipe, architecture, is_robust)
# model_id is passed straight to get_model -> robustbench.load_model for unknown
# names. These are distinct RECIPES/ARCHITECTURES, the point being that none share
# Salman-RN50-PGD-AT. Adjust ids/availability to your local RobustBench zoo.
DEFAULT_PANEL = [
    ("ResNet50",                          "Standard RN50",     "none",      "RN50",       False),
    ("Engstrom2019Robustness_ImageNet",   "Engstrom",          "PGD-AT",    "RN50",       True),
    ("Salman2020Do_R50",                  "Salman R50",        "PGD-AT",    "RN50",       True),
    # ("Wong2020Fast",                    "Wong fast-AT",      "Fast-AT",   "RN50",       True),  # dropped: gdrive rate-limit + 288px + RN50-redundant
    ("Mo2022When_ViT-B",                  "Mo2022",            "ViT-AT",    "ViT-B",      True),
    ("ARES_ConvNeXt_B",                   "ARES ConvNeXt",     "AT",        "ConvNeXt-B", True),
    ("Singh2023Revisiting_ViT-B-ConvStem","Singh2023",         "AT",        "ViT-B-CvSt", True),
]

DEFAULT_TARGETS = ["Swin_B_ImageNet", "ConvNeXt_B_ImageNet"]

# Locally-supplied checkpoints that are NOT RobustBench model ids (e.g. uploaded
# to a server folder). panel_id -> (filename under <dir>/imagenet/Linf/, timm arch).
LOCAL_CKPTS = {
    "ARES_ConvNeXt_B": ("ARES_ConvNext_Base_AT.pth", "convnext_base"),
}
_LOCAL_DIRS = [d for d in [os.environ.get("MODEL_ROOT"),
                           os.path.join(os.getcwd(), "assets", "models"),
                           "/autodl-tmp/models", "/root/autodl-tmp/models",
                           "/root/robustbench/models"] if d]


def _find_ckpt(filename):
    for base in _LOCAL_DIRS:
        p = os.path.join(base, "imagenet", "Linf", filename)
        if os.path.exists(p):
            return p
    return None


class _NormWrap(torch.nn.Module):
    """Feed [0,1] images; resize to 224 and apply ImageNet normalization.
    If clean_acc comes out near-zero, the checkpoint likely bakes its own
    normalization -- set normalize=False to feed [0,1] directly."""
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.model(x)


def _load_local(mid, device, normalize=True):
    import timm
    fname, arch = LOCAL_CKPTS[mid]
    path = _find_ckpt(fname)
    if path is None:
        raise FileNotFoundError(f"{fname} not found under {_LOCAL_DIRS} imagenet/Linf")
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    sd = {(key[7:] if key.startswith("module.") else key): v for key, v in sd.items()}
    net = timm.create_model(arch, pretrained=False, num_classes=1000)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"{mid}: {len(missing)} weights of arch '{arch}' were not present in "
            f"{os.path.basename(path)} and would stay at their random "
            f"initialisation (first: {missing[:3]}). Refusing to attack from a "
            f"partially initialised model.")
    print(f"  [local-ckpt {mid}] arch={arch}, all weights loaded, "
          f"{len(unexpected)} unused key(s) in the checkpoint")
    return _NormWrap(net, normalize=normalize).to(device).eval()


def get_source_model(mid, device):
    """Dispatch: local custom checkpoint, else RobustBench/get_model."""
    if mid in LOCAL_CKPTS:
        return _load_local(mid, device)
    return get_model(mid, dataset="imagenet", device=device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_examples", type=int, default=500)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--eps_attack", type=float, default=16 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--diversity_prob", type=float, default=1.0)
    p.add_argument("--resize_rate", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lgc_batch", type=int, default=64,
                   help="images used to estimate per-source LGC")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--targets", type=str, nargs="+", default=None)
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help="override panel; 'id|label|recipe|arch|is_robust' "
                        "(is_robust = 1/0), or just 'id' (assumed robust)")
    p.add_argument("--results_dir", type=str, default="results/recipe_panel")
    return p.parse_args()


def parse_panel(spec_list):
    panel = []
    for s in spec_list:
        parts = s.split("|")
        if len(parts) >= 5:
            mid, lbl, rec, arch, rob = parts[:5]
            panel.append((mid.strip(), lbl.strip(), rec.strip(), arch.strip(),
                          rob.strip() in ("1", "true", "True")))
        else:
            mid = parts[0].strip()
            lbl = parts[1].strip() if len(parts) > 1 else mid
            panel.append((mid, lbl, "unknown", "unknown", True))
    return panel


def attack_and_eval(src_model, tgt_model, x, y, attack_name, args, device):
    if attack_name == "mifgsm":
        attack = get_attack("mifgsm", src_model, eps=args.eps_attack,
                            alpha=args.alpha, steps=args.steps)
    else:
        attack = get_attack("difgsm", src_model, eps=args.eps_attack,
                            alpha=args.alpha, steps=args.steps,
                            resize_rate=args.resize_rate,
                            diversity_prob=args.diversity_prob)
    n_correct, n_total = 0, 0
    for i in tqdm(range(0, x.size(0), args.batch_size),
                  desc=f"    {attack_name}", leave=False):
        xb = x[i:i + args.batch_size].to(device)
        yb = y[i:i + args.batch_size].to(device)
        x_adv = attack(xb, yb)
        with torch.no_grad():
            preds = tgt_model(x_adv).argmax(dim=1)
        n_correct += (preds == yb).sum().item()
        n_total += xb.size(0)
    return 1.0 - n_correct / n_total


def source_lgc(src_model, x, y, args, device):
    """Mean LGC over a small calibration batch (Eq. 1, K=5)."""
    xb = x[:args.lgc_batch].to(device)
    yb = y[:args.lgc_batch].to(device)
    vals = []
    for i in range(0, xb.size(0), args.batch_size):
        _, lgc = estimate_p_lgc(src_model, xb[i:i + args.batch_size],
                                yb[i:i + args.batch_size], device, K=5)
        vals.append(lgc.detach().cpu())
    return float(torch.cat(vals).mean())


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    panel = parse_panel(args.sources) if args.sources else DEFAULT_PANEL
    targets = args.targets if args.targets else DEFAULT_TARGETS

    print("=" * 80)
    print("Cross-recipe robustness panel (Scissors replication across recipes/targets)")
    print("=" * 80)
    print(f"  N={args.n_examples}, n_seeds={args.n_seeds}, steps={args.steps}, "
          f"eps_attack={args.eps_attack*255:.0f}/255")
    print(f"  DI: p={args.diversity_prob}, resize_rate={args.resize_rate}")
    print(f"  Targets: {targets}")
    print(f"  Panel ({len(panel)} sources):")
    for mid, lbl, rec, arch, rob in panel:
        print(f"    - {lbl:<16} [{mid}]  recipe={rec}, arch={arch}, "
              f"{'robust' if rob else 'standard'}")
    print()

    seed_everything(42)
    print(f"[Data] Loading ImageNet validation ({args.n_examples})...")
    x_test, y_test = load_dataset("imagenet", args.n_examples)

    # Load targets once
    tgt_models = {}
    for t in targets:
        try:
            tgt_models[t] = get_model(t, dataset="imagenet", device=device)
            tgt_models[t].eval()
        except Exception as e:
            print(f"  [skip target] {t}: {e}")
    if not tgt_models:
        print("[Error] No targets loaded.")
        return

    results = []
    t0 = time.time()

    for mid, lbl, recipe, arch, is_robust in panel:
        print(f"\n[Source] {lbl}  [{mid}]  recipe={recipe}, arch={arch}")
        try:
            src_model = get_source_model(mid, device)
            src_model.eval()
        except Exception as e:
            print(f"  [skip source] failed to load: {e}")
            continue

        # Clean-acc sanity (catches a silently-misloaded local checkpoint:
        # near-zero acc => state_dict key mismatch or normalization mismatch).
        with torch.no_grad():
            nc = 0
            for i in range(0, x_test.size(0), args.batch_size):
                xb = x_test[i:i + args.batch_size].to(device)
                nc += (src_model(xb).argmax(1).cpu()
                       == y_test[i:i + args.batch_size]).sum().item()
        clean_acc = nc / x_test.size(0)
        flag = "  <-- LOW! check ckpt load / normalization" if clean_acc < 0.30 else ""
        print(f"  clean_acc={clean_acc*100:.1f}%{flag}")

        try:
            lgc = source_lgc(src_model, x_test, y_test, args, device)
        except Exception as e:
            print(f"  [warn] LGC failed ({e}); set NaN")
            lgc = float("nan")
        print(f"  LGC={lgc:.3f}")

        for tname, tgt_model in tgt_models.items():
            same_family = arch.split("-")[0].lower() in tname.lower()
            # MI-FGSM (deterministic -> 1 seed); DI-FGSM (n_seeds)
            seed_everything(42)
            mi = attack_and_eval(src_model, tgt_model, x_test, y_test,
                                 "mifgsm", args, device)
            di_seeds = []
            for s in range(args.n_seeds):
                seed_everything(42 + s * 100)
                di_seeds.append(attack_and_eval(src_model, tgt_model, x_test,
                                                y_test, "difgsm", args, device))
            di = float(np.mean(di_seeds))
            di_sem = float(np.std(di_seeds, ddof=1) / np.sqrt(len(di_seeds))
                           if len(di_seeds) > 1 else 0.0)
            d_asr = di - mi
            tag = ("OK" if (is_robust and d_asr < 0) or
                   (not is_robust and d_asr > 0) else "FLAG")
            fam = " (same-family!)" if same_family else ""
            print(f"    -> {tname:<22} MI={mi*100:6.2f}  DI={di*100:6.2f}  "
                  f"D_ASR={d_asr*100:+6.2f}  [{tag}]{fam}")
            results.append({
                "source": mid, "label": lbl, "recipe": recipe, "arch": arch,
                "is_robust": is_robust, "lgc": lgc, "clean_acc": clean_acc,
                "target": tname,
                "same_family": same_family, "mi_asr": mi, "di_asr": di,
                "di_sem": di_sem, "d_asr": d_asr, "n_seeds": args.n_seeds,
                "n_examples": args.n_examples,
            })
        del src_model
        torch.cuda.empty_cache()

    if not results:
        print("[Error] No results.")
        return

    df = pd.DataFrame(results)
    csv = os.path.join(args.results_dir, "recipe_panel.csv")
    df.to_csv(csv, index=False)
    report(df, args, time.time() - t0)
    print(f"\n[Saved] {csv}")


def report(df, args, elapsed):
    lines = []

    def log(m=""):
        print(m)
        lines.append(m)

    log("\n" + "=" * 92)
    log("SUMMARY  (D_ASR = ASR(DI) - ASR(MI); robust expects <0, standard >0)")
    log("=" * 92)
    log(f"{'Source':<16}{'recipe':<10}{'arch':<12}{'LGC':>6}  "
        f"{'target':<20}{'MI':>7}{'DI':>7}{'D_ASR':>8}")
    log("-" * 92)
    for _, r in df.iterrows():
        log(f"{r['label']:<16}{r['recipe']:<10}{r['arch']:<12}{r['lgc']:>6.2f}  "
            f"{r['target']:<20}{r['mi_asr']*100:>7.2f}{r['di_asr']*100:>7.2f}"
            f"{r['d_asr']*100:>+8.2f}")

    rob = df[df["is_robust"]]
    std = df[~df["is_robust"]]
    # Exclude same-family source/target pairs from the headline replication count
    rob_xf = rob[~rob["same_family"]]

    log("\n" + "=" * 92)
    log("REPLICATION VERDICT")
    log("=" * 92)
    n_rob = len(rob_xf)
    n_hurt = int((rob_xf["d_asr"] < 0).sum())
    log(f"  Robust cross-family (source,target) pairs with DI HARM (D_ASR<0): "
        f"{n_hurt}/{n_rob}")
    if n_rob:
        recipes = sorted(rob_xf["recipe"].unique())
        archs = sorted(rob_xf["arch"].unique())
        tgts = sorted(rob_xf["target"].unique())
        log(f"  spanning recipes={recipes}, archs={archs}, targets={tgts}")
    n_std = len(std)
    n_help = int((std["d_asr"] > 0).sum())
    log(f"  Standard (source,target) pairs with DI BENEFIT (D_ASR>0): {n_help}/{n_std}")

    # LGC-tracking: does sign(D_ASR) follow LGC regime rather than recipe?
    valid = df.dropna(subset=["lgc"])
    if len(valid) > 3:
        hi = valid[valid["lgc"] > 0.92]
        lo = valid[valid["lgc"] <= 0.92]
        hi_hurt = int((hi["d_asr"] < 0).sum())
        lo_help = int((lo["d_asr"] > 0).sum())
        log(f"  LGC-tracking: of LGC>0.92 pairs, {hi_hurt}/{len(hi)} show harm; "
            f"of LGC<=0.92 pairs, {lo_help}/{len(lo)} show benefit.")
        log("  (If the sign follows LGC across DIFFERENT recipes/archs, the effect "
            "is a gradient-regime property, not a Salman-RN50-PGD-AT artifact.)")

    log("\n  SCOPE: strength is controlled only on the Salman spectrum "
        "(run_eps_sweep). This")
    log("  panel establishes cross-recipe/-target REPLICATION of the sign and its "
        "LGC-tracking,")
    log("  NOT a second controlled strength sweep (no public fixed-backbone "
        "spectrum exists).")
    log(f"\n  runtime: {elapsed/60:.1f} min")

    with open(os.path.join(args.results_dir, "recipe_panel_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
