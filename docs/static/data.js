// All numbers below are transcribed from the accepted TMLR manuscript.
// Each block names the table it comes from so the page can be re-checked
// against the paper. Differences between two ASRs are in percentage points (pp);
// % is used only for rates.

// Tab. 7 -- controlled epsilon-sweep (fixed ResNet-50 + PGD-AT, target Swin-B,
// N=500, 3 seeds, all-images basis).
const EPS_SWEEP = [
  { eps: "0",   label: "0 (standard)", mi: 41.2, di: 54.2, d: +13.0, note: "helps" },
  { eps: "0.5", label: "0.5/255",      mi: 88.4, di: 82.9, d:  -5.5, note: "crossover" },
  { eps: "1",   label: "1/255",        mi: 86.0, di: 79.1, d:  -6.9, note: "" },
  { eps: "2",   label: "2/255",        mi: 78.4, di: 65.9, d: -12.5, note: "max harm" },
  { eps: "4",   label: "4/255",        mi: 62.2, di: 51.3, d: -10.9, note: "" },
  { eps: "8",   label: "8/255",        mi: 44.4, di: 37.8, d:  -6.6, note: "" },
];

// Tab. 12 -- 2x2 factorial over 6 sources x 3 targets, N=1000, 5 seeds, p=1,
// all-images basis, r>1. Effects in pp with 95% image-paired bootstrap CIs.
// NOTE: the earlier "resize accounts for 67% of the harm" attribution was
// withdrawn; it was not a valid additive decomposition. These are the
// main effects and the interaction, reported separately.
const FACTORIAL = {
  robust: [
    { name: "resize (main effect)",      est: -9.71, lo: -10.43, hi: -9.01, key: true },
    { name: "translation (main effect)", est: -2.32, lo:  -2.84, hi: -1.84 },
    { name: "interaction",               est: -0.47, lo:  -1.07, hi: +0.13, zero: true },
    { name: "both cells",                est: -12.03, lo: -12.96, hi: -11.14 },
    { name: "full DI operator",          est: -14.19, lo: -15.22, hi: -13.22 },
  ],
  standard: [
    { name: "resize (main effect)",      est: +4.38, lo: +3.77, hi: +4.99 },
    { name: "translation (main effect)", est: +8.47, lo: +7.71, hi: +9.24 },
    { name: "interaction",               est: -2.65, lo: -3.54, hi: -1.77 },
    { name: "both cells",                est: +12.85, lo: +11.67, hi: +14.06 },
    { name: "full DI operator",          est: +11.49, lo: +10.35, hi: +12.63 },
  ],
};

// Tab. 9 -- gradient geometry, 12 ImageNet models (N=200, K=5 probes at 1/255).
// HF is a ratio of log-magnitudes, not of energies.
const MODELS = [
  { name: "Swin-B",       type: "standard", hf: 0.55, lgc: 0.36 },
  { name: "ResNet50",     type: "standard", hf: 0.55, lgc: 0.64 },
  { name: "DenseNet121",  type: "standard", hf: 0.42, lgc: 0.74 },
  { name: "CLIP ViT-B/32", type: "vlm",     hf: 0.47, lgc: 0.72 },
  { name: "InceptionV3",  type: "standard", hf: 0.39, lgc: 0.81 },
  { name: "ResNet18",     type: "standard", hf: 0.48, lgc: 0.83 },
  { name: "VGG16",        type: "standard", hf: 0.50, lgc: 0.84 },
  { name: "ConvNeXt-B",   type: "standard", hf: 0.57, lgc: 0.87 },
  { name: "ViT-B/16",     type: "standard", hf: 0.37, lgc: 0.90 },
  { name: "Engstrom",     type: "robust",   hf: 0.31, lgc: 0.98 },
  { name: "Salman2020",   type: "robust",   hf: 0.35, lgc: 0.98 },
  { name: "Mo2022",       type: "robust",   hf: 0.17, lgc: 1.00 },
];

// Tab. 5 -- ten attacks, Engstrom / ResNet50 -> Swin-B, N=1000, 5 seeds,
// all-images basis. D = DI - noDI in pp.
const ATTACKS = [
  { name: "MI-FGSM",  venue: "CVPR'18",    rob: -1.7,  std: +0.2 },
  { name: "NI-FGSM",  venue: "ICLR'20",    rob: -0.7,  std: +0.4 },
  { name: "VMI-FGSM", venue: "CVPR'21",    rob: -0.6,  std: +1.9 },
  { name: "Admix",    venue: "ICCV'21",    rob: -3.2,  std: +1.7 },
  { name: "SSA",      venue: "ECCV'22",    rob: -15.6, std: -0.6 },
  { name: "SIA",      venue: "ICCV'23",    rob: -5.6,  std: +1.1 },
  { name: "GRA",      venue: "ICCV'23",    rob: -0.6,  std: +1.0 },
  { name: "PGN",      venue: "NeurIPS'23", rob: -1.8,  std: +1.5 },
  { name: "BSR",      venue: "CVPR'24",    rob: -5.1,  std: +0.4 },
  { name: "AdaMSI",   venue: "AAAI'24",    rob: -1.8,  std: +0.4 },
];

// Tab. 6 -- the same panel with dependence taken into account. The attacks are
// not ten independent observations, so the tally is not a binomial experiment.
const PANEL_STATS = {
  rhoBar: "0.12-0.15",
  effectiveN: "4.2-4.9",
  swinRobust:   { d: -3.92, lo: -4.81, hi: -3.04 },
  swinStandard: { d: +0.83, lo: +0.27, hi: +1.40 },
  cnxRobust:    { d: -6.25, lo: -7.30, hi: -5.23 },
  cnxStandard:  { d: +1.38, lo: +0.68, hi: +2.06 },
  resolvedSwin: "4 of 10 per side",
  resolvedCnx:  "9 of 10 on the robust side",
};

// Sec. 5 -- CG-DI. tau frozen at its published value before the held-out runs.
const CGDI = {
  tau: 0.92, pOn: 0.8, pOff: 0, K: 5, probeScale: "1/255",
  costPP: 0.95, costLo: 0.50, costHi: 1.42,
  presets: [
    { name: "Swin-B",       lgc: 0.36 },
    { name: "ResNet-50",    lgc: 0.64 },
    { name: "ViT-B/16",     lgc: 0.90 },
    { name: "Engstrom",     lgc: 0.98 },
    { name: "Mo2022",       lgc: 1.00 },
  ],
};

// Tab. 19 -- held-out evaluation, seven surrogates never used as sources,
// N=1000, 5 seeds, r=0.9, source-and-target clean-correct basis. D in pp.
const HELDOUT = [
  { name: "ARES ConvNeXt-AT",  type: "robust",   lgc: 0.999, pred: "p=0",   d: -2.17, lo: -3.16, hi: -1.16 },
  { name: "Singh ConvStem-AT", type: "robust",   lgc: 1.000, pred: "p=0",   d: -0.81, lo: -1.74, hi: +0.15 },
  { name: "Swin-B",            type: "standard", lgc: 0.390, pred: "p=0.8", d: +23.30, lo: +21.26, hi: +25.27 },
  { name: "DenseNet-121",      type: "standard", lgc: 0.759, pred: "p=0.8", d: +14.15, lo: +12.73, hi: +15.52 },
  { name: "InceptionV3",       type: "standard", lgc: 0.814, pred: "p=0.8", d:  +9.76, lo:  +8.36, hi: +11.18 },
  { name: "ConvNeXt-B",        type: "standard", lgc: 0.875, pred: "p=0.8", d: +16.23, lo: +14.78, hi: +17.78 },
  { name: "ViT-B/16",          type: "standard", lgc: 0.902, pred: "p=0.8", d: +13.23, lo: +11.56, hi: +14.97 },
];

// Headline: Tab. 2, Engstrom -> four cross-family targets, all-images basis.
const HEADLINE = { costPP: 10.3, base: 76.0, withDI: 65.7, stdBase: 44.0, stdWithDI: 58.6 };

// Fig. 1 curves, exported by scripts/plot_fig1_scissors.py (the same script
// that draws the figure), so the page and the figure cannot drift apart.
const FIG1 = {"ps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "panels": [{"title": "ImageNet   (4 targets)", "rows": [{"label": "ResNet-50 (standard)", "base": 43.92, "mean": [0.0, 3.7, 6.48, 8.82, 10.24, 11.66, 12.68, 13.62, 14.11, 14.43, 14.88], "lo": [0.0, 3.16, 5.69, 7.84, 9.18, 10.52, 11.45, 12.33, 12.78, 13.07, 13.47], "hi": [0.0, 4.26, 7.3, 9.84, 11.36, 12.83, 13.92, 14.91, 15.42, 15.84, 16.3]}, {"label": "Engstrom (robust)", "base": 76.05, "mean": [0.0, -0.78, -1.4, -2.13, -2.82, -3.66, -4.58, -5.9, -6.92, -8.43, -9.88], "lo": [0.0, -1.1, -1.86, -2.7, -3.5, -4.42, -5.39, -6.8, -7.86, -9.48, -10.98], "hi": [0.0, -0.47, -0.96, -1.58, -2.16, -2.9, -3.76, -5.01, -5.98, -7.4, -8.77]}, {"label": "Salman $\\epsilon$=2 (robust)", "base": 82.08, "mean": [0.0, -0.98, -1.62, -2.24, -2.55, -3.08, -3.66, -4.3, -4.94, -6.16, -7.48], "lo": [0.0, -1.32, -2.11, -2.82, -3.2, -3.8, -4.44, -5.13, -5.88, -7.13, -8.51], "hi": [0.0, -0.64, -1.16, -1.67, -1.91, -2.37, -2.89, -3.48, -4.07, -5.19, -6.46]}]}, {"title": "CIFAR-10   (2 targets)", "rows": [{"label": "ResNet-18 (standard)", "base": 83.9, "mean": [0.0, 1.55, 2.47, 3.54, 3.92, 3.82, 3.42, 2.84, 2.36, 1.55, 0.5], "lo": [0.0, 1.02, 1.63, 2.51, 2.74, 2.52, 2.03, 1.39, 0.87, 0.02, -1.08], "hi": [0.0, 2.1, 3.33, 4.63, 5.12, 5.12, 4.79, 4.31, 3.85, 3.09, 2.09]}, {"label": "Engstrom (robust)", "base": 29.1, "mean": [0.0, -0.82, -1.77, -2.47, -3.24, -4.06, -4.95, -5.88, -6.51, -7.54, -8.44], "lo": [0.0, -1.26, -2.4, -3.28, -4.12, -5.07, -6.01, -7.04, -7.74, -8.83, -9.83], "hi": [0.0, -0.39, -1.14, -1.73, -2.37, -3.09, -3.92, -4.77, -5.32, -6.26, -7.11]}, {"label": "Rice (robust)", "base": 24.9, "mean": [0.0, -0.69, -1.26, -1.89, -2.49, -3.06, -3.81, -4.69, -5.51, -6.4, -7.08], "lo": [0.0, -1.08, -1.9, -2.67, -3.39, -4.0, -4.82, -5.82, -6.73, -7.71, -8.43], "hi": [0.0, -0.31, -0.66, -1.14, -1.65, -2.15, -2.81, -3.6, -4.33, -5.15, -5.76]}]}]};
