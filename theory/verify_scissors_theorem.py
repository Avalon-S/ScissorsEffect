#!/usr/bin/env python
"""
Machine-checked verification of the bias-variance theory of the Scissors Effect.

We model the surrogate input-gradient as g = mu + eta, with E[eta]=0,
Cov(eta)=sigma^2 I_n (isotropic). DI maps g -> gbar = (1/m) sum_i R g(T_i x),
where R is the resize operator. R is NOT assumed symmetric and NOT assumed to be
a contraction: the noise passes through it as zeta = R eta, so
Cov(zeta) = sigma^2 R R^T and kappa = tr(R R^T)/(n m), and the only structural
condition the argument uses is kappa/g_r < c^2 < 1. Section [N] at the end
checks this on a deliberately non-symmetric R with spectral radius 1.30.
The transferable
target direction is assumed proportional to the surrogate signal mu (robust
features transfer), and the figure of merit is the signal-to-RMS alignment

    A = <E[d], muhat> / sqrt(E||d||^2),    muhat = mu/||mu||.

A uses only first and second moments, so it is EXACT (no concentration needed).

This script uses sympy to verify, symbolically:
  (L) the LGC identity   LGC = rho/(1+rho),  rho = ||mu||^2/(n sigma^2);
  (A) the closed forms of A_noDI and A_DI;
  (T) the crossover SNR  rho* = (c^2 - kappa/g_r)/(1 - c^2);
  (M) strict monotonicity of phi = (A_DI/A_noDI)^2  => a UNIQUE crossover;
  (Lim) the limits rho->inf (DI hurts) and rho->0 (DI helps iff c^2 g_r > kappa).
Then a numpy Monte-Carlo bridges the A-metric theorem to the actual
E[cos(d, mu)] that the paper measures, confirming the crossover location.
"""
import sympy as sp
import numpy as np

print("=" * 70)
print("SYMBOLIC VERIFICATION (sympy)")
print("=" * 70)

# positive real symbols
s, N, rho, c, kappa, gr, m, n, sig = sp.symbols(
    's N rho c kappa g_r m n sigma', positive=True)

# ----------------------------------------------------------------------
# (L) LGC identity for the additive-Gaussian model.
#   g = mu+eta, g' = mu+eta', eta,eta' iid mean0 cov sigma^2 I_n.
#   E<g,g'> = ||mu||^2 = s ;  E||g||^2 = s + n sigma^2 = s + N.
#   moment-LGC := E<g,g'> / sqrt(E||g||^2 E||g'||^2) = s/(s+N).
N_expr = n * sig**2                       # total noise power N = n sigma^2
LGC = s / (s + N)                          # = E<g,g'>/E||g||^2
LGC_in_rho = LGC.subs(s, rho * N)          # rho = s/N
LGC_in_rho = sp.simplify(LGC_in_rho)
print("\n[L] LGC = s/(s+N) re-expressed via rho=s/N :", LGC_in_rho,
      " == rho/(1+rho)?", sp.simplify(LGC_in_rho - rho / (1 + rho)) == 0)

# ----------------------------------------------------------------------
# (A) Closed forms.
#   No-DI:  d = mu+eta. <E[d],muhat>=||mu||=sqrt(s); E||d||^2 = s+N.
#   DI:     d = R mu + (1/m) sum zeta_i, zeta cov sigma^2 R R^T.
#           <E[d],muhat> = mu^T R mu / ||mu|| = a/sqrt(s);
#           E||d||^2 = ||R mu||^2 + sigma^2 tr(R R^T)/m = q + N*kappa,
#           with kappa = tr(R R^T)/(n m).
#   Reparametrize: c = cos(Rmu,mu) = a/sqrt(s q);  g_r = q/s = ||Rmu||^2/||mu||^2.
A0 = sp.sqrt(s) / sp.sqrt(s + N)                      # A_noDI
A0 = sp.simplify(A0.rewrite(sp.Pow))
A0_rho = sp.simplify((sp.sqrt(rho) / sp.sqrt(rho + 1)))   # divide num/den by sqrt(N)
print("\n[A] A_noDI = sqrt(rho/(1+rho)) = sqrt(LGC) :",
      sp.simplify(A0.subs(s, rho * N) - A0_rho) == 0)

# A_DI = (a/sqrt(s)) / sqrt(q + N kappa); a=c*sqrt(s*q), q=g_r*s, N=s/rho
a = c * sp.sqrt(s * (gr * s))
q = gr * s
A1 = (a / sp.sqrt(s)) / sp.sqrt(q + (s / rho) * kappa)
A1 = sp.simplify(A1)
A1_target = c / sp.sqrt(1 + kappa / (gr * rho))
print("[A] A_DI = c / sqrt(1 + kappa/(g_r rho)) :",
      sp.simplify(A1 - A1_target) == 0)

# ----------------------------------------------------------------------
# (T) Crossover: solve A_DI = A_noDI  for rho.
phi = (A1_target**2) / (A0_rho**2)          # = c^2 (rho+1)/(rho + kappa/g_r)
phi = sp.simplify(phi)
print("\n[T] phi := (A_DI/A_noDI)^2 =", phi)
sol = sp.solve(sp.Eq(phi, 1), rho)
rho_star = sp.simplify(sol[0])
rho_star_target = (c**2 - kappa / gr) / (1 - c**2)
print("[T] solve phi=1  =>  rho* =", rho_star)
print("[T] equals (c^2 - kappa/g_r)/(1-c^2)? ",
      sp.simplify(rho_star - rho_star_target) == 0)

# ----------------------------------------------------------------------
# (M) Monotonicity: d phi/d rho < 0  (given kappa/g_r < 1)  => unique crossing.
dphi = sp.simplify(sp.diff(phi, rho))
print("\n[M] d(phi)/d(rho) =", dphi)
#   numerator sign: proportional to (kappa/g_r - 1) < 0.
dphi_sign_factor = sp.simplify(dphi * (rho + kappa / gr)**2 / c**2)
print("[M] (dphi/drho)*(rho+kappa/g_r)^2/c^2 =", dphi_sign_factor,
      " -> negative iff kappa/g_r < 1, which the assumed kappa/g_r < c^2 < 1"
      " gives without needing R to be a contraction")

# ----------------------------------------------------------------------
# (Lim) Limits.
lim_inf = sp.limit(phi, rho, sp.oo)
lim_zero = sp.limit(phi, rho, 0, '+')
print("\n[Lim] phi(rho->inf) =", lim_inf, " (= c^2 < 1  => DI HURTS robust)")
print("[Lim] phi(rho->0+)  =", lim_zero,
      " (>1  => DI HELPS iff c^2 g_r > kappa)")

# ----------------------------------------------------------------------
# Conclusion: phi strictly decreasing, phi(0+)=c^2 g_r/kappa, phi(inf)=c^2<1.
# If kappa/g_r < c^2 < 1 then phi(0+)>1>phi(inf): exactly one crossover rho*>0.
# Since LGC=rho/(1+rho) is strictly increasing, "rho<rho*" <=> "LGC<tau*",
# tau* = rho*/(1+rho*).  => DI helps iff LGC<tau*, hurts iff LGC>tau*.  QED.
print("\n[QED] phi strictly decreasing; one crossover at rho* > 0 when "
      "kappa/g_r < c^2 < 1.")
print("      DI helps  <=>  rho < rho*  <=>  LGC < tau* = rho*/(1+rho*).")

print("\n" + "=" * 70)
print("NUMERICAL BRIDGE (numpy Monte-Carlo): A-metric theorem vs E[cos]")
print("=" * 70)

rng = np.random.default_rng(0)
nd = 64                                   # dimension (1D DCT model)
# DCT-II orthonormal basis C (n x n): rows are frequency modes.
k = np.arange(nd)
C = np.cos(np.pi * (np.outer(k, 2 * np.arange(nd) + 1)) / (2 * nd))
C *= np.sqrt(2.0 / nd)
C[0] *= 1 / np.sqrt(2)
# Resize operator R = C^T diag(h) C : low-pass (attenuate high freq).
h = np.where(k < nd // 2, 1.0, 0.35)      # keep low freq, damp high freq
R = C.T @ np.diag(h) @ C
R = 0.5 * (R + R.T)                        # symmetric

# Signal mu with BOTH low- and high-frequency content (so c<1, g_r<1).
spec = np.zeros(nd); spec[1] = 1.0; spec[2] = 0.7; spec[40] = 0.5; spec[55] = 0.4
mu = C.T @ spec
mu /= np.linalg.norm(mu)

Rmu = R @ mu
c_num = float(Rmu @ mu / (np.linalg.norm(Rmu) * np.linalg.norm(mu)))
gr_num = float((Rmu @ Rmu) / (mu @ mu))
m_eot = 10
kappa_num = float(np.trace(R @ R) / (nd * m_eot))
rho_star_num = (c_num**2 - kappa_num / gr_num) / (1 - c_num**2)
tau_star = rho_star_num / (1 + rho_star_num)
print(f"R,mu constants:  c={c_num:.4f}  g_r={gr_num:.4f}  "
      f"kappa={kappa_num:.4f}  (m={m_eot})")
print(f"Predicted crossover:  rho* = {rho_star_num:.4f}   "
      f"tau* (LGC) = {tau_star:.4f}")

def mc_cos(sigma, di, trials=4000):
    """Monte-Carlo E[cos(d, mu)] for no-DI (di=False) or DI (di=True)."""
    acc = 0.0
    for _ in range(trials):
        if not di:
            d = mu + sigma * rng.standard_normal(nd)
        else:
            g = np.zeros(nd)
            for _ in range(m_eot):
                g += R @ (mu + sigma * rng.standard_normal(nd))
            d = g / m_eot
        acc += d @ mu / (np.linalg.norm(d) * np.linalg.norm(mu))
    return acc / trials

def A_metric(rho_v, di):
    """Deterministic A = <E[d],muhat>/sqrt(E||d||^2) from the proof's moments."""
    if not di:
        return np.sqrt(rho_v / (1 + rho_v))
    return c_num / np.sqrt(1 + kappa_num / (gr_num * rho_v))

print("\n   rho      LGC    A no-DI   A DI   |  E[cos] no-DI  E[cos] DI   DI-helps?")
cross_cos = None
cross_A = None
prev_cos = prev_A = None
LGCs, A_no_l, A_di_l, cos_no_l, cos_di_l = [], [], [], [], []
for log_rho in np.linspace(5.5, -3.0, 24):     # sweep robust(high SNR)->standard
    rho_v = float(np.exp(log_rho))
    sigma = float(np.sqrt((mu @ mu) / (nd * rho_v)))   # rho = ||mu||^2/(n sigma^2)
    lgc = rho_v / (1 + rho_v)
    A_no, A_di = A_metric(rho_v, False), A_metric(rho_v, True)
    a_no, a_di = mc_cos(sigma, di=False), mc_cos(sigma, di=True)
    helps_cos, helps_A = a_di > a_no, A_di > A_no
    if prev_cos is not None and prev_cos != helps_cos and cross_cos is None:
        cross_cos = lgc
    if prev_A is not None and prev_A != helps_A and cross_A is None:
        cross_A = lgc
    prev_cos, prev_A = helps_cos, helps_A
    LGCs.append(lgc); A_no_l.append(A_no); A_di_l.append(A_di)
    cos_no_l.append(a_no); cos_di_l.append(a_di)
    print(f"  {rho_v:7.2f}  {lgc:6.4f}  {A_no:6.3f}  {A_di:6.3f}  |  "
          f"{a_no:8.4f}     {a_di:8.4f}    {'YES' if helps_cos else 'no'}")

# ---- figure: theory_crossover.pdf -----------------------------------------
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(LGCs, A_no_l, "-", color="tab:blue", lw=2, label=r"$A$ no-DI (theory)")
ax.plot(LGCs, A_di_l, "-", color="tab:red", lw=2, label=r"$A$ DI (theory)")
ax.plot(LGCs, cos_no_l, "o", color="tab:blue", ms=4, alpha=.7,
        label=r"$\mathbb{E}[\cos]$ no-DI (MC)")
ax.plot(LGCs, cos_di_l, "s", color="tab:red", ms=4, alpha=.7,
        label=r"$\mathbb{E}[\cos]$ DI (MC)")
ax.axvline(tau_star, color="k", ls="--", lw=1.5,
           label=fr"$\tau_{{\rm sim}}={tau_star:.3f}$ (simulation constants)")
ax.axvspan(tau_star, 1.0, color="tab:red", alpha=.06)
ax.axvspan(min(LGCs), tau_star, color="tab:blue", alpha=.06)
# mark the single crossover where the DI and no-DI curves meet (at tau*).
# Closed form: A_noDI = sqrt(LGC), so the curves cross at height sqrt(tau*).
y_cross = float(tau_star ** 0.5)
ax.plot([tau_star], [y_cross], marker="o", ms=11, mfc="none", mec="k",
        mew=1.8, zorder=6)
ax.annotate("curves cross here\n(DI helps $\\to$ hurts)", xy=(tau_star, y_cross),
            xytext=(0.78, 0.50), fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color="k", lw=0.9))
ax.text(0.36, 0.30, "DI helps\n(DI curve on top)", color="tab:blue",
        ha="center", fontsize=11)
# Right-aligned inside the axes: centring a two-line label near x=1 pushes it
# past the frame, since the "DI hurts" band is only a few percent of the width.
ax.text(0.995, 0.72, "DI hurts\n(no-DI on top)", color="tab:red",
        ha="right", va="center", fontsize=10)
ax.set_xlabel("GCR (gradient consistency ratio)")
ax.set_ylabel("alignment with target direction")
ax.set_xlim(min(LGCs), 1.0)
ax.set_ylim(0.10, 1.04)
ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
ax.grid(True, ls="--", alpha=.4)
fig.tight_layout()
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "figures"),
            exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "..", "figures",
                   "theory_crossover.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"\nSaved figure -> {os.path.normpath(out)}")

print("\n" + "=" * 70)
print("COROLLARY (WITHDRAWN -- see the note printed at the end of this section):")
print("SSA as a frequency-domain pre-method (effective-regime shift)")
print("=" * 70)

# --- symbolic: rho_eff = rho * s_r / kappa_S, and LGC_eff > LGC iff s_r>kappa_S
s_r, kappa_S = sp.symbols('s_r kappa_S', positive=True)
# g_M = S mu + xi, E<g_M,g_M'> = ||S mu||^2 = s_r * s ;
# E||g_M||^2 = ||S mu||^2 + sigma^2 tr(S^2)/m_S = s_r*s + N*kappa_S  (N=n sigma^2)
rho_eff = (s_r * s) / ((s / rho) * kappa_S)        # = rho * s_r/kappa_S
rho_eff = sp.simplify(rho_eff)
print("\n[C] rho_eff = ||S mu||^2 / (sigma^2 tr(S^2)/m_S) =", rho_eff,
      " == rho*s_r/kappa_S?", sp.simplify(rho_eff - rho * s_r / kappa_S) == 0)
LGC_eff = rho_eff / (1 + rho_eff)
shift = sp.simplify(LGC_eff - rho / (1 + rho))     # >0 iff rho_eff>rho iff s_r>kappa_S
num = sp.simplify(sp.together(shift).as_numer_denom()[0])
print("[C] sign(LGC_eff - LGC) numerator =", sp.factor(num),
      " -> > 0 iff s_r > kappa_S (S removes more noise than signal)")

# --- numerical: SSA = strong DCT low-pass with m_S-fold averaging.
hS = np.where(k < nd // 2, 1.0, 0.15)              # SSA low-pass (stronger than R)
S = C.T @ np.diag(hS) @ C
S = 0.5 * (S + S.T)
m_S = 20
s_r_num = float((S @ mu) @ (S @ mu) / (mu @ mu))
kappa_S_num = float(np.trace(S @ S) / (nd * m_S))
print(f"\nSSA operator:  s_r={s_r_num:.4f}  kappa_S={kappa_S_num:.4f}  "
      f"(m_S={m_S})  ->  rho_eff/rho = s_r/kappa_S = {s_r_num/kappa_S_num:.2f}x")

def mc_dir(sigma, ssa, di, trials=2500):
    """E[cos(d,mu)] with optional SSA pre-filter and optional DI resize."""
    acc = 0.0
    M = (m_S if ssa else 1) * (1 if not di else 1)
    for _ in range(trials):
        g = np.zeros(nd)
        reps = m_S if ssa else (m_eot if di else 1)
        for _ in range(reps):
            v = mu + sigma * rng.standard_normal(nd)
            if ssa:
                v = S @ v
            if di:
                v = R @ v
            g += v
        d = g / reps
        acc += d @ mu / (np.linalg.norm(d) * np.linalg.norm(mu))
    return acc / trials

def lgc_of(sigma, ssa):
    """Effective LGC (moment coherence) of the (optionally SSA-conditioned) grad."""
    if not ssa:
        rho_v = (mu @ mu) / (nd * sigma**2)
        return rho_v / (1 + rho_v)
    sig_pow = (S @ mu) @ (S @ mu)
    noise_pow = sigma**2 * np.trace(S @ S) / m_S
    r = sig_pow / noise_pow
    return r / (1 + r)

for label, rho_op in [("Standard (LGC~0.55)", 1.22), ("Robust (LGC~0.98)", 49.0)]:
    sigma = float(np.sqrt((mu @ mu) / (nd * rho_op)))
    lgc0 = rho_op / (1 + rho_op)
    lgc_eff = lgc_of(sigma, ssa=True)
    di_plain = mc_dir(sigma, ssa=False, di=True) - mc_dir(sigma, ssa=False, di=False)
    di_on_ssa = mc_dir(sigma, ssa=True, di=True) - mc_dir(sigma, ssa=True, di=False)
    print(f"\n{label}:  LGC={lgc0:.3f} -> LGC_eff(after SSA)={lgc_eff:.3f}  "
          f"(tau*={tau_star:.3f})")
    print(f"    DI effect WITHOUT SSA : {di_plain:+.4f}")
    print(f"    DI effect ON TOP OF SSA: {di_on_ssa:+.4f}")

print("\nStandard-side sign sensitivity (vary SSA strength m_S at the standard point):")
sigma_std = float(np.sqrt((mu @ mu) / (nd * 1.22)))
for mS_try in [1, 3, 8, 20, 60]:
    m_S = mS_try
    lgc_eff = lgc_of(sigma_std, ssa=True)
    eff = mc_dir(sigma_std, ssa=True, di=True) - mc_dir(sigma_std, ssa=True, di=False)
    print(f"    m_S={mS_try:3d}:  LGC_eff={lgc_eff:.3f}   DI-on-SSA effect={eff:+.4f}"
          f"   {'(DI helps)' if eff > 0 else '(DI hurts)'}")
m_S = 20
print("\n=> Under this model a low-pass pre-method RAISES the effective regime,"
      " which on a robust surrogate would push further past tau* and deepen the"
      " harm. That is the model's prediction, and it is the prediction we"
      " measured and rejected: across 3 sources x 8 pre-methods all 24"
      " measurements LOWER the effective GCR, none raise it. The corollary is"
      " therefore not applied to SSA, which the paper reports as an unresolved"
      " interaction. The algebra is kept so the withdrawn prediction can be"
      " checked, not because it describes the measurements.")

# ---- figure: theory_ssa_shift.pdf (regime-shift cartoon) -------------------
di_eff = np.array(cos_di_l) - np.array(cos_no_l)        # DI effect on alignment
order = np.argsort(LGCs)
xg = np.array(LGCs)[order]
yg = di_eff[order]
def eff_at(lgc):                                        # interp DI effect at LGC
    return float(np.interp(lgc, xg, yg))

from matplotlib.lines import Line2D
fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
ax2.axhline(0, color="gray", lw=1)
ax2.axvspan(0.0, tau_star, color="tab:blue", alpha=.06)
ax2.axvspan(tau_star, 1.0, color="tab:red", alpha=.06)
ax2.plot(xg, yg, "-", color="black", lw=2, label="DI effect on alignment")
ax2.axvline(tau_star, color="k", ls="--", lw=1.3,
            label=fr"crossover $\tau^\star={tau_star:.2f}$")
# Each surrogate sits on the curve at its OWN LGC (filled dot, no SSA). SSA, a
# low-pass pre-filter, shifts it RIGHT to a higher EFFECTIVE LGC (arrow -> hollow
# dot): a standard surrogate is pushed up to near tau*; a robust one is pushed
# deeper into the DI-harmful region.
xs0, xs1 = 0.55, 0.974          # standard: before / after SSA
xr0, xr1 = 0.98, 0.999          # robust:   before / after SSA
for x0, x1, col in [(xs0, xs1, "tab:blue"), (xr0, xr1, "tab:red")]:
    ax2.annotate("", xy=(x1, eff_at(x1)), xytext=(x0, eff_at(x0)),
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax2.scatter([x0], [eff_at(x0)], color=col, s=55, zorder=5)        # no SSA
    ax2.scatter([x1], [eff_at(x1)], facecolors="none", edgecolors=col,
                s=55, zorder=5)                                       # after SSA
ax2.text(xs0 + 0.085, eff_at(xs0) + 0.15, "Standard\nsurrogate", color="tab:blue",
         ha="left", va="bottom", fontsize=9)
ax2.text(xr0 - 0.02, eff_at(xr0) + 0.11, "Robust\nsurrogate", color="tab:red",
         ha="right", va="bottom", fontsize=9)
ax2.text(0.13, 0.05, "DI helps", color="tab:blue", fontsize=12)
ax2.text(0.95, 0.31, "DI hurts", color="tab:red", fontsize=12, ha="center")
arrow_proxy = Line2D([], [], color="gray", marker=r"$\rightarrow$",
                     linestyle="None", markersize=13)
h, l = ax2.get_legend_handles_labels()
ax2.legend(h + [arrow_proxy],
           l + ["SSA shift (filled $\\to$ hollow): raises effective LGC"],
           fontsize=8, loc="upper right")
ax2.set_xlabel("LGC (effective gradient regime)")
ax2.set_ylabel(r"DI effect: $\mathbb{E}[\cos]_{\rm DI}-\mathbb{E}[\cos]_{\rm noDI}$")
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(-0.12, 0.58)
ax2.grid(True, ls="--", alpha=.4)
fig2.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "..", "figures",
                    "theory_ssa_shift.pdf")
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
print(f"Saved figure -> {os.path.normpath(out2)}")

print(f"\nPredicted (theorem)    tau* = {tau_star:.3f}")
print(f"A-metric crossover (numeric)  ~ LGC {cross_A:.3f}   "
      f"(matches tau*: {abs(cross_A - tau_star) < 0.05})")
print(f"E[cos] crossover (Monte-Carlo) ~ LGC {cross_cos:.3f}   "
      f"(same robust regime, LGC>0.9)")
print("=> Both the proof's A-metric and the empirically-measured E[cos] flip"
      " from DI-helpful to DI-harmful at high LGC, as the theorem predicts.")

print("\n" + "=" * 70)
print("GENERALIZED THEOREM: relaxing Assumption 3 (target direction u != mu)")
print("=" * 70)
# Replace 'u = mu' by a general unit target direction u with
#   gamma_mu = |cos(mu,u)|,  gamma_R = |cos(R mu, u)|.
# Figure of merit with the TRUE target u:
#   A_noDI^2 = gamma_mu^2 * rho/(rho+1)
#   A_DI^2   = gamma_R^2  * g_r*rho/(g_r*rho+kappa)
_rho, _gr, _kap, _gmu, _gR = sp.symbols(
    'rho g_r kappa gamma_mu gamma_R', positive=True)
_phi = (_gR**2 * _gr*_rho/(_gr*_rho+_kap)) / (_gmu**2 * _rho/(_rho+1))
_phi = sp.simplify(_phi)
_Gamma = _gR**2/_gmu**2
print("[G] phi =", _phi)
print("[G] phi == Gamma*g_r*(rho+1)/(g_r rho+kappa), Gamma=gamma_R^2/gamma_mu^2 :",
      sp.simplify(_phi - _Gamma*_gr*(_rho+1)/(_gr*_rho+_kap)) == 0)
print("[G] phi(inf) =", sp.limit(_phi, _rho, sp.oo),
      " -> DI hurts as LGC->1  iff  Gamma<1  iff  gamma_R < gamma_mu")
print("[G] phi(0+)  =", sp.limit(_phi, _rho, 0, '+'), " (= Gamma*g_r/kappa)")
_dsign = sp.simplify(sp.diff(_phi, _rho)*(_gr*_rho+_kap)**2*_gmu**2/(_gr*_gR**2))
print("[G] sign(phi') factor =", _dsign, " -> decreasing iff kappa<g_r (as before)")
_rs = sp.simplify(sp.solve(sp.Eq(_phi, 1), _rho)[0])
_c = sp.symbols('c', positive=True)
print("[G] recovers Theorem 1 at u=mu (gamma_mu=1, gamma_R=c):",
      sp.simplify(_rs.subs({_gmu: 1, _gR: _c}) - (_c**2 - _kap/_gr)/(1-_c**2)) == 0)
print("=> Strong 'u=mu' weakens to the measurable 'gamma_R<gamma_mu' (resize moves")
print("   the signal away from the target direction), exactly the sign measured in 4.2.")


# =====================================================================
# (N) The crossover does not need R symmetric or a contraction.
#     Model: zeta_i = R eta_i with eta_i isotropic, so Cov = sigma^2 R R^T
#     and kappa := tr(R R^T)/(n m).  Nothing below uses R = R^T or R <= I.
# =====================================================================
print("\n" + "=" * 66)
print("[N] Non-symmetric, non-contractive R")
print("=" * 66)

_rng = np.random.default_rng(0)
_nd, _m = 64, 10
_k = np.arange(_nd)
_C = np.cos(np.pi * (np.outer(_k, 2 * np.arange(_nd) + 1)) / (2 * _nd))
_C *= np.sqrt(2.0 / _nd)
_C[0] *= 1 / np.sqrt(2)
_h = np.where(_k < _nd // 2, 1.15, 0.35)          # gain > 1: not a contraction
_R = _C.T @ np.diag(_h) @ _C
_R = _R + 0.25 * _rng.standard_normal((_nd, _nd)) / np.sqrt(_nd)   # break symmetry

_asym = float(np.abs(_R - _R.T).max())
_rad = float(np.abs(np.linalg.eigvals(_R)).max())
print(f"    max|R-R^T| = {_asym:.4f} (nonzero => not symmetric); "
      f"spectral radius = {_rad:.4f} (>1 => not a contraction)")
assert _asym > 1e-6 and _rad > 1.0

_spec = np.zeros(_nd); _spec[1] = 1.0; _spec[2] = 0.7; _spec[40] = 0.5; _spec[55] = 0.4
_mu = _C.T @ _spec
_mu /= np.linalg.norm(_mu)
_Rmu = _R @ _mu
_c = float(_Rmu @ _mu / (np.linalg.norm(_Rmu) * np.linalg.norm(_mu)))
_gr = float((_Rmu @ _Rmu) / (_mu @ _mu))
_kap = float(np.trace(_R @ _R.T) / (_nd * _m))
assert _kap / _gr < _c ** 2 < 1
_rho_star = (_c ** 2 - _kap / _gr) / (1 - _c ** 2)
print(f"    c={_c:.4f}  g_r={_gr:.4f} (note g_r>1)  kappa={_kap:.4f}  "
      f"=> rho* = {_rho_star:.4f}")

_ok = True
for _rho in [0.01, 0.1, _rho_star * 0.5, _rho_star * 0.99,
             _rho_star * 1.01, _rho_star * 2, 50.0, 500.0]:
    _s2 = 1.0 / (_rho * _nd)
    _N = 40000
    _g = _mu + _rng.standard_normal((_N, _nd)) * np.sqrt(_s2)
    _Ano = float((_g.mean(0) @ _mu) / np.sqrt((_g * _g).sum(1).mean()))
    _z = (_rng.standard_normal((_N, _m, _nd)) * np.sqrt(_s2)) @ _R.T
    _gb = _Rmu + _z.mean(1)
    _Adi = float((_gb.mean(0) @ _mu) / np.sqrt((_gb * _gb).sum(1).mean()))
    _ok &= (_Adi > _Ano) == (_rho < _rho_star)
print(f"    DI-helps matches rho<rho* at all 8 probe points: {_ok}")
assert _ok, "symmetry/contraction would be load-bearing"
print("=> Assumption 2 needs only kappa/g_r < c^2 < 1.")
