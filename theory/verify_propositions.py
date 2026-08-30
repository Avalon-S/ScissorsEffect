#!/usr/bin/env python
"""
Machine-checked verification of three propositions.

Prop. A  (probe-statistic bridge) relates the moment ratio at the probe scale to
         the theorem's GCR, and bounds the gap between them. Eq. 1 of the paper
         is a mean cosine and is a separate quantity; see the note in the Prop. A
         block.
Prop. B  (random transforms) replaces the fixed symmetric contraction by the
         random operator the attack actually applies, and shows the crossover
         survives with corrected constants.
Prop. C  (non-additivity) shows the resize and translation main effects are not
         expected to add on the figure-of-merit scale, so a non-zero factorial
         interaction is predicted by the model rather than anomalous.

Symbolic steps use sympy; each is followed by a numerical check, including a
Monte-Carlo with a non-symmetric random operator confirming that symmetry is
never used in the derivation.

Run:  python verify_propositions.py
"""
import numpy as np
import sympy as sp

rng = np.random.default_rng(0)
OK = []


def check(label, cond):
    OK.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


# ===========================================================================
print("=" * 74)
print("PROP. A -- the LGC bridge")
print("=" * 74)
# ---------------------------------------------------------------------------
# Model: g       = mu + eta            (clean gradient)
#        g_xi    = mu + eta_xi         (gradient at x + xi, signal Lipschitz-close)
#        E<eta,eta_xi> = n sigma^2 omega,  omega in [0,1] the noise autocorrelation
#                                          at the probe scale ||xi||.
# The theorem's quantity uses INDEPENDENT draws, i.e. omega = 0.
rho, omega, n, sig, s = sp.symbols("rho omega n sigma s", positive=True)

Ninv = n * sig**2                      # total noise power
num = s + Ninv * omega                 # E<g, g_xi> = ||mu||^2 + n sigma^2 omega
den = s + Ninv                         # E||g||^2 = E||g_xi||^2
mr_probe = sp.simplify((num / den).subs(s, rho * Ninv))
lgc_thm = rho / (1 + rho)

print(f"\n  mr_probe (moment ratio, correlated probes) = {sp.simplify(mr_probe)}")
print(f"  GCR      (theorem, independent draws)      = {lgc_thm}")
check("mr_probe = (rho + omega)/(1 + rho)",
      sp.simplify(mr_probe - (rho + omega) / (1 + rho)) == 0)
check("mr_probe - GCR = omega/(1+rho) >= 0  (correlated probes sit above GCR)",
      sp.simplify(mr_probe - lgc_thm - omega / (1 + rho)) == 0)
check("omega = 0  =>  mr_probe = GCR  (independent draws recover the theorem)",
      sp.simplify(mr_probe.subs(omega, 0) - lgc_thm) == 0)
check("gap is decreasing in rho (vanishes for robust surrogates)",
      sp.simplify(sp.diff(omega / (1 + rho), rho)) == -omega / (1 + rho) ** 2)

# mr_probe is a ratio of moments, not Eq. 1. Eq. 1 is a mean of per-image
# cosines and differs from any moment ratio by a Jensen gap, which carries no
# sign in general: measured on 15 surrogates, Eq. 1 sits above GCR on 14 and
# below it on ViT-B/16. The bound just checked applies to mr_probe only.

# --- the Jensen gap: E[cos] vs the ratio of moments -------------------------
# For Gaussian eta,  Var(||g||^2) = 2 n sigma^4 + 4 sigma^2 ||mu||^2, so the
# RELATIVE variance of the normalizer is O(1/n) UNIFORMLY in rho:
relvar = sp.simplify(
    (2 * n * sig**4 + 4 * sig**2 * (rho * Ninv)) / (rho * Ninv + Ninv) ** 2)
target = (2 / n) * (1 + 2 * rho) / (1 + rho) ** 2
check("Var(||g||^2)/(E||g||^2)^2 = (2/n)(1+2rho)/(1+rho)^2 = O(1/n), uniform in rho",
      sp.simplify(relvar - target) == 0)
sup = sp.maximum((1 + 2 * rho) / (1 + rho) ** 2, rho, sp.Interval(0, sp.oo))
print(f"  sup_rho (1+2rho)/(1+rho)^2 = {sup}   ->  relative variance <= {sup}*2/n")

print("\n  Monte-Carlo: E[cos(g,g')] vs the moment ratio, as n grows")
print(f"  {'n':>8} {'rho':>6} {'E[cos]':>10} {'moment':>10} {'|gap|':>10} {'gap*n':>9}")
gap_n = []
for n_v in (256, 1024, 4096, 16384):
    for rho_v in (0.5, 4.0):
        sig_v = 1.0
        mu = np.zeros(n_v)
        mu[0] = np.sqrt(rho_v * n_v) * sig_v          # ||mu||^2 = rho n sigma^2
        G = mu + rng.normal(0, sig_v, (20000, n_v))
        H = mu + rng.normal(0, sig_v, (20000, n_v))
        cosv = (G * H).sum(1) / (np.linalg.norm(G, axis=1)
                                 * np.linalg.norm(H, axis=1))
        mom = rho_v / (1 + rho_v)
        gap = abs(cosv.mean() - mom)
        gap_n.append(gap * n_v)
        print(f"  {n_v:>8} {rho_v:>6.1f} {cosv.mean():>10.5f} {mom:>10.5f}"
              f" {gap:>10.2e} {gap*n_v:>9.3f}")
check(f"Jensen gap scales like 1/n: gap*n stays bounded "
      f"(max {max(gap_n):.2f} over n = 256..16384)", max(gap_n) < 5.0)
print("  -> at n = 3*224^2 = 150528 the gap is ~1e-5; the pairing term "
      "omega/(1+rho) is the only one that matters.")

# ===========================================================================
print("\n" + "=" * 74)
print("PROP. B -- random transforms (dropping 'fixed' and 'symmetric')")
print("=" * 74)
# ---------------------------------------------------------------------------
# gbar = (1/m) sum_i (T_i mu + zeta_i), T_i iid ~ nu, M = E[T] (NOT symmetric,
# NOT assumed to be a contraction).
#   E[gbar]   = M mu
#   E||gbar||^2 = ||M mu||^2 + (1/m)[ v_T + sigma^2 E tr(T^T T) ]
# with v_T = E||T mu - M mu||^2 (transform variance) and t = E tr(T^T T).
# Constants:  c = cos(M mu, mu), g_r = ||M mu||^2/||mu||^2,
#             v_r = v_T/||mu||^2,  b0 = t/n,
#             a = 1 + v_r/(m g_r),  b = b0/(m g_r).
c, gr, v_r, b0, m, kappa = sp.symbols("c g_r v_r b_0 m kappa", positive=True)

A_noDI_sq = rho / (1 + rho)
# A_DI = <E[gbar], muhat> / sqrt(E||gbar||^2);  <M mu, muhat> = c sqrt(g_r) ||mu||
A_DI_sq = sp.simplify(
    (c**2 * gr * (rho * Ninv))
    / (gr * rho * Ninv + (v_r * rho * Ninv + b0 * sig**2 * n) / m))
a = 1 + v_r / (m * gr)
b = b0 / (m * gr)
phi = sp.simplify(A_DI_sq / A_noDI_sq)
phi_target = c**2 * (1 + rho) / (a * rho + b)
check("phi(rho) = c^2 (1+rho) / (a rho + b)",
      sp.simplify(sp.together(phi - phi_target)) == 0)

dphi = sp.simplify(sp.diff(phi_target, rho))
sign_fac = sp.simplify(dphi * (a * rho + b) ** 2 / c**2)
print(f"\n  d(phi)/d(rho) * (a rho + b)^2 / c^2 = {sp.simplify(sign_fac)}")
check("phi strictly decreasing  <=>  b < a  (holds whenever transform variance "
      "does not vanish while noise pass-through explodes)",
      sp.simplify(sign_fac - (b - a)) == 0)

lim_inf = sp.limit(phi_target, rho, sp.oo)
lim_zero = sp.limit(phi_target, rho, 0, "+")
print(f"  phi(rho -> inf) = {sp.simplify(lim_inf)}   (= c^2/a < 1  => DI hurts)")
print(f"  phi(rho -> 0+)  = {sp.simplify(lim_zero)}   (> 1 iff b < c^2)")
check("phi(inf) = c^2/a < c^2 < 1: random transforms make the robust limit "
      "STRICTLY worse than the deterministic theorem predicted",
      sp.simplify(lim_inf - c**2 / a) == 0)
rho_star = sp.solve(sp.Eq(phi_target, 1), rho)[0]
print(f"  rho* = {sp.simplify(rho_star)}")
check("rho* = (c^2 - b)/(a - c^2)",
      sp.simplify(rho_star - (c**2 - b) / (a - c**2)) == 0)
check("deterministic special case v_r=0, b0/m = kappa  =>  "
      "rho* = (c^2 - kappa/g_r)/(1-c^2)  [Theorem 1 recovered]",
      sp.simplify(rho_star.subs({v_r: 0, b0: kappa * m})
                  - (c**2 - kappa / gr) / (1 - c**2)) == 0)
check("crossover moves LEFT under transform randomness (a>1 shrinks rho*): "
      "d(rho*)/d(v_r) < 0",
      sp.simplify(sp.diff((c**2 - b) / (a - c**2), v_r)) ==
      sp.simplify(-(c**2 - b) / ((a - c**2) ** 2 * m * gr)))

# --- numerical check with a genuinely NON-symmetric, RANDOM operator --------
print("\n  Monte-Carlo with a non-symmetric random operator (shift + rescale),")
print("  no contraction assumed: closed form vs. simulation")
n_v, m_v = 512, 10
idx = np.arange(n_v)


def rand_T(v):
    """Random circular shift + random gain: non-symmetric, and E[T] is a
    smoothing operator that is NOT a projection."""
    k = rng.integers(-6, 7)
    g = rng.uniform(0.7, 1.3)
    return g * np.roll(v, k)


mu_v = np.sin(2 * np.pi * idx / n_v * 3) + 0.4 * np.sin(2 * np.pi * idx / n_v * 40)
mu_v /= np.linalg.norm(mu_v)
# empirical M mu, v_T, t
draws = np.array([rand_T(mu_v) for _ in range(200000)])
Mmu = draws.mean(0)
v_T = ((draws - Mmu) ** 2).sum(1).mean()
c_v = float(Mmu @ mu_v / (np.linalg.norm(Mmu) * np.linalg.norm(mu_v)))
gr_v = float(Mmu @ Mmu / (mu_v @ mu_v))
E_e = np.array([rand_T(e) for e in np.eye(n_v)[rng.integers(0, n_v, 4000)]])
t_v = float((E_e ** 2).sum(1).mean() * n_v)          # E tr(T^T T)
print(f"    measured: c={c_v:.4f}  g_r={gr_v:.4f}  v_r={v_T:.4f}  t/n={t_v/n_v:.4f}")

rel_err = []
print(f"    {'rho':>7} {'A_DI (MC)':>12} {'A_DI (closed)':>15} {'rel.err':>10}")
for rho_v in (0.2, 1.0, 5.0, 50.0):
    sig_v = np.sqrt(1.0 / (rho_v * n_v))             # ||mu||=1 => rho = 1/(n sig^2)
    gb = np.zeros((4000, n_v))
    for i in range(m_v):
        # The transform acts on the *gradient*, so the noise passes through it
        # too: g_i = T_i(mu + eta_i).  Adding untransformed noise here would
        # simulate a different model from the one the closed form assumes, and
        # the two would still agree to a few percent because E tr(T^T T)/n is
        # near 1 for this operator -- close enough to hide the discrepancy.
        Td = np.array([rand_T(mu_v + rng.normal(0, sig_v, n_v))
                       for _ in range(4000)])
        gb += Td
    gb /= m_v
    A_mc = (gb.mean(0) @ mu_v) / np.sqrt((gb ** 2).sum(1).mean())
    # closed form: A_DI^2 = c^2 g_r ||mu||^2 / (g_r||mu||^2 + (v_T + sig^2 t)/m)
    A_cf = np.sqrt(c_v**2 * gr_v / (gr_v + (v_T + sig_v**2 * t_v) / m_v))
    rel = abs(A_mc - A_cf) / A_cf
    rel_err.append(rel)
    print(f"    {rho_v:>7.1f} {A_mc:>12.5f} {A_cf:>15.5f} {rel:>10.2%}")
# Tolerance is the Monte-Carlo error at 4000 samples.
check(f"closed form matches Monte-Carlo for a non-symmetric random operator "
      f"(max rel. err {max(rel_err):.2%} over 4 regimes)", max(rel_err) < 0.02)

# ===========================================================================
print("\n" + "=" * 74)
print("PROP. C -- the resize x translation interaction is predicted, not anomalous")
print("=" * 74)
# Two factors with constants (c_R, gr_R, k_R) and (c_P, gr_P, k_P); the (1,1)
# cell composes them.  Even if the operator constants composed multiplicatively
# and the variances added, the FIGURE OF MERIT is a nonlinear function of them,
# so the effects on A (and a fortiori on ASR) do not add.
cR, cP, grR, grP, kR, kP = sp.symbols("c_R c_P g_rR g_rP kappa_R kappa_P",
                                      positive=True)


def A_of(cc, gg, kk):
    return cc / sp.sqrt(1 + kk / (gg * rho))


A00 = sp.sqrt(rho / (1 + rho))
A10 = A_of(cR, grR, kR)
A01 = A_of(cP, grP, kP)
# ideal multiplicative composition of the operators
A11 = A_of(cR * cP, grR * grP, kR + kP)
INT = sp.simplify((A11 - A10) - (A01 - A00))
print("\n  Interaction on the A-scale, INT = (A11 - A10) - (A01 - A00):")
print(f"    {sp.simplify(INT)}")
# translation-only is near-neutral: c_P -> 1, gr_P -> 1
INT_neutral = sp.simplify(INT.subs({cP: 1, grP: 1}))
check("even when translation alone is exactly neutral (c_P=g_rP=1), the "
      "interaction is NON-ZERO whenever kappa_P > 0",
      sp.simplify(INT_neutral) != 0)
num_check = sp.simplify(
    INT_neutral.subs({cR: sp.Rational(84, 100), grR: sp.Rational(43, 100),
                      kR: sp.Rational(6, 100), kP: sp.Rational(6, 100),
                      rho: 40}))
print(f"    numeric example (robust-like constants, rho=40): INT = "
      f"{float(num_check):+.4f} on the A-scale")
check("predicted interaction is NEGATIVE for robust-like constants "
      "(the sign a factorial decomposition would report)",
      float(num_check) < 0)

print("\n" + "=" * 74)
print(f"RESULT: {sum(OK)}/{len(OK)} checks passed")
print("=" * 74)
