# E8 pre-registration: why does aggressive resize reverse the standard side?

Written 2026-08-05 02:00 CST, while `results/cifar10_correlation_rates/` is still
empty (job 98 is on surrogate 7 of 13; that script writes nothing until all 13
finish). No rate other than r=0.6 has been observed with a disjoint target.

## The observation to be explained

With the target changed from Engstrom (which is itself in the surrogate pool, so
that row was white-box) to RobustBench's naturally trained WRN-28-10 (outside the
pool), the CIFAR-10 standard side reverses at r=0.6:

| pool | standard mean | robust mean | Pearson(LGC, D) |
|---|---|---|---|
| target Engstrom, r=0.6 | +1.42 | -2.81 | -0.669 (p=0.012) |
| target Standard, r=0.6 | -6.61 | -7.96 | -0.345 (p=0.249) |

Two explanations are on the table and they make different predictions.

**H1, transform strength.** DI's benefit is variance reduction on a noisy
gradient; its cost is the distortion the transform itself introduces. r=0.6 on a
32x32 image shrinks it to 19x19 and zero-pads back, leaving 35% of the canvas as
signal, so the distortion term dominates for every surrogate. Under H1 the
standard side is positive at mild r and crosses to negative as r falls; the
scissors is a moderate-r phenomenon.

**H2, target headroom.** Against this target the standard surrogates already
succeed on 77-93% of images, so there is no room for a gain to appear, and any
transform cost shows up as a loss. Under H2 the standard side is negative at
every r, including near-identity ones, and the earlier positive numbers came from
a target on which standard surrogates scored only 12.6-13.3%.

## Predictions, and what falsifies what

Rates run: 0.97, 0.95, 0.9, 0.8, 0.7, 0.6. Standard surrogates: DenseNet121
(LGC 0.8994), ResNet50 (0.9189), ResNet18 (0.9355), VGG16 (0.9414).

**P1 (discriminates H1 from H2).** Sign of D for the four standard surrogates at
r=0.97 and r=0.95.
- At least 3 of 4 positive at r>=0.95, and at least 3 of 4 negative at r=0.6
  -> supports H1.
- 3 or more of 4 negative at r=0.97 -> supports H2 and falsifies H1.
- Anything else -> neither is supported; report the sweep and claim no mechanism.

**P2 (tests H1 specifically, and cannot be rescued by tuning).** If H1 holds,
a noisier gradient should tolerate more distortion before the cost wins, so the
crossover rate r* should be ordered by LGC: DenseNet121 < ResNet50 < ResNet18 <
VGG16. Scored as Spearman(LGC, r*) over the four standard surrogates.
- rho >= +0.8 (i.e. at most one adjacent swap) -> P2 passes.
- Anything lower -> P2 fails, and we report the reversal as observed but not
  explained by gradient noise. LGC values are already fixed and r* comes from the
  data, so there is no free parameter here.

**P3 (the operational question, independent of H1/H2).** Pearson(LGC, D) over all
13 surrogates, per rate.
- Significant and negative (p<0.05) at r=0.9 -> the CIFAR-10 correlation survives
  at the paper's default operator and is reported with that scope.
- Not significant at r=0.9 -> the CIFAR-10 LGC correlation is withdrawn and the
  LGC evidence rests on ImageNet alone.

## What we will not do

We will not choose a rate after seeing the results and present it as the setting
the paper always meant. r=0.9 is the operator used throughout the submitted paper
and is the one P3 is scored on; r=0.6 is the aggressive setting the scope
discussion uses. Both are reported whatever they show.

The caveat that survives either way: D in percentage points is not comparable
across baselines that differ by 60 points (77-93% standard vs 21-29% robust), so
the standard-vs-robust gap on CIFAR-10 is reported as a direction, not as a
magnitude to be compared against ImageNet's.
