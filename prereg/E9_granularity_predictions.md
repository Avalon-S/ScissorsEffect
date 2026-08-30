# E9 pre-registration: is DI's benefit limited by how many distinct transforms it can draw?

Written 2026-08-05 05:10 CST, before any run with `--scale_grid` exists.

## Why

E8 falsified the transform-strength account: on CIFAR-10, against a target
outside the surrogate pool, all four standard surrogates lose 2-5 pp even at
r=0.97, a near-identity resize. So there is no "mild resize helps, aggressive
resize hurts" crossover to appeal to, and the CIFAR-10 standard blade is flat at
best. The remaining informal explanation, that 32x32 is simply too small, is
untested and in any case predicts the *absence* of a benefit, not a cost.

One concrete mechanism does predict both, and can be tested on ImageNet without
changing dataset, models, or resolution. DI's benefit is variance reduction over
the transforms it averages. The number of distinct scales it can draw is
`S - floor(rS) + 1`:

| input | r=0.9 | distinct scales |
|---|---|---|
| ImageNet 224 | 201 | 24 |
| CIFAR-10 32 | 28 | 5 |

If the benefit needs a rich transform distribution, then coarsening the scale set
on ImageNet to 32x32's granularity should remove most of the standard-side gain,
while leaving the robust side harmed as before.

## Design

Sources: ResNet50 (standard) and Engstrom (robust). Targets: the four
cross-family ones. N=1000, 5 seeds, r=0.9, p=1. Arms: p0 plus n_scales in
{5, 8, 12, 24}. Only the number of distinct resize scales varies; padding
offsets keep their full range, so exactly one variable moves. 24 scales spans
the same integers the unrestricted operator draws from.

## Predictions

**G0, internal check.** D(ns24) for ResNet50 must reproduce the unrestricted
operator's +14.88 [+13.45, +16.27] measured in `results/fig1_imagenet_n1000`
(all-image basis) to within the width of that interval. If it does not, the
arm is not measuring what it claims and nothing else here is interpretable.

**G1, the test.** D(ns24) - D(ns5) for ResNet50, image-paired bootstrap.
- CI excludes zero and the difference is positive -> granularity is part of the
  mechanism, and we report the magnitude.
- CI contains zero -> granularity is NOT the mechanism. We drop it and say the
  CIFAR-10 reversal is observed and unexplained.
- Negative and resolved -> also a falsification, reported as such.

**G2, control.** Engstrom's D stays negative at all four granularities. If
coarsening the scale set makes the robust side stop being harmed, then the arm is
changing more than intended and G1 cannot be read as a benefit-side result.

**G3, shape.** If G1 passes, D should be monotone non-decreasing in n_scales
over {5, 8, 12, 24}. A non-monotone pattern is reported as such rather than
smoothed; it would mean the count of scales is not the operative variable even
if the endpoints differ.

## What we will not do

We will not convert a positive G1 into a claim that resolution explains the
CIFAR-10 result. G1 tests granularity on ImageNet only. Transferring it to
CIFAR-10 would need the reverse experiment (enriching the scale set at 32x32,
which is impossible at r=0.9 since only 5 integers exist between 28 and 32), so
at most we can say the two are consistent, and we will say only that.

We will not report this as a mechanism for the Scissors Effect itself. It
concerns the size of the benefit on the standard side, not the sign difference
between the two source types.
