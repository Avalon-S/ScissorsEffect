# E10 pre-registration: is the CIFAR-10 standard side positive at the conventional p?

Written 2026-08-05, before `results/fig1_cifar10_convp_a/` or `_b/` exist.

## Why

DI's conventional transformation probability is $p=0.5$: it is torchattacks'
default, and our own appendix states it ("DI-FGSM: transformation probability
p=0.5 (default) unless swept"). Our headline CIFAR-10 numbers are reported at
p=1, and the rate sweep at p=0.8.

The Figure 1 p-sweep shows the CIFAR-10 standard source rising to an interior
optimum and falling back, on both targets independently and with resolved
intervals: ResNet-18 to VGG-16 peaks at +6.18 (p=0.5) and is +3.48 at p=1; to
DenseNet-121 it peaks at +1.70 (p=0.4) and is -2.48 at p=1. ImageNet has no such
turn: its standard curve is monotone to p=1.

If that shape holds across standard surrogates, then "we do not claim a two-sided
effect at 32x32", which we posted on 5 Aug, understates what we measured, and the
manuscript would otherwise claim more than the posted reply does.

Only one standard surrogate has been measured this way so far, so the claim is
not yet supportable either way.

## Design

Two runs, each with sources disjoint from targets, so the four genuine CIFAR-10
standard models are all covered as sources without any white-box arm.

- A: sources ResNet-18, ResNet-50 (standard) and Engstrom (robust control);
  targets VGG-16, DenseNet-121, RobustBench `Standard` WRN-28-10.
- B: sources VGG-16, DenseNet-121 (standard) and Rice (robust control);
  targets ResNet-18, ResNet-50, `Standard`.

N=1000, 5 seeds, r=0.9, p grid 0 to 1 in steps of 0.1, ASR over all images.
Target clean accuracies are asserted before use.

## Predictions

**H1.** At p=0.5, averaged over its three targets, each of the four standard
surrogates has D > 0 with an image-paired bootstrap CI excluding zero.

- 4 of 4 -> the effect is two-sided at 32x32 at the conventional p. We edit the
  posted sentence, in the direction of claiming more, and say plainly that we had
  measured only p=1 and p=0.8.
- 2 or 3 of 4 -> report as surrogate-dependent. No post edit; the manuscript
  states which surrogates and which targets.
- 0 or 1 of 4 -> the Figure 1 shape is specific to ResNet-18. No post edit, and
  the manuscript reports it as a single-surrogate observation.

**H2, shape.** For each standard surrogate, D(p=0.5) > D(p=1), image-paired CI
excluding zero. This is the interior-optimum claim and is separate from H1: a
surrogate can be positive at both without a turn.

**H3, control.** Engstrom in run A and Rice in run B stay negative at p=0.5.
If a robust source turns positive at the conventional p, the two-sided reading is
about p rather than about surrogate type, and we report that instead.

**H4, ordering.** Peak height decreases as the target's p=0 ASR rises. Scored as
Spearman over the (target, peak) pairs available across both runs, pooled over
standard sources. This is the only mechanism-flavoured prediction here and it is
exploratory in the sense that it came from three points already seen; it is
registered so that it can fail.

## What we will not do

We will not move the paper's headline from p=1 to p=0.5 on the strength of this.
p=1 stays the headline because it is what the submitted tables use and because
ImageNet is monotone there; p=0.5 is reported alongside as the conventional
setting, whatever it shows.

We will not describe an interior optimum as a recovery of the withdrawn $p^*$
claim. What was withdrawn is that the argmax location is statistically
identifiable. A resolved difference between two grid points is a weaker and
different statement, and the revision will say so in those words.
