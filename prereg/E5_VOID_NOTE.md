# E5_predictions_<first timestamp>.json is VOID

Reason: the stage-1 measurement it was derived from computed GCR_eff using the
variance across *method draws only*. For `method="none"` the transform is the
identity, so all m draws coincide, the variance is exactly zero, and
GCR_eff(none) = 1.0000 by construction (observed on all three surrogates).
Against that baseline every random method necessarily appears to shift the
surrogate leftward, so checks (i), (ii) and (v) were guaranteed to fail
regardless of what SSA does. The file therefore tests an artefact, not
Corollary 2.

The theorem's rho is an input-noise signal-to-noise ratio. The corrected
measurement (`48_E5_measure_v2`) perturbs the input on every draw with the same
probe scale E1 uses (1/255), so `none` reduces to E1's rho and all method arms
share one noise source.

Integrity: the void file is retained, not deleted, with its original hash. The
defect was found and the correction queued while stage 2 was still running and
BEFORE any stage-2 attack outcome had been inspected.
