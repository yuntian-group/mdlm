# Staged dependence debugging

The purpose is to establish what the joint layer can learn before scaling its
language experiments. No projected results are included.

## Completed mechanism checks

- `tiny/results.json`: 81 exact-distribution runs on AB/BA, AA/BB, and
  independent two-token targets. The original shared scorer cannot fit AB/BA
  with uniform unaries; the restriction is exact, not a failed seed.
- `tiny/rank2_results.json`: 18 runs of the separate-endpoint scorer at half
  rank. With the same 274 active parameters, initialization scale 0.25, and
  100 static warmup updates, it learns both dependent targets in all three
  seeds. No-warmup failures remain in the file.
- `tiny/figure_matched.pdf`: seed 1 fixed before inspecting examples. The
  rightmost panel samples independently from the exact same node marginals.

The proof assumes two tokens, uniform unaries, full vocabulary, and one edge.
It does not say that every shared-factor model has positive association in
every language context.

## Real-backbone checks

The committed launch sequence is `deploy/gcloud/run_staged_debug_v1.sh`.
It first audits serial versus batched backbone outputs, then fits three heads
to 32 fixed corruptions, and evaluates saved factorial checkpoints separately.
The overfit run uses the exact same cached serial hidden states, logits,
candidates, masks, and minibatch order in all arms. The released backbone
computes internally in BF16; storing the outputs in FP32 does not change that.

Shared rank 16 and separate-endpoint rank 8 each train 840,640 parameters.
The candidate-only unary adapter trains 827,312: comparable, not identical.
All preserve the backbone's residual conditional distribution. Therefore,
zero training loss is generally impossible when the target is outside top-K.
`scripts/report_staged_support_floor.py` measures the corresponding lower
bound. A loss plateau above zero is not by itself an optimizer failure.

The data preparation script selects 96 distinct source documents without
model scores: 32 debug-fit, 32 dev, and 32 test. Their source document IDs are
recorded for exclusion from future benchmark claims. The debug-fit subset is
deliberately trained on. This is a memorization diagnostic, not a benchmark.

## Direct dependence controls

`evaluation/dependence_diagnostics.py` separates the joint score into singleton
scores and observed edge-dependence terms. Marginal-preserving shrinkage uses
one development-selected lambda; test observations never choose it.

`evaluation/fixed_group_sampling.py` reveals exactly 1, 2, 4, 8, or 16 tokens
per call using the same precomputed reveal order in all sampling modes. It
records edges whose endpoints are revealed together, preserves context, and
avoids empty calls. This is a distribution/NFE control, not a wall-clock
benchmark: its callback may compute the structured head even in backbone mode.

These tools are separate from the original production decoder. Existing
factorial and generation artifacts remain results for the original scorer.
