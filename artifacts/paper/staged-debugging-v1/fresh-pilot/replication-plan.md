# Unchanged-seed replication, 2026-09-08

The completed seed-1 reserved-test comparison shows useful dependence for the
separate-factor head (+0.002837 nats per masked token, paired document 95% CI
[0.001925, 0.003746]), but no overall advantage over the trained unary adapter
(-0.000129, [-0.002618, 0.002322]). At 90% masking it improves on the unary
adapter by 0.008257 [0.004239, 0.012153]; the unary adapter wins at 25% and
50%. These are the pilot results, not outcomes for additional seeds.

Replicate the entire experiment at training seeds 2 and 3. Keep all four mask
rates, all three arms, the 2,048/64/128 source-document split, 1,000 updates,
learning rate 0.0003, and the original development selection rule unchanged.
Do not restrict evaluation to 90% masking or select a checkpoint by dependence
gain. Each seed has fresh training order and masks; masks stay paired across
its arms. Report every seed and mask rate, whether it agrees with seed 1 or not.

This replication was chosen after reading the seed-1 test result. The original
test artifact and its seal remain immutable. New seeds select checkpoints on
development data only. A separate aggregate seal and test export can re-score
the three fixed seeds without changing any seed-1 selection, then use the
implemented crossed seed/document bootstrap. This is replication on the same
reserved documents, not a new independent test corpus or a new tuning search.

The existing L4 VM and dedicated experiment disk are reused. The original
absolute shutdown deadline, 2026-09-08 04:49:24 UTC, is unchanged. A bounded
sequential queue runs seeds 2 and 3; an interrupted or failed run is retained
and cannot pass the complete-run selection sealer.

The first queue was preempted at 03:01:44 UTC during seed 2's initial
development evaluation, before any training updates were saved. Its
step-zero checkpoint and partial outputs are retained. The retry uses fresh
`retry1-v1` output directories and the same initialization and settings;
this is a restart from initialization, not an optimizer resume. The original
absolute shutdown deadline is restored after VM restart.
