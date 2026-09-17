# Separate-endpoint restart recovery

The original DD attempt was preempted before its first checkpoint and then
failed when its existing output directory blocked the retry. A later retry
also exposed a missing jq dependency. The final runner uses Python JSON reads
and explicit recovery checks.

- Lock each run against concurrent writers; keep every attempt separate.
- Select the newest readable full checkpoint with compatible architecture,
  optimizer and scheduler state, falling back if the newest file is corrupt.
- Restart from zero if interrupted before any checkpoint; reject incompatible
  or entirely corrupt checkpoint sets.
- Stop at 7,000 total updates and preserve completed generation outputs.
- Resume rank-8 and rank-16 runs only with their matching architecture.

This is recovery behavior, not evidence that a pre-checkpoint interrupted run
was resumed exactly. Historical server identifiers and storage locations are
omitted. Final evaluation outcomes appear in the experiment notes.
