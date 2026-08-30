# Paper result artifacts

This directory contains the machine-readable records and compact checkpoints
behind the reported experiments.  It is evidence for the public code release;
an anonymous submission archive should omit Git metadata and author-specific
operational notes.

- `g1-heldout/` contains all five post-tuning confirmation seeds (9--13), all
  six ablations, exact enumeration records, training histories, the frozen run
  manifest, and the frozen gate result.
- `structural-oracle/` contains the raw per-condition JSON/CSV records, gate,
  hashes, and privacy-sanitized manifest for the finite-data table-fit oracle.
- `kernel-profile/` contains the CUDA low-rank versus pre-materialized-dense
  inference profile used in the paper.
- `released-backbone-preflight/` contains the strict raw-weight loading and
  length-1024 structured-forward compatibility check.  It is not a language
  quality or latency result.
- `owt-paired-pilot/` is an exploratory three-seed diagnostic.  Its historical
  runs predate cryptographic input/corruption digests, so it is deliberately
  excluded from submission evidence and from the manuscript's measured
  claims.

The frozen real-text confirmation is added only after its input/corruption
digests, checkpoint identity, and paired diagnostics have been validated.
