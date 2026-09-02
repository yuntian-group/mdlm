# Replicated K=128 topology-by-factor factorial

This directory records the complete `causal_primary` grid: four structured
adapter arms, three training seeds, four pinned corpora, four mask rates, three
paired corruption seeds, and 256 validation batches per evaluation job.  The
compiled plan contains 12 training jobs, 12 exports, and 576 evaluations.

The primary static/static-minus-dynamic/dynamic conditional-NLL contrast is
0.00856781 with a 95% train-seed/document hierarchical bootstrap interval of
[0.00760308, 0.00927484].  Dynamic factors improve NLL under both topology
settings.  Dynamic topology does not: fixed/dynamic-minus-dynamic/dynamic is
-0.00296005 [-0.00418204, -0.00205660], while static/static-minus-
dynamic/fixed is 0.00006114 [-0.00003198, 0.00013836].  All ten prespecified
contrasts are retained in `analysis.json`, including adverse and null results.

The analysis validates all 600 job markers and 470,736 conditional records.
Pairing, finite-value, no-edge identity, support monotonicity, nonempty-
topology, and degree/component-preserving permutation gates pass.  The exact
clean source revision is `a574aca873d6de66a3b847c13efd1c7bc4efb66b` and the
compiled-plan SHA-256 is
`b75f5f213a3fc51247e1d96042710a0d1e01f7ae7049ccfd2dc26111afdfc7a9`.
The canonical analysis identity is
`4023c56860fd0d33df43c4aedb7787a715d4ba3c30f650e39890a1d85d735d82`;
the serialized `analysis.json` file SHA-256 is
`ca027caea9ab99a31ae09ce51dda08c1bc0710cda9e386642193dddb468dff7d`.

The grid finished across partition disks on A100 and L4 workers.  Because some
jobs were independently rerun and their committed bytes were not identical,
the original conflict-rejecting consolidator first stopped.  `selection.json`
then records a frozen base-first, ordered-source choice for every job and every
candidate marker hash; `consolidation.json` records the resulting one-source-
per-job 600-job tree.  Their file SHA-256 values are
`4b0ccdb84b1bc7d9e4aff542c2f82bb291724eb538820ac40ba921133f790658`
and `6fe0051c23999727a3c8679265dfb06b14fda187660126532d1cef48fdff8c9b`.
