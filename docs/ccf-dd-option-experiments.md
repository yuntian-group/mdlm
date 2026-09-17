# Two independent dynamic/dynamic screens

Both experiments use the usual `mdlm` environment, a frozen released MDLM
backbone, shared rank-16 factors, seed 1, length 1024, batch size 4, and the
original 3e-4 learning rate after 50 warmup steps. They train fresh to 7000
updates, validate/save every 500, export the step-7000 adapter, and generate
exactly one sample at seed 91001 with 1000 transitions plus final cleanup.
The sample is scored by the pinned GPT-2-large reference model.

| Script | FiLM hidden width | Candidate top-K |
| --- | --- | --- |
| `scripts/train_ccf_dd_film_7k.sh` | 128 | 128 |
| `scripts/train_ccf_dd_topk256_7k.sh` | 0 (original affine FiLM) | 256 |

They are separate Slurm jobs and can execute concurrently when Slurm grants
resources. Each requests one GPU, four CPUs, and 30 GB host memory. They do
not inspect, stop, or modify other users' processes. Incompatible/problematic
nodes `cluster-node`, `cluster-node`, `cluster-node`, and `cluster-node` are excluded.

The opt-in model setting is:

```text
++model.structured_decoder.factor_conditioner_hidden_dim=128
```

The default, zero, retains the old parameter names, initialization, and adapter
identity hashes. A positive width replaces the additive hidden/time linear
projections with `[LayerNorm(h), time_embedding] -> Linear(width) -> SiLU ->
Linear(2 * rank)`. Its output still supplies FiLM shift and scale, with the
same positive scale and softplus factor transform. Width 128 adds 84,096
parameters to the shared rank-16 production head. This option alone does
not remove the shared-token-factor ordering restriction.

The setting is authenticated in new adapter manifests; loading with the wrong
width is rejected. Both shared and separate endpoint tables support it, but
these two jobs deliberately use shared tables. Top-K 256 needs no additional
architecture option: set `model.structured_decoder.top_k=256` for training
and generation. It increases explicit candidate support and associated work,
not the number of learned parameters.

Submission uses an isolated code snapshot rather than modifying the working
repository used by existing jobs:

```bash
cd /path/to/isolated/snapshot
sbatch --export=ALL,CCF_CODE_ROOT="$PWD" scripts/train_ccf_dd_film_7k.sh
sbatch --export=ALL,CCF_CODE_ROOT="$PWD" scripts/train_ccf_dd_topk256_7k.sh
```

Outputs are under `tree_mdlm_cache/runs/ccf_dd_<variant>_s001_step7000_job<ID>`.
Read `generation/samples.jsonl` for the text and per-sample `reference_lm`
scores, `generation/summary.json` for aggregate metrics, and
`training/checkpoints` for intermediate checkpoints. Existing output trees
are never overwritten; an interrupted run needs an explicit resume workflow.
One generated sample is a sanity check, not evidence of a reliable improvement.

Historical results and sample counts: `docs/experiments/03-endpoints-and-conditioners.md`.
