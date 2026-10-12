# Evaluating saved generations

`evaluation.py` scores saved samples; it does not generate, regenerate, select,
or reject them. The default external evaluator is **GPT-2 Large**, not the
generator's own likelihood. This is generation perplexity (also called reverse
PPL), **not held-out data perplexity under the diffusion model**.

## Scoring convention

The scoring path follows the GPT-2 branch of
[BD3-LM's `metrics.py`](https://github.com/kuleshov-group/bd3lms/blob/1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7/metrics.py),
from *Block Diffusion* (Arriola et al.), under the repository's Apache-2.0
license. It retains the source's FP32 next-token cross entropy, one-sequence
scoring batches, 1,024-token evaluator context, and stride of 512. The adapted
implementation replaces stateful TorchMetrics accumulation with explicit
per-sample records and FP64 pooled summation.

1. Decode the saved generator token IDs with `skip_special_tokens=False`.
   Keep the initial context token and internal EOS tokens; do not truncate the
   string at the first EOS or join articles by deleting their boundary tokens.
2. Retokenize those strings using the evaluator tokenizer, with right padding,
   EOS as padding, and an explicit maximum length (default 1,024).
3. Score token positions after position zero. EOS token IDs have zero target
   weight, including padding. They are **still present in the input context**.
   The attention mask distinguishes real EOS from padding.
4. For lengths above 1,024, slide the 1,024-token context by 512 and score only
   newly reached targets. No target is counted twice. Setting `--max-length
   2048` permits more retokenized targets; it does not give GPT-2 a 2,048-token
   context or create a longer generation. Generation length remains whatever
   was saved. The default truncation can discard retokenized suffixes.

The output reports both:

- `pooled_token_ppl = exp(sum(token NLL) / sum(scored target counts))`.
  FP32 token losses are summed in FP64, then pooled across the declared sample
  population (accepted by default; every raw attempt in raw mode).
- `legacy_mean_sequence_ppl`: arithmetic mean of per-sequence perplexities.
  Each sequence PPL uses the upstream FP32 reduction/exponentiation. This is
  generally different from pooled PPL, even at the same generated length,
  because retokenization and EOS masking can change the scored target counts.

All-EOS samples have no scored targets. The default accepted-population path
raises an explicit error; raw-population evaluation records their undefined
scores without replacing or omitting those attempts (see below).
The API accepts an already loaded evaluator; the CLI uses
FP32 weights and logits. Changing precision, evaluator/tokenizer revision,
decoding, EOS handling, context, or sample population changes the evaluation.
CPU metric tests alone do not establish real-checkpoint numerical equivalence
on a GPU.

## Saved-sample interface

```json
{
  "schema": "additive_chain_saved_generations_v1",
  "eos_token_id": 50256,
  "samples_requested": 2,
  "generation_timing_complete": true,
  "setup_seconds": null,
  "warmup_seconds": null,
  "attempts": [
    {"attempt_id": 0, "accepted": true,
     "token_ids": [50256, 15496, 995, 0],
     "new_tokens": 3, "generation_seconds": 1.2},
    {"attempt_id": 1, "accepted": false,
     "token_ids": [50256, 0, 0, 0],
     "new_tokens": 3, "generation_seconds": 1.1},
    {"attempt_id": 2, "accepted": true,
     "token_ids": [50256, 1212, 318, 257, 1332],
     "new_tokens": 4, "generation_seconds": 1.3}
  ]
}
```

This short example demonstrates the format, not the 1,024-token experimental
population or its acceptance filter. Preserve **every raw attempt**, in order,
including rejections. Each attempt is one sequence. `token_ids` includes its
initial supplied context and all generated tokens, also for rejected attempts.
`new_tokens` excludes supplied context/BOS; omit it if unknown rather than
inferring a length. Optional `text` is checked against retained-special-token
decoding. An incomplete accepted population is reported as `partial`, not as a
finished evaluation. Zero accepted samples yield no quality score.

### Fixed raw populations

For a fixed-size, unfiltered comparison, generate with `population="raw"` or
the generation CLI's `--population raw`. Its schema is
`additive_chain_saved_raw_generations_v1`; `samples_requested` means raw
attempts, not accepted outputs. The scorer then evaluates **every** attempt,
including those whose saved `accepted` flag is false. The flag remains a
diagnostic: scoring does not repeat generation or fill rejected slots. A null
`text` for a rejected attempt is reconstructed from its saved IDs, retaining EOS.

The result uses `additive_chain_raw_generation_metrics_v1`. `quality` describes
all raw attempts, and `quality_subsets.accepted` / `.rejected` describe the two
subsets of that same population. All three report `samples`, `defined` and
`undefined`. A sequence with no non-EOS next-token targets keeps its attempt
ID, status `undefined`, zero scored targets and null sequence PPL. If any member
is undefined, that population's PPL fields are null; any calculable score is
explicitly labeled `defined_subset`, never silently called the full score.
An empty subset also has null PPL. `quality_population="complete"` means all
requested attempts are present, not that every score is defined.

Raw populations cannot exceed their declared size. A shorter saved prefix is
reported as partial. Do not change an accepted-run schema to raw while leaving
its requested accepted count unchanged: explicitly choose the raw prefix and
declare its size when converting historical results. Acceptance, source IDs
and elapsed work must remain traceable.

## Timing and diagnostics

`generation_seconds` must measure the complete attempt, synchronized before
and after on a GPU: backbone, candidate and pair preparation, inference,
sampling, decoding, and acceptance checks. Include rejected attempts. Exclude
model loading, initial setup, warmup, external quality scoring, and measured
durable-file I/O; report those exclusions when using the result. Do not sum
overlapping component timers into an end-to-end time.

`seconds_per_accepted_sequence` is the sum of **all** attempt durations divided
by accepted sequences. Throughput uses actual newly generated token counts,
not the requested length. Missing elapsed values or an unmeasured hard-kill
tail require `generation_timing_complete=false`; the module then omits rates.
Known interrupted work should remain in the reported experiment accounting;
this compact interchange format cannot reconstruct interruptions from files.

Setup and warmup are separate optional totals over all invocations, including
resumes. If either is unknown, setup/warmup-amortized latency is null, not the
steady-state value relabeled as an amortized number. Loading is still excluded
from that secondary measure.

Diagnostics use raw generator token IDs, before decoding or scoring:

- empirical token entropy in nats, including initial context and EOS;
- fraction of distinct tokens;
- repeated-4gram fraction: `1 - unique_4grams / total_4grams` (zero below four
  tokens), counting overlapping 4grams;
- EOS count and per-attempt last-256-token empirical entropy.

Raw, accepted and rejected diagnostic means are separate. The original
fixed-length sampler rejected a last-256 entropy below four nats; evaluation
reports the FP32 statistic but **does not reapply the filter**. Predictive model
entropy is a different quantity. These diagnostics help detect repetition;
low generation PPL by itself does not establish coherent or diverse text.

## Use and dependencies

```python
from additive_chain.evaluation import aggregate_scores, summarize_attempts

# Scores returned by score_token_batch/evaluate_saved:
quality = aggregate_scores(score_rows)
timing = summarize_attempts(attempts, eos_token_id=50256,
    samples_requested=64, timing_complete=True)
```

The CLI requires PyTorch and Hugging Face `transformers` (plus the tokenizer
and model dependencies for the installed version). It only uses already cached
or local model assets: `local_files_only=True`, no remote code, no downloads.
The defaults pin GPT-2 Large revision
`32b71b12589c2f8d625668d2335a01cac3249519` and the generator's GPT-2 tokenizer
revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

```bash
python -m additive_chain.evaluation samples.json --output metrics.json --device cuda
```

For custom assets, explicitly supply `--scorer`, `--scorer-revision`,
`--generator-tokenizer`, and `--tokenizer-revision`. The output records these
choices, input/module SHA256, device, and PyTorch version. These are useful
provenance, not cryptographic authentication of arbitrary local model weights.
Pin and verify model/tokenizer files separately for a strict reproduction.
The output path must be new. Quality-scoring time is separate from saved
generation time.

```bash
python -m pytest -q tests/test_additive_chain_evaluation.py
```

These tests run on CPU with a deterministic mock evaluator and tokenizer.
They compare the original sliding-window arithmetic, EOS/context and padding
behavior, aggregation, rejection-aware timing, and CLI load flags. They do not
download checkpoints, run real GPT-2 Large, or claim GPU scoring results.
Raw-mode tests also check fixed population sizes, scoring of rejected attempts,
explicit undefined rows, accepted/rejected subsets and unchanged arithmetic
for samples with defined scores.
