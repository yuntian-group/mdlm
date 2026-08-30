# Paper and Experiment Strategy

Status: planning draft, 2026-08-30. This document is the contract between the
paper claims and the experiments. It should be revised whenever evidence
contradicts the current story.

## 1. Decision: pivot from a fixed chain CRF to contextual coupling forests

The old idea correctly identifies a real problem: a masked diffusion language
model (MDLM) predicts clean tokens with a product of marginals, even when it
commits several mutually dependent tokens in one denoising step. However,
"replace factorized predictions with a tractable joint model" is no longer a
novel claim. Discrete Copula Diffusion, EDLM, ADJUST, CoDD, FeF-DLLM, and
CoCommit all address the same marginal--joint gap. CoDD is especially close: it
multiplies the backbone's contextual unaries by a tractable, static structural
prior and samples jointly.

A fixed context-dependent chain is also too incremental. Fast Structured
Decoding (Sun et al., 2019) already combines globally normalized chain CRFs,
top-K candidate truncation, low-rank/context-dependent transitions, and exact
dynamic programming; Cascaded Text Generation with Markov Transformers
(Deng and Rush, 2020) is another close precursor.

The defensible paper is therefore about **adapting the dependency topology to
the corrupted context and diffusion time**:

> A masked-diffusion backbone predicts a sparse forest over currently unknown
> positions and context-conditioned token-pair potentials on its edges. Within
> a top-K candidate lattice, the resulting denoising distribution is globally
> normalized and supports exact sum-product marginals and exact joint sampling.

The linear chain remains an important ablation and implementation milestone,
not the headline contribution.

Working title:

> **From Conflict Graphs to Coupling Forests: Joint Denoising in Diffusion
> Language Models**

## 2. What the paper may and may not claim

### Safe target claims

1. Context- and time-dependent graphical output structure improves recovery of
   dependencies that a factorized denoiser misses.
2. Forest structure permits exact normalization, marginals, and joint sampling
   inside the candidate lattice.
3. The largest gains should occur at high mask ratios and aggressive parallel
   commitment, where conditional total correlation is largest.
4. Modeling and scheduling are complementary: a structured sampler can model
   selected dependencies, while DEMASK/DAPD/PUNT-style scheduling can avoid
   dependencies that remain outside the forest.
5. A contextual forest should improve the quality--latency frontier relative
   to a matched factorized MDLM and a static structured prior.

### Claims to avoid

- first structured or joint diffusion denoiser;
- first identification of the factorization barrier;
- dense full-vocabulary pairwise modeling (the residual state preserves token
  support, but only retained candidates receive explicit pair couplings);
- exact data likelihood unless the induced reverse process and variational
  bound are derived and tested;
- universal long-range dependency modeling;
- state of the art before matched experiments exist.

## 3. Method the experiments must instantiate

Let `M_t` be the masked positions in `x_t`. A backbone produces hidden states
`h_i`, unary token logits `u_i`, and pairwise dependency scores. Candidate sets
`C_i` retain the top K unary tokens. A residual state aggregates all remaining
unary mass and decodes from the normalized tail distribution. Pair factors are
one whenever either endpoint is residual. This retains full token support and
keeps training and inference candidate sets target-independent; recall and
retained mass are still reported as approximation diagnostics.

From a sparse candidate edge graph (local-window edges plus a few
attention/dependency proposals), the model selects a bounded-component maximum
spanning forest `F_theta(x_t,t)`. The conditional energy is

    E(z; x_t,t) = sum_i u_i(z_i)
                  + sum_(i,j in F) psi_ij(z_i,z_j; x_t,t).

The pair potential is a low-rank mixture whose gates depend on `h_i`, `h_j`,
and a time embedding. The forest is globally normalized. Tree sum-product gives
the log partition and node/edge marginals; ancestral tree sampling gives one
joint clean assignment. The absorbing reverse posterior then decides which
tokens are committed. Observed tokens are evidence, never latent variables.

Complexity should be reported as `O(|M_t| K + |E_t| K^2)` for dense potentials
or `O(|M_t| K + |E_t| K R)` when the *positive factor itself* has nonnegative
rank R. Low-rank log potentials do not obtain the latter bound. Forest
construction and every runtime component are measured. Components are capped
so the structured layer can be batched and parallelized.

Training objective:

- structured denoising negative log likelihood on forward-corrupted examples;
- unary auxiliary loss to keep candidate recall high;
- a dependency/topology supervision or distillation loss based on conditional
  influence estimates;
- optional end-to-end fine-tuning after a frozen-backbone adapter phase.

The draft must distinguish this target method from the current recovery code,
which trains a locally normalized directed Markov model and samples independent
marginals.

## 4. Paper organization

1. **Introduction.** The quality--parallelism tradeoff; why selection-only and
   static-coupling methods leave a gap; contributions and scope.
2. **Background and closest work.** MDLM clean-data prediction, conditional
   total correlation, CoDD/DCD/EDLM/ADJUST, dependency-aware schedulers, and
   earlier structured non-autoregressive decoding.
3. **Contextual coupling forests.** Candidate lattice, dynamic topology,
   pairwise potentials, global normalization, exact inference, joint reverse
   sampling, and complexity.
4. **Why topology must adapt.** A controlled construction plus the distinction
   between modeling and avoiding dependencies.
5. **Experiments.** Controlled recovery, MDLM-scale language modeling, a modern
   7--8B adapter, efficiency, and ablations.
6. **Analysis and limitations.** Candidate truncation, tree bias, topology
   discreteness, compute, and claims that remain conditional on results.
7. **Conclusion.** Conservative summary tied to measured evidence.

The main mechanism figure should contrast factorized mixing, a static prior,
and a context-dependent forest on two contexts with different dependency
graphs. The main empirical figure should be quality versus measured wall-clock
latency.

## 5. Backward-designed experiment program

### Stage A: mathematical and implementation correctness

Run exhaustive enumeration for tiny vocabularies and lengths:

- log partition, node/edge marginals, and gradients;
- empirical frequencies from joint tree sampling;
- reverse-mixture probabilities for absorbing diffusion;
- clamping and top-K behavior.

Every quantity must agree with enumeration to numerical tolerance. This stage
also needs an architecture-count/no-edge control; its retained pair parameters
are inactive, so it is not an active-capacity-matched unary adapter.

### Stage B: controlled mechanism experiments

Use three seeds on:

- ambiguous pairs with identical unigram marginals but incompatible joint
  modes;
- first-order Markov languages with adjustable coupling strength;
- a nonlocal copy/agreement task whose relevant edge changes with context.
- a cyclic/XOR counterexample that exposes the forest family's limitation.

Compare factorized, locally normalized chain, global natural-order chain,
static forest, and contextual forest. Report exact KL/TV where available,
conditional NLL, edge recovery, mutual-information recovery, invalid-pair rate,
and speed. Full vocabulary removes top-K as a confound.

Decision gate: do not scale unless joint sampling beats independent sampling
from the same checkpoint and the contextual forest beats both the factorized
and static-structure models on the nonlocal task.

### Stage C: MDLM-scale controlled language modeling

Use the released `kuleshov-group/mdlm-owt` checkpoint in the
`yuntian-group/mdlm` codebase. Freeze the backbone first and train output
adapters on OpenWebText. This makes the closest comparisons cheap and fair.
Retain OpenWebText because MDLM, SEDD, and AR checkpoints are public; evaluate
continuations on WikiText-103. Add text8 with its full 35-character vocabulary
as an exact-lattice sanity benchmark.

Minimum matched baselines:

- original factorized MDLM, with both its original and confidence reveal rules;
- architecture-count/no-edge control (legacy experiment identifier:
  `parameter_matched_independent`; pair parameters inactive);
- locally normalized chain;
- globally normalized natural-order chain;
- static forest/transition prior;
- contextual forest with independent-marginal versus joint sampling;
- CoDD and a context-modulated CoDD variant following its appendix;
- one dependency-aware scheduler (DEMASK, DAPD, or PUNT), alone and composed
  with the contextual forest.

Evaluate 1, 4, 8, 16, 32, 64, and 128 denoiser calls; K in 8, 16, 32, 64, and 128;
lengths 256, 512, and 1024; batch sizes 1 and 8.

Metrics are separated into:

- **fit:** valid diffusion bound/estimated NLL, structured conditional NLL by
  mask-ratio bin, calibration, candidate recall;
- **generated distribution:** MAUVE, two reference-AR perplexities (labeled as
  proxies), distinct-n, Self-BLEU, repetition-2/4, n-gram JS divergence, and
  adjacent/nonlocal agreement;
- **efficiency:** latency, throughput, denoiser calls, finalized tokens per
  call, peak memory, and structured-layer share of latency.

Decision gates:

- recall@64 at least 99% on masked targets;
- confidence-gated factorized MDLM does not explain the gain;
- parameter matching does not erase the gain;
- contextual structure beats a static prior, especially at high mask ratios;
- per-step cost remains below 4x and yields a better matched-quality latency.

### Stage D: modern transfer experiment

Primary target: Dream-7B; secondary target: LLaDA-8B. Freeze the backbone and
train only the structured adapter first. Use the public competitor settings so
that CoDD, ADJUST, DEMASK/DAPD, Fast-dLLM/APD, and the serial reference are
meaningful comparisons.

Tasks:

- ParallelBench PB80/PB90 and complete accuracy-versus-tokens-per-step curves;
- IFEval;
- GSM8K and MATH-500;
- HumanEval and MBPP;
- WikiText-103 continuation;
- random-span and multi-span infilling.

GPQA is optional because it is costly and weakly diagnostic of the mechanism.

### Stage E: essential ablations

- no edges / natural chain / static forest / contextual forest;
- fixed topology with dynamic factors / dynamic topology with fixed factors /
  dynamic topology with dynamic factors;
- position-only / context-conditioned / context-and-time-conditioned edges;
- context-shuffled and timestep-shuffled topology controls;
- local candidate graph / attention proposals / learned proposals;
- local versus global normalization;
- independent marginals versus exact joint sampling;
- unary-only versus exact globally temperature-scaled sampling;
- K, pair-potential rank, component-size cap, and mask-ratio bins;
- frozen versus jointly tuned backbone;
- scheduler alone, forest alone, and their composition.

## 6. Expected-result placeholders for the first draft

These values are targets for planning, not observations:

- near-zero invalid ambiguous pairs for the exact joint sampler versus about
  50% under matched independent marginals;
- no material likelihood loss at conservative step counts;
- the clearest gain at 8--32 denoiser calls and high mask ratios;
- 10--30% relative reduction in dependency/repetition errors;
- MAUVE improvement on the order of 0.02--0.05;
- 5--15% lower reference-model generative perplexity at aggressive decoding;
- 1.3--2.5x structured-layer overhead per denoiser call, recovered by fewer
  calls at matched quality;
- contextual topology consistently better than a static prior on nonlocal and
  reasoning-sensitive evaluations.

All such values must appear in the paper through a visible `\placeholder{}`
macro. Nothing bracketed as predicted may remain in a submitted manuscript.

## 7. Code ownership in `yuntian-group/mdlm`

- `models/structured_decoder.py`: new unary, dependency, topology, and
  low-rank pair-potential heads; keep `models/crf_decoder.py` as the legacy
  locally normalized ablation.
- `structured_utils.py`: forest construction, tree partition/marginals,
  ancestral joint sampling, clamping, and reverse-mixture tests.
- `diffusion.py`: structured loss, joint reverse sampler, matched reveal rules,
  and metric hooks.
- `models/dit.py`: hidden-state/unary interface and safe checkpoint loading.
- `dataloader.py`: synthetic languages, text8, deterministic corruptions, and
  fixed evaluation splits.
- `configs/model/contextual-forest-small.yaml`: method configuration; do not
  silently redefine the existing CRF config.
- `configs/experiment/`: named pilot, ablation, and scale configurations.
- `evaluation/`: quality, dependency, infilling, memorization, and latency
  evaluation.
- `tests/test_structured_forest.py`: exhaustive checks.
- `scripts/`: resumable pilot/full launchers, with data and checkpoints written
  to the experiment's new persistent GCloud disk.

The current branch is a recovery baseline, not evidence for the paper: its
32-example smoke test verifies mechanics but its collapsed samples rule out any
positive empirical claim.

## 8. Compute order

1. CPU/GPU unit tests and synthetic tasks.
2. One L4/G4 smoke run with a hard time limit.
3. Text8 and frozen MDLM-OWT adapter pilots.
4. Only after decision gates pass, the full MDLM-scale matrix.
5. Only after the main effect survives, Dream-7B transfer and expensive
   baseline reproduction.

Every cloud run should use the requested new persistent disk, explicit labels,
resumable checkpoints, and a hard stop. The retained old debug disk is not to
be reused for the new experiment series.
