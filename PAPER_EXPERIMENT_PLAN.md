# Paper and Experiment Strategy

Status: implementation and verified-evidence snapshot, 2026-08-30. This
document is the contract between the paper claims and the experiments. It must
be revised whenever evidence contradicts the current story.

## 0. Current execution status

Completed and usable as evidence:

- exact dense/enumerated versus low-rank forest normalization, marginals,
  gradients, joint-sampling, hard-constraint, and CUDA-repeatability tests;
- a protocol frozen after development on seeds 1--8 and evaluated once on
  held-out seeds 9--13: 900 steps, 300 shared-factor warmup steps, factor
  initialization standard deviation 0.25, and factor initialization seed 1729;
- the held-out contextual-matching result at commit `c8d4b70`: contextual TV
  0.05660, invalid mass 0.00509, edge F1 1.0, and an all-positive paired TV
  improvement whose 95% bootstrap interval is [0.62089, 0.62821];
- the optimized inference profile at commit `9a7129f`, using one L4,
  PyTorch 2.5.1+cu121, three warmups and ten measured calls bracketed by CUDA
  synchronization per backend: exact agreement to machine precision and
  2.67x--10.07x dense over low-rank speed ratios across the reported shapes;
- an opt-in real-data DIT bridge with frozen-backbone loading, streaming
  OpenWebText training, finite validation, conditional-denoising-NLL metrics,
  topology-teacher coverage, and factorized/marginal/joint sampling modes;
- two-step text8 train/resume and structured-sampling plumbing checks; and
- a strict full-length structured pass using the pinned released raw MDLM-OWT
  export (no EMA), with finite logits/marginals and normalized nodes.

Not yet evidence and therefore not a current paper claim:

- a trained OpenWebText adapter improvement on paired fixed validation data;
- generated-text quality or matched-quality end-to-end latency;
- CoDD, scheduler, and modern 7--8B comparisons; and
- any diffusion ELBO, data likelihood, MAUVE, reference-model perplexity, or
  state-of-the-art result for the contextual forest.

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

Current title:

> **Contextual Coupling Forests for Joint Denoising in Diffusion Language
> Models**

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

### Stage A: mathematical and implementation correctness (complete)

Run exhaustive enumeration for tiny vocabularies and lengths:

- log partition, node/edge marginals, and gradients;
- empirical frequencies from joint tree sampling;
- reverse-mixture probabilities for absorbing diffusion;
- clamping and top-K behavior.

Every quantity agrees with enumeration to numerical tolerance. The implemented
architecture-count/no-edge control retains inactive pair parameters, so it is
not described as an active-capacity-matched unary adapter.

### Stage B: controlled mechanism experiments (partially complete)

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

The frozen-protocol context-switching-matching experiment passes this gate on
held-out seeds 9--13. The broader ambiguous-pair, adjustable Markov, and XOR
learned-task matrix remains pending; oracle/table-fit checks are mechanism
sanity evidence and must not be conflated with learned-adapter results.

### Stage C: MDLM-scale controlled language modeling (plumbing complete;
quality pending)

Use the released `kuleshov-group/mdlm-owt` checkpoint in the
`yuntian-group/mdlm` codebase, pinned to revision
`d0958fa851335ece6c15260ce0025f030673c0fb` and verified against the released
`model.safetensors` SHA256. The public artifact contains raw `backbone.*`
weights and no EMA. Freeze the backbone first and train output adapters on
streaming OpenWebText. This makes the closest comparisons cheap and fair.
Retain OpenWebText because MDLM, SEDD, and AR checkpoints are public; evaluate
paired fixed examples/corruptions on WikiText-103. Add text8 with its full
35-character vocabulary as an exact-lattice sanity benchmark.

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

### Stage D: modern transfer experiment (pending)

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

### Stage E: essential ablations (pending)

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

## 6. Measured evidence and unresolved hypotheses

The only current learned quality result is the frozen-protocol synthetic
held-out experiment described in Section 0. The optimized forest profile is a
kernel-only measurement: it excludes candidate/factor and forest construction,
so it cannot be presented as end-to-end decoding latency. The released-backbone
and text8 runs are plumbing checks and cannot be presented as language-quality
evidence.

The earlier numerical targets for MAUVE, reference-model perplexity,
dependency/repetition error, end-to-end overhead, and denoiser-call regimes
have been removed. Those cells remain unclaimed until frozen-protocol paired
experiments produce artifacts. Future hypotheses remain qualitative:

- benefits should concentrate at high mask ratios and aggressive commitment;
- contextual topology should matter most when the relevant dependency changes
  with corrupted context;
- a scheduler and a structured output distribution may be complementary; and
- candidate truncation and forest bias may limit gains on long-range cycles.

No predicted numeric result may be inserted into a submission table or prose.

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
- `scripts/`: resumable pilot/full launchers, immutable release preparation,
  profile provenance, and paired artifact aggregation; large data and
  checkpoints belong on a dedicated persistent experiment volume.

The branch now contains paper-usable exact-inference, held-out synthetic, and
kernel-profile evidence. The original 32-example recovery smoke remains only a
historical mechanics check, and real-data plumbing still supports no positive
language-quality claim.

## 8. Compute order

1. **Complete:** CPU/GPU unit tests and the frozen-protocol held-out synthetic
   task.
2. **Complete:** bounded GPU text8 and released-backbone plumbing checks.
3. **Next:** a short frozen MDLM-OWT adapter screen with paired fixed
   validation artifacts and its factorized control.
4. **Conditional:** only after that gate passes, the full MDLM-scale matrix.
5. **Conditional:** only after the real-data main effect survives, a modern
   7--8B transfer and expensive baseline reproduction.

Remote runs use an experiment-dedicated persistent volume, explicit run labels,
resumable checkpoints, immutable source revisions, and a hard stop. Public
artifacts record hardware/software provenance without private account,
funding, project, instance, or resource identifiers.
