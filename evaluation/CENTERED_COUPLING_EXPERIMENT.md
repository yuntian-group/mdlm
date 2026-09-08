# Learn dependence without moving the unary probabilities

Status: standalone head and correctness tests implemented. No real-text
training or generation result has been measured for this head.

The completed fresh-training pilot finds positive dependence gain, but the
separate-factor head also worsens singleton scores relative to the trained
unary adapter. This experiment isolates those two effects by construction.

## Model

`FrozenUnaryCenteredForestHead` deep-copies and freezes a candidate-only unary
adapter. Let its normalized token probabilities be `b_i`. For each edge of
the natural-order chain of masked positions, learn separate endpoint features
`f_i` and `g_j` with zero mean under `b_i` and `b_j`. Residual-state features
are zero. The edge ratio is

    phi_ij(a,b) = 1 + eta * mean_r f_i(a,r) g_j(b,r).

Features are bounded by one and `0 < eta < 1`, so all ratios are positive.
The joint distribution is `q(x) = product_i b_i(x_i) product_ij phi_ij(x_i,x_j)`.
Each edge ratio integrates to one at either endpoint. Removing leaves proves
that the forest is normalized and has exactly the prescribed marginals.
Observed log scores need only the unary scores and observed edge ratios;
training does not need a partition-function computation.

The implementation exposes positive factors of rank `2*d+1` to the existing
inference and sampling backends. It preserves the selected-test evaluator's
FP32 compact unary lattice followed by FP64 normalization, and normalizes the
within-residual backbone conditional separately in FP64. Recomputing compact
residual mass would change the frozen reference and is not allowed.

At the default `d=8`, `eta=.95`, and GPT-2-sized vocabulary, the new coupling
has 827,312 trainable parameters, in addition to the frozen unary adapter's
827,312 parameters. The positive log-ratio per edge is bounded by
`log(1+eta)`; this family cannot represent arbitrary dependence.

## Gates before a language-model claim

1. **Exactness.** Enumeration, both forest inference backends, direct scores,
   gradients, empty tails, inactive nodes, initialization, and three-seed
   AB/BA learning are covered by the unit tests. Initial scores must equal
   those of the copied unary adapter, including tail targets.
2. **Small training fit.** On a fixed training-only subset, freeze an
   authenticated unary checkpoint and fit coupling parameters. Log the
   joint-minus-unary gain and maximum marginal error at every evaluation.
   Do not inspect reserved-test documents to debug this stage.
3. **Fresh corruptions.** Fit on training documents with fresh masks and choose
   checkpoints using development joint NLL. Include the independence
   checkpoint. Keep the current pilot's test files immutable; any later use
   of these same documents is a follow-up comparison after observing the
   pilot, not new independent confirmation.
4. **Budget-matched control.** Compare against spending the same additional
   trainable parameters and updates on a second unary correction. The frozen
   base plus coupling costs twice the original unary adapter's parameter
   count. A win over the frozen base alone does not establish a better use
   of that budget.
5. **Simultaneous decisions.** The backbone bridge and specialized FP64
   residual decoder now have generation unit tests. With identical contexts,
   reveal order, and RNG, tested group-size-one trajectories agree exactly
   with the selected-test-precision frozen unary reference, including after
   nonzero couplings. Existing ordinary-output sampler paths are unchanged.
   This does not claim bitwise equivalence between FP32 and FP64 residual
   normalization. Add an authenticated real-text launch, then test
   group sizes 2/4/8/16, logging committed edges, quality, and wall time.
   Use a genuinely untouched corpus for the final confirmation.

No production head, training runner, checkpoint, or paper result is replaced
by this prototype. Relevant tests are `test_centered_coupling.py` and
`test_centered_forest.py`.
