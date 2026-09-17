"""CPU-only mechanism probes; no trained-checkpoint or generation claims.

Uses this repository's head/inference and small explicitly enumerated laws.
The numerical examples and reveal-kernel calculation were constructed for
this diagnosis; no external implementation was consulted or copied.
Run from the repository root: python scripts/diagnose_ccf_objective.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.structured_decoder import ContextualCouplingForestHead
from structured_objective import (full_vocabulary_marginals,
                                  infer_structured_distribution,
                                  structured_token_log_probability)


def main():
    torch.manual_seed(20260916)
    torch.set_num_threads(1)
    head = ContextualCouplingForestHead(
        hidden_size=8, vocab_size=4, top_k=2, rank=2,
        time_embed_dim=4, topology_dim=4, local_window=1,
        num_anchor_slots=1, contextual_neighbors=1,
        component_size_cap=2, topology_mode='fixed').double()
    logits = torch.zeros(1, 2, 4, dtype=torch.float64)
    active = torch.ones(1, 2, dtype=torch.bool)
    output = head(torch.randn(1, 2, 8, dtype=torch.float64), logits,
                  torch.tensor([0.5], dtype=torch.float64), active)

    def masses(explicit_factor, backend):
        # All explicit pairs have the SAME factor; residual rows remain one.
        endpoint = torch.full_like(output.pair_left_factors,
                                   (explicit_factor / head.rank) ** 0.5)
        changed = replace(output, pair_left_factors=endpoint,
                          pair_right_factors=endpoint.clone())
        inference = infer_structured_distribution(changed, active, backend)
        marginal = full_vocabulary_marginals(changed, logits, active, inference)
        return marginal, marginal.gather(-1, output.candidate_ids).sum(-1)

    backend_results = {}
    for backend in ('dense', 'low_rank'):
        neutral, _ = masses(1., backend)
        _, shifted = masses(4., backend)
        torch.testing.assert_close(neutral, logits.softmax(-1), atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(shifted, torch.full_like(shifted, 5 / 7),
                                   atol=1e-7, rtol=1e-7)
        backend_results[backend] = {
            'neutral_max_probability_error': (neutral - logits.softmax(-1)).abs().max().item(),
            'base_explicit_mass': 0.5,
            'constant_factor_4_explicit_mass': shifted.tolist(),
        }

    head.topology_mode = 'dynamic'
    dynamic = head(torch.randn(1, 2, 8, dtype=torch.float64), logits,
                   torch.tensor([0.5], dtype=torch.float64), active)
    joint_loss = -structured_token_log_probability(
        dynamic, logits, torch.tensor([[0, 1]]), active).sum()
    factor_grad, topology_grad = torch.autograd.grad(
        joint_loss, (head.token_factor_embedding.weight,
                     head.topology_hidden_projection.weight), allow_unused=True)
    assert factor_grad is not None and factor_grad.abs().max() > 0
    assert topology_grad is None

    # True correlated pair, uniform independent backbone, imperfect learned
    # joint. The learned joint improves joint CE but degrades both marginals.
    truth = torch.tensor([[.49, .01], [.01, .49]], dtype=torch.float64)
    base = torch.full_like(truth, .25)
    learned = torch.tensor([[.65, .01], [.01, .33]], dtype=torch.float64)

    def cross_entropies(prediction):
        joint = -(truth * prediction.log()).sum().item()
        singleton_sum = sum(
            -(truth.sum(dim=axis) * prediction.sum(dim=axis).log()).sum().item()
            for axis in (0, 1))
        return joint, singleton_sum

    base_joint, base_singletons = cross_entropies(base)
    learned_joint, learned_singletons = cross_entropies(learned)
    assert learned_joint < base_joint
    assert learned_singletons > base_singletons
    thinning = []
    for reveal in (0.001, 0.01, 0.1, 1.):
        # Reveal-pattern probabilities are identical for both models, so
        # their CE cancels in the difference. Normalize by expected 2*r
        # revealed tokens. This is exact for the two-site example.
        base_ce = ((1-reveal)*base_singletons + reveal*base_joint) / 2
        learned_ce = ((1-reveal)*learned_singletons + reveal*learned_joint) / 2
        thinning.append({'reveal_probability': reveal,
                         'both_revealed_probability': reveal**2,
                         'base_token_ce': base_ce,
                         'learned_token_ce': learned_ce,
                         'learned_minus_base': learned_ce-base_ce})
    assert thinning[0]['learned_minus_base'] > 0
    assert thinning[-1]['learned_minus_base'] < 0
    print(json.dumps({
        'scope': 'Synthetic CPU probes, not measured trained-model failures',
        'torch': torch.__version__,
        'neutral_and_residual_probes': backend_results,
        'joint_nll_gradient_probe': {
            'factor_embedding_max_abs_gradient': factor_grad.abs().max().item(),
            'topology_hidden_projection_gradient_is_none': topology_grad is None,
            'note': 'Separate gold-reveal auxiliary loss, not joint NLL, trains topology.',
        },
        'joint_ce': {'base': base_joint, 'learned': learned_joint},
        'singleton_ce_per_token': {'base': base_singletons/2,
                                   'learned': learned_singletons/2},
        'reveal_kernel': thinning,
    }, indent=2))


if __name__ == '__main__':
    main()
