import math

import pytest
import torch

from additive_chain.candidates import CompactCandidates
from additive_chain.counts import SparseCountPotential
from additive_chain.iwslt_spans import compare_span, span_population


def setup_pair():
    # Joint masses [[9,1],[1,9]]: equal marginals but correlated decisions.
    potential = SparseCountPotential(torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64), torch.tensor([0, 3]), torch.tensor([8., 8.], dtype=torch.float64))
    ids = torch.tensor([[[0, 1], [0, 1]]])
    unary = torch.full((1, 2, 2), -math.log(2), dtype=torch.float64)
    return CompactCandidates(ids, unary, 2), potential


def test_joint_span_probability_not_product_of_identical_marginals():
    candidates, potential = setup_pair()
    result = compare_span(candidates, potential, torch.tensor([[0, 0]]), draws=2000)
    assert result["joint_gold_probability"] == pytest.approx(.45)
    assert result["independent_gold_probability"] == pytest.approx(.25)
    joint_agree = sum(a == b for a, b in result["joint_draws"]) / 2000
    independent_agree = sum(a == b for a, b in result["independent_draws"]) / 2000
    assert .86 < joint_agree < .94
    assert .45 < independent_agree < .55


def test_missing_gold_stays_zero_without_support_augmentation():
    candidates, potential = setup_pair()
    result = compare_span(candidates, potential, torch.tensor([[0, 2]]))
    assert not result["supported"]
    assert result["joint_gold_probability"] == result["independent_gold_probability"] == 0
    assert result["joint_gold_log_probability"] is None


def test_visible_neighbor_is_conditioned_not_marginalized():
    candidates, potential = setup_pair()
    candidates.log_unary[0, 0] = torch.tensor([0., -torch.inf])
    result = compare_span(candidates, potential, torch.tensor([[0, 0]]))
    assert result["joint_gold_probability"] == pytest.approx(.9)
    assert result["independent_gold_probability"] == pytest.approx(.9)


def test_zero_pair_strength_and_single_site_agree():
    candidates, _ = setup_pair()
    potential = SparseCountPotential(torch.ones(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float64))
    for length in (1, 2):
        packet = CompactCandidates(candidates.candidate_ids[:, :length], candidates.log_unary[:, :length], 2)
        result = compare_span(packet, potential, torch.zeros((1, length), dtype=torch.long))
        assert result["joint_gold_log_probability"] == pytest.approx(result["independent_gold_log_probability"])


def test_population_is_reproducible_and_only_length_filtered():
    rows = [dict(id=str(i), target_ids=list(range(i + 1))) for i in range(12)]
    a, eligible = span_population(rows, 4, 5, 11)
    b, _ = span_population(rows, 4, 5, 11)
    assert a == b and eligible == 9 and len(a) == 5
    assert all(0 <= start <= len(row["target_ids"]) - 4 for row, start in a)
