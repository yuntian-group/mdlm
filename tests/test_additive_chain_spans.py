"""Exhaustive selected-joint-law tests; no weights, GPU, or seed sweeps."""

import itertools

import pytest
import torch

from additive_chain.spans import relevant_span_context


MASK = 3


def selected_law(tokens, logits, transition, reveal):
    """Enumerate a tiny complete chain and aggregate its selected-token law."""
    supports = [list(range(MASK)) if token == MASK else [token]
                for token in tokens.tolist()]
    paths = torch.tensor(list(itertools.product(*supports)), dtype=torch.long)
    score = torch.zeros(paths.shape[0], dtype=torch.float64)
    for pos in range(len(tokens)):
        if tokens[pos] == MASK:
            score += logits[pos, paths[:, pos]]
    if len(tokens) > 1:
        score += transition[paths[:, :-1], paths[:, 1:]].sum(-1)
    probabilities = score.softmax(0)
    selected = reveal & tokens.eq(MASK)
    law = {}
    for path, probability in zip(paths[:, selected].tolist(), probabilities):
        key = tuple(path)
        law[key] = law.get(key, 0.) + float(probability)
    return law


def test_group_selection_batch_boundaries_and_visible_reveals():
    tokens = torch.tensor([[3, 3, 1, 3, 3, 2, 3, 3],
                           [1, 3, 2, 3, 3, 0, 3, 1],
                           [3, 3, 3, 3, 3, 3, 3, 3]])
    reveal = torch.tensor([[False, True, False, False, False, False, False, False],
                           [True, False, False, True, False, False, True, True],
                           [False, False, False, False, True, False, False, False]])
    actual = relevant_span_context(tokens, reveal, mask_id=MASK)
    expected = torch.tensor([[3, 3, 1, 0, 0, 2, 0, 0],
                              [1, 0, 2, 3, 3, 0, 3, 1],
                              [3, 3, 3, 3, 3, 3, 3, 3]])
    assert torch.equal(actual, expected)
    assert torch.equal(tokens, torch.tensor([[3, 3, 1, 3, 3, 2, 3, 3],
        [1, 3, 2, 3, 3, 0, 3, 1], [3, 3, 3, 3, 3, 3, 3, 3]]))


@pytest.mark.parametrize('row', [
    [3], [0], [3, 3, 3], [3, 1, 3, 3], [3, 3, 1, 3, 3],
    [0, 3, 1, 2, 3, 3], [3, 0, 3, 1, 3], [0, 1, 2, 0],
])
def test_every_reveal_subset_preserves_exact_selected_joint(row):
    tokens = torch.tensor([row])
    length = len(row)
    # Correlated, asymmetric pair matrix: not a separable/product toy.
    transition = torch.tensor([[1.7, -.8, .2], [-1.1, 1.4, .5], [.9, -.7, 1.2]],
                              dtype=torch.float64)
    logits = (torch.arange(length * MASK, dtype=torch.float64).reshape(length, MASK)
              .mul(.73).sin())
    for subset in itertools.product((False, True), repeat=length):
        reveal = torch.tensor([subset])
        context = relevant_span_context(tokens, reveal, mask_id=MASK)
        expected = selected_law(tokens[0], logits, transition, reveal[0])
        actual = selected_law(context[0], logits, transition, reveal[0])
        assert actual.keys() == expected.keys()
        for key in expected:
            assert abs(expected[key] - actual[key]) < 2e-14


def test_multiple_batch_selected_laws_do_not_cross_rows():
    tokens = torch.tensor([[3, 3, 0, 3, 1, 3], [3, 1, 3, 3, 0, 3]])
    reveal = torch.tensor([[True, False, False, False, False, True],
                           [False, False, True, False, False, False]])
    logits = torch.tensor([[.2, -.3, .1]] * 6, dtype=torch.float64)
    edge = torch.tensor([[2., -1., .1], [-.5, 1.2, .2], [.7, .3, -1.]], dtype=torch.float64)
    context = relevant_span_context(tokens, reveal, mask_id=MASK)
    for row in range(2):
        expected = selected_law(tokens[row], logits, edge, reveal[row])
        actual = selected_law(context[row], logits, edge, reveal[row])
        assert max(abs(expected[key] - actual[key]) for key in expected) < 2e-14


def test_empty_reveal_and_all_visible_cases():
    tokens = torch.tensor([[3, 1, 3], [0, 1, 2]])
    context = relevant_span_context(tokens, torch.zeros_like(tokens, dtype=torch.bool), mask_id=MASK)
    assert context.tolist() == [[0, 1, 0], [0, 1, 2]]
    assert torch.equal(relevant_span_context(tokens[1:], torch.ones_like(tokens[1:], dtype=torch.bool), mask_id=MASK), tokens[1:])


def test_distinct_dummy_and_shape_validation():
    tokens = torch.tensor([[3, 0, 3]])
    reveal = torch.zeros_like(tokens, dtype=torch.bool)
    assert relevant_span_context(tokens, reveal, mask_id=MASK, dummy_token_id=2).tolist() == [[2, 0, 2]]
    with pytest.raises(ValueError):
        relevant_span_context(tokens, reveal, mask_id=MASK, dummy_token_id=MASK)
    with pytest.raises(ValueError):
        relevant_span_context(tokens.float(), reveal, mask_id=MASK)
    with pytest.raises(ValueError):
        relevant_span_context(tokens, reveal.long(), mask_id=MASK)

