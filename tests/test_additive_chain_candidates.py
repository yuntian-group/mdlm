"""Bitwise candidate/independent-draw agreement with the frozen full sort."""
import pytest
import torch

from additive_chain.candidates import extract_compact_candidates as reference
from additive_chain.fast_candidates import extract_fast_candidates as fast


def compare(raw, tokens, *, mask_id, cap, p):
    old = reference(raw, tokens, mask_id=mask_id, top_k=cap, nucleus_p=p)
    new = fast(raw, tokens, mask_id=mask_id, top_k=cap, nucleus_p=p)
    assert torch.equal(new.candidate_ids, old.candidate_ids)
    assert torch.equal(new.log_unary, old.log_unary)
    assert torch.equal(new.diagnostics['support_sizes'], old.diagnostics['support_sizes'])
    # Diagnostic only: CUDA sum kernels can round the same retained mass
    # differently after row regrouping. This value never enters support,
    # normalized unaries, or sampling. Keep every model-law check bitwise.
    torch.testing.assert_close(new.diagnostics['retained_original_mass'],
        old.diagnostics['retained_original_mass'], atol=2e-15, rtol=2e-14)
    assert new.diagnostics['nucleus_sizes'] is None
    assert new.diagnostics['nucleus_original_mass'] is None
    assert new.diagnostics['diagnostic_scope'] == 'retained_support_only'
    # Identical support order is required to preserve baseline random draws.
    a = torch.Generator(device=raw.device).manual_seed(93)
    b = torch.Generator(device=raw.device).manual_seed(93)
    first = torch.multinomial(old.log_unary.flatten(0, 1).exp(), 1, generator=a)
    second = torch.multinomial(new.log_unary.flatten(0, 1).exp(), 1, generator=b)
    assert torch.equal(first, second)
    return new


def examples(device):
    g = torch.Generator(device=device).manual_seed(38)
    raw = torch.randn(2, 9, 33, generator=g, device=device).log_softmax(-1)
    tokens = torch.full((2, 9), 32, dtype=torch.long, device=device)
    tokens[0, 0], tokens[1, 5] = 1, 7
    raw[:, 1, :12] = -2.
    raw[:, 2, :5] = -torch.inf
    raw[:, 3] = -torch.inf
    raw[:, 3, 5] = 0.
    raw[:, 4, :4] = torch.tensor([0., -0., 0., -0.], device=device)
    # Normalization makes small FP32 differences disappear next to 1e30:
    # this row must fall back instead of ranking original small scores.
    raw[:, 6] = torch.arange(33, device=device).float()
    raw[:, 6, 20] = 1e30
    return raw, tokens


def test_cpu_all_boundary_tie_and_dtype_cases():
    raw, tokens = examples('cpu')
    for dtype in (torch.float32, torch.float64):
        for cap in (1, 2, 7, 33, 50):
            for p in (.05, .5, .9, 1.):
                compare(raw.to(dtype), tokens, mask_id=32, cap=cap, p=p)
    uniform = torch.zeros(1, 2, 11)
    tokens = torch.full((1, 2), 10, dtype=torch.long)
    result = compare(uniform, tokens, mask_id=10, cap=8, p=.5)
    assert result.diagnostics['reference_fallback_rows'] == 2


def test_cpu_rejects_invalid_inputs_and_keeps_input_unchanged():
    raw, tokens = examples('cpu')
    previous = raw.clone()
    compare(raw, tokens, mask_id=32, cap=7, p=.9)
    assert torch.equal(raw, previous)
    for value in (torch.nan, torch.inf):
        wrong = raw.clone()
        wrong[0, 1, 0] = value
        with pytest.raises(ValueError):
            fast(wrong, tokens, mask_id=32, top_k=7)
    with pytest.raises(ValueError):
        fast(raw, tokens, mask_id=32, top_k=0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA candidate gate required')


@cuda
def test_cuda_boundary_and_tie_cases():
    raw, tokens = examples('cuda')
    for cap in (1, 2, 7, 33, 50):
        for p in (.05, .5, .9, 1.):
            compare(raw, tokens, mask_id=32, cap=cap, p=p)
    uniform = torch.zeros(1, 2, 11, device='cuda')
    tokens = torch.full((1, 2), 10, dtype=torch.long, device='cuda')
    result = compare(uniform, tokens, mask_id=10, cap=8, p=.5)
    assert result.diagnostics['reference_fallback_rows'] == 2


@cuda
@pytest.mark.parametrize('cap', [500, 1000, 2000])
def test_cuda_real_vocab_bitwise_candidates_and_baseline_draw(cap):
    g = torch.Generator(device='cuda').manual_seed(537)
    raw = torch.randn(2, 19, 50258, device='cuda', generator=g).log_softmax(-1)
    tokens = torch.full((2, 19), 50257, dtype=torch.long, device='cuda')
    tokens[0, 1], tokens[1, 3] = 7, 13
    raw[:, 5, 100:500] = -7.
    raw[:, 8, :200] = -torch.inf
    for p in (.2, .9, 1.):
        compare(raw, tokens, mask_id=50257, cap=cap, p=p)

