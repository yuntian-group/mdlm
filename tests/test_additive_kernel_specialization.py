"""Avoid recompiling learned CRF kernels for each target length.

The optional CUDA check restores only the old specialization annotations in a
temporary module. Both versions then receive the same potentials and random
noise, so their messages and draws can be compared directly.
"""
import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import additive_chain._additive_kernels as kernels
import additive_chain._count_inference as count
from additive_chain.candidates import CompactCandidates
from additive_chain.chain import marginals
from additive_chain.counts import SparseCountPotential
from additive_chain.cuda import dense_pair_reference, prepare_additive_factors


RUNTIME_ARGUMENTS = {
    "_pair_tile": ["original_length"],
    "_message_kernel": ["original_length", "pos", "width"],
    "_sample_kernel": ["original_length", "width"],
}


def kernel_source():
    return Path(kernels.__file__).read_text()


def old_specialization_source(source):
    """Restore the previous annotations without copying a second kernel body."""
    for names in RUNTIME_ARGUMENTS.values():
        marker = "@triton.jit(do_not_specialize=" + repr(names) + ")"
        assert source.count(marker) == 1
        source = source.replace(marker, "@triton.jit")
    marker = "original_length, states: tl.constexpr"
    assert source.count(marker) == 3
    return source.replace(marker, "original_length: tl.constexpr, states: tl.constexpr")


def kernel_functions(source):
    return {node.name: node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name in RUNTIME_ARGUMENTS}


def test_sequence_layout_does_not_specialize_learned_kernels():
    functions = kernel_functions(kernel_source())
    assert set(functions) == set(RUNTIME_ARGUMENTS)
    for name, expected in RUNTIME_ARGUMENTS.items():
        function = functions[name]
        assert len(function.decorator_list) == 1
        decorator = function.decorator_list[0]
        assert isinstance(decorator, ast.Call)
        assert ast.unparse(decorator.func) == "triton.jit"
        assert {arg.arg: ast.literal_eval(arg.value) for arg in decorator.keywords} == {
            "do_not_specialize": expected}
        arguments = {argument.arg: argument for argument in function.args.args}
        for argument in expected:
            assert arguments[argument].annotation is None
        # These still determine tile shape, reduction order, and score law.
        for argument in ("states", "rank", "rank_block", "kind"):
            assert ast.unparse(arguments[argument].annotation) == "tl.constexpr"


def test_legacy_reference_changes_only_specialization_not_math():
    source = kernel_source()
    current = kernel_functions(source)
    reference = kernel_functions(old_specialization_source(source))
    for name in current:
        assert ast.dump(ast.Module(body=current[name].body, type_ignores=[])) == ast.dump(
            ast.Module(body=reference[name].body, type_ignores=[]))


def load_legacy_module(tmp_path):
    path = tmp_path / "legacy_additive_specialization.py"
    path.write_text(old_specialization_source(kernel_source()))
    name = "additive_chain._legacy_additive_specialization"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Dataclass and Triton inspect the defining module.
    spec.loader.exec_module(module)
    return module


cuda = pytest.mark.skipif(not torch.cuda.is_available() or kernels.triton is None,
                         reason="Explicit NVIDIA CUDA/Triton parity check")


@cuda
@pytest.mark.parametrize("width,length,rank", [(3, 7, 64), (500, 4, 64), (16, 17, 128)])
def test_cuda_runtime_layout_matches_legacy_messages_draws_and_dense_oracle(
        tmp_path, width, length, rank):
    legacy = load_legacy_module(tmp_path)
    generator = torch.Generator(device="cuda").manual_seed(337)
    vocab = width + 2
    ids = torch.arange(width, device="cuda")[None, None].expand(2, length, width).clone()
    unary = torch.randn(2, length, width, generator=generator, device="cuda",
                        dtype=torch.float64).log_softmax(-1)
    for position in sorted({0, length // 2, length - 1}):
        ids[1, position] = vocab
        ids[1, position, 0] = 1
        unary[1, position] = -torch.inf
        unary[1, position, 0] = 0.
    candidates = CompactCandidates(ids, unary, vocab)
    potential = SparseCountPotential(
        torch.linspace(.6, .9, vocab, device="cuda", dtype=torch.float64),
        torch.linspace(.7, .4, vocab, device="cuda", dtype=torch.float64),
        torch.tensor([1, vocab], device="cuda"),
        torch.tensor([.3, .6], device="cuda", dtype=torch.float64), log_scale=7.)
    left = torch.randn(2, length, width, rank, device="cuda", generator=generator) * .02
    right = torch.randn(2, length, width, rank, device="cuda", generator=generator) * .02
    packet = count.prepare_packet(candidates)
    factors = prepare_additive_factors(packet, potential, left, right)
    dense = dense_pair_reference(candidates, potential, left, right)
    expected = marginals(unary.float().double(), dense.double(), return_edges=False).node
    actual = torch.zeros_like(unary)
    actual.scatter_(-1, unary.argmax(-1, keepdim=True), 1.)
    for packed, rows, starts, lengths, positions, valid in count._chunks(packet, factors, 3):
        alpha = kernels._messages(packed, factors, rows, starts, lengths)
        old_alpha = legacy._messages(packed, factors, rows, starts, lengths)
        beta = kernels._messages(packed, factors, rows, starts, lengths, backward=True)
        old_beta = legacy._messages(packed, factors, rows, starts, lengths, backward=True)
        assert torch.equal(alpha, old_alpha)
        assert torch.equal(beta, old_beta)
        # A shared noise tensor tests the FFBS mapping, without distributional
        # tolerances or extra random-seed experiments.
        noise = torch.empty_like(alpha, dtype=torch.float64).exponential_(
            generator=generator).log_()
        sample = kernels._sample_with_noise(alpha, factors, rows, starts, lengths, noise)
        old_sample = legacy._sample_with_noise(old_alpha, factors, rows, starts, lengths, noise)
        assert torch.equal(sample, old_sample)
        probability = (alpha + beta).log_softmax(-1).exp()
        batches = rows[:, None].expand_as(positions)
        actual[batches[valid], positions[valid]] = probability[valid].double()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
