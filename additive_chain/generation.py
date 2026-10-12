"""Finite-step masked generation with additive token-chain proposals.

This is a standalone port of the experiment's finite-step scheduler and reveal
adapter, not BD3-LM's first-hitting sampler. No model/checkpoint loading, gold
tokens, training, or external quality scorer is involved. See GENERATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import torch

from .candidates import extract_compact_candidates
from .counts import SparseCountPotential
from .evaluation import decode_retaining_eos, token_diagnostics
from .heads import concatenate_scaled_factors
from .spans import relevant_span_context


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _coefficient(value):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "Stage strengths must be finite and nonnegative")
    return float(value)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def stream_seed(namespace, attempt_id, stream, *, window_id=0):
    """Exact attempt/window/stream SHA key; independent of method and acceptance."""
    _require(isinstance(namespace, str) and bool(namespace)
             and type(attempt_id) is int and attempt_id >= 0
             and type(window_id) is int and window_id >= 0
             and stream in ("reveal", "proposal"), "Invalid explicit random stream key")
    key = f"{namespace}:attempt-{attempt_id}:window-{window_id}:{stream}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def initial_tokens(*, length=1024, mask_id, bos_token_id, device="cpu"):
    """One supplied BOS/EOS context ID, followed by masks; no forced final EOS."""
    _require(type(length) is int and length >= 2 and type(mask_id) is int
             and type(bos_token_id) is int and 0 <= bos_token_id < mask_id,
             "Require length>=2 and a clean initial context ID")
    tokens = torch.full((1, length), mask_id, device=device, dtype=torch.long)
    tokens[:, 0] = bos_token_id
    return tokens


@dataclass
class WindowResult:
    tokens: torch.Tensor
    stats: dict


class _CountCache:
    """Own a count-only table under the same interface as StaticGlobalCache."""
    def __init__(self, potential, table):
        self.potential, self.table = potential, table
        self.signature, self.closed = potential.signature(), False

    def assert_current(self, potential):
        _require(not self.closed and potential is self.potential
                 and potential.signature() == self.signature, "Static count source changed/closed")

    def prepared_factors(self, packet):
        from ._dense_inference import DenseFactors
        self.assert_current(self.potential)
        return DenseFactors(packet.candidate_ids, self.table)

    def close(self):
        self.table, self.closed = None, True


class ChainDecoder:
    """Stage composition and selected-position proposals on unchanged support.

    ``count_statistics`` contains left_counts/right_counts/pair_keys/pair_counts
    from training text. Count strength is applied once when constructing the
    powered smoothed potential; learned strengths scale LEFT factors once.
    ``prepare()`` is explicit setup outside generation timing, even for fused
    inference. Heads must already be frozen FP32 modules on the chosen device.
    """
    def __init__(self, *, mask_id, device="cpu", count_statistics=None,
                 count_strength=0., count_mode="conditional", smoothing=.1,
                 global_head=None, global_strength=0., contextual_head=None,
                 contextual_strength=0., sampling="joint", backend="fused",
                 top_k=500, nucleus_p=.9, fast_candidates=False,
                 prune_irrelevant=True, segment_batch_size=32,
                 strict_unary_validation=False):
        _require(type(mask_id) is int and mask_id > 0, "Mask must be the last vocabulary ID")
        _require(sampling in ("joint", "own-marginal"), "Unknown sampling selector")
        _require(backend in ("fused", "static"), "Choose fused or explicit static backend")
        _require(count_mode in ("conditional", "pmi") and type(smoothing) in (int, float)
                 and 0 < smoothing <= 1, "Invalid count mode or smoothing")
        _require(type(top_k) is int and 1 <= top_k <= 2048, "top_k must lie in1..2048")
        _require(type(nucleus_p) in (int, float) and 0 < nucleus_p <= 1,
                 "nucleus_p must lie in(0,1]")
        _require(type(segment_batch_size) is int and segment_batch_size > 0,
                 "segment_batch_size must be positive")
        _require(type(strict_unary_validation) is bool, "strict_unary_validation must be bool")
        self.mask_id = mask_id
        self.device = torch.empty(0, device=device).device
        self.count_strength = _coefficient(count_strength)
        self.count_mode, self.smoothing = count_mode, float(smoothing)
        self.global_strength = _coefficient(global_strength)
        self.contextual_strength = _coefficient(contextual_strength)
        self.global_head, self.contextual_head = global_head, contextual_head
        self.sampling, self.backend = sampling, backend
        self.top_k, self.nucleus_p = top_k, float(nucleus_p)
        self.fast_candidates, self.prune_irrelevant = fast_candidates, prune_irrelevant
        self.segment_batch_size = segment_batch_size
        self.strict_unary_validation = strict_unary_validation
        self.active = bool(self.count_strength or self.global_strength or self.contextual_strength)
        for coefficient, head in ((self.global_strength, global_head),
                                  (self.contextual_strength, contextual_head)):
            if coefficient:
                _require(head is not None and not head.training
                    and all(not p.requires_grad and p.dtype == torch.float32 and p.device == self.device
                            for p in head.parameters()), "Active heads must be frozen FP32 on the decoder device")
        ranks = [h.rank for c, h in ((self.global_strength, global_head),
                 (self.contextual_strength, contextual_head)) if c]
        _require(sum(ranks) <= 128, "Combined learned rank exceeds fused kernel128")
        vocab = mask_id + 1
        if self.count_strength:
            _require(count_statistics is not None, "Nonzero count strength requires raw training counts")
            keys = ("left_counts", "right_counts", "pair_keys", "pair_counts")
            _require(set(count_statistics) == set(keys), "Unexpected count-statistics fields")
            arrays = [count_statistics[k].to(self.device) for k in keys]
            self.potential = SparseCountPotential.from_counts(*arrays, mode=count_mode,
                strength=self.count_strength, smoothing=smoothing)
            _require(self.potential.vocab_size == vocab, "Count and denoiser vocabulary differ")
        else:
            ones = torch.ones(vocab, dtype=torch.float64, device=self.device)
            self.potential = SparseCountPotential(ones, ones.clone(),
                torch.empty(0, dtype=torch.long, device=self.device), ones[:0], 0.)
        self.cache, self.prepared, self.closed = None, False, False
        self.setup_seconds = None

    @torch.no_grad()
    def prepare(self, **cache_budgets):
        """Build/prime count lookup once; time separately, never inside a step."""
        _require(not self.closed and not self.prepared, "Decoder already prepared or closed")
        synchronize(self.device)
        started = time.perf_counter()
        if self.active:
            if self.backend == "static":
                from . import static
                if self.global_strength:
                    self.cache = static.prepare_static_global_cache(self.potential,
                        self.global_head, self.global_strength, **cache_budgets)
                else:
                    self.cache = _CountCache(self.potential,
                        static.prepare_static_count_cache(self.potential, **cache_budgets))
            else:
                _require(not cache_budgets, "Cache budgets apply only to static setup")
                from ._count_inference import prepare_count_cache
                prepare_count_cache(self.potential)
        synchronize(self.device)
        self.setup_seconds = time.perf_counter() - started
        self.prepared = True
        return {"backend": self.backend, "active": self.active,
                "seconds": self.setup_seconds, "excluded_from_generation_timing": True,
                "static_table_prepared": self.cache is not None and self.cache.table is not None,
                "zero_global_fused_fallback": self.cache is not None and self.cache.table is None}

    def close(self):
        if self.cache is not None:
            self.cache.close()
        self.cache, self.closed = None, True
        if self.active and self.backend == "fused":
            from .cuda import clear_count_cache
            clear_count_cache()

    def _candidates(self, log_probs, tokens):
        extract = extract_compact_candidates
        if self.fast_candidates:
            from .fast_candidates import extract_compact_candidates as extract
        return extract(log_probs, tokens, mask_id=self.mask_id,
                       nucleus_p=self.nucleus_p, top_k=self.top_k)

    @torch.no_grad()
    def reveal(self, output, tokens, reveal_mask, generator):
        _require(self.prepared and not self.closed, "Call prepare() outside the timed loop")
        log_probs = output if isinstance(output, torch.Tensor) else output.log_probs
        hidden = None if isinstance(output, torch.Tensor) else output.hidden
        _require(log_probs.shape == (*tokens.shape, self.mask_id + 1)
                 and log_probs.dtype == torch.float32 and log_probs.device == tokens.device,
                 "Expected native FP32 log probabilities")
        # Candidate extraction already rejects NaN/+Inf and checks every used
        # masked row's clean normalizer. Do not duplicate a full [B,L,V] scan
        # and synchronization at every production step. The stricter option
        # also diagnoses invalid rows that this reveal does not use.
        if self.strict_unary_validation:
            _require(not bool(torch.isnan(log_probs).any() | torch.isposinf(log_probs).any()),
                     "Log probabilities may contain finite values or -inf only")
            supported = torch.isfinite(log_probs)
            supported[..., self.mask_id] = False
            _require(not bool((tokens.eq(self.mask_id) & ~supported.any(-1)).any()),
                     "Every masked position needs at least one finite clean token")
        _require(reveal_mask.dtype == torch.bool and reveal_mask.shape == tokens.shape
                 and reveal_mask.device == tokens.device, "Invalid reveal mask")
        selected = reveal_mask & tokens.eq(self.mask_id)
        _require(bool(selected.any()), "Empty reveals must skip forward/proposal")
        if not self.active:
            # Match the baseline's selected-row-only FP64 categorical draws.
            c = self._candidates(log_probs[selected][None], tokens[selected][None])
            slots = torch.multinomial(c.log_unary[0].exp(), 1, generator=generator)
            result = tokens.clone()
            result[selected] = c.candidate_ids[0].gather(-1, slots).squeeze(-1)
        else:
            context = relevant_span_context(tokens, selected, mask_id=self.mask_id) \
                if self.prune_irrelevant else tokens
            c = self._candidates(log_probs, context)
            terms = []
            if self.backend == "fused" and self.global_strength:
                terms.append((self.global_strength, self.global_head.node_factors(c.candidate_ids, hidden)))
            if self.contextual_strength:
                _require(hidden is not None and hidden.shape[:2] == tokens.shape,
                         "Contextual features must come from this same masked forward")
                terms.append((self.contextual_strength,
                              self.contextual_head.node_factors(c.candidate_ids, hidden)))
            left, right = concatenate_scaled_factors(terms) if terms else (None, None)
            settings = {"segment_batch_size": self.segment_batch_size}
            if self.backend == "static":
                from .static_contextual import sample_static_contextual, static_contextual_marginals
                sample = lambda: sample_static_contextual(c, self.potential, self.cache,
                    left, right, generator, **settings)
                marginals = lambda: static_contextual_marginals(c, self.potential, self.cache,
                    left, right, **settings)
            else:
                from .cuda import sample_additive_count, additive_count_marginals
                sample = lambda: sample_additive_count(c, self.potential, left, right, generator, **settings)
                marginals = lambda: additive_count_marginals(c, self.potential, left, right, **settings)
            if self.sampling == "joint":
                result = torch.where(selected, sample(), tokens)
            else:
                # Production chain marginals are FP32. Only selected sites draw.
                probability = marginals()[selected]
                _require(probability.dtype == torch.float32 and bool(torch.isfinite(probability).all())
                         and bool((probability >= 0).all()) and bool((probability.sum(-1) > 0).all()),
                         "Invalid own-chain marginals")
                slots = torch.multinomial(probability, 1, generator=generator)
                result = tokens.clone()
                result[selected] = c.candidate_ids[selected].gather(-1, slots).squeeze(-1)
        _require(not bool(result[selected].eq(self.mask_id).any())
                 and torch.equal(result[~selected], tokens[~selected]),
                 "Proposal must fill selected masks and preserve every other position")
        return result


@torch.no_grad()
def sample_window(tokens, forward, decoder, *, steps, reveal_generator, proposal_generator):
    """The original finite linear-mask schedule, with no extra cleanup pass.

    FP64 reveal uniforms are drawn for ALL positions at EVERY scheduled step.
    Only a nonempty reveal calls the frozen forward (once) and proposal. The
    final step reveals every remaining mask. Never consume proposal randomness
    in the reveal stream. Returned NFEs are observed, not assumed equal to T.
    """
    _require(tokens.ndim == 2 and tokens.dtype == torch.long and min(tokens.shape) > 0
             and tokens.device == decoder.device
             and bool(((tokens >= 0) & (tokens <= decoder.mask_id)).all()), "Invalid token window")
    _require(type(steps) is int and steps > 0, "steps must be positive")
    _require(isinstance(reveal_generator, torch.Generator)
             and isinstance(proposal_generator, torch.Generator)
             and reveal_generator is not proposal_generator, "Two distinct explicit generators required")
    _require(reveal_generator.device == tokens.device and proposal_generator.device == tokens.device,
             "Generators and tokens must share a device")
    current, counts, schedule, nfes = tokens.clone(), [], [], 0
    for index in range(steps):
        t, s = (steps - index) / steps, (steps - index - 1) / steps
        probability = 1. if index == steps - 1 else 1. - s / t
        uniform = torch.rand(tokens.shape, dtype=torch.float64,
                             device=tokens.device, generator=reveal_generator)
        reveal = (uniform < probability) & current.eq(decoder.mask_id)
        number = int(reveal.sum().item())
        if number:
            output = forward(current, t)
            nfes += 1
            current = decoder.reveal(output, current, reveal, proposal_generator)
        counts.append(number)
        schedule.append({"step": index + 1, "t": t, "s": s, "revealed": number,
                         "actual_forward_calls": nfes})
    _require(not bool(current.eq(decoder.mask_id).any())
             and torch.equal(current[tokens.ne(decoder.mask_id)], tokens[tokens.ne(decoder.mask_id)]),
             "Finite schedule left masks or changed visible context")
    return WindowResult(current, {"scheduled_steps": steps, "executed_steps": steps,
        "actual_forward_calls": nfes, "reveal_counts": counts, "schedule": schedule,
        "schedule_name": "linear_mask_T_plus_one_boundaries_no_cleanup"})


@torch.no_grad()
def generate_attempt(forward, decoder, tokenizer, *, attempt_id, stream_namespace,
                     steps=32, length=1024):
    """One timed raw attempt; retains rejected tokens and every EOS in text."""
    _require(decoder.prepared and not decoder.closed, "Prepare decoder before attempts")
    synchronize(decoder.device)
    started = time.perf_counter()
    seeds = {kind: stream_seed(stream_namespace, attempt_id, kind) for kind in ("reveal", "proposal")}
    generators = {kind: torch.Generator(device=decoder.device).manual_seed(seed)
                  for kind, seed in seeds.items()}
    tokens = initial_tokens(length=length, mask_id=decoder.mask_id,
        bos_token_id=tokenizer.bos_token_id, device=decoder.device)
    result = sample_window(tokens, forward, decoder, steps=steps,
        reveal_generator=generators["reveal"], proposal_generator=generators["proposal"])
    raw = result.tokens[0].cpu().tolist()
    # Match the native device's FP32 entropy check, before converting to CPU.
    _, counts = torch.unique(result.tokens[:, -256:], return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
    accepted = not entropy < 4.
    text = decode_retaining_eos(tokenizer, raw) if accepted else None
    diagnostics = token_diagnostics(raw, tokenizer.eos_token_id)
    synchronize(decoder.device)
    elapsed = time.perf_counter() - started
    return {"attempt_id": attempt_id, "accepted": accepted, "token_ids": raw,
            "text": text, "new_tokens": length - 1, "generation_seconds": elapsed,
            "last256_entropy_nats": entropy,
            "rejection_reason": None if accepted else "last256_entropy_below_4",
            "diagnostics": diagnostics, "stream_seeds": seeds, "window": result.stats}


def generate_saved(forward, decoder, tokenizer, *, samples, stream_namespace,
                   steps=32, length=1024, warmup=False):
    """Finite one-window run returning the portable evaluation JSON structure.

    Stops at the requested accepted count OR ten consecutive rejections for a
    sample. Warmup uses separate attempt1_000_000_000 and is excluded. Caller
    serializes the returned dictionary outside generation timing. This is not
    a durable/resumable job runner; exceptions propagate rather than inventing
    an accepted result or complete timing for interrupted work.
    """
    _require(type(samples) is int and samples > 0, "samples must be positive")
    warmup_seconds = None
    if warmup:
        warmup_seconds = generate_attempt(forward, decoder, tokenizer,
            attempt_id=1_000_000_000, stream_namespace=stream_namespace,
            steps=steps, length=length)["generation_seconds"]
    rows, accepted, consecutive = [], 0, 0
    while accepted < samples and consecutive < 10:
        row = generate_attempt(forward, decoder, tokenizer, attempt_id=len(rows),
            stream_namespace=stream_namespace, steps=steps, length=length)
        rows.append(row)
        if row["accepted"]:
            accepted, consecutive = accepted + 1, 0
        else:
            consecutive += 1
    return {"schema": "additive_chain_saved_generations_v1",
        "eos_token_id": tokenizer.eos_token_id, "samples_requested": samples,
        "generation_timing_complete": True, "setup_seconds": decoder.setup_seconds,
        "warmup_seconds": warmup_seconds, "attempts": rows,
        "status": "complete" if accepted == samples else "retry_limit",
        "generation_protocol": {"steps": steps, "length": length, "initial_context_tokens": 1,
            "stream_namespace": stream_namespace, "count_strength": decoder.count_strength,
            "count_mode": decoder.count_mode, "smoothing": decoder.smoothing,
            "global_strength": decoder.global_strength, "contextual_strength": decoder.contextual_strength,
            "sampling": decoder.sampling, "backend": decoder.backend, "top_k": decoder.top_k,
            "top_p": decoder.nucleus_p, "prune_irrelevant": decoder.prune_irrelevant,
            "fast_candidates": decoder.fast_candidates, "segment_batch_size": decoder.segment_batch_size,
            "strict_unary_validation": decoder.strict_unary_validation,
            "device": str(decoder.device), "torch_version": str(torch.__version__),
            "max_consecutive_rejections": 10}}
