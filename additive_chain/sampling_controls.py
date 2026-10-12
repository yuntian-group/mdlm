"""Cheap independent controls applied AFTER diffusion candidate selection.

These are decoding controls, not learned unary heads or chain marginals. The
candidate extractor must run on the original diffusion probabilities first.
No count-pair table, dynamic program, or additional backbone pass is needed.
"""
from __future__ import annotations

import hashlib
import math

import torch


class IndependentControl:
    """Frequency and/or temperature correction on unchanged candidate support.

    For original sequence position i, the categorical weights are
    exp((u_i(v) + alpha * 1[i>0] * log J_R(v)) / temperature).
    J_R is the destination marginal returned by smoothed_unigram_log_probs,
    not simply the add-one backoff distribution. The first ORIGINAL position
    has no incoming edge; the first selected position usually does.

    The individual frequency and temperature laws reproduce the independent
    experiment adapters. Their explicit composition is supported but was not
    measured in those experiments. Identity settings use the baseline exp()
    path exactly, without a second normalization or any extra random draw.
    """
    def __init__(self, *, frequency_strength=0., log_marginal=None, temperature=1.):
        if (type(frequency_strength) not in (int, float)
                or not math.isfinite(frequency_strength) or frequency_strength < 0):
            raise ValueError('Frequency strength must be finite and nonnegative')
        if (type(temperature) not in (int, float)
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError('Temperature must be finite and positive')
        if frequency_strength and log_marginal is None:
            raise ValueError('Positive frequency strength requires log J_R')
        if log_marginal is not None:
            if (not isinstance(log_marginal, torch.Tensor) or log_marginal.ndim != 1
                    or not log_marginal.numel() or log_marginal.dtype != torch.float64
                    or not bool(torch.isfinite(log_marginal).all())
                    or abs(float(log_marginal.logsumexp(0))) > 1e-10):
                raise ValueError('Expected a finite normalized FP64 log marginal[V]')
            # Own the static input: caller mutation cannot silently change the law.
            # Keep a versioned tensor even if the caller prepares inference
            # inside torch.inference_mode(); ordinary inference tensors have
            # no version counter for the mutation guard below.
            with torch.inference_mode(False):
                log_marginal = log_marginal.detach().clone()
        self.frequency_strength = float(frequency_strength)
        self.temperature = float(temperature)
        self._log_marginal = log_marginal
        self._marginal_version = None if log_marginal is None else log_marginal._version
        # Bind the caller-supplied vector even outside the count-file CLI. This
        # copies only V FP64 values to CPU, once, before generation is timed.
        self._marginal_identity = None if log_marginal is None else dict(
            sha256=hashlib.sha256(log_marginal.contiguous().cpu().numpy().tobytes()).hexdigest(),
            dtype=str(log_marginal.dtype), shape=list(log_marginal.shape))

    @property
    def active(self):
        return bool(self.frequency_strength or self.temperature != 1.)

    def validate_for(self, *, vocab_size, device):
        """Validate static metadata once, outside the decoding loop."""
        if self._log_marginal is not None:
            if (self._log_marginal.numel() != vocab_size
                    or self._log_marginal.device != torch.device(device)):
                raise ValueError('Count marginal must match decoder vocabulary and device')

    def identity(self):
        return dict(frequency_strength=self.frequency_strength, temperature=self.temperature,
            log_marginal=None if self._marginal_identity is None else dict(
                self._marginal_identity, shape=list(self._marginal_identity['shape'])),
            support='original diffusion nucleus then candidate cap; never recomputed after correction',
            law='q_i(v) proportional to exp((u_i(v) + alpha*1[i>0]*log J_R(v))/temperature)',
            incoming_edge='original sequence position i>0, not packed selected-row index',
            pair_inference=False, combined_control_measured=False)

    def probabilities(self, candidates, selected):
        """Transform authenticated extractor output, preserving IDs and padding.

        candidates contains flattened selected rows [1,N,K], in boolean-index
        order from selected[B,L]. This trusted hot path checks tensor metadata
        but does not rescan the vocabulary or synchronize scalar CUDA checks.
        Candidate validity is established by the decoder's extractor.
        """
        unary, ids = candidates.log_unary, candidates.candidate_ids
        if (selected.ndim != 2 or selected.dtype != torch.bool
                or unary.ndim != 3 or unary.shape[0] != 1 or unary.dtype != torch.float64
                or ids.shape != unary.shape or ids.dtype != torch.long
                or unary.device != ids.device or unary.device != selected.device):
            raise ValueError('Expected FP64 selected candidates[1,N,K] and bool selected[B,L]')
        if not self.active:
            return unary[0].exp()
        adjusted = unary
        if self.frequency_strength:
            marginal = self._log_marginal
            if (marginal.device != unary.device or marginal._version != self._marginal_version):
                raise ValueError('Prepared count marginal changed or is on another device')
            positions = torch.arange(selected.shape[1], device=selected.device)[None]
            incoming = positions.expand_as(selected).gt(0)[selected][None]
            if incoming.shape != unary.shape[:2]:
                raise ValueError('Candidates must contain exactly the selected positions')
            vocab = marginal.numel()
            valid = ids.ge(0) & ids.lt(vocab)
            correction = self.frequency_strength * marginal[ids.clamp(0, vocab - 1)]
            correction = torch.where(valid & incoming[..., None], correction, 0.)
            adjusted = unary + correction.to(unary.dtype)
        if self.temperature != 1.:
            adjusted = adjusted / self.temperature
        return adjusted[0].softmax(-1)
