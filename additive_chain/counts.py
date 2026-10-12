"""Sparse count potentials for the optional additive CUDA backend.

Represent W[a,b] = exp(log_scale) * (left[a]*right[b] + correction[a,b]).
Only observed corrections are stored; no V-by-V tensor is allocated. This is
the exact powered smoothed conditional (or PMI) potential, not a low-rank
approximation to its exponential. Counts must come from training data only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SparseCountPotential:
    left: torch.Tensor
    right: torch.Tensor
    pair_keys: torch.Tensor
    row_values: torch.Tensor
    log_scale: float = 0.

    def __post_init__(self):
        if any(t.is_inference() for t in (self.left,self.right,self.pair_keys,self.row_values)):
            raise ValueError('Construct count statistics outside torch.inference_mode for mutation tracking')
        if (self.left.ndim != 1 or self.left.numel() < 1 or self.right.shape != self.left.shape
                or self.pair_keys.ndim != 1 or self.row_values.shape != self.pair_keys.shape
                or self.pair_keys.dtype != torch.long
                or any(t.dtype != torch.float64 for t in (self.left,self.right,self.row_values))
                or any(t.device != self.left.device for t in (self.right,self.pair_keys,self.row_values))):
            raise ValueError('Count background/corrections must be FP64, keys int64, on one device')
        if (not math.isfinite(self.log_scale)
                or any(not bool(t.isfinite().all()) for t in (self.left,self.right,self.row_values))
                or bool((self.left <= 0).any() | (self.right <= 0).any() | (self.row_values < 0).any())):
            raise ValueError('Count background must be positive; corrections nonnegative and finite')
        keys=self.pair_keys
        if bool((keys < 0).any() | (keys >= self.vocab_size**2).any()
                | (keys[1:] <= keys[:-1]).any()):
            raise ValueError('Pair keys must be sorted, unique, and in vocabulary range')
        if any(t.requires_grad for t in (self.left,self.right,self.row_values)):
            raise ValueError('Count potentials are frozen, not learned tensors')

    @property
    def vocab_size(self):
        return self.left.numel()

    def signature(self):
        """Cheap mutation/device guard for an amortized GPU lookup cache."""
        return (self.vocab_size, self.log_scale, tuple((id(t),t._version,t.device,t.dtype)
                for t in (self.left,self.right,self.pair_keys,self.row_values)))

    @classmethod
    def from_reference(cls, potential):
        """Read an existing sparse-plus-background potential without dense V².

        A compatibility input must expose left/right FP64 vectors, a CSR
        correction matrix ``sparse``, and finite ``log_scale``. Its CSR entries
        must agree with optional sorted ``pair_keys``. No count normalization
        or strength change happens here.
        """
        if isinstance(potential, cls):
            return potential
        csr=potential.sparse
        if csr.layout != torch.sparse_csr:
            raise ValueError('Reference corrections must be CSR')
        rows=torch.repeat_interleave(torch.arange(potential.vocab_size,device=potential.left.device),
            csr.crow_indices()[1:]-csr.crow_indices()[:-1],output_size=csr.values().numel())
        keys=rows*potential.vocab_size+csr.col_indices()
        if hasattr(potential,'pair_keys') and not torch.equal(potential.pair_keys,keys):
            raise ValueError('Reference keys disagree with CSR correction order')
        return cls(potential.left,potential.right,keys,csr.values(),potential.log_scale)

    @classmethod
    def from_counts(cls, left_counts, right_counts, pair_keys, pair_counts,
                    *, mode='conditional', strength=1., smoothing=.1):
        """Build the established add-one-marginal/independence-backoff model.

        N=sum counts; bL=(cL+1)/(N+V), bR=(cR+1)/(N+V).
        J(a,b)=(1-eps)*c(a,b)/N + eps*bL(a)*bR(b).
        Conditional: W=[J/JL]^strength. PMI: W=[J/(JL*JR)]^strength.
        For N=0 set eps=1. Floating dynamic-range failure raises; no clipping.
        """
        if mode not in ('conditional','pmi'):
            raise ValueError('mode must be conditional or pmi')
        if isinstance(strength,bool) or not math.isfinite(strength) or strength < 0:
            raise ValueError('strength must be finite and nonnegative')
        if isinstance(smoothing,bool) or not math.isfinite(smoothing) or not 0 < smoothing <= 1:
            raise ValueError('smoothing must lie in (0,1]')
        if pair_keys.dtype != torch.long:
            raise ValueError('Pair keys must be int64')
        lc=left_counts.detach().to(dtype=torch.float64).clone()
        rc=right_counts.detach().to(device=lc.device,dtype=torch.float64).clone()
        keys=pair_keys.detach().to(device=lc.device).clone()
        cnt=pair_counts.detach().to(device=lc.device,dtype=torch.float64).clone()
        if lc.ndim != 1 or lc.numel() == 0 or rc.shape != lc.shape:
            raise ValueError('Endpoint counts must have the same nonempty vector shape')
        if keys.ndim != 1 or cnt.shape != keys.shape:
            raise ValueError('Pair keys/counts must have equal vector shapes')
        if any(not bool(t.isfinite().all()) or bool((t < 0).any()) for t in (lc,rc,cnt)):
            raise ValueError('Counts must be finite and nonnegative')
        vocab=lc.numel()
        if bool((keys < 0).any() | (keys >= vocab*vocab).any()
                | (keys[1:] <= keys[:-1]).any() | (cnt <= 0).any()):
            raise ValueError('Pair keys must be sorted/unique/in range with positive counts')
        rows,cols=keys//vocab,keys%vocab
        observed_l,observed_r=torch.zeros_like(lc),torch.zeros_like(rc)
        observed_l.index_add_(0,rows,cnt)
        observed_r.index_add_(0,cols,cnt)
        if not torch.allclose(lc,observed_l) or not torch.allclose(rc,observed_r):
            raise ValueError('Endpoint counts must be marginals of pair counts')
        n=float(lc.sum())
        eps=float(smoothing) if n else 1.
        b_l,b_r=(lc+1.)/(n+vocab),(rc+1.)/(n+vocab)
        empirical_scale=(1.-eps)/max(n,1.)
        m_l,m_r=empirical_scale*lc+eps*b_l,empirical_scale*rc+eps*b_r
        log_left=strength*(math.log(eps)+b_l.log()-m_l.log())
        log_right=strength*b_r.log()
        if mode=='pmi':
            log_right=log_right-strength*m_r.log()
        log_background=log_left[rows]+log_right[cols]
        if keys.numel() and empirical_scale > 0 and strength > 0:
            log_ratio=(math.log(empirical_scale)+cnt.log()-math.log(eps)
                       -b_l[rows].log()-b_r[cols].log())
            delta=strength*torch.nn.functional.softplus(log_ratio)
            log_correction=log_background+delta+torch.log(-torch.expm1(-delta))
            observed_max=(log_background+delta).max()
        else:
            log_correction=torch.full_like(cnt,-torch.inf)
            observed_max=lc.new_tensor(-torch.inf)
        scale=float(torch.maximum(log_left.max()+log_right.max(),observed_max))
        pivot=float(log_left.max())
        left=(log_left-pivot).exp()
        right=(log_right+pivot-scale).exp()
        correction=(log_correction-scale).exp()
        keep=correction > 0
        return cls(left,right,keys[keep],correction[keep],scale)

    def entries(self, rows, columns):
        """Scale-free pair weights for broadcastable valid int64 token IDs."""
        if (rows.dtype != torch.long or columns.dtype != torch.long
                or rows.device != self.left.device or columns.device != self.left.device
                or bool((rows < 0).any() | (rows >= self.vocab_size).any()
                        | (columns < 0).any() | (columns >= self.vocab_size).any())):
            raise ValueError('Lookup requires valid int64 token IDs on the potential device')
        values=self.left[rows]*self.right[columns]
        if self.pair_keys.numel():
            query=rows*self.vocab_size+columns
            index=torch.searchsorted(self.pair_keys,query.contiguous()).clamp_max(self.pair_keys.numel()-1)
            values=values+torch.where(self.pair_keys[index]==query,self.row_values[index],0.)
        return values

    def log_scores(self, rows, columns):
        return self.entries(rows,columns).log()+self.log_scale
