"""Utilities for first-order CRF inference.

Provides the forward-backward algorithm for computing per-position
marginals of a first-order chain, used during CRF-MDLM sampling.
"""

import torch
import torch.nn.functional as F


def forward_backward(emission_0, transitions):
  """Forward-backward algorithm for first-order CRF marginals.

  Computes exact per-position marginals in log-space for
  numerical stability.

  Args:
    emission_0: (batch, K) log probs at position 0.
        emission_0[:, k] = log P(x_{0,0} = candidate k | START, x_t)
    transitions: (batch, N-1, K, K) log transition probs.
        transitions[:, j, u, v] =
          log P(pos j+1 = v-th candidate | pos j = u-th candidate)

  Returns:
    marginals: (batch, N, K) per-position marginal probabilities
        normalised over the K candidates at each position.
  """
  batch, N_minus_1, K, _ = transitions.shape
  N = N_minus_1 + 1
  device = emission_0.device
  dtype = emission_0.dtype

  # --- Forward pass ---
  # alpha[i, k] = log sum_{x_{0:i-1}} P(x_{0:i}, x_{0,i}=k | x_t)
  log_alphas = []
  log_alpha = emission_0                           # (batch, K)
  log_alphas.append(log_alpha)

  for j in range(N_minus_1):
    # log_alpha:              (batch, K_prev)
    # transitions[:, j]:      (batch, K_prev, K_curr)
    # result:                 (batch, K_curr)
    T = transitions[:, j]
    log_alpha = torch.logsumexp(
      log_alpha.unsqueeze(-1) + T, dim=1)
    log_alphas.append(log_alpha)

  log_alphas = torch.stack(log_alphas, dim=1)       # (batch, N, K)

  # --- Backward pass ---
  # beta[i, k] = log sum_{x_{i+1:N-1}} P(x_{i+1:N-1} | x_{0,i}=k)
  log_betas_rev = []
  log_beta = torch.zeros(
    batch, K, device=device, dtype=dtype)
  log_betas_rev.append(log_beta)

  for j in range(N_minus_1 - 1, -1, -1):
    T = transitions[:, j]                           # (batch, K_prev, K_curr)
    # For each u (K_prev), sum over v (K_curr):
    #   T[u, v] + beta[v]
    log_beta = torch.logsumexp(
      T + log_beta.unsqueeze(1), dim=-1)            # (batch, K_prev)
    log_betas_rev.append(log_beta)

  log_betas_rev.reverse()
  log_betas = torch.stack(log_betas_rev, dim=1)     # (batch, N, K)

  # --- Marginals ---
  # P(x_{0,i} = k) proportional to alpha[i,k] * beta[i,k]
  log_marginals = log_alphas + log_betas
  marginals = F.softmax(log_marginals, dim=-1)

  return marginals
