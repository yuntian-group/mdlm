"""Additive token-chain models with exact dense reference inference."""

from .chain import (ChainMarginals, joint_nll, log_partition, marginals,
                    path_score, sample_ffbs)
from .composition import compose_pair_scores, factors_to_pair_scores, sample_tokens
from .heads import (ContextualAdditiveHead, GlobalAdditiveHead,
                    concatenate_scaled_factors)
from .independent import (IndependentAdditiveHead, add_independent_residual,
                          choose_width, parameter_count)
from .training import (BlockRevealLoss, TrainingSupport, block_reveal_nll,
                       prepare_training_support)

__all__ = [
    'BlockRevealLoss', 'ChainMarginals', 'ContextualAdditiveHead',
    'GlobalAdditiveHead', 'IndependentAdditiveHead', 'TrainingSupport',
    'add_independent_residual', 'block_reveal_nll', 'choose_width',
    'compose_pair_scores', 'concatenate_scaled_factors',
    'factors_to_pair_scores', 'joint_nll', 'log_partition', 'marginals',
    'parameter_count', 'path_score', 'prepare_training_support',
    'sample_ffbs', 'sample_tokens',
]
