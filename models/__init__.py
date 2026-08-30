import importlib

from . import dit
from . import ema
from . import crf_decoder


def __getattr__(name):
  """Load optional backbones only when selected.

  In particular, the autoregressive backbone still requires FlashAttention,
  but importing the CRF backbone no longer imports it as a side effect.
  """
  if name in {'dimamba', 'autoregressive'}:
    module = importlib.import_module(f'{__name__}.{name}')
    globals()[name] = module
    return module
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
