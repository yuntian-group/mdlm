import importlib.util
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def _load_dataloader_module():
  """Load dataloader.py without requiring Lightning/timm in unit tests."""
  fake_utils = types.ModuleType('utils')
  fake_utils.get_logger = logging.getLogger
  fake_utils.fsspec_exists = lambda _path: False
  fake_utils.fsspec_mkdirs = lambda _path, exist_ok=True: None
  module_path = Path(__file__).resolve().parents[1] / 'dataloader.py'
  spec = importlib.util.spec_from_file_location(
    '_streaming_test_dataloader', module_path)
  module = importlib.util.module_from_spec(spec)
  missing = object()
  prior_utils = sys.modules.get('utils', missing)
  sys.modules['utils'] = fake_utils
  try:
    with mock.patch.object(
        os, 'sched_getaffinity', create=True, return_value={0}):
      spec.loader.exec_module(module)
  finally:
    if prior_utils is missing:
      del sys.modules['utils']
    else:
      sys.modules['utils'] = prior_utils
  return module


dataloader = _load_dataloader_module()
import torch


def _load_diffusion_module():
  """Load Diffusion.on_train_start without the optional Lightning stack."""
  fake_hydra = types.ModuleType('hydra')
  fake_hydra.__path__ = []
  fake_hydra_utils = types.ModuleType('hydra.utils')
  fake_hydra.utils = fake_hydra_utils

  fake_lightning = types.ModuleType('lightning')
  fake_lightning.LightningModule = object

  fake_torchmetrics = types.ModuleType('torchmetrics')
  fake_torchmetrics.Metric = object
  fake_torchmetrics.MetricCollection = object
  fake_torchmetrics.aggregation = types.SimpleNamespace(MeanMetric=object)

  fake_utils = types.ModuleType('utils')
  fake_utils.fsspec_exists = lambda _path: False

  replacements = {
    'dataloader': dataloader,
    'hydra': fake_hydra,
    'hydra.utils': fake_hydra_utils,
    'lightning': fake_lightning,
    'torchmetrics': fake_torchmetrics,
    'utils': fake_utils,
  }
  missing = object()
  prior = {name: sys.modules.get(name, missing)
           for name in replacements}
  sys.modules.update(replacements)
  module_path = Path(__file__).resolve().parents[1] / 'diffusion.py'
  spec = importlib.util.spec_from_file_location(
    '_streaming_test_diffusion', module_path)
  module = importlib.util.module_from_spec(spec)
  try:
    spec.loader.exec_module(module)
  finally:
    for name, prior_module in prior.items():
      if prior_module is missing:
        del sys.modules[name]
      else:
        sys.modules[name] = prior_module
  return module


class _CapturingDataLoader:
  def __init__(self, dataset, **kwargs):
    self.dataset = dataset
    self.kwargs = kwargs
    self.tokenizer = None


def _config(*, streaming, valid='wikitext103'):
  return types.SimpleNamespace(
    data=types.SimpleNamespace(
      train='openwebtext',
      valid=valid,
      wrap=True,
      cache_dir='/unused-cache',
      streaming=streaming),
    loader=types.SimpleNamespace(
      global_batch_size=2,
      eval_global_batch_size=2,
      batch_size=2,
      eval_batch_size=2,
      num_workers=0,
      pin_memory=False),
    trainer=types.SimpleNamespace(
      num_nodes=1,
      accumulate_grad_batches=1),
    model=types.SimpleNamespace(length=1024))


class DataloaderStreamingTest(unittest.TestCase):

  def _get_dataloaders(self, config, **kwargs):
    calls = []

    def fake_get_dataset(dataset_name, tokenizer, **dataset_kwargs):
      calls.append((dataset_name, dataset_kwargs))
      return [0, 1]

    with mock.patch.object(
        dataloader, 'get_dataset', side_effect=fake_get_dataset), \
        mock.patch.object(
          dataloader.torch.cuda, 'device_count', return_value=1), \
        mock.patch.object(
          dataloader.torch.utils.data,
          'DataLoader', _CapturingDataLoader):
      loaders = dataloader.get_dataloaders(
        config, tokenizer=object(), **kwargs)
    return loaders, calls

  def test_training_streaming_flag_is_propagated_and_disables_shuffle(self):
    (train_loader, _), calls = self._get_dataloaders(
      _config(streaming=True), skip_valid=True)

    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][1]['mode'], 'train')
    self.assertIs(calls[0][1]['streaming'], True)
    self.assertIs(train_loader.kwargs['shuffle'], False)
    self.assertIsInstance(
      train_loader.dataset, torch.utils.data.IterableDataset)

    (train_loader, _), calls = self._get_dataloaders(
      _config(streaming=False), skip_valid=True)
    self.assertIs(calls[0][1]['streaming'], False)
    self.assertIs(train_loader.kwargs['shuffle'], True)

  def test_validation_and_test_splits_are_always_finite_map_style(self):
    for valid_name, expected_mode in (
        ('wikitext103', 'validation'),
        ('text8', 'test'),
        ('lm1b', 'test'),
        ('ag_news', 'test')):
      with self.subTest(valid_name=valid_name):
        (_, valid_loader), calls = self._get_dataloaders(
          _config(streaming=True, valid=valid_name), skip_train=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['mode'], expected_mode)
        self.assertIs(calls[0][1]['streaming'], False)
        self.assertIs(valid_loader.kwargs['shuffle'], False)

  def test_streaming_dataset_never_reuses_or_writes_map_style_cache(self):
    class FakeStreamingDataset:
      def __init__(self):
        self.map_calls = []
        self.saved_paths = []

      def map(self, function, **kwargs):
        self.map_calls.append((function, kwargs))
        return self

      def remove_columns(self, columns):
        return self

      def save_to_disk(self, path):
        self.saved_paths.append(path)

      def with_format(self, format_name):
        self.format_name = format_name
        return self

    class FakeTokenizer:
      bos_token = '[BOS]'
      eos_token = '[EOS]'

      def encode(self, token):
        return [1]

    stream = FakeStreamingDataset()
    with mock.patch.object(
        dataloader.utils, 'fsspec_exists', return_value=True), \
        mock.patch.object(
          dataloader.datasets, 'load_from_disk') as load_from_disk, \
        mock.patch.object(
          dataloader.datasets, 'load_dataset',
          return_value={'train': stream}) as load_dataset:
      result = dataloader.get_dataset(
        'openwebtext',
        tokenizer=FakeTokenizer(),
        wrap=True,
        mode='train',
        cache_dir='/unused-cache',
        streaming=True)

    self.assertIs(result, stream)
    load_from_disk.assert_not_called()
    load_dataset.assert_called_once_with(
      'openwebtext', cache_dir='/unused-cache', streaming=True)
    self.assertEqual(stream.saved_paths, [])
    self.assertEqual(len(stream.map_calls), 2)
    self.assertEqual(stream.format_name, 'torch')

  def test_on_train_start_preserves_streaming_loader(self):
    diffusion = _load_diffusion_module()

    class SourceDataset:
      def __iter__(self):
        yield {'input_ids': torch.tensor([1, 2])}
        yield {'input_ids': torch.tensor([3, 4])}

    stream = dataloader._TorchIterableDatasetAdapter(SourceDataset())
    original_loader = torch.utils.data.DataLoader(
      stream, batch_size=1, num_workers=0)
    combined_loader = types.SimpleNamespace(
      flattened=[original_loader])
    trainer = types.SimpleNamespace(
      _accelerator_connector=types.SimpleNamespace(
        use_distributed_sampler=False,
        is_distributed=False),
      fit_loop=types.SimpleNamespace(
        _combined_loader=combined_loader))

    model = diffusion.Diffusion.__new__(diffusion.Diffusion)
    model.ema = None
    model.trainer = trainer
    model.on_train_start()

    self.assertIs(combined_loader.flattened[0], original_loader)
    batch = next(iter(combined_loader.flattened[0]))
    torch.testing.assert_close(
      batch['input_ids'], torch.tensor([[1, 2]]))


if __name__ == '__main__':
  unittest.main()
