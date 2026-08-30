from pathlib import Path
import hashlib
import tempfile
import unittest

from safetensors.torch import load_file
import torch

from scripts.export_structured_adapter import (
  export_adapter,
  load_adapter_into_head,
  load_adapter_state,
)


class ExportStructuredAdapterTest(unittest.TestCase):

  def _sha256(self, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

  def _checkpoint(self, path: Path) -> dict[str, torch.Tensor]:
    adapter = {
      'weight': torch.arange(6.0).reshape(2, 3),
      'bias': torch.tensor([1.0, 2.0]),
    }
    torch.save({
      'global_step': 7,
      'state_dict': {
        'backbone.layer.weight': torch.ones(1),
        **{
          f'structured_head.{key}': value
          for key, value in adapter.items()
        },
      },
    }, path)
    return adapter

  def test_export_is_portable_and_strictly_rehydrates_head(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      expected = self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest_path = root / 'adapter-manifest.json'

      manifest = export_adapter(
        checkpoint,
        output,
        manifest_path,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7,
        expected_backbone_tensors=1)

      self.assertEqual(set(load_file(output)), set(expected))
      self.assertEqual(manifest['source_checkpoint_global_step'], 7)
      self.assertEqual(manifest['adapter_tensor_count'], 2)
      self.assertEqual(manifest['omitted_frozen_backbone_tensor_count'], 1)
      self.assertNotIn(str(root), manifest_path.read_text())
      head = torch.nn.Linear(3, 2)
      load_adapter_into_head(
        head, output, expected_sha256=manifest['adapter_sha256'])
      for key, value in expected.items():
        torch.testing.assert_close(head.state_dict()[key], value)

  def test_export_fails_closed_on_state_or_provenance_mismatch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest = root / 'manifest.json'
      with self.assertRaisesRegex(ValueError, 'global-step mismatch'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=8, expected_backbone_tensors=1)

      payload = torch.load(
        checkpoint, map_location='cpu', weights_only=False)
      payload['state_dict']['noise.weight'] = torch.ones(1)
      torch.save(payload, checkpoint)
      with self.assertRaisesRegex(ValueError, 'outside backbone'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7, expected_backbone_tensors=1)

  def test_hash_verification_and_overwrite_protection(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'last.ckpt'
      self._checkpoint(checkpoint)
      output = root / 'adapter.safetensors'
      manifest = root / 'manifest.json'
      export_adapter(
        checkpoint, output, manifest,
        expected_checkpoint_sha256=self._sha256(checkpoint),
        expected_global_step=7, expected_backbone_tensors=1)
      with self.assertRaisesRegex(FileExistsError, '--force'):
        export_adapter(
          checkpoint, output, manifest,
          expected_checkpoint_sha256=self._sha256(checkpoint),
          expected_global_step=7, expected_backbone_tensors=1)
      with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
        load_adapter_state(output, expected_sha256='0' * 64)

  def test_checkpoint_hash_is_checked_before_deserialization(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / 'untrusted.ckpt'
      checkpoint.write_bytes(b'not a pickle checkpoint')
      with self.assertRaisesRegex(ValueError, 'checkpoint SHA256 mismatch'):
        export_adapter(
          checkpoint,
          root / 'adapter.safetensors',
          root / 'manifest.json',
          expected_checkpoint_sha256='0' * 64,
          expected_global_step=7,
          expected_backbone_tensors=1)


if __name__ == '__main__':
  unittest.main()
