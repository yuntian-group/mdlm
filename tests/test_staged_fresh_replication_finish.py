"""The replication finish workflow scores only newly selected seeds."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from deploy.gcloud import finish_staged_fresh_replication_v1 as finish


class ReplicationFinishTest(unittest.TestCase):
  def test_seal_all_runs_but_score_only_new_unique_checkpoint_files(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      files = []
      for seed in (2, 3):
        for arm, step in [('shared', 900), ('directional', 900), ('unary', 1000)]:
          files.append({'arm': arm, 'training_seed': seed,
                         'path': str(root / f'fresh-replication-seed{seed}-retry1-v1'
                                     / 'checkpoints' / f'step-{step:06d}.pt')})
      sidecar = {'selected_checkpoint_files': files, 'selection_file_sha256': 'a' * 64}
      result = ({'overall': {}}, {'training_seeds': [1, 2, 3], 'records': 4608})
      with mock.patch.object(finish, 'seal', return_value=({}, sidecar)) as seal, \
           mock.patch.object(finish, 'evaluate') as evaluate, \
           mock.patch.object(finish, 'aggregate', return_value=result) as aggregate, \
           mock.patch.object(finish, 'file_sha256', return_value='b' * 64), \
           mock.patch.dict(finish.os.environ), redirect_stdout(io.StringIO()):
        finish.main(['--experiment-root', str(root)])
      self.assertEqual(seal.call_count, 2)
      self.assertEqual([path.name for path in seal.call_args_list[0].args[0]],
                       ['fresh-replication-seed2-retry1-v1', 'fresh-replication-seed3-retry1-v1'])
      self.assertEqual([path.name for path in seal.call_args_list[1].args[0]],
                       ['fresh-pilot-resume-v1', 'fresh-replication-seed2-retry1-v1',
                        'fresh-replication-seed3-retry1-v1'])
      command = evaluate.call_args.args[0]
      selected = [command[index + 1] for index, item in enumerate(command) if item == '--selected-checkpoint']
      self.assertEqual(len(selected), 4)
      self.assertEqual(selected, sorted({item['path'] for item in files}))
      self.assertFalse(any('fresh-pilot-resume-v1' in path for path in selected))
      shards = aggregate.call_args.args[4]
      self.assertEqual(len(shards), 2)
      self.assertEqual(shards[0][1], finish.PILOT_MANIFEST_SHA)
      self.assertEqual(shards[0][3], finish.PILOT_TEST_SHA)
      self.assertEqual(shards[0][2], root / 'fresh-test-v1')
      self.assertEqual(aggregate.call_args.args[5], root / 'fresh-test-seed123-v1')

  def test_incomplete_training_stops_before_test_access(self):
    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(finish, 'seal', side_effect=ValueError('incomplete')) as seal, \
         mock.patch.object(finish, 'evaluate') as evaluate, \
         mock.patch.object(finish, 'aggregate') as aggregate, mock.patch.dict(finish.os.environ):
      with self.assertRaisesRegex(ValueError, 'incomplete'):
        finish.main(['--experiment-root', directory])
      self.assertEqual(seal.call_count, 1)
      evaluate.assert_not_called()
      aggregate.assert_not_called()

  def test_resumed_seed2_keeps_seed3_in_its_fresh_namespace(self):
    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(finish, 'seal', side_effect=ValueError('incomplete')) as seal, \
         mock.patch.object(finish, 'evaluate') as evaluate, mock.patch.dict(finish.os.environ):
      with self.assertRaisesRegex(ValueError, 'incomplete'):
        finish.main(['--experiment-root', directory, '--seed2-resumed'])
      self.assertEqual([path.name for path in seal.call_args.args[0]],
                       ['fresh-replication-seed2-resume-v1',
                        'fresh-replication-seed3-retry1-v1'])
      evaluate.assert_not_called()

  def test_existing_outputs_are_never_overwritten(self):
    with tempfile.TemporaryDirectory() as directory, mock.patch.object(finish, 'seal') as seal:
      (Path(directory) / 'fresh-selection-seed23-v1').mkdir()
      with self.assertRaises(FileExistsError):
        finish.main(['--experiment-root', directory])
      seal.assert_not_called()


if __name__ == '__main__':
  unittest.main()
