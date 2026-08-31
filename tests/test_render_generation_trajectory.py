import unittest

from scripts.render_generation_trajectory import _decode_segments


class _Tokenizer:

  def decode(self, token_ids, **unused_kwargs):
    return ''.join(chr(96 + value) for value in token_ids)


class RenderGenerationTrajectoryTest(unittest.TestCase):

  def test_consecutive_masks_are_counted_and_resolved_runs_are_decoded(self):
    rendered = _decode_segments(
      _Tokenizer(), [1, 2, 3, 4, 5], [False, True, True, False, False])
    self.assertEqual(rendered, 'a ⟦2 masked⟧ de')

  def test_all_resolved_tokens_decode_as_one_run(self):
    rendered = _decode_segments(
      _Tokenizer(), [1, 2, 3], [False, False, False])
    self.assertEqual(rendered, 'abc')


if __name__ == '__main__':
  unittest.main()
