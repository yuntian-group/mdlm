from types import SimpleNamespace

import numpy as np
import pytest

from additive_chain.iwslt_readout import paired_difference
from additive_chain.iwslt_test_readout import paired_seed_metric


class Accuracy:
    def _extract_corpus_statistics(self, hypotheses, references):
        return [[int(a == b), 1] for a, b in zip(hypotheses, references[0])]

    def _compute_score_from_stats(self, stats):
        return SimpleNamespace(score=stats[0] / stats[1])


def examples(correct):
    return [dict(id=str(i), reference='yes', hypothesis='yes' if ok else 'no') for i, ok in enumerate(correct)]


def test_bootstrap_is_paired_over_sentences_not_seed_copies():
    a, b = examples([1, 0, 1, 1, 0]), examples([0, 0, 1, 0, 1])
    one = paired_seed_metric([a], [b], Accuracy(), replicates=100)
    three = paired_seed_metric([a, a, a], [b, b, b], Accuracy(), replicates=100)
    np.testing.assert_allclose(one['interval_95'], three['interval_95'])
    assert three['examples'] == 5 and three['decoding_seeds'] == 3
    brute = paired_difference(a, b, lambda h, r: sum(x == y for x, y in zip(h, r)) / len(r), replicates=100)
    np.testing.assert_allclose(one['interval_95'], brute['interval_95'])


def test_opposite_seed_effects_cancel_using_shared_sentence_indices():
    a, b = examples([1, 0, 1]), examples([0, 1, 0])
    result = paired_seed_metric([a, b], [b, a], Accuracy(), replicates=100)
    assert result['delta'] == 0 and result['interval_95'] == [0, 0]


def test_rejects_unpaired_seeds_or_sentence_order():
    a = examples([1, 0, 1])
    with pytest.raises(ValueError):
        paired_seed_metric([a], [], Accuracy())
    with pytest.raises(ValueError, match='ordering'):
        paired_seed_metric([a, a], [a, a[::-1]], Accuracy())


def test_cached_sacrebleu_matches_direct_corpus_bootstrap():
    metrics = pytest.importorskip('sacrebleu.metrics')
    a = [dict(id='1', reference='this is a simple test sentence', hypothesis='this is a small test sentence'),
         dict(id='2', reference='a second example is here', hypothesis='another example is here'),
         dict(id='3', reference='the cat sat on the mat', hypothesis='the cat sat on the mat')]
    b = [dict(r, hypothesis='this example is not the same') for r in a]
    for metric in [metrics.BLEU(lowercase=True, tokenize='13a'), metrics.CHRF()]:
        cached = paired_seed_metric([a], [b], metric, replicates=100)
        direct = paired_difference(a, b, lambda h, r: metric.corpus_score(h, [r]).score, replicates=100)
        np.testing.assert_allclose(cached['delta'], direct['joint_minus_independent'], atol=1e-10)
        np.testing.assert_allclose(cached['interval_95'], direct['interval_95'], atol=1e-10)
