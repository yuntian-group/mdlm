import pytest

from additive_chain.iwslt_readout import paired_difference


def exact_match(hypotheses, references):
    return sum(a == b for a, b in zip(hypotheses, references)) / len(references)


def test_identical_predictions_have_zero_paired_uncertainty():
    rows = [dict(id=str(i), reference="yes", hypothesis="yes" if i % 2 else "no") for i in range(20)]
    result = paired_difference(rows, rows, exact_match, replicates=100)
    assert result["joint_minus_independent"] == 0
    assert result["interval_95"] == [0, 0]


def test_pairing_rejects_different_order_and_references():
    rows = [dict(id="a", reference="yes", hypothesis="no"), dict(id="b", reference="no", hypothesis="no")]
    with pytest.raises(ValueError, match="Matched IDs"):
        paired_difference(rows, rows[::-1], exact_match)
    with pytest.raises(ValueError, match="Matched IDs"):
        paired_difference(rows, [rows[0], dict(rows[1], reference="other")], exact_match)


def test_paired_difference_sign_and_reproducibility():
    a = [dict(id=str(i), reference="yes", hypothesis="yes") for i in range(20)]
    b = [dict(r, hypothesis="no") for r in a]
    result = paired_difference(a, b, exact_match, replicates=100)
    assert result == paired_difference(a, b, exact_match, replicates=100)
    assert result["joint_minus_independent"] == 1
    assert result["interval_95"] == [1, 1]
