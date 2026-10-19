import pytest

from additive_chain.iwslt_diagnostics import diagnostics


def row(identifier="a", length=6, generated=5):
    return dict(id=identifier, predicted_length=length, token_ids=list(range(generated)),
                hypothesis="a b c d a b c d", reference="a b c d")


def test_length_counts_include_eos_and_keep_every_prediction():
    predictions = [row(), row("b", length=3, generated=3), row("c", length=8, generated=2)]
    data = [dict(id=r["id"], source="german", target_ids=[1, 2, 3, 4, 5]) for r in predictions]
    result = diagnostics(predictions, data)
    assert len(result["examples"]) == 3
    assert result["summary"]["mean_reference_tokens"] == 6
    assert result["summary"]["no_eos"] == 1
    assert result["summary"]["early_eos"] == 1
    assert result["summary"]["mean_absolute_length_error"] == pytest.approx(5 / 3)
    assert result["examples"][0]["repeated_4grams"] == 1


def test_mismatched_or_duplicate_ids_and_invalid_lengths_are_rejected():
    data = [dict(id="a", source="german", target_ids=[1])]
    with pytest.raises(KeyError):
        diagnostics([row("b")], data)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics([row(), row()], data)
    with pytest.raises(ValueError, match="Invalid"):
        diagnostics([row(length=1, generated=3)], data)
