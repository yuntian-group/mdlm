from collections import Counter

import pytest

from additive_chain.iwslt_residual_train import TrainingConfig, epoch_groups, update_plateau


def test_epoch_covers_every_row_and_preserves_long_examples():
    rows = [dict(source_ids=[0] * (i % 17 + 1), target_ids=[0] * (i % 90 + 1)) for i in range(1073)]
    rows.append(dict(source_ids=[1], target_ids=[1] * 407))
    config = TrainingConfig(microbatch=8, target_token_budget=128)
    groups = epoch_groups(rows, 1, config)
    observed = [i for group in groups for batch in group for i in batch]
    assert Counter(observed) == Counter(range(len(rows)))
    assert groups == epoch_groups(rows, 1, config)
    assert groups != epoch_groups(rows, 2, config)
    for group in groups:
        assert len(group) <= config.accumulate
        for batch in group:
            assert len(batch) <= config.microbatch
            assert len(batch) == 1 or len(batch) * max(len(rows[i]["target_ids"]) + 1
                                                     for i in batch) <= config.target_token_budget


def test_plateau_and_epoch_zero_selection():
    state = dict(best_dev=None, bad_checks=0, history=[], update=0)
    assert update_plateau(state, {"nll_per_selected_token": 2.}, 0)
    assert not update_plateau(state, {"nll_per_selected_token": 2.1}, 1)
    assert state["bad_checks"] == 1
    assert update_plateau(state, {"nll_per_selected_token": 1.9}, 1)
    assert state["bad_checks"] == 0


def test_training_configuration_bounds():
    TrainingConfig().validate()
    for changed in (dict(rank=128), dict(max_epochs=0), dict(dev_examples=8000),
                    dict(learning_rate=float("nan")), dict(kind="other")):
        with pytest.raises(ValueError):
            TrainingConfig(**changed).validate()
