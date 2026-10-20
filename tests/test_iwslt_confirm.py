import json
import math

import pytest

from additive_chain.backbone import file_sha256
from additive_chain.iwslt_confirm import (atomic_json, cell_name, check_output,
    dev_cells, select_settings, test_cells as make_test_cells)


def development():
    return [dict(c, bleu=(20 if c['strength'] == .1 else 18) +
                 (1 if c['sampling'] == 'own-marginal' else 0))
            for step in (8, 32) for c in dev_cells(step)]


def test_shortlist_is_complete_and_unique():
    cells = development()
    assert len(cells) == len(set(cell_name(c) for c in cells)) == 12
    assert {c['seed'] for c in cells} == {101}
    for bad in (cells[:-1], cells + [cells[0]], cells[:-1] + [cells[0]]):
        with pytest.raises(ValueError, match='grid'):
            select_settings(bad)


def test_shared_selection_uses_both_samplers_not_best_joint():
    cells = development()
    for c in cells:
        if c['strength'] == .2:
            c['bleu'] = 25 if c['sampling'] == 'joint' else 10
    chosen = select_settings(cells)
    assert chosen['8']['strength'] == chosen['32']['strength'] == .1
    assert chosen['8']['baseline_temperature'] == .85
    cells[0]['bleu'] = math.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        select_settings(cells)


def test_test_grid_uses_frozen_settings_and_all_three_seeds():
    frozen = dict(selected=select_settings(development()), test_seeds=[101, 202, 303])
    for step in (8, 32):
        cells = make_test_cells(frozen, step)
        assert len(cells) == 9
        for seed in (101, 202, 303):
            group = [c for c in cells if c['seed'] == seed]
            assert len(group) == 3
            a, b = [c for c in group if c['strength'] > 0]
            assert {k: v for k, v in a.items() if k != 'sampling'} == {k: v for k, v in b.items() if k != 'sampling'}
    with pytest.raises(ValueError, match='Unregistered'):
        make_test_cells(dict(frozen, test_seeds=[101]), 8)


def test_output_check_rejects_missing_rows_and_modified_bytes(tmp_path):
    cell = dev_cells(8)[0]
    name = cell_name(cell)
    identity = dict(cell, ids=['one', 'two'])
    output = tmp_path / (name + '.json')
    receipt = tmp_path / (name + '-identity.json')
    payload = dict(predictions=[dict(id='one'), dict(id='two')], summary=dict(examples=2))
    atomic_json(output, payload)
    atomic_json(receipt, dict(identity=identity, output_sha256=file_sha256(output)))
    assert check_output(tmp_path, cell, identity) == payload
    output.write_text(output.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        check_output(tmp_path, cell, identity)
    atomic_json(output, dict(payload, predictions=[dict(id='one')]))
    atomic_json(receipt, dict(identity=identity, output_sha256=file_sha256(output)))
    with pytest.raises(ValueError, match='Incomplete'):
        check_output(tmp_path, cell, identity)
