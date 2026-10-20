import json
import math
from types import SimpleNamespace

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


def test_freeze_validates_full_dev_and_does_not_require_test_data(tmp_path, monkeypatch):
    metric_class = pytest.importorskip('sacrebleu.metrics').BLEU
    from additive_chain import iwslt_confirm as confirm
    monkeypatch.setitem(confirm.COUNTS, 'dev', 2)
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'dev.jsonl').write_text('\n'.join(json.dumps(dict(id=str(i))) for i in range(2)) + '\n')
    atomic_json(data / 'manifest.json', dict(splits=dict(dev=dict(sha256=file_sha256(data / 'dev.jsonl')))))
    trained = tmp_path / 'model.pt'
    trained.write_bytes(b'fixture model identity only')
    development_root = tmp_path / 'dev'
    ids = [r['id'] for r in confirm.ordered_rows(data, 'dev')]
    for steps in (8, 32):
        root = development_root / f'step-{steps}'
        root.mkdir(parents=True)
        for cell in dev_cells(steps):
            reference = 'this is the correct translation of a sentence'
            hyp = reference if cell['strength'] == .2 or cell['temperature'] == .85 else 'wrong'
            rows = [dict(id=i, reference=reference, hypothesis=hyp) for i in ids]
            bleu = metric_class(lowercase=True, tokenize='13a').corpus_score([hyp]*2, [[reference]*2]).score
            identity = dict(cell, split='dev', ids=ids, model_sha256=file_sha256(trained),
                            data_sha256=file_sha256(data / 'manifest.json'),
                            source_sha256=confirm.source_hashes(), selection_sha256=None)
            path = root / (cell_name(cell) + '.json')
            atomic_json(path, dict(predictions=rows, summary=dict(examples=2, bleu=bleu)))
            atomic_json(root / (cell_name(cell) + '-identity.json'),
                        dict(identity=identity, output_sha256=file_sha256(path)))
        atomic_json(root / 'summary.json', dict(completed=6, expected=6))
    args = SimpleNamespace(data=data, trained=trained, development=development_root, output=tmp_path / 'frozen')
    confirm.freeze(args)
    selection = confirm.read(args.output / 'selection.json')
    assert selection['selected']['8']['strength'] == .2
    assert selection['selected']['32']['baseline_temperature'] == .85
    assert len(selection['development_artifacts']) == 24
    confirm.freeze(args)
    atomic_json(args.output / 'selection.json', dict(selection, selected={}))
    with pytest.raises(ValueError, match='Refusing to change'):
        confirm.freeze(args)
