"""eval/data/csv_demo_200/dataset_200.csv: a 200-row CSV for Dataset Testing's upload. Valid against csv-schema@1, the intended route mix, a hidden label on every row,
and most Model C rows above 0.6 (selected on purpose; see eval/csv_demo_dataset.py)."""
import collections
import csv
import io
import json

import pytest

from eval import csv_demo_dataset as D
from services.app import test_datasets as tds

pytestmark = pytest.mark.skipif(not D.FILE.exists(), reason='dataset not built (python -m eval.csv_demo_dataset build)')


def test_the_file_is_accepted_by_the_upload_validator(monkeypatch):
    monkeypatch.setattr(tds, '_store', lambda text: 'x')                      # validating must not write to the uploads folder
    check = tds.validate_csv(D.FILE.read_text(encoding='utf-8'))
    assert check['valid'] and check['errors'] == [] and check['warnings'] == []
    assert check['summary'] == {'rows': 200, 'failed_rows': 200, 'history_rows': 0, 'has_ground_truth': True, 'mode': 'verified'}


def test_routes_labels_and_columns():
    text = D.FILE.read_text(encoding='utf-8')
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 200 and len({r['payment_id'] for r in rows}) == 200 and all(r['status'] == 'failed' and r['true_cause'] in tds.CAUSES for r in rows)
    assert set(rows[0]) == set(D.COLUMNS) and set(tds.REQUIRED) <= set(rows[0])
    manifest = json.loads((D.OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['routes'] == {'CLEAR_AND_CONSISTENT': 80, 'MISSING_OR_GENERIC': 100, 'CONTRADICTORY': 20} and manifest['model_c_rows'] == 100
    assert manifest['share_above_0.6'] >= 0.8 and manifest['rule_matches_hidden_label'] == 1.0


def test_the_model_c_confidence_is_reproducible_from_the_csv_alone():
    rows = list(csv.DictReader(io.StringIO(D.FILE.read_text(encoding='utf-8'))))
    res = D.evaluate(rows)
    conf = [r['confidence'] for r in res.values() if 'confidence' in r]
    assert len(conf) == 100 and sum(c > 0.6 for c in conf) / len(conf) >= 0.8
    assert collections.Counter(r['route'] for r in res.values()) == {'CLEAR_AND_CONSISTENT': 80, 'MISSING_OR_GENERIC': 100, 'CONTRADICTORY': 20}
