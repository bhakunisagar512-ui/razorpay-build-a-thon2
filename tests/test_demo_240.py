"""eval/data/demo_240/entries.csv: 240 revenue events (60 per scenario; payment failures 20 rule / 35 Model C / 5 contradictory), valid for the revenue-events upload, unique, and with
at least 32 of the 35 Model C rows above 0.60 when scored by the real model (selected rows: see eval/demo_240.py)."""
import collections
from pathlib import Path

import pytest

from eval import demo_240 as D
from services.app.revenue import catalog as C, event_csv, routing

pytestmark = pytest.mark.skipif(not D.CSV_FILE.exists(), reason='dataset not built (python -m eval.demo_240 build)')
NEEDS_MODEL = pytest.mark.skipif(not (Path('eval/artifacts/phase4') / 'model.joblib').exists(), reason='frozen Model C artifact not present')


def test_the_file_passes_the_revenue_events_validator_and_has_the_requested_shape():
    entries, errors, _warnings, summary = event_csv.parse(D.CSV_FILE.read_text(encoding='utf-8'))
    assert errors == [] and summary['rows'] == 240
    assert summary['by_scenario'] == {s: 60 for s in ('CHECKOUT_ABANDONMENT', 'SUBSCRIPTION_FAILURE', 'OVERDUE_RECEIVABLE', 'PAYMENT_FAILURE')}
    assert len({e['entry_id'] for e in entries}) == 240 and not any(e['scenario'] == 'EDGE_CASE' for e in entries)
    keys = [tuple(sorted(e['request'].items(), key=str)) for e in entries]
    assert len(set(keys)) == 240                                                                  # no duplicated records


def test_the_payment_failure_routes_are_20_rule_35_model_c_5_contradictory_by_the_existing_router():
    rows = [r for r in D.load_rows() if r['event_type'] == C.PAYMENT_FAILURE]
    routes = collections.Counter(routing.assess({k: (r[k] or None) for k in routing.INFO_FIELDS}).route for r in rows)
    assert routes == {routing.CLEAR: 20, routing.GENERIC: 35, routing.CONTRADICTORY: 5}
    assert all(r['expected_route'] == routing.assess({k: (r[k] or None) for k in routing.INFO_FIELDS}).route for r in rows)
    assert len({r['method'] for r in rows}) == 3 and len({r['issuer_id'] for r in rows}) >= 8 and len({r['gw_error_code'] for r in rows}) == 3


@NEEDS_MODEL
def test_at_least_32_of_the_35_model_c_rows_score_above_0_60_and_the_model_is_called_once_per_row():
    entries, _e, _w, _s = event_csv.parse(D.CSV_FILE.read_text(encoding='utf-8'))
    generic = [e for e in entries if e['event_type'] == C.PAYMENT_FAILURE and routing.assess({k: e['request'].get(k) for k in routing.INFO_FIELDS}).route == routing.GENERIC]
    assert len(generic) == 35
    scored = [D.score(e['request'], e['entry_id']) for e in generic]
    assert all(calls == 1 for *_x, calls in scored) and sum(conf > 0.60 for _c, _g, conf, _n in scored) >= 32
    assert len({e['request']['gw_error_step'] + e['request'].get('error_reason', '') + e['request']['method'] for e in generic}) > 20        # a variety of evidence, not one repeated input


def test_no_hidden_label_is_in_the_file():
    assert 'evaluation_label' not in D.CSV_FILE.read_text(encoding='utf-8').splitlines()[0]
