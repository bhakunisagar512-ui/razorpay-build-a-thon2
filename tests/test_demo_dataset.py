"""The 203-entry demo dataset (eval/demo_dataset.py): valid against the existing controlled-input rules, the intended mix, no hidden label, Model C confidence only where
Model C runs, and the runner and the dataset agree on the simulated customer's behaviours."""
import json
import collections
from pathlib import Path

import pytest

from eval import demo_dataset as D
from services.app.revenue import catalog as C, routing

pytestmark = pytest.mark.skipif(not (D.OUT / 'entries.jsonl').exists(), reason='demo dataset not built (python -m eval.demo_dataset build)')


def test_the_dataset_is_valid_and_has_the_requested_mix():
    result = D.validate()
    assert result['problems'] == [] and result['entries'] == 203
    entries = D.load()
    assert collections.Counter(e['scenario'] for e in entries) == {'CHECKOUT_ABANDONMENT': 50, 'SUBSCRIPTION_FAILURE': 50, 'OVERDUE_RECEIVABLE': 50, 'PAYMENT_FAILURE': 50, 'EDGE_CASE': 3}
    assert {e['event_type'] for e in entries} == set(C.EVENT_TYPES)


def test_every_payment_failure_takes_its_intended_route_and_only_generic_ones_need_model_c():
    for e in D.load():
        if e['event_type'] != C.PAYMENT_FAILURE:
            assert not e['expected']['model_c'] and 'precheck' not in e
            continue
        route = routing.assess({k: e['request'].get(k) for k in routing.INFO_FIELDS}).route
        assert route == e['expected']['route'] and e['expected']['model_c'] == (route == routing.GENERIC)
        assert e['expected']['workflow'] == (route != routing.CONTRADICTORY)


def test_at_least_eighty_percent_of_the_model_c_entries_were_confident_under_every_precheck_id():
    ml = [e for e in D.load() if e['expected']['model_c']]
    lows = [min(s[2] for s in e['precheck']['scores']) for e in ml]
    assert len(ml) == 26 and sum(c > 0.6 for c in lows) / len(ml) >= 0.8
    assert all(len(e['precheck']['scores']) == len(D.PRECHECK_IDS) for e in ml)


def test_the_hidden_label_is_not_in_the_dataset_and_the_edge_cases_are_the_designed_ones():
    entries = D.load()
    assert not any('evaluation_label' in e['request'] for e in entries)
    edge = [e for e in entries if e['scenario'] == 'EDGE_CASE']
    assert [e['request']['customer_id'] for e in edge][0] == 'cust_syn_05' and edge[1]['request']['failure_time'] == 't_2230' and edge[2]['expected']['model_c']


def test_the_runner_uses_the_same_customer_behaviours_and_only_the_public_api():
    text = (Path('eval') / 'run_demo_dataset.py').read_text(encoding='utf-8')
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_demo_dataset', Path('eval') / 'run_demo_dataset.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.BEHAVIOUR == D.BEHAVIOUR
    assert 'services.app' not in text and 'sqlalchemy' not in text and 'psycopg' not in text          # it talks to the API only
