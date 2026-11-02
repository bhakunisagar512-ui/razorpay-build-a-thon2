"""Automated Dataset Testing of the revenue pipeline (Phase 9), in-process: reproducibility, accounting, one Model C call at most,
hidden-truth isolation, and the rule that a correct diagnosis is not recovered revenue. No database, no Temporal."""
import copy
from pathlib import Path

import pytest

from services.app.revenue import catalog as C, engine as E, fixtures, routing, runs, service as svc


def run_all(dataset, limit=None):
    out = []
    for key, spec in runs.specs(dataset, limit):
        row = runs.run_case(spec)
        row['case'] = key
        out.append(row)
    return out


@pytest.fixture(scope='module')
def fixture_results():
    return run_all(fixtures.NAME)


@pytest.fixture(scope='module')
def mix_results():
    return run_all(runs.MIX)


def test_dataset_options_include_the_prebuilt_datasets_and_the_two_controlled_ones():
    names = {d['name']: d for d in runs.dataset_options()}
    assert {'normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime', fixtures.NAME, runs.MIX} <= set(names)
    assert names[fixtures.NAME]['records'] == 66 and names[runs.MIX]['records'] == 48


def test_runs_are_reproducible():
    for ds in (fixtures.NAME, runs.MIX):
        a, b = run_all(ds), run_all(ds)
        assert a == b and runs.summarise(a) == runs.summarise(b)


def test_accounting_adds_up_and_recovery_is_only_a_simulated_payment(fixture_results, mix_results):
    for results in (fixture_results, mix_results):
        m = runs.summarise(results)
        assert m[C.RECOVERED] + m[C.FAILED] + m['human_interventions'] == m['total_cases'] == len(results)
        assert m['simulated_recovered_amount_minor'] == sum(r['amount_minor'] for r in results if r['status'] == C.RECOVERED)
        for r in results:
            decisions = [s.get('decision') for s in r['steps'] if s['status'] == 'EXECUTED']
            if r['status'] == C.RECOVERED and r['reason'] == 'original_payment_succeeded':
                assert 'ACCEPTED' not in decisions and r['recovered_amount_minor'] == r['amount_minor']     # the original payment went through: counted once, nobody asked to pay again
            elif r['status'] == C.RECOVERED:
                assert decisions.count('ACCEPTED') == 1 and decisions[-1] == 'ACCEPTED' and r['recovered_amount_minor'] == r['amount_minor']   # counted once, then stopped
            else:
                assert 'ACCEPTED' not in decisions and r['recovered_amount_minor'] == 0
        assert 'simulated' in m['note'].lower() and 'not recovered revenue' in m['diagnosis_match']['note'].lower()


def test_all_three_routes_are_exercised_and_model_c_is_called_at_most_once_per_case(fixture_results):
    by = {}
    for r in fixture_results:
        by.setdefault(r['route'], []).append(r)
    assert set(by) == set(routing.ROUTES) and {k: len(v) for k, v in by.items()} == {routing.CLEAR: 24, routing.GENERIC: 24, routing.CONTRADICTORY: 18}
    assert all(r['model_c_calls'] == 0 and r['diagnosis_method'] == 'rule' for r in by[routing.CLEAR])
    assert all(r['model_c_calls'] == 1 and r['diagnosis_method'] == 'model_c' for r in by[routing.GENERIC])
    assert all(r['model_c_calls'] == 0 and r['status'] == C.HUMAN and r['executed'] == 0 and r['plan_actions'] == [] for r in by[routing.CONTRADICTORY])
    assert all(r['route'] == r['expected_route'] for r in fixture_results)
    assert runs.summarise(fixture_results)['max_model_c_calls_per_case'] == 1


def test_no_safety_violations_in_the_controlled_datasets(fixture_results, mix_results):
    assert runs.summarise(fixture_results)['safety_violations'] == 0 and runs.summarise(mix_results)['safety_violations'] == 0


def test_a_violation_is_detected_when_a_hard_decline_would_be_retried(monkeypatch):
    monkeypatch.setattr(E, 'simulate', lambda *a, **k: 'REJECTED')                              # nothing is accepted early, so the appended retry is reached
    clear = [r for r in fixtures.records() if r['expected_route'] == routing.CLEAR]
    rec = next(r for r in clear if svc.decide(svc.fixture_spec(r, 'evt_v'))[1]['steps'][0]['action'] != 'human_escalation')      # a plan that does not end at once in an escalation
    spec = svc.fixture_spec(rec, 'evt_v')
    diagnosis, plan = svc.decide(spec)
    plan = copy.deepcopy(plan)
    plan['steps'].append({'idx': 9, 'action': 'delayed_retry', 'label': 'x', 'delay_s': 0, 'channel': None, 'customer_facing': False, 'system_retry': True, 'payment_opportunity': True})
    spec['hidden_cause'] = 'CARD_EXPIRED'
    outcome = E.execute(event_id='e', event_type=C.PAYMENT_FAILURE, customer_id='c', amount_minor=100, plan=plan, consent={k: True for k in C.CHANNELS}, sim_start=spec['sim_start'], hidden_cause='CARD_EXPIRED')
    assert 'hard_decline_retried' in runs.case_result(spec, diagnosis, plan, outcome)['violations']


def test_a_correct_diagnosis_is_not_recovered_revenue(monkeypatch):
    monkeypatch.setattr(E, 'simulate', lambda *a, **k: 'REJECTED')                              # nobody ever pays
    monkeypatch.setattr(E, 'verify_state', lambda *a, **k: 'FAILED')                            # and no timed-out original payment turns out to have succeeded
    results = run_all(fixtures.NAME)
    m = runs.summarise(results)
    assert m['diagnosis_match']['matched'] > 0 and m[C.RECOVERED] == 0 and m['simulated_recovered_amount_minor'] == 0 and m['simulated_recovery_percentage'] == 0.0
    assert m[C.FAILED] + m['human_interventions'] == m['total_cases']


def test_the_hidden_cause_never_changes_a_diagnosis_or_a_plan():
    for rec in fixtures.records()[:30]:
        a, b = svc.fixture_spec(rec, 'evt_a'), svc.fixture_spec({**rec, 'hidden_cause': 'RISK_BLOCKED' if rec['hidden_cause'] != 'RISK_BLOCKED' else 'TD_PSP_TIMEOUT'}, 'evt_a')
        assert a['hidden_cause'] != b['hidden_cause']
        da, pa = svc.decide(a)
        db, pb = svc.decide(b)
        assert da == db and pa == pb


def test_every_event_type_runs_through_the_automated_executor(mix_results):
    m = runs.summarise(mix_results)
    assert set(m['by_event_type']) == set(C.EVENT_TYPES) and m['by_event_type'][C.CHECKOUT] == 12 and m['total_cases'] == 48
    assert m['route_distribution'] == {routing.CLEAR: 12} and m['model_c_calls'] == 0


def test_a_prebuilt_dataset_subset_runs_and_balances():
    from services.app import test_datasets as tds
    if not [d for d in tds.list_datasets() if d['available']]:
        pytest.skip('prebuilt datasets are not built')
    results = run_all('mixed_difficult', 120)
    m = runs.summarise(results)
    assert m['total_cases'] == 120 and set(m['route_distribution']) <= set(routing.ROUTES) and m['max_model_c_calls_per_case'] <= 1
    assert m['model_c_calls'] == m['route_distribution'].get(routing.GENERIC, 0) and m[C.RECOVERED] + m[C.FAILED] + m['human_interventions'] == 120
    assert all(r['model_c_calls'] == (r['route'] == routing.GENERIC) for r in results)


def test_run_files_are_stored_apart_and_nothing_historical_is_touched(tmp_path, monkeypatch):
    from services.app.settings import settings
    monkeypatch.setattr(settings, 'ui_data_dir', str(tmp_path))
    doc = runs.execute('rv20260101T000000-abcd', runs.MIX, 12, 0)
    assert doc['status'] == 'completed' and doc['metrics']['total_cases'] == 12 and (tmp_path / 'revenue_runs' / 'rv20260101T000000-abcd.json').exists()
    assert not (tmp_path / 'runs').exists()                                                     # the existing evaluation runs folder is never used
    got = runs.get_run('rv20260101T000000-abcd')
    assert got['metrics']['total_cases'] == 12 and 'cases' not in got
    page = runs.cases('rv20260101T000000-abcd', None, None, 0, 5)
    recovered = doc['metrics'][C.RECOVERED]
    assert page['total'] == 12 and len(page['items']) == 5 and runs.cases('rv20260101T000000-abcd', None, C.RECOVERED, 0, 50)['total'] == recovered
    assert runs.cases('rv20260101T000000-abcd', 'CLEAR_AND_CONSISTENT', None, 0, 50)['total'] == 0           # the first 12 mix cases are not payment failures
    assert runs.get_run('not-a-run-id') is None and runs.cases('rv20990101T000000-ffff', None, None, 0, 5) is None


def test_an_original_payment_found_to_have_succeeded_is_counted_once_and_is_not_a_payment_request(monkeypatch):
    monkeypatch.setattr(E, 'simulate', lambda *a, **k: 'REJECTED')
    monkeypatch.setattr(E, 'verify_state', lambda *a, **k: 'SUCCEEDED')
    results = run_all(fixtures.NAME)
    paid = [r for r in results if r['reason'] == 'original_payment_succeeded']
    assert paid and all(r['status'] == C.RECOVERED and r['recovered_amount_minor'] == r['amount_minor'] and not any('decision' in s for s in r['steps']) for r in paid)
    assert all(r['plan_actions'][0] == 'verify_payment_state' for r in paid)                 # only plans that verify first can end this way
