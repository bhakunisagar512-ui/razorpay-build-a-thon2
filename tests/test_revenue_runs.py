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


# ------------------------------------------------------------------------------------------------- the 203-entry revenue-event dataset
@pytest.fixture(scope='module')
def demo_results():
    return run_all(runs.DEMO)


def test_the_203_entry_dataset_is_selectable_and_keeps_its_distribution(demo_results):
    option = {d['name']: d for d in runs.dataset_options()}[runs.DEMO]
    assert option['available'] and option['records'] == 203 and option['kind'] == 'events'
    m = runs.summarise(demo_results)
    assert m['total_cases'] == 203 and {k: v['cases'] for k, v in m['by_scenario'].items()} == {
        'CHECKOUT_ABANDONMENT': 50, 'SUBSCRIPTION_FAILURE': 50, 'OVERDUE_RECEIVABLE': 50, 'PAYMENT_FAILURE': 50, 'EDGE_CASE': 3}
    assert list(m['by_scenario']) == list(runs.SCENARIO_ORDER)
    assert all(v[C.RECOVERED] + v[C.FAILED] + v[C.HUMAN] == v['cases'] for v in m['by_scenario'].values())
    assert m[C.RECOVERED] + m[C.FAILED] + m['human_interventions'] == 203 and m['safety_violations'] >= 0


def test_model_c_runs_once_only_on_the_generic_route_and_only_those_cases_have_a_confidence(demo_results):
    for r in demo_results:
        if r['route'] == routing.GENERIC:
            assert r['model_c_calls'] == 1 and r['diagnosis_method'] == 'model_c' and 0 <= r['model_c_confidence'] <= 1
        else:
            assert r['model_c_calls'] == 0 and r['model_c_confidence'] is None             # nothing is invented for rule, plan-only or contradictory cases
    m = runs.summarise(demo_results)
    assert m['model_c_calls'] == 26 and m['max_model_c_calls_per_case'] == 1 and m['model_c_confidence']['cases'] == 26
    assert m['by_scenario']['CHECKOUT_ABANDONMENT']['model_c_confidence'] is None and m['by_scenario']['SUBSCRIPTION_FAILURE']['model_c_confidence'] is None
    assert m['by_scenario']['PAYMENT_FAILURE']['model_c_calls'] + m['by_scenario']['EDGE_CASE']['model_c_calls'] == 26
    assert m['route_distribution'] == {routing.CLEAR: 22, routing.GENERIC: 26, routing.CONTRADICTORY: 4}


def test_the_203_entry_run_is_reproducible_and_the_temporal_sample_covers_every_scenario(demo_results):
    again = run_all(runs.DEMO)
    assert runs.summarise(again) == runs.summarise(demo_results) and [r['model_c_confidence'] for r in again] == [r['model_c_confidence'] for r in demo_results]
    picked = [key for key, _ in runs.specs(runs.DEMO, 50, sample=True)]
    assert len(picked) == 50 and len(set(picked)) == 50
    scenario_of = {e['entry_id']: e['scenario'] for e in runs.demo_entries()}
    assert {scenario_of[k] for k in picked} == set(runs.SCENARIO_ORDER)
    assert [k for k, _ in runs.specs(runs.DEMO, 5)] == [e['entry_id'] for e in runs.demo_entries()[:5]]          # "first N" is unchanged when no sample is asked for


def test_the_entry_view_shows_inputs_and_no_hidden_label():
    page = runs.demo_entry_page(0, 5, 'PAYMENT_FAILURE')
    assert page['total'] == 50 and page['scenarios']['EDGE_CASE'] == 3 and len(page['items']) == 5
    assert all('evaluation_label' not in i['inputs'] and 'behaves as' not in i['purpose'] for i in page['items'])



# ------------------------------------------------------------------------------------------------- events CSV upload, acceptance rate, eligibility, progress
DEMO_CSV = Path('eval/data/demo_203/entries.csv')


@pytest.fixture()
def uploads(tmp_path, monkeypatch):
    from services.app.settings import settings
    monkeypatch.setattr(settings, 'ui_data_dir', str(tmp_path))
    return tmp_path


def test_the_attached_events_csv_validates_and_keeps_its_meaning(uploads):
    from services.app.revenue import event_csv
    if not DEMO_CSV.exists():
        pytest.skip('demo dataset not built')
    res = event_csv.validate(DEMO_CSV.read_text(encoding='utf-8'))
    assert res['valid'] and res['errors'] == [] and res['summary']['rows'] == 203 and res['events_csv_id']
    assert res['summary']['by_scenario'] == {'CHECKOUT_ABANDONMENT': 50, 'SUBSCRIPTION_FAILURE': 50, 'OVERDUE_RECEIVABLE': 50, 'PAYMENT_FAILURE': 50, 'EDGE_CASE': 3}
    assert any('ignored columns' in w for w in res['warnings'])                                  # customer_behaviour / purpose / expected_* are not inputs
    entries = event_csv.load(res['events_csv_id'])
    original = {e['entry_id']: e['request'] for e in runs.demo_entries()}
    assert len(entries) == 203 and all(e['request'] == {k: v for k, v in original[e['entry_id']].items() if v is not None} for e in entries)    # nothing added, nothing changed


def test_the_events_csv_is_validated_row_by_row_and_never_confused_with_the_payment_failure_csv(uploads):
    from services.app.revenue import event_csv
    header = 'entry_id,event_type,customer_id,amount_rupees,failure_time,checkout_stage,invoice_id,method\n'
    bad = header + 'a,CHECKOUT_ABANDONMENT,cust_nope,999,t_1500,payment_page,,\n' + 'b,OVERDUE_RECEIVABLE,cust_syn_01,500,t_1200,,inv_syn_03,\n' + 'c,NOT_A_TYPE,cust_syn_01,,,,,\n'
    res = event_csv.validate(bad)
    assert not res['valid'] and res['events_csv_id'] is None
    found = {(e['row'], e['column']) for e in res['errors']}
    assert (2, 'customer_id') in found and (3, 'amount_rupees') in found and (4, 'event_type') in found        # a field that does not belong to the type is refused, not ignored
    pf = 'payment_id,occurred_at,status,customer_id,issuer_id,amount_minor,method\npay_1,2026-05-04T10:15:00+00:00,failed,c,i,100,upi\n'
    res = event_csv.validate(pf)
    assert not res['valid'] and res['errors'][0]['column'] == 'event_type' and 'payment-failure CSVs use the other upload' in res['errors'][0]['message']


def test_a_payment_failure_row_without_a_diagnosis_scenario_is_accepted_and_still_routed_by_the_router(uploads):
    from services.app.revenue import event_csv
    text = ('event_type,customer_id,amount_rupees,issuer_id,failure_time,method,instrument_hint,gw_error_code,gw_error_source,gw_error_step,error_reason\n'
            'PAYMENT_FAILURE,cust_syn_01,499,issuer_01,t_1200,upi,na,GATEWAY_ERROR,beneficiary_bank,payment_debit_response,\n'
            'PAYMENT_FAILURE,cust_syn_01,499,issuer_01,t_1200,upi,na,BAD_REQUEST_ERROR,customer,payment_authentication,card_expired\n')
    res = event_csv.validate(text)
    assert res['valid'] and res['summary']['rows'] == 2
    entries = event_csv.load(res['events_csv_id'])
    specs_ = [spec for _k, spec in runs.specs('events:' + res['events_csv_id'], None)]
    assert 'diagnosis_scenario' not in specs_[0]['payload']                                      # a scenario the file never stated is not recorded
    assert [routing.assess(s['info']).route for s in specs_] == [routing.GENERIC, routing.CONTRADICTORY] and entries[0]['entry_id'] == 'row_0001'


@pytest.fixture(scope='module')
def acceptance_runs():
    out = {}
    for pct in (0, 50, 80, 100):
        accept = {'pct': float(pct), 'seed': 1}
        decided = [(k, s, *svc.decide(s)) for k, s in runs.specs(runs.DEMO, None, offset_days=runs.sim_offset_days('rvTEST-aaaa'))]
        assignment = runs.assign_acceptance(decided, accept)
        rows = []
        for key, spec, diagnosis, plan in decided:
            runs.stamp_acceptance(spec, accept, assignment.get(key))
            row = runs.run_case(spec, (diagnosis, plan))
            row['case'] = key
            rows.append(row)
        out[pct] = (rows, runs.summarise(rows, accept, sum(a['will'] for a in assignment.values())), assignment)
    return out


def test_the_acceptance_rate_is_applied_to_eligible_cases_exactly_and_changes_the_outcomes(acceptance_runs):
    eligible = sum(r['eligible'] for r in acceptance_runs[80][0])
    assert eligible == 184
    for pct in (0, 50, 80, 100):
        rows, m, assignment = acceptance_runs[pct]
        sim = m['simulation']
        assert sim['eligible_cases'] == 184 and sim['cases_assigned_to_accept'] == round(pct / 100 * 184) and sim['acceptance_rate_selected_pct'] == pct
        assert len(assignment) == 184 and sum(a['will'] for a in assignment.values()) == round(pct / 100 * 184)
        assert sim['simulated_accepts'] <= sim['cases_assigned_to_accept'] and sim['recovery_opportunities'] == sim['simulated_accepts'] + sim['simulated_rejects']
        assert sim['successful_recoveries'] == sim['recoveries_by_accept'] + sim['recoveries_by_verified_original_payment'] == m[C.RECOVERED]
        assert m['safety_violations'] == 0 and m['model_c_calls'] == 26
    accepts = [acceptance_runs[p][1]['simulation']['simulated_accepts'] for p in (0, 50, 80, 100)]
    assert accepts[0] == 0 and accepts == sorted(accepts) and accepts[3] > accepts[1] > 0                      # the rate really drives what happens
    recovered = [acceptance_runs[p][1][C.RECOVERED] for p in (0, 50, 80, 100)]
    assert recovered == sorted(recovered)
    assert acceptance_runs[0][1][C.RECOVERED] == acceptance_runs[0][1]['simulation']['recoveries_by_verified_original_payment']     # at 0%, only a verified original payment can "recover" anything


def test_an_accept_is_never_an_automatic_recovery_and_safety_rules_still_apply(acceptance_runs):
    rows = acceptance_runs[100][0]
    for r in rows:
        if r['route'] == routing.CONTRADICTORY or not r['eligible']:
            assert r['accepts'] == 0 and r['status'] in (C.HUMAN, C.FAILED)                           # no opportunity, so no ACCEPT, whatever the rate
        if r['status'] == C.RECOVERED and r['reason'] != 'original_payment_succeeded':
            assert r['accepts'] == 1 and r['reason'].startswith('accepted_step_')
        assert 'contacted_without_consent' not in r['violations'] and 'hard_decline_retried' not in r['violations']
    assert any(r['route'] != routing.CONTRADICTORY and not r['eligible'] and r['status'] == C.HUMAN for r in rows)                  # directly escalated cases stay escalated at 100%


def test_the_assignment_is_reproducible_depends_on_the_seed_and_never_on_the_hidden_label():
    def assigned(seed):
        decided = [(k, s, *svc.decide(s)) for k, s in runs.specs(runs.DEMO, 60)]
        return runs.assign_acceptance(decided, {'pct': 50.0, 'seed': seed})
    assert assigned(7) == assigned(7) and assigned(7) != assigned(8)
    import inspect
    assert 'hidden' not in inspect.getsource(runs.assign_acceptance).split('"""')[2]


def test_engine_configured_response_follows_the_assignment_and_the_in_process_and_workflow_paths_share_it():
    a = {'pct': 50.0, 'seed': 1, 'will': True, 'ordinal': 1}
    assert [E.configured_response(a, 'k', i, 3) for i in range(3)] == ['REJECTED', 'ACCEPTED', 'REJECTED']
    assert [E.configured_response({**a, 'will': False}, 'k', i, 3) for i in range(3)] == ['REJECTED'] * 3
    assert E.configured_response(a, 'k', 0, 0) == 'REJECTED'
    assert E.accept_params({'sim_acceptance_pct': 80, 'sim_seed': 3, 'sim_accepts': True, 'sim_accept_at': 2}) == {'pct': 80.0, 'seed': 3, 'will': True, 'ordinal': 2} and E.accept_params({'sim_key': 'x'}) is None
    import inspect
    from services.app.workflows import revenue_activities
    assert 'configured_response' in inspect.getsource(revenue_activities.rv_simulate_response) and 'configured_response' in inspect.getsource(E.execute)


def test_the_run_clock_offset_keeps_the_calendar_so_features_and_rules_see_the_same_times():
    from datetime import date, timedelta
    d0 = date(2026, 8, 10)
    offsets = {runs.sim_offset_days(f'rv2026110{i}T000000-{i:04x}') for i in range(200)}
    assert len(offsets) > 20 and runs.sim_offset_days(None) == 0
    for off in offsets:
        d1 = d0 + timedelta(days=off)
        assert (d1.weekday(), d1.month, d1.day) == (d0.weekday(), d0.month, d0.day), off          # same weekday and same date: the hour, quiet hours and days-to-month-end are unchanged
    assert runs.sim_offset_days('rv20261104T000000-abcd') == runs.sim_offset_days('rv20261104T000000-abcd')


def test_a_full_in_process_run_of_the_events_dataset_records_progress_eligibility_and_per_case_details(uploads):
    doc = runs.execute('rv20260101T000000-beef', runs.DEMO, None, 0, {'pct': 80.0, 'seed': 1})
    assert doc['status'] == 'completed' and doc['metrics']['simulation']['eligible_cases'] == 184 and doc['acceptance'] == {'pct': 80.0, 'seed': 1}
    state = runs._state['rv20260101T000000-beef']['progress']
    assert state['phase'] == 'done' and state['total'] == 203 and state['evaluated'] == 203 and state['eligible'] == 184
    page = runs.cases('rv20260101T000000-beef', None, None, 0, 20)
    assert page['total'] == 203 and len(page['items']) == 20
    for r in page['items']:
        assert r['inputs'] and r['diagnosis']['method'] and 'steps' in r['plan'] and 'eligible' in r
        assert (r['model_c_confidence'] is not None) == (r['diagnosis']['model_c'] is not None)
    assert len(runs.cases('rv20260101T000000-beef', None, None, 200, 20)['items']) == 3
