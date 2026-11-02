"""LIVE end-to-end checks of the revenue pipeline against the running stack: FastAPI -> PostgreSQL -> Temporal -> a worker on the revenue queue.

Run from the host while the stack is up and a worker serves the revenue queue:

    RUN_LIVE_REVENUE=1 python -m pytest tests/test_revenue_live.py -q

Every test creates NEW synthetic events (unique generated ids) and never edits or deletes anything that exists. Skipped unless RUN_LIVE_REVENUE=1.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get('REVENUE_API_BASE', 'http://localhost:8000')
pytestmark = pytest.mark.skipif(os.environ.get('RUN_LIVE_REVENUE') != '1', reason='live stack test: set RUN_LIVE_REVENUE=1 (needs the API, Temporal and a revenue worker)')

OTP = dict(method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='customer', gw_error_step='payment_authentication', error_reason='payment_cancelled')
FUNDS = dict(method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='issuer_bank', gw_error_step='payment_debit_response', error_reason='insufficient_funds')
RISK = dict(method='upi', instrument_hint='na', gw_error_code='GATEWAY_ERROR', gw_error_source='issuer_bank', gw_error_step='payment_debit_response', error_reason='payment_risk_check_failed')
BASE_PF = dict(event_type='PAYMENT_FAILURE', customer_id='cust_syn_01', amount_rupees=499, issuer_id='issuer_01', failure_time='t_1200')
OTHER = {
    'CHECKOUT_ABANDONMENT': dict(event_type='CHECKOUT_ABANDONMENT', customer_id='cust_syn_01', amount_rupees=999, failure_time='t_1500', checkout_stage='payment_page'),
    'SUBSCRIPTION_FAILURE': dict(event_type='SUBSCRIPTION_FAILURE', customer_id='cust_syn_06', plan_id='plan_pro_monthly', method='card', failure_kind='CARD_EXPIRED', failure_time='t_1200'),
    'OVERDUE_RECEIVABLE': dict(event_type='OVERDUE_RECEIVABLE', customer_id='cust_syn_07', invoice_id='inv_syn_03', failure_time='t_0930'),
}


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b'null')
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'null')


def wait(fn, timeout=40, what='condition'):
    end = time.time() + timeout
    while time.time() < end:
        value = fn()
        if value:
            return value
        time.sleep(0.4)
    raise AssertionError(f'timed out waiting for {what}')


def create(body, timeout_s=None):
    s, ev = call('POST', '/revenue/events' + (f'?timeout_s={timeout_s}' if timeout_s else ''), body)
    assert s == 201, ev
    return ev


def view(eid):
    return call('GET', f'/revenue/events/{eid}')[1]


def open_request(eid, n=1, timeout=40):
    """Wait until the n-th payment request of the event exists and is PENDING; returns it."""
    def find():
        reqs = view(eid)['requests']
        return reqs[n - 1] if len(reqs) >= n and reqs[n - 1]['status'] == 'PENDING' else None
    return wait(find, timeout, f'pending request #{n}')


def settled(eid, timeout=40):
    return wait(lambda: (lambda v: v if v['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED') else None)(view(eid)), timeout, 'workflow completion')


@pytest.mark.parametrize('body', [{**BASE_PF, **OTP}, *OTHER.values()], ids=['PAYMENT_FAILURE', *OTHER])
def test_each_event_type_creates_a_payment_opportunity_and_accept_recovers_it(body):
    ev = create(body)
    assert ev['event_type'] == body['event_type'] and ev['status'] == 'PENDING' and ev['plan']['steps'] and ev['simulated'] is True
    req = open_request(ev['event_id'])
    page = call('GET', f"/revenue/requests/{req['request_id']}")[1]
    assert page['amount_minor'] == ev['amount_minor'] and page['workflow_id'] == f"revenue-{ev['event_id']}" and page['recovery_action'] and page['simulated']
    s, res = call('POST', f"/revenue/requests/{req['request_id']}/respond", {'decision': 'ACCEPT'})
    assert s == 200 and res['changed'] and res['status'] == 'ACCEPTED'
    done = settled(ev['event_id'])
    assert done['status'] == 'SIMULATED_RECOVERED' and done['recovered_amount_minor'] == ev['amount_minor'] and done['workflow_state'] == 'WORKFLOW_COMPLETED'
    assert [s['status'] for s in done['plan']['steps']][0] == 'EXECUTED' and all(s['status'] in ('EXECUTED', 'CANCELLED') for s in done['plan']['steps'])
    assert any(t['action'] == 'simulated_recovered' for t in done['timeline'])


def test_accept_stops_temporal_and_cancels_the_remaining_actions():
    ev = create({**BASE_PF, **OTP})
    open_request(ev['event_id'])
    call('POST', f"/revenue/requests/{view(ev['event_id'])['requests'][0]['request_id']}/respond", {'decision': 'ACCEPT'})
    done = settled(ev['event_id'])
    assert [s['status'] for s in done['plan']['steps']] == ['EXECUTED', 'CANCELLED'] and len(done['requests']) == 1


def test_reject_continues_to_the_next_action_and_a_later_accept_recovers():
    ev = create({**BASE_PF, **FUNDS})                                                            # plan: pay with another method, then a reminder
    first = open_request(ev['event_id'])
    assert call('POST', f"/revenue/requests/{first['request_id']}/respond", {'decision': 'REJECT'})[1]['status'] == 'REJECTED'
    second = open_request(ev['event_id'], 2)
    mid = view(ev['event_id'])
    assert mid['status'] == 'PENDING' and mid['diagnosis'] == ev['diagnosis'] and mid['diagnosis']['model_c_calls'] == 0      # the diagnosis did not change and no model ran
    call('POST', f"/revenue/requests/{second['request_id']}/respond", {'decision': 'ACCEPT'})
    done = settled(ev['event_id'])
    assert done['status'] == 'SIMULATED_RECOVERED' and [r['status'] for r in done['requests']] == ['REJECTED', 'ACCEPTED'] and done['recovered_amount_minor'] == ev['amount_minor']


def test_rejecting_everything_ends_simulated_failed_and_never_re_diagnoses():
    ev = create({**BASE_PF, **OTP})
    for n in (1, 2):
        call('POST', f"/revenue/requests/{open_request(ev['event_id'], n)['request_id']}/respond", {'decision': 'REJECT'})
    done = settled(ev['event_id'])
    assert (done['status'], done['status_reason'], done['recovered_amount_minor']) == ('SIMULATED_FAILED', 'plan_exhausted', 0)
    assert done['diagnosis'] == ev['diagnosis'] and done['diagnosis']['model_c_calls'] == 0 and len(done['requests']) == 2   # no extra cycle, no third request
    assert [t['action'] for t in done['timeline']].count('diagnosis') == 1


def test_duplicate_conflicting_and_late_responses_change_nothing():
    ev = create({**BASE_PF, **OTP})
    rid = open_request(ev['event_id'])['request_id']
    assert call('POST', f'/revenue/requests/{rid}/respond', {'decision': 'ACCEPT'})[1]['changed'] is True
    again = call('POST', f'/revenue/requests/{rid}/respond', {'decision': 'ACCEPT'})
    assert again[0] == 200 and again[1]['changed'] is False and again[1]['duplicate'] is True
    assert call('POST', f'/revenue/requests/{rid}/respond', {'decision': 'REJECT'})[0] == 409
    done = settled(ev['event_id'])
    assert done['recovered_amount_minor'] == ev['amount_minor'] and sum(r['status'] == 'ACCEPTED' for r in done['requests']) == 1
    assert call('POST', '/revenue/requests/rq_does_not_exist/respond', {'decision': 'ACCEPT'})[0] == 404
    assert call('POST', f'/revenue/requests/{rid}/respond', {'decision': 'MAYBE'})[0] == 422


def test_simultaneous_accepts_are_counted_once():
    ev = create({**BASE_PF, **OTP})
    rid = open_request(ev['event_id'])['request_id']
    out = []
    threads = [threading.Thread(target=lambda: out.append(call('POST', f'/revenue/requests/{rid}/respond', {'decision': 'ACCEPT'}))) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(1 for s, b in out if s == 200 and b['changed']) == 1 and all(s == 200 for s, _ in out)
    done = settled(ev['event_id'])
    assert done['recovered_amount_minor'] == ev['amount_minor'] and sum(1 for t in done['timeline'] if t['action'] == 'simulated_recovered') == 1


def test_an_unanswered_request_times_out_and_a_late_response_is_refused():
    ev = create({**BASE_PF, **OTP}, timeout_s=3)                                                 # configurable demo timeout
    first = open_request(ev['event_id'])
    second = open_request(ev['event_id'], 2, timeout=30)                                         # after 3 s the first expires and the plan moves on
    assert view(ev['event_id'])['requests'][0]['status'] == 'EXPIRED'
    s, body = call('POST', f"/revenue/requests/{first['request_id']}/respond", {'decision': 'ACCEPT'})
    assert s == 409 and 'EXPIRED' in json.dumps(body)
    done = settled(ev['event_id'], timeout=30)                                                   # the second also times out: bounded execution
    assert done['status'] == 'SIMULATED_FAILED' and done['recovered_amount_minor'] == 0 and second['request_id'] != first['request_id']


def test_risk_block_and_missing_consent_escalate_to_a_human_without_a_payment_request():
    risk = settled(create({**BASE_PF, **RISK})['event_id'])
    assert (risk['status'], risk['status_reason'], risk['requests']) == ('HUMAN_INTERVENTION', 'risk_blocked', [])
    no_consent = settled(create({**BASE_PF, **FUNDS, 'customer_id': 'cust_syn_05'})['event_id'])
    assert (no_consent['status'], no_consent['status_reason'], no_consent['requests']) == ('HUMAN_INTERVENTION', 'no_safe_recovery_action', [])
    kinds = {i['id']: i['kind'] for i in call('GET', '/interventions?limit=500')[1]['items']}
    assert kinds[risk['event_id']] == 'SAFETY_ESCALATION' and kinds[no_consent['event_id']] == 'PLAN_INTERVENTION'


def test_quiet_hours_delay_the_message_on_the_simulated_clock():
    ev = create({**BASE_PF, **OTP, 'failure_time': 't_2230'})
    req = open_request(ev['event_id'], timeout=40)
    step = view(ev['event_id'])['plan']['steps'][0]
    assert step['status'] == 'EXECUTED' and step['sim_at']
    from datetime import datetime, timedelta, timezone
    local = datetime.fromisoformat(step['sim_at']).astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert 9 <= local.hour < 21, local
    call('POST', f"/revenue/requests/{req['request_id']}/respond", {'decision': 'REJECT'})


def _fixture(route):
    from services.app.revenue import fixtures
    return next(r for r in fixtures.records() if r['expected_route'] == route)


def test_contradictory_dataset_record_goes_straight_to_a_human_and_never_starts_a_workflow():
    assert next(d for d in call('GET', '/revenue/catalog')[1]['datasets'] if d['name'] == 'routing_fixtures')['records'] == 66
    s, created = call('POST', '/revenue/events/from-dataset', {'dataset': 'routing_fixtures', 'payment_id': _fixture('CONTRADICTORY')['payment_id']})
    assert s == 201 and created['route'] == 'CONTRADICTORY'
    assert (created['status'], created['workflow_state'], created['plan']['steps'], created['diagnosis']['model_c_calls']) == ('HUMAN_INTERVENTION', 'NOT_REQUIRED', [], 0)
    assert created['diagnosis']['conflicts'] and 'evaluation' not in created and created['requests'] == []
    case = call('GET', f"/interventions/revenue/{created['event_id']}")[1]
    assert case['read_only'] and case['kind'] == 'DIAGNOSIS_INTERVENTION' and case['reason'] == 'contradictory_gateway_information' and case['actions_available'] == []
    listed = call('GET', '/interventions?limit=500')[1]
    assert any(i['id'] == created['event_id'] for i in listed['items']) and listed['total'] >= 1 and listed['read_only'] is True


def drain(eid, seconds=30):
    """Answer every payment request of an event with REJECT until its workflow completes (leaves nothing waiting on the demo timeout)."""
    end = time.time() + seconds
    while time.time() < end:
        v = view(eid)
        if v['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED'):
            return v
        for r in v['requests']:
            if r['status'] == 'PENDING':
                call('POST', f"/revenue/requests/{r['request_id']}/respond", {'decision': 'REJECT'})
        time.sleep(0.5)
    raise AssertionError('the workflow did not complete')


def test_a_missing_or_generic_fixture_calls_model_c_once_and_a_reject_does_not_call_it_again():
    s, ev = call('POST', '/revenue/events/from-dataset', {'dataset': 'routing_fixtures', 'payment_id': _fixture('MISSING_OR_GENERIC')['payment_id']})
    assert s == 201 and ev['route'] == 'MISSING_OR_GENERIC' and ev['diagnosis']['method'] == 'model_c' and ev['diagnosis']['model_c_calls'] == 1
    done = drain(ev['event_id'])                                                                # every payment opportunity is rejected
    assert done['diagnosis'] == ev['diagnosis'] and done['diagnosis']['model_c_calls'] == 1
    assert [t['action'] for t in done['timeline']].count('diagnosis') == 1
    assert done['status'] in ('SIMULATED_FAILED', 'HUMAN_INTERVENTION', 'SIMULATED_RECOVERED')


PSP = dict(method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='customer_psp', gw_error_step='payment_request', error_reason='upi_app_technical_error')


def test_a_timeout_is_verified_first_and_nobody_is_asked_to_pay_while_the_original_may_have_succeeded():
    seen = set()
    for _ in range(14):
        ev = create({**BASE_PF, **PSP})
        assert [s['action'] for s in ev['plan']['steps']] == ['verify_payment_state', 'payment_reminder'] and ev['diagnosis']['cause'] == 'TD_PSP_TIMEOUT'
        v = wait(lambda: (lambda x: x if any(t['action'] == 'payment_state_verified' for t in x['timeline']) else None)(view(ev['event_id'])), 30, 'the payment state check')
        state = next(t['payload']['state'] for t in v['timeline'] if t['action'] == 'payment_state_verified')
        seen.add(state)
        done = drain(ev['event_id'])
        if state == 'SUCCEEDED':                                                               # the original payment went through
            assert (done['status'], done['status_reason'], done['recovered_amount_minor'], done['requests']) == ('SIMULATED_RECOVERED', 'original_payment_succeeded', ev['amount_minor'], [])
        elif state == 'UNRESOLVED':                                                           # cannot tell: no payment request, a person checks
            assert (done['status'], done['status_reason'], done['requests']) == ('HUMAN_INTERVENTION', 'payment_state_unverified', [])
            kinds = {i['id']: i['kind'] for i in call('GET', '/interventions?limit=500')[1]['items']}
            assert kinds[ev['event_id']] == 'SAFETY_ESCALATION'
        else:                                                                                  # confirmed failed: only now a payment request, and never a retry
            assert [r['step_idx'] for r in done['requests']] == [1] and not any(s['system_retry'] for s in done['plan']['steps'])
        times = [t['at'] for t in done['timeline'] if t['action'] in ('payment_state_verified', 'payment_request_created')]
        assert times == sorted(times)                                                          # a request never precedes the verification
    assert len(seen) >= 2, seen


def test_insufficient_funds_gets_a_delayed_retry_not_an_immediate_one():
    ev = create({**BASE_PF, **FUNDS})
    assert [s['action'] for s in ev['plan']['steps']] == ['alternative_payment_method', 'payment_reminder', 'delayed_retry']
    done = drain(ev['event_id'], 40)
    assert (done['status'], done['status_reason']) == ('SIMULATED_FAILED', 'plan_exhausted') and [r['title'] for r in done['requests']][-1] == 'Automatic retry'
    steps = {s['action']: s for s in done['plan']['steps']}
    from datetime import datetime
    gap = (datetime.fromisoformat(steps['delayed_retry']['sim_at']) - datetime.fromisoformat(steps['alternative_payment_method']['sim_at'])).total_seconds()
    assert sum(s['delay_s'] for s in done['plan']['steps']) == 3 * 86400                       # the plan itself is the fixed 3 days
    assert 3 * 86400 <= gap <= 14 * 86400, gap / 86400                                         # never sooner than 3 days; the weekly contact cap (shared by earlier test events of the same customer on nearby simulated days) can only push it later


def test_the_audit_chain_stays_valid_and_labels_stay_hidden():
    assert call('GET', '/audit/verify')[1]['valid'] is True
    labelled = create({**BASE_PF, **OTP, 'evaluation_label': 'BD_USER_ABANDONED_OTP'})
    assert labelled['evaluation']['label'] == 'BD_USER_ABANDONED_OTP' and labelled['evaluation']['diagnosis_matches_label'] is True       # the tester's own label, compared afterwards
    assert 'BD_USER_ABANDONED_OTP' not in json.dumps(labelled['inputs'])                           # and never part of the stored inputs
    ds = call('POST', '/revenue/events/from-dataset', {'dataset': 'routing_fixtures', 'payment_id': _fixture('CLEAR_AND_CONSISTENT')['payment_id']})[1]
    assert 'evaluation' not in ds and 'hidden' not in json.dumps(ds).lower()


def test_invalid_custom_events_are_refused_with_field_errors():
    for body, field in (({**BASE_PF, **OTP, 'issuer_id': 'HDFC'}, 'issuer_id'), ({**BASE_PF, **OTP, 'error_reason': 'payment_failed'}, 'error_reason'),
                        ({**BASE_PF, **OTP, 'error_reason': None}, 'error_reason'), ({**BASE_PF, **OTP, 'gw_error_step': 'payment_authorization'}, 'gw_error_step'),
                        ({**OTHER['CHECKOUT_ABANDONMENT'], 'amount_rupees': 123}, 'amount_rupees'), ({**OTHER['CHECKOUT_ABANDONMENT'], 'method': 'upi'}, 'method')):
        s, body_ = call('POST', '/revenue/events', body)
        assert s == 422 and any(e['loc'] == ['body', field] for e in body_['detail']), (field, body_)
    assert call('POST', '/revenue/events', {**OTHER['CHECKOUT_ABANDONMENT'], 'free_text_note': 'hello'})[0] == 422        # unknown fields are rejected


def test_removed_endpoints_are_gone_and_historical_interventions_are_read_only():
    for method, path in (('POST', '/webhooks/razorpay'), ('POST', '/simulate/failure'), ('POST', '/diagnose'), ('POST', '/tests/payments'), ('GET', '/tests/runs'),
                         ('GET', '/metrics/uplift'), ('GET', '/payments/pay_x/timeline'), ('GET', '/interventions/temporal/pay_x')):
        assert call(method, path, {} if method == 'POST' else None)[0] in (404, 405), (method, path)
    status, hist = call('GET', '/interventions/historical?limit=5')
    assert status == 200 and hist['read_only'] is True and hist['total'] >= len(hist['items'])
    assert call('POST', '/interventions/historical', {})[0] in (404, 405) and call('DELETE', '/interventions/historical')[0] in (404, 405)


GENERIC_IN = dict(BASE_PF, method='upi', instrument_hint='na', gw_error_code='GATEWAY_ERROR', gw_error_source='beneficiary_bank', gw_error_step='payment_debit_response', error_reason=None)
CONTRA_IN = dict(BASE_PF, method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='customer', gw_error_step='payment_authentication', error_reason='card_expired')
CLEAR_IN = dict(BASE_PF, **OTP)


def test_the_three_diagnosis_scenarios_run_through_the_production_pipeline():
    clear = create({**CLEAR_IN, 'diagnosis_scenario': 'clear'})
    assert clear['route'] == 'CLEAR_AND_CONSISTENT' and clear['diagnosis']['method'] == 'rule' and clear['diagnosis']['model_c_calls'] == 0
    generic = create({**GENERIC_IN, 'diagnosis_scenario': 'generic', 'evaluation_label': 'TD_ISSUER_DOWN'})
    d = generic['diagnosis']
    assert generic['route'] == 'MISSING_OR_GENERIC' and d['method'] == 'model_c' and d['model_c_calls'] == 1 and d['model_c']['decision'] and generic['inputs']['diagnosis_scenario'] == 'generic'
    contra = create({**CONTRA_IN, 'diagnosis_scenario': 'contradictory'})
    assert contra['route'] == 'CONTRADICTORY' and contra['diagnosis']['model_c_calls'] == 0 and contra['diagnosis']['method'] == 'none' and contra['plan']['steps'] == []
    assert contra['workflow_state'] in ('NOT_REQUIRED', 'NOT_STARTED') and not contra.get('workflow_id')                            # no Temporal workflow for a contradictory record
    for ev in (clear, generic):                                                                                                   # the other two really run in Temporal
        got = wait(lambda e=ev: (v := view(e['event_id'])) and v['workflow_id'] and v, what='the workflow to start')
        assert got['workflow_id'].startswith('revenue-') or got['workflow_id']
    assert contra['event_id'] in {i['id'] for i in call('GET', '/interventions?limit=500')[1]['items']}
    assert 'evaluation_label' not in json.dumps(view(generic['event_id'])['inputs'])                                              # the hidden label is not echoed as an input


def test_a_scenario_cannot_be_used_to_smuggle_in_inputs_it_does_not_allow():
    status, body = call('POST', '/revenue/events', {**CLEAR_IN, 'diagnosis_scenario': 'generic'})
    assert status == 422
    status, body = call('POST', '/revenue/events', {**GENERIC_IN, 'diagnosis_scenario': 'clear'})
    assert status == 422
    status, body = call('POST', '/revenue/events', {**CLEAR_IN, 'diagnosis_scenario': 'route=MODEL_C'})
    assert status == 422
    status, body = call('POST', '/revenue/events', {**CLEAR_IN, 'route': 'MISSING_OR_GENERIC'})
    assert status == 422                                                                                                          # unknown fields (a route, a prediction) are refused


def test_the_label_does_not_change_the_route_or_the_number_of_model_c_calls():
    """(The exact probabilities can differ a little between two events: the simulated failure day is derived from the event id and Model C sees calendar features.
    Label independence at identical event id is proven in tests/test_diagnosis_scenarios.py.)"""
    a = create({**GENERIC_IN, 'diagnosis_scenario': 'generic'})
    b = create({**GENERIC_IN, 'diagnosis_scenario': 'generic', 'evaluation_label': 'RISK_BLOCKED'})
    pick = lambda ev: (ev['route'], ev['diagnosis']['method'], ev['diagnosis']['model_c_calls'], ev['diagnosis']['rule_id'])
    assert pick(a) == pick(b) == ('MISSING_OR_GENERIC', 'model_c', 1, 'NEEDS_MODEL')
