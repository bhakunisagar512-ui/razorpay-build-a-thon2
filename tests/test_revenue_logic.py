"""Revenue-risk pipeline, pure logic (no database, no Temporal): event validation, the three-route router, one-time Model C use,
deterministic planning, the stopping rules and the in-process executor. Phases 2, 3, 5, 6."""
import inspect
from datetime import datetime, timedelta, timezone

import pytest

from services.app.domain import custom_input_rules as R, guards as G
from services.app.revenue import catalog as C, engine as E, fixtures, planner, routing, runs, service as svc

IST = G.IST
ALL = {k: True for k in C.CHANNELS}
UPI_CLEAR = dict(method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='customer', gw_error_step='payment_authentication', error_reason='payment_cancelled')


def info(**o):
    return {**UPI_CLEAR, **o}


# ------------------------------------------------------------------------------------------------- Phase 2: four event types, controlled input
CUSTOM = {
    C.CHECKOUT: dict(customer_id='cust_syn_01', amount_rupees=499, failure_time='t_1200', checkout_stage='payment_page'),
    C.SUBSCRIPTION: dict(customer_id='cust_syn_01', plan_id='plan_pro_monthly', method='card', failure_kind='CARD_EXPIRED', failure_time='t_1200'),
    C.OVERDUE: dict(customer_id='cust_syn_01', invoice_id='inv_syn_03', failure_time='t_1200'),
    C.PAYMENT_FAILURE: dict(customer_id='cust_syn_01', amount_rupees=499, issuer_id='issuer_01', failure_time='t_1200', **UPI_CLEAR),
}


@pytest.mark.parametrize('etype', C.EVENT_TYPES)
def test_every_event_type_validates_diagnoses_and_plans(etype):
    assert svc.validate_custom(etype, CUSTOM[etype]) == []
    spec = svc.custom_spec(etype, CUSTOM[etype], event_id='evt_test')
    diagnosis, plan = svc.decide(spec)
    assert plan['steps'] and planner.validate(plan) == [] and spec['sim_start'].tzinfo is not None
    assert diagnosis['model_c_calls'] == 0                                                     # none of these four custom events needs Model C


def test_events_accept_only_catalog_values_and_reject_everything_else():
    for etype in C.EVENT_TYPES:
        good = CUSTOM[etype]
        assert svc.validate_custom(etype, {**good, 'customer_id': 'cust_made_up'})[0]['field'] == 'customer_id'
        assert svc.validate_custom(etype, {**good, 'failure_time': '25:99'})
        assert svc.validate_custom(etype, {k: v for k, v in good.items() if k != 'customer_id'})[0]['field'] == 'customer_id'   # required
        assert svc.validate_custom(etype, {**good, 'invoice_id': 'inv_syn_01'}) if etype != C.OVERDUE else True               # a field of another type is refused
    assert svc.validate_custom(C.CHECKOUT, {**CUSTOM[C.CHECKOUT], 'amount_rupees': 123})                                    # only the predefined amounts
    assert svc.validate_custom(C.PAYMENT_FAILURE, {**CUSTOM[C.PAYMENT_FAILURE], 'issuer_id': 'HDFC'})                       # no arbitrary issuer
    assert svc.validate_custom(C.SUBSCRIPTION, {**CUSTOM[C.SUBSCRIPTION], 'method': 'upi'})                                 # an expired card cannot fail a UPI subscription
    assert svc.validate_custom('REFUND_REQUEST', {})[0]['field'] == 'event_type'


def test_event_ids_are_generated_and_the_catalog_has_no_free_text():
    ids = {svc.new_event_id(t) for t in C.EVENT_TYPES for _ in range(3)}
    assert len(ids) == 12 and all(i.split('_')[0] in C.EVENT_PREFIX.values() for i in ids)
    view = C.catalog_view()
    assert len(view['event_types']) == 4 and view['issuers'] == [f'issuer_{i:02d}' for i in range(10)] and len(view['customers']) == 8


# ------------------------------------------------------------------------------------------------- Custom Testing scope: clear and consistent only
def test_custom_scope_offers_only_inputs_that_route_as_clear():
    for m in R.METHODS:
        rows = R.scenarios(m, 'clear')
        assert rows and rows < R.scenarios(m)
        for hint, code, source, step, reason in rows:
            a = routing.assess({'method': m, 'instrument_hint': hint, 'gw_error_code': code, 'gw_error_source': source, 'gw_error_step': step, 'error_reason': reason})
            assert a.route == routing.CLEAR and a.cause == R.clear_reason_cause(m, reason), (m, reason)
        offered = {r[4] for r in rows}
        assert None not in offered and not offered & {'payment_failed', 'payment_declined', 'debit_declined', 'card_declined'}


def test_the_dropdown_vocabulary_is_not_broadened_and_the_default_scope_is_unchanged():
    res = R.build_options({'method': 'upi'}, scope='clear')
    reasons = [o['value'] for o in next(f for f in res['fields'] if f['field'] == 'error_reason')['options']]
    assert None not in reasons and 'payment_failed' not in reasons and 'payment_cancelled' in reasons
    assert None in [o['value'] for o in next(f for f in R.build_options({'method': 'upi'})['fields'] if f['field'] == 'error_reason')['options']]    # default scope as before
    assert svc.validate_custom(C.PAYMENT_FAILURE, {**CUSTOM[C.PAYMENT_FAILURE], 'error_reason': None})                       # missing reason: not for Custom Testing
    assert svc.validate_custom(C.PAYMENT_FAILURE, {**CUSTOM[C.PAYMENT_FAILURE], 'error_reason': 'payment_failed'})            # generic: not for Custom Testing


# ------------------------------------------------------------------------------------------------- Phase 3: the three routes
def test_clear_and_consistent_uses_a_rule_and_never_calls_model_c(monkeypatch):
    calls = []
    monkeypatch.setattr(routing, 'model_c_diagnose', lambda f: calls.append(f))
    for reason, cause in (('insufficient_funds', 'BD_INSUFFICIENT_FUNDS'), ('bank_technical_error', 'TD_ISSUER_DOWN'), ('upi_app_technical_error', 'TD_PSP_TIMEOUT'),
                          ('payment_risk_check_failed', 'RISK_BLOCKED'), ('incorrect_pin', 'BD_USER_ABANDONED_OTP')):
        step = {'insufficient_funds': 'payment_debit_response', 'bank_technical_error': 'payment_debit_request', 'upi_app_technical_error': 'payment_request',
                'payment_risk_check_failed': 'payment_debit_response', 'incorrect_pin': 'payment_authentication'}[reason]
        source = {'incorrect_pin': 'customer'}.get(reason, 'issuer_bank')
        code = 'BAD_REQUEST_ERROR' if reason in ('insufficient_funds', 'upi_app_technical_error', 'incorrect_pin') else 'GATEWAY_ERROR'
        d = routing.diagnose(info(error_reason=reason, gw_error_step=step, gw_error_source=source, gw_error_code=code), features_provider=lambda: pytest.fail('no features needed'))
        assert (d['route'], d['method'], d['cause'], d['model_c_calls']) == (routing.CLEAR, 'rule', cause, 0) and d['rule_id'] == f'REASON:{reason}' and d['explanation']
    assert not calls


def test_the_same_reason_can_be_clear_on_one_method_and_ambiguous_on_another():
    upi = routing.assess(info(error_reason='authentication_failed'))
    card = routing.assess({**UPI_CLEAR, 'method': 'card', 'instrument_hint': 'valid', 'gw_error_step': 'payment_authentication', 'error_reason': 'authentication_failed'})
    assert upi.route == routing.CLEAR and card.route == routing.GENERIC


def test_missing_or_generic_calls_model_c_exactly_once_and_uses_its_gate(monkeypatch):
    calls = []

    def fake(features):
        calls.append(features)
        return {'status': 'scored', 'cause': 'BD_INSUFFICIENT_FUNDS', 'confidence': .93, 'decision': 'reliable_diagnosis', 'route': 'AUTO_RETRY', 'top_cause': 'BD_INSUFFICIENT_FUNDS',
                'model_id': 'm', 'probabilities': {c: 0.0 for c in R.CAUSE_INFO} | {'BD_INSUFFICIENT_FUNDS': .93, 'TD_ISSUER_DOWN': .07}}
    monkeypatch.setattr(routing, 'model_c_diagnose', fake)
    for reason in (None, 'payment_failed', 'payment_declined', 'debit_declined'):
        calls.clear()
        code = R.codes_for_reason(reason)[0]                                                    # each generic reason with a code it can really carry
        d = routing.diagnose(info(error_reason=reason, gw_error_code=code), features_provider=lambda: {'row': 1})
        assert (d['route'], d['method'], d['model_c_calls'], d['cause']) == (routing.GENERIC, 'model_c', 1, 'BD_INSUFFICIENT_FUNDS') and len(calls) == 1


def test_model_c_abstention_uses_the_conservative_reminder_not_a_guess(monkeypatch):
    monkeypatch.setattr(routing, 'model_c_diagnose', lambda f: {'status': 'scored', 'cause': 'AMBIGUOUS', 'confidence': .5, 'decision': 'low_confidence', 'route': 'REVIEW',
                                                                'top_cause': 'BD_USER_ABANDONED_OTP', 'model_id': 'm', 'probabilities': {'BD_USER_ABANDONED_OTP': .5, 'BD_INSUFFICIENT_FUNDS': .5}})
    d = routing.diagnose(info(error_reason=None), features_provider=lambda: {})
    plan = planner.build(C.PAYMENT_FAILURE, cause=d['cause'], consent=ALL)
    assert d['cause'] == 'AMBIGUOUS' and [s['action'] for s in plan['steps']] == ['verify_payment_state', 'payment_reminder'] and plan['escalate_on_exhaustion']


def test_contradictory_goes_to_a_human_without_model_c(monkeypatch):
    monkeypatch.setattr(routing, 'model_c_diagnose', lambda f: pytest.fail('Model C must not run for contradictory input'))
    cases = [info(gw_error_step='payment_authorization'), info(error_reason='insufficient_funds', gw_error_code='GATEWAY_ERROR'), info(error_reason='card_expired'),
             info(instrument_hint='valid'), info(error_reason='insufficient_funds', gw_error_code='SERVER_ERROR', gw_error_step='payment_debit_response'),
             info(error_reason='Customer abandoned authentication without completing the payment.'),
             {**UPI_CLEAR, 'method': 'card', 'instrument_hint': 'expired', 'error_reason': 'insufficient_funds', 'gw_error_source': 'issuer_bank', 'gw_error_step': 'payment_authorization'}]
    for case in cases:
        d = routing.diagnose(case, features_provider=lambda: pytest.fail('no features for a contradiction'))
        assert d['route'] == routing.CONTRADICTORY and d['cause'] is None and d['model_c_calls'] == 0 and d['conflicts'] and all(c['fields'] and c['message'] for c in d['conflicts'])
        spec = {'event_type': C.PAYMENT_FAILURE, 'info': case, 'features_provider': lambda: pytest.fail('no features'), 'consent': ALL}
        plan = svc.decide(spec)[1]
        assert plan['steps'] == []                                                              # no automatic recovery action


def test_missing_is_not_contradictory():
    for reason in (None, 'payment_failed', ''):
        assert routing.assess(info(error_reason=reason)).route == routing.GENERIC
    assert routing.assess(info(error_reason='missing')).route == routing.GENERIC               # the dataset spelling of "no reason"


def test_every_in_distribution_scenario_is_routable_and_never_contradictory():
    for m in R.METHODS:
        for hint, code, source, step, reason in R.scenarios(m):
            a = routing.assess({'method': m, 'instrument_hint': hint, 'gw_error_code': code, 'gw_error_source': source, 'gw_error_step': step, 'error_reason': reason})
            assert a.route in (routing.CLEAR, routing.GENERIC), (m, hint, code, source, step, reason, a.conflicts)


def test_routing_fixtures_cover_all_three_routes_as_designed():
    recs = fixtures.records()
    assert len(recs) == 66 and {r['expected_route'] for r in recs} == set(routing.ROUTES)
    for r in recs:
        assert routing.assess(r).route == r['expected_route'], r['payment_id']


def test_model_c_wrapper_adds_nothing_to_the_frozen_model():
    """The router's call returns the frozen adapter's probabilities and gate verdict, plus only the independent-risk rule."""
    import json
    from pathlib import Path
    from services.app import test_datasets as tds
    from services.app.adapters import diagnosis_v4
    from services.app.domain.safety_policy_v2 import detect_independent_risk
    golden = json.loads((Path(__file__).parent / 'fixtures' / 'model_c_golden.json').read_text(encoding='utf-8'))
    for case in golden['predictions']:
        row = tds.derive_single({**case['input'], 'payment_id': 'parity', 'status': 'failed'})
        mine = routing.model_c_diagnose(row)
        d = diagnosis_v4.diagnose(gw_error_code=row['gw_error_code'], gw_error_source=row['gw_error_source'], gw_error_step=row['gw_error_step'], method=row['method'],
                                  instrument_hint=row['instrument_hint'], features=row)
        assert mine['probabilities'] == d.probabilities == case['expected']['probabilities']
        risky = detect_independent_risk(d.cause, d.probabilities)
        assert (mine['cause'], mine['decision']) == (('RISK_BLOCKED', 'independent_risk_detection') if risky else (d.cause, d.decision))


# ------------------------------------------------------------------------------------------------- Phase 5: one diagnosis, never a second one
def test_a_rejected_recovery_never_triggers_another_diagnosis(monkeypatch):
    spec = svc.fixture_spec(next(r for r in fixtures.records() if r['expected_route'] == routing.GENERIC), 'evt_once')
    calls = []
    real = routing.model_c_diagnose
    monkeypatch.setattr(routing, 'model_c_diagnose', lambda f: (calls.append(1), real(f))[1])
    monkeypatch.setattr(E, 'simulate', lambda *a, **k: 'REJECTED')                              # every payment opportunity is rejected
    result = runs.run_case(spec)
    assert len(calls) == 1 and result['model_c_calls'] == 1 and result['status'] in (C.FAILED, C.HUMAN)
    assert result['executed'] >= 1 and all(s.get('decision', 'REJECTED') == 'REJECTED' for s in result['steps'])


def test_the_workflow_and_executor_have_no_way_to_diagnose():
    from services.app.workflows import revenue, revenue_activities
    for module in (revenue, revenue_activities, E):
        src = inspect.getsource(module)
        for banned in ('diagnosis_v4', 'model_c_diagnose', 'routing', 'next_hypothesis', 'reassess', 'decide_v2'):
            assert banned not in src.replace('# ', ''), (module.__name__, banned)


# ------------------------------------------------------------------------------------------------- Phase 6: deterministic plans
def test_plans_follow_the_documented_policy():
    steps = lambda ev, cause=None, consent=ALL, **p: [s['action'] for s in planner.build(ev, cause=cause, consent=consent, params=p)['steps']]
    assert steps(C.PAYMENT_FAILURE, 'BD_INSUFFICIENT_FUNDS') == ['alternative_payment_method', 'payment_reminder', 'delayed_retry']
    assert steps(C.PAYMENT_FAILURE, 'CARD_EXPIRED') == ['card_update_request', 'alternative_payment_method']
    assert steps(C.PAYMENT_FAILURE, 'TD_ISSUER_DOWN') == ['delayed_retry', 'payment_reminder']
    assert steps(C.PAYMENT_FAILURE, 'TD_PSP_TIMEOUT') == ['verify_payment_state', 'payment_reminder']
    assert steps(C.PAYMENT_FAILURE, 'RISK_BLOCKED') == ['human_escalation']
    assert steps(C.CHECKOUT) == ['checkout_reminder', 'checkout_reminder'] and steps(C.OVERDUE, days_overdue=7) == ['invoice_reminder', 'invoice_reminder']
    assert len(steps(C.OVERDUE, days_overdue=60)) == 3 and steps(C.SUBSCRIPTION, 'CARD_EXPIRED')[0] == 'card_update_request'
    assert planner.build(C.OVERDUE, consent=ALL, params={'days_overdue': 60})['escalate_on_exhaustion'] and not planner.build(C.OVERDUE, consent=ALL, params={'days_overdue': 7})['escalate_on_exhaustion']


def test_every_customer_facing_action_carries_a_payment_opportunity_and_a_consented_channel():
    for ev in C.EVENT_TYPES:
        for cause in (None, *C.FAILURE_KIND_IDS, 'AMBIGUOUS'):
            for consent in (ALL, {'whatsapp': False, 'sms': True, 'email': False}, {k: False for k in C.CHANNELS}):
                plan = planner.build(ev, cause=cause, consent=consent, params={'days_overdue': 30})
                assert planner.validate(plan) == [], (ev, cause, consent)
                for s in plan['steps']:
                    if s['customer_facing']:
                        assert s['payment_opportunity'] and consent[s['channel']]
    assert 'Complete payment' in planner.PAYMENT_MESSAGE


def test_hard_causes_are_never_retried_and_an_uncertain_state_is_never_retried():
    for ev in (C.PAYMENT_FAILURE, C.SUBSCRIPTION):
        for cause in ('CARD_EXPIRED', 'RISK_BLOCKED', 'TD_PSP_TIMEOUT', 'BD_USER_ABANDONED_OTP', 'AMBIGUOUS'):
            assert not any(s['system_retry'] for s in planner.build(ev, cause=cause, consent=ALL)['steps']), cause
    bad = planner.build(C.PAYMENT_FAILURE, cause='CARD_EXPIRED', consent=ALL)
    bad['steps'].append({**bad['steps'][0], 'idx': 9, 'system_retry': True})
    assert any('hard-decline' in p for p in planner.validate(bad))


def test_no_consent_means_escalation_not_an_invented_channel():
    plan = planner.build(C.PAYMENT_FAILURE, cause='BD_INSUFFICIENT_FUNDS', consent={k: False for k in C.CHANNELS})
    assert [s['action'] for s in plan['steps']] == ['human_escalation'] and not plan['escalate_on_exhaustion']
    assert planner.build(C.CHECKOUT, consent={k: False for k in C.CHANNELS})['steps'][0]['action'] == 'human_escalation'


def test_the_planner_is_deterministic_and_reads_no_hidden_label():
    a = planner.build(C.PAYMENT_FAILURE, cause='TD_ISSUER_DOWN', consent=ALL)
    assert a == planner.build(C.PAYMENT_FAILURE, cause='TD_ISSUER_DOWN', consent=ALL)
    assert 'truth' not in inspect.getsource(planner) and 'hidden' not in inspect.getsource(planner).lower().replace('no hidden label', '')


# ------------------------------------------------------------------------------------------------- stopping rules and the executor
def at(h, m=0, day=10):
    return datetime(2026, 5, day, h, m, tzinfo=IST)


def test_stopping_rules_keep_the_existing_constants_and_order():
    step = {'channel': 'whatsapp', 'system_retry': False}
    st = E.new_state(ALL)
    assert E.check(st, step, at(12)) == (True, None)
    assert E.check(st, step, at(22)) == (False, 'quiet_hours') and E.check(st, step, at(8, 59)) == (False, 'quiet_hours')
    assert E.next_permitted('quiet_hours', at(22), st) == at(9, day=11) and E.next_permitted('quiet_hours', at(7), st) == at(9)
    assert E.check(E.new_state({**ALL, 'whatsapp': False}), step, at(12)) == (False, 'consent_required')
    st['contacts'] = G.MAX_CONTACTS_PER_PAYMENT
    assert E.check(st, step, at(12)) == (False, 'max_contact_attempts')
    retry = {'channel': None, 'system_retry': True}
    st2 = E.new_state(ALL); st2['retries'] = G.MAX_RETRIES_PER_PAYMENT
    assert E.check(st2, retry, at(22)) == (False, 'max_retries')                                # a retry has no channel, so quiet hours do not apply to it
    st3 = E.new_state(ALL, [at(11) - timedelta(hours=i) for i in range(G.MAX_CUSTOMER_CONTACTS_PER_WEEK)])
    assert E.check(st3, step, at(12)) == (False, 'max_customer_contacts_per_week') and E.next_permitted('max_customer_contacts_per_week', at(12), st3) > at(12)
    paid = E.new_state(ALL); paid['paid'] = True
    assert E.check(paid, step, at(12)) == (False, 'order_paid_elsewhere')


def _run(cause_plan, consent=ALL, accept_at=None, start=at(12), verify='FAILED'):
    plan = cause_plan

    def fake(event_type, action, key, idx, hidden=None):
        return 'ACCEPTED' if idx == accept_at else 'REJECTED'
    old, old_verify = E.simulate, E.verify_state
    E.simulate = fake
    E.verify_state = lambda *a, **k: verify
    try:
        return E.execute(event_id='e', event_type=C.PAYMENT_FAILURE, customer_id='c', amount_minor=49900, plan=plan, consent=consent, sim_start=start)
    finally:
        E.simulate, E.verify_state = old, old_verify


def test_accept_stops_remaining_actions_and_counts_the_amount_once():
    plan = planner.build(C.PAYMENT_FAILURE, cause='BD_USER_ABANDONED_OTP', consent=ALL)
    out = _run(plan, accept_at=0)
    assert (out['status'], out['recovered_amount_minor'], len(out['steps'])) == (C.RECOVERED, 49900, 1)
    out2 = _run(plan, accept_at=1)                                                              # reject the first, accept a later opportunity
    assert (out2['status'], out2['recovered_amount_minor'], [s['decision'] for s in out2['steps']]) == (C.RECOVERED, 49900, ['REJECTED', 'ACCEPTED'])


def test_reject_continues_and_an_exhausted_plan_ends_simulated_failed_or_escalates():
    plan = planner.build(C.PAYMENT_FAILURE, cause='BD_USER_ABANDONED_OTP', consent=ALL)
    out = _run(plan)
    assert out['status'] == C.FAILED and out['executed'] == 2 and out['recovered_amount_minor'] == 0
    amb = _run(planner.build(C.PAYMENT_FAILURE, cause='AMBIGUOUS', consent=ALL))
    assert (amb['status'], amb['reason']) == (C.HUMAN, 'plan_exhausted_unresolved')
    risk = _run(planner.build(C.PAYMENT_FAILURE, cause='RISK_BLOCKED', consent=ALL))
    assert (risk['status'], risk['reason'], risk['executed']) == (C.HUMAN, 'risk_blocked', 1)


def test_unsafe_actions_are_suppressed_not_converted_and_nothing_safe_left_escalates():
    plan = planner.build(C.PAYMENT_FAILURE, cause='BD_USER_ABANDONED_OTP', consent=ALL)
    no_wa = _run(plan, consent={'whatsapp': False, 'sms': True, 'email': False})                # first step's channel is WhatsApp: suppressed; the SMS step still runs
    assert [s['status'] for s in no_wa['steps']] == ['SUPPRESSED', 'EXECUTED']
    none_left = _run(plan, consent={k: False for k in C.CHANNELS})
    assert (none_left['status'], none_left['reason'], none_left['executed']) == (C.HUMAN, 'no_safe_recovery_action', 0)


def test_quiet_hours_wait_on_the_simulated_clock_instead_of_failing():
    plan = planner.build(C.CHECKOUT, consent=ALL)
    out = _run(plan, start=at(21, 30))                                                          # the first reminder is due inside quiet hours
    first = datetime.fromisoformat(out['steps'][0]['sim_at']).astimezone(IST)
    assert first.hour >= 9 and first.day == 11 and out['steps'][0]['status'] == 'EXECUTED'


def test_the_simulator_is_deterministic_and_only_a_payment_counts_as_recovery():
    a = [E.simulate(C.CHECKOUT, 'checkout_reminder', 'k', i) for i in range(50)]
    assert a == [E.simulate(C.CHECKOUT, 'checkout_reminder', 'k', i) for i in range(50)] and {'ACCEPTED', 'REJECTED'} == set(a)
    assert E.simulate(C.PAYMENT_FAILURE, 'delayed_retry', 'k', 0, 'CARD_EXPIRED') == 'REJECTED'  # a retry on an expired card never succeeds


# ------------------------------------------------------------------------------------------------- planner review (PSP timeout, insufficient funds, issuer downtime)
from datetime import date


def plan_for(cause, today=date(2026, 3, 10), ev=C.PAYMENT_FAILURE):
    return planner.build(ev, cause=cause, consent=ALL, params={'today': today})


def cumulative(plan):
    out, t = [], 0
    for s in plan['steps']:
        t += s['delay_s']
        out.append((s['action'], t))
    return out


def test_a_timeout_is_verified_before_anyone_is_asked_to_pay_and_is_never_retried():
    for cause in ('TD_PSP_TIMEOUT', 'AMBIGUOUS'):
        plan = plan_for(cause)
        assert plan['steps'][0]['action'] == 'verify_payment_state' and not any(s['system_retry'] for s in plan['steps']) and planner.validate(plan) == []
        assert all(s['action'] != 'verify_payment_state' or not s['payment_opportunity'] for s in plan['steps'])     # the lookup itself requests nothing
    bad = plan_for('TD_PSP_TIMEOUT')
    bad['steps'] = bad['steps'][1:]                                                          # a plan that asks for payment before verifying
    assert any('before the payment state is verified' in p for p in planner.validate(bad))


def test_the_payment_state_check_resolves_before_any_payment_is_requested():
    plan = plan_for('TD_PSP_TIMEOUT')
    paid = _run(plan, verify='SUCCEEDED')                                                    # the original payment turned out to have succeeded
    assert (paid['status'], paid['reason'], paid['recovered_amount_minor'], paid['executed']) == (C.RECOVERED, 'original_payment_succeeded', 49900, 1)
    assert not any('decision' in s for s in paid['steps'])                                   # no payment request was ever made, and the amount is counted once
    unresolved = _run(plan, verify='UNRESOLVED')                                             # it cannot be told: a person checks, nobody is asked to pay again
    assert (unresolved['status'], unresolved['reason'], unresolved['recovered_amount_minor']) == (C.HUMAN, 'payment_state_unverified', 0)
    assert not any('decision' in s for s in unresolved['steps']) and unresolved['executed'] == 1
    failed = _run(plan, verify='FAILED', accept_at=1)                                        # only a CONFIRMED failure leads to a payment request
    assert [s['action'] for s in failed['steps']] == ['verify_payment_state', 'payment_reminder'] and failed['steps'][0]['verified'] == 'FAILED' and failed['status'] == C.RECOVERED


def test_the_simulated_status_lookup_is_deterministic_and_only_a_timeout_is_uncertain():
    states = [E.verify_state(f'k{i}') for i in range(600)]
    assert states == [E.verify_state(f'k{i}') for i in range(600)]
    share = {s: states.count(s) / 600 for s in ('SUCCEEDED', 'FAILED', 'UNRESOLVED')}
    assert abs(share['SUCCEEDED'] - .35) < .07 and abs(share['FAILED'] - .45) < .07 and abs(share['UNRESOLVED'] - .20) < .07
    for cause in ('BD_INSUFFICIENT_FUNDS', 'CARD_EXPIRED', 'TD_ISSUER_DOWN', 'RISK_BLOCKED', 'BD_USER_ABANDONED_OTP'):
        assert {E.verify_state(f'k{i}', cause) for i in range(50)} == {'FAILED'}               # a decline the gateway reported is a confirmed failure
    assert {E.verify_state(f'k{i}', 'TD_PSP_TIMEOUT') for i in range(300)} == {'SUCCEEDED', 'FAILED', 'UNRESOLVED'}


def test_insufficient_funds_is_never_retried_at_once_and_assumes_no_salary_date():
    first = None
    for day in (1, 3, 6, 7, 10, 20, 28):
        plan = plan_for('BD_INSUFFICIENT_FUNDS', date(2026, 3, day))
        acts = cumulative(plan)
        assert [a for a, _ in acts] == ['alternative_payment_method', 'payment_reminder', 'delayed_retry'] and planner.validate(plan) == []
        assert acts[-1][1] == planner.MIN_FUNDS_RETRY_GAP_S == 3 * 86400, (day, acts)         # one fixed, documented delay; the calendar date changes nothing
        assert sum(s['system_retry'] for s in plan['steps']) == 1
        first = first or plan['steps']
        assert plan['steps'] == first                                                         # identical for every date: no salary cycle is inferred
    assert not hasattr(planner, 'next_salary_window_s') and 'salary' not in plan['rationale'].lower().replace('no salary date is known, so none is assumed', '')
    bad = plan_for('BD_INSUFFICIENT_FUNDS')
    bad['steps'].insert(0, {**bad['steps'][-1], 'idx': 0, 'delay_s': 3600})
    assert any('within 3 days' in p for p in planner.validate(bad))


def test_issuer_downtime_is_retried_only_after_the_outage_has_had_time_to_clear():
    plan = plan_for('TD_ISSUER_DOWN')
    assert cumulative(plan)[0] == ('delayed_retry', 2 * 3600) and planner.validate(plan) == []
    bad = plan_for('TD_ISSUER_DOWN')
    bad['steps'][0]['delay_s'] = 1800
    assert any('bank outage' in p for p in planner.validate(bad))


def test_every_diagnosis_maps_to_a_justified_plan_with_the_retry_and_contact_limits_respected():
    justified = {'BD_INSUFFICIENT_FUNDS': 'alternative_payment_method', 'BD_USER_ABANDONED_OTP': 'payment_reminder', 'CARD_EXPIRED': 'card_update_request',
                 'TD_ISSUER_DOWN': 'delayed_retry', 'TD_PSP_TIMEOUT': 'verify_payment_state', 'RISK_BLOCKED': 'human_escalation', 'AMBIGUOUS': 'verify_payment_state'}
    for cause, first in justified.items():
        plan = plan_for(cause)
        assert plan['steps'][0]['action'] == first and planner.validate(plan) == [] and plan['rationale']
        retries = sum(s['system_retry'] for s in plan['steps'])
        contacts = sum(bool(s['channel']) for s in plan['steps'])
        assert retries <= 1 and retries <= G.MAX_RETRIES_PER_PAYMENT and contacts <= G.MAX_CONTACTS_PER_PAYMENT
        if cause in ('CARD_EXPIRED', 'RISK_BLOCKED', 'BD_USER_ABANDONED_OTP', 'TD_PSP_TIMEOUT', 'AMBIGUOUS'):
            assert retries == 0
