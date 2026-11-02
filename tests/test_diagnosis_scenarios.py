"""Custom Testing "Diagnosis scenario" selector: it chooses which inputs can be BUILT, never the route. The router, the frozen Model C and the planner are
the production ones; these tests prove the selector cannot force a diagnosis, that each route behaves as specified, and that the hidden label is inert."""
import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.app import main
from services.app.domain import custom_input_rules as R
from services.app.revenue import catalog as C, routing, service as svc

NEEDS_MODEL = pytest.mark.skipif(not (Path('eval/artifacts/phase4') / 'model.joblib').exists(), reason='frozen Model C artifact not present')

BASE = {'event_type': C.PAYMENT_FAILURE, 'customer_id': 'cust_syn_01', 'issuer_id': 'issuer_01', 'amount_rupees': 499, 'failure_time': 't_1200', 'method': 'upi', 'instrument_hint': 'na'}
CLEAR_IN = {**BASE, 'gw_error_code': 'BAD_REQUEST_ERROR', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication', 'error_reason': 'payment_cancelled'}
GENERIC_IN = {**BASE, 'gw_error_code': 'GATEWAY_ERROR', 'gw_error_source': 'beneficiary_bank', 'gw_error_step': 'payment_debit_response', 'error_reason': None}
CONTRA_IN = {**BASE, 'gw_error_code': 'BAD_REQUEST_ERROR', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication', 'error_reason': 'card_expired'}


def body(inputs, scenario, label=None):
    return {**inputs, 'diagnosis_scenario': scenario, **({'evaluation_label': label} if label else {})}


def decide(inputs, scenario, label=None):
    f = body(inputs, scenario, label)
    assert svc.validate_custom(C.PAYMENT_FAILURE, f) == [], svc.validate_custom(C.PAYMENT_FAILURE, f)
    return svc.decide(svc.custom_spec(C.PAYMENT_FAILURE, f, 'evt_t'))


# ------------------------------------------------------------------------------------------------- what each scenario lets the tester build
def info(method, row):
    hint, code, source, step, reason = row
    return dict(method=method, instrument_hint=hint, gw_error_code=code, gw_error_source=source, gw_error_step=step, error_reason=reason)


def test_every_input_the_clear_scope_offers_is_routed_clear_and_every_generic_input_to_model_c():
    for method in R.METHODS:
        assert {routing.assess(info(method, r)).route for r in R.scenarios(method, 'clear')} == {routing.CLEAR}
        assert {routing.assess(info(method, r)).route for r in R.scenarios(method, 'generic')} == {routing.GENERIC}
        assert R.scenarios(method, 'clear').isdisjoint(R.scenarios(method, 'generic'))


def test_the_generic_scope_offers_missing_and_generic_reasons_but_no_clear_evidence():
    opts = R.build_options({'method': 'upi'}, None, 'generic')
    reasons = {o['value'] for o in next(f for f in opts['fields'] if f['field'] == 'error_reason')['options']}
    assert None in reasons and 'payment_failed' in reasons and 'payment_cancelled' not in reasons            # "no reason provided" is offered; a clear reason is withheld
    hints = {o['value'] for o in R.build_options({'method': 'card'}, None, 'generic')['fields'][1]['options']}
    assert 'expired' not in hints
    withheld = next(f for f in opts['fields'] if f['field'] == 'error_reason')['withheld']
    assert any(w['value'] == 'payment_cancelled' and 'router would decide by rule' in w['why'] for w in withheld)


def test_the_conflict_scope_offers_every_vocabulary_value_so_contradictions_can_be_built_but_never_free_text():
    opts = R.build_options({'method': 'upi'}, None, 'conflict')
    by = {f['field']: {o['value'] for o in f['options']} for f in opts['fields']}
    assert set(R.ALL_REASONS) | {None} == by['error_reason'] and set(R.ALL_SOURCES) == by['gw_error_source'] and set(R.ALL_HINTS) == by['instrument_hint']
    assert 'card_expired' in by['error_reason'] and 'valid' in by['instrument_hint']                         # card-only values on UPI: the contradiction
    assert R.validate({**{k: v for k, v in CONTRA_IN.items() if k in R.OBSERVABLE}, 'method': 'upi'}, scope='conflict') == []
    assert R.validate({**{k: v for k, v in CONTRA_IN.items() if k in R.OBSERVABLE}, 'method': 'upi', 'gw_error_source': 'made_up'}, scope='conflict')       # unknown token: rejected


def test_the_options_endpoint_previews_the_router_verdict_without_calling_model_c(monkeypatch):
    calls = routing.model_c_call_count()
    client = TestClient(main.app)
    for scope, sel, expected in (('clear', CLEAR_IN, routing.CLEAR), ('generic', GENERIC_IN, routing.GENERIC), ('conflict', CONTRA_IN, routing.CONTRADICTORY)):
        res = client.post('/tests/custom-options', json={**{k: v for k, v in sel.items() if k in R.OBSERVABLE or k == 'method'}, 'scope': scope}).json()
        assert res['complete'] and res['route_preview']['route'] == expected, (scope, res.get('cleared'), res.get('route_preview'))
    assert routing.model_c_call_count() == calls                                                          # a preview never runs the model


# ------------------------------------------------------------------------------------------------- the selector cannot override the router
def test_inputs_that_do_not_fit_the_selected_scenario_are_refused_not_rerouted():
    assert svc.validate_custom(C.PAYMENT_FAILURE, body(CLEAR_IN, 'generic'))                            # a clear reason cannot be built as "missing / generic"
    assert svc.validate_custom(C.PAYMENT_FAILURE, body(GENERIC_IN, 'clear'))                            # and a missing reason cannot be built as "clear"
    assert svc.validate_custom(C.PAYMENT_FAILURE, body(CONTRA_IN, 'clear')) and svc.validate_custom(C.PAYMENT_FAILURE, body(CONTRA_IN, 'generic'))
    assert svc.validate_custom(C.PAYMENT_FAILURE, body(CLEAR_IN, 'nonsense'))


def test_the_contradictory_scenario_does_not_force_a_contradiction():
    """Consistent evidence built under the contradictory scenario is still decided by rule: the route comes from the inputs alone."""
    diagnosis, _ = decide(CLEAR_IN, 'contradictory')
    assert diagnosis['route'] == routing.CLEAR and diagnosis['method'] == 'rule' and diagnosis['model_c_calls'] == 0


def test_the_scenario_value_never_reaches_routing_diagnosis_or_the_planner():
    for module in ('routing', 'planner', 'engine'):
        assert 'diagnosis_scenario' not in (Path('services/app/revenue') / f'{module}.py').read_text(encoding='utf-8'), module
    for inputs in (CLEAR_IN, GENERIC_IN, CONTRA_IN):
        results = []
        for scenario in svc.DIAGNOSIS_SCENARIOS:
            if svc.validate_custom(C.PAYMENT_FAILURE, body(inputs, scenario)):
                continue
            spec = svc.custom_spec(C.PAYMENT_FAILURE, body(inputs, scenario), 'evt_same')
            results.append(svc.decide(spec) if routing.assess(spec['info']).route != routing.GENERIC else routing.assess(spec['info']).as_dict())
        assert len(results) >= 1 and all(r == results[0] for r in results)


# ------------------------------------------------------------------------------------------------- the three routes, through the production decision path
def test_clear_and_consistent_is_decided_by_rule_with_zero_model_c_calls():
    before = routing.model_c_call_count()
    diagnosis, plan = decide(CLEAR_IN, 'clear')
    assert (diagnosis['route'], diagnosis['method'], diagnosis['model_c_calls']) == (routing.CLEAR, 'rule', 0) and diagnosis['cause'] == 'BD_USER_ABANDONED_OTP'
    assert routing.model_c_call_count() == before and plan['steps']


@NEEDS_MODEL
def test_missing_or_generic_calls_the_frozen_model_exactly_once_behind_its_gate():
    before = routing.model_c_call_count()
    diagnosis, plan = decide(GENERIC_IN, 'generic')
    assert routing.model_c_call_count() == before + 1 and diagnosis['model_c_calls'] == 1
    assert (diagnosis['route'], diagnosis['method']) == (routing.GENERIC, 'model_c')
    gate = diagnosis['model_c']
    assert gate['decision'] and gate['route'] and gate['model_id'].startswith('phase4-hgb-deployment_core') and diagnosis['cause']      # a cause, or AMBIGUOUS when the gate abstains
    assert plan['steps']                                                                                  # the conservative / cause plan, built once


def test_contradictory_goes_to_human_intervention_without_model_c():
    before = routing.model_c_call_count()
    diagnosis, plan = decide(CONTRA_IN, 'contradictory')
    assert (diagnosis['route'], diagnosis['method'], diagnosis['model_c_calls'], diagnosis['cause']) == (routing.CONTRADICTORY, 'none', 0, None)
    assert diagnosis['conflicts'] and plan['steps'] == [] and routing.model_c_call_count() == before


# ------------------------------------------------------------------------------------------------- the hidden label is inert
@NEEDS_MODEL
@pytest.mark.parametrize('inputs,scenario', [(CLEAR_IN, 'clear'), (GENERIC_IN, 'generic'), (CONTRA_IN, 'contradictory')])
def test_the_hidden_evaluation_label_never_changes_the_diagnosis_or_the_plan(inputs, scenario):
    reference = decide(inputs, scenario, None)
    for label in R.labels_for('upi'):
        got = decide(inputs, scenario, label)
        assert got == reference, label
    spec = svc.custom_spec(C.PAYMENT_FAILURE, body(inputs, scenario, 'RISK_BLOCKED'), 'evt_l')
    assert spec['hidden_cause'] == 'RISK_BLOCKED' and 'RISK_BLOCKED' not in str(spec['info']) and 'evaluation_label' not in spec['payload']


def test_the_label_is_used_only_by_the_simulator_and_the_after_the_fact_comparison():
    for module in ('routing', 'planner'):
        text = (Path('services/app/revenue') / f'{module}.py').read_text(encoding='utf-8')
        assert 'hidden_cause' not in text and 'evaluation_label' not in text and 'true_cause' not in text, module
    spec = svc.custom_spec(C.PAYMENT_FAILURE, body(GENERIC_IN, 'generic', 'CARD_EXPIRED'), 'evt_k')
    changed = copy.deepcopy(spec)
    changed['hidden_cause'] = 'TD_ISSUER_DOWN'
    # decide() reads the spec, so prove it ignores the label field itself: identical output for different hidden causes (Model C stubbed to avoid a second inference)
    routing_calls = []
    original = routing.model_c_diagnose
    routing.model_c_diagnose = lambda feats: (routing_calls.append(1), {'status': 'scored', 'cause': 'AMBIGUOUS', 'confidence': 0.0, 'decision': 'stub', 'route': 'REVIEW',
                                                                        'probabilities': {}, 'top_cause': None, 'model_id': 'stub'})[1]
    try:
        assert svc.decide(spec) == svc.decide(changed)
    finally:
        routing.model_c_diagnose = original
    assert len(routing_calls) == 2                                                                         # one call per decide(), never more
