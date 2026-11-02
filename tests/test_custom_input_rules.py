"""Custom payment input compatibility (Change 1): rules, option filtering, API validation, feature pass-through, label independence.
No database, no Temporal and no recovery workflow is started by any test here."""
import inspect
import json
import random
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.app import main, test_api, test_datasets as tds
from services.app.revenue import service as svc
from services.app.domain import custom_input_rules as R
from services.app.domain.failure_vocab import ERROR_CODES, SOURCES, STEPS

SCENARIO = dict(method='upi', instrument_hint='na', gw_error_code='BAD_REQUEST_ERROR', gw_error_source='customer',
                gw_error_step='payment_authentication', error_reason='payment_cancelled')
BODY = dict(customer_id='cust_t', issuer_id='issuer_01', amount_minor=51900, occurred_at='2026-05-04T10:15:00+00:00', **SCENARIO)
TRUTH_KEYS = {'true_cause', 'actual_cause', 'synthetic_truth', 'ground_truth'}


@pytest.fixture(scope='module')
def client():
    return TestClient(main.app)


def opts(selected, changed=None):
    return R.build_options(selected, changed)


def values(res, field):
    return [o['value'] for o in next(f for f in res['fields'] if f['field'] == field)['options']]


# ------------------------------------------------------------------------------------------------- 1. each method and its options
def test_every_method_exposes_only_its_own_vocabulary():
    for m in R.METHODS:
        rows = R.scenarios(m)
        assert {r[2] for r in rows} <= set(SOURCES[m]) and {r[3] for r in rows} <= set(STEPS[m])
        assert {r[1] for r in rows} <= set(ERROR_CODES)
    assert {r[0] for r in R.scenarios('upi')} == {r[0] for r in R.scenarios('netbanking')} == {'na'}
    assert {r[0] for r in R.scenarios('card')} == {'valid', 'unknown', 'expired'} and R.default_hint('card') == 'unknown'
    assert R.default_hint('upi') == R.default_hint('netbanking') == 'na'
    assert {r[1] for r in R.scenarios('upi')} == {'BAD_REQUEST_ERROR', 'GATEWAY_ERROR'}                 # no UPI scenario produces server_error
    assert 'SERVER_ERROR' in {r[1] for r in R.scenarios('card')} | {r[1] for r in R.scenarios('netbanking')}
    assert 'payment_authorization' not in {r[3] for r in R.scenarios('upi')}                             # a card / netbanking step
    assert 'customer_psp' not in {r[2] for r in R.scenarios('card')}                                      # a UPI source


def test_labels_are_method_scoped_and_card_expired_is_card_only():
    assert set(R.labels_for('card')) == set(R.CAUSE_INFO)
    for m in ('upi', 'netbanking'):
        assert 'CARD_EXPIRED' not in R.labels_for(m) and len(R.labels_for(m)) == 5


def test_card_only_reasons_are_never_offered_for_upi_or_netbanking():
    for m in ('upi', 'netbanking'):
        assert not {r[4] for r in R.scenarios(m)} & R.CARD_ONLY_REASONS
        res = opts({'method': m})
        reasons = values(res, 'error_reason')
        assert not set(reasons) & R.CARD_ONLY_REASONS
        why = {w['value']: w['why'] for w in next(f for f in res['fields'] if f['field'] == 'error_reason')['withheld']}
        assert 'Card-only' in why['card_expired'] and m.upper()[:3] in why['card_expired'].upper()
    assert {'card_expired', 'incorrect_cvv'} <= {r[4] for r in R.scenarios('card')}
    assert R.CARD_ONLY_REASONS - {'card_expired', 'incorrect_cvv'} == {'incorrect_card_expiry_date', 'card_not_enrolled'}   # documented, no model signal
    assert not {'incorrect_card_expiry_date', 'card_not_enrolled'} & {r[4] for r in R.scenarios('card')}


def test_every_documented_value_has_a_description():
    assert set(R.ALL_REASONS) <= set(R.REASON_INFO) and set(R.ALL_STEPS) <= set(R.STEP_INFO) and set(R.ALL_SOURCES) <= set(R.SOURCE_INFO)


# ------------------------------------------------------------------------------------------------- 2. dependent dropdowns
def test_method_comes_first_and_disables_the_rest():
    res = opts({})
    assert values(res, 'method') == ['upi', 'card', 'netbanking']
    assert all(not f['enabled'] for f in res['fields'] if f['field'] != 'method') and not res['complete']


def test_selecting_source_and_step_narrows_reasons_in_both_directions():
    base = set(values(opts({'method': 'upi'}), 'error_reason'))
    by_source = set(values(opts({'method': 'upi', 'gw_error_source': 'customer'}), 'error_reason'))
    both = set(values(opts({'method': 'upi', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication'}), 'error_reason'))
    assert both < by_source < base and 'payment_cancelled' in both and 'insufficient_funds' not in both
    # reverse direction: a reason narrows the steps, and the steps narrow the reasons
    steps = values(opts({'method': 'upi', 'error_reason': 'insufficient_funds'}), 'gw_error_step')
    assert 'payment_debit_response' in steps and 'payment_authentication' not in steps
    # the error code follows the reason, and the reason follows the code
    assert values(opts({'method': 'card', 'error_reason': 'server_error'}), 'gw_error_code') == ['SERVER_ERROR']
    assert set(values(opts({'method': 'card', 'gw_error_code': 'SERVER_ERROR'}), 'error_reason')) <= {'server_error', None}
    assert values(opts({'method': 'upi', 'error_reason': 'payment_cancelled'}), 'gw_error_code') == ['BAD_REQUEST_ERROR', 'GATEWAY_ERROR']
    assert values(opts({'method': 'upi', 'error_reason': 'debit_declined'}), 'gw_error_code') == ['GATEWAY_ERROR']
    # an expired-card hint only goes with what the CARD_EXPIRED profile produces
    assert 'card_expired' in values(opts({'method': 'card', 'instrument_hint': 'expired'}), 'error_reason')
    assert 'incorrect_otp' not in values(opts({'method': 'card', 'instrument_hint': 'expired'}), 'error_reason')


def test_every_offered_option_leads_to_a_valid_scenario():
    rng = random.Random(7)
    for m in R.METHODS:
        for _ in range(60):
            row = rng.choice(sorted(R.scenarios(m), key=str))
            sel = {'method': m, **dict(zip(R.OBSERVABLE, row))}
            assert R.validate({**sel, 'true_cause': None}) == []
            fields = rng.sample(R.OBSERVABLE, 3)
            part = {'method': m, **{f: sel[f] for f in fields}}
            res = opts(part)
            assert all(sel[f] in values(res, f) for f in R.OBSERVABLE)                                   # the generating tuple stays reachable


# ------------------------------------------------------------------------------------------------- 3 + 4. clearing and preserving
def test_switching_method_clears_only_what_no_longer_fits_and_never_substitutes():
    sel = {'method': 'card', 'instrument_hint': 'valid', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication', 'error_reason': 'card_expired'}
    res = opts({**sel, 'method': 'upi', 'true_cause': 'CARD_EXPIRED'}, 'method')
    cleared = {c['field'] for c in res['cleared']}
    assert cleared == {'instrument_hint', 'error_reason', 'true_cause'}
    assert res['selected']['gw_error_source'] == 'customer' and res['selected']['gw_error_step'] == 'payment_authentication'   # still valid: kept
    assert 'true_cause' not in res['selected'] and res['selected']['instrument_hint'] == 'na' and res['applied_defaults'][0]['field'] == 'instrument_hint'
    assert 'error_reason' not in res['selected']                                                         # cleared, not replaced by another reason
    back = opts({**res['selected'], 'method': 'card'}, 'method')                                          # and back to card: 'na' is not a card value
    assert [c['field'] for c in back['cleared']] == ['instrument_hint'] and back['selected']['instrument_hint'] == 'unknown'


def test_the_evaluation_label_is_never_replaced_by_another_cause():
    for label in R.CAUSE_INFO:
        res = opts({'method': 'netbanking', 'true_cause': label}, 'method')
        assert res['selected'].get('true_cause') in (label, None)
        assert (label == 'CARD_EXPIRED') == any(c['field'] == 'true_cause' for c in res['cleared'])


def test_a_new_choice_keeps_compatible_selections_and_clears_conflicts():
    sel = {'method': 'upi', 'instrument_hint': 'na', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication', 'error_reason': 'payment_cancelled'}
    kept = opts({**sel, 'gw_error_source': 'customer_psp'}, 'gw_error_source')
    assert kept['cleared'] == [] and kept['selected']['error_reason'] == 'payment_cancelled'
    conflict = opts({**sel, 'error_reason': 'insufficient_funds'}, 'error_reason')                       # the changed field wins; the step yields
    assert [c['field'] for c in conflict['cleared']] == ['gw_error_step'] and conflict['selected']['error_reason'] == 'insufficient_funds'
    assert conflict['cleared'][0]['why']


# ------------------------------------------------------------------------------------------------- 5. API: 422 on incompatible input (POST /revenue/events)
EVENT = dict(event_type='PAYMENT_FAILURE', customer_id='cust_syn_01', amount_rupees=499, issuer_id='issuer_01', failure_time='t_1200', **SCENARIO)


def post(client, **over):
    return client.post('/revenue/events', json={**EVENT, **over})


def test_unsupported_combinations_return_422_with_a_field_error(client):
    for over, field in ((dict(gw_error_step='payment_authorization'), 'gw_error_step'),                   # card / netbanking step on UPI
                        (dict(error_reason='card_expired'), 'error_reason'),                              # card-only reason on UPI
                        (dict(error_reason='Customer abandoned authentication without completing the payment.'), 'error_reason'),   # free text
                        (dict(error_reason=None), 'error_reason'),                                        # missing: not for Custom Testing
                        (dict(error_reason='payment_failed'), 'error_reason'),                            # generic: not for Custom Testing
                        (dict(instrument_hint='valid'), 'instrument_hint'),
                        (dict(gw_error_code='SERVER_ERROR'), 'gw_error_code'),
                        (dict(gw_error_source='nonsense'), 'gw_error_source'),
                        (dict(evaluation_label='CARD_EXPIRED'), 'true_cause'),
                        (dict(issuer_id='HDFC'), 'issuer_id'),
                        (dict(method='card', instrument_hint='na', gw_error_source='customer_psp', gw_error_step='payment_authentication'), 'gw_error_source'),
                        (dict(method='card', instrument_hint='na'), 'instrument_hint'),
                        (dict(method='netbanking', instrument_hint='na', error_reason='incorrect_cvv', gw_error_step='payment_authentication'), 'error_reason')):
        r = post(client, **over)
        assert r.status_code == 422, over
        assert any(e['loc'] == ['body', field] and e['msg'] for e in r.json()['detail']), (over, r.json()['detail'])
    assert post(client, method='cheque').status_code == 422
    assert client.post('/revenue/events', json={**EVENT, 'free_text_note': 'hello'}).status_code == 422          # unknown fields are rejected


def test_the_investigated_scenario_is_valid_and_the_old_inputs_are_rejected():
    assert svc.validate_custom('PAYMENT_FAILURE', EVENT) == []
    old = svc.validate_custom('PAYMENT_FAILURE', {**EVENT, 'gw_error_step': 'payment_authorization', 'instrument_hint': 'valid',
                                                  'error_reason': 'Customer abandoned authentication without completing the payment.'})
    assert {e['field'] for e in old} >= {'gw_error_step', 'instrument_hint', 'error_reason'}


def test_options_endpoint_is_read_only_and_matches_the_rules(client):
    r = client.post('/tests/custom-options', json={'method': 'upi', 'gw_error_source': 'customer', 'changed': 'gw_error_source'})
    assert r.status_code == 200 and r.json()['selected']['instrument_hint'] == 'na'
    assert values(r.json(), 'error_reason') == values(R.build_options({'method': 'upi', 'gw_error_source': 'customer'}), 'error_reason')
    none_reason = client.post('/tests/custom-options', json={'method': 'upi', 'error_reason': None}).json()
    assert 'error_reason' in none_reason['selected'] and none_reason['selected']['error_reason'] is None


# ------------------------------------------------------------------------------------------------- 6. the selected values reach feature engineering unchanged
def test_selected_values_reach_the_feature_row_unchanged():
    f = tds.derive_single({**BODY, 'payment_id': 'p1', 'status': 'failed'})
    assert (f['method'], f['instrument_hint'], f['gw_error_code'], f['gw_error_source'], f['gw_error_step'], f['error_reason']) == \
        ('upi', 'na', 'BAD_REQUEST_ERROR', 'customer', 'payment_authentication', 'payment_cancelled')
    assert f['reason_family'] == 'customer_session' and f['is_generic_reason'] == 0 and f['step_ordinal'] == 2 and f['cold_start_features'] is True
    rng = random.Random(3)
    for m in R.METHODS:
        for row in rng.sample(sorted(R.scenarios(m), key=str), 25):
            sel = dict(zip(R.OBSERVABLE, row))
            f = tds.derive_single({**BODY, **sel, 'method': m, 'payment_id': 'p', 'status': 'failed'})
            assert (f['instrument_hint'], f['gw_error_code'], f['gw_error_source'], f['gw_error_step']) == (sel['instrument_hint'], sel['gw_error_code'], sel['gw_error_source'], sel['gw_error_step'])
            assert f['error_reason'] == sel['error_reason'] or (sel['error_reason'] is None and f['error_reason'] in (None, 'missing'))


# ------------------------------------------------------------------------------------------------- 7. the evaluation label never reaches inference
def test_the_label_never_changes_features_diagnosis_or_plan():
    specs = [svc.custom_spec('PAYMENT_FAILURE', {**EVENT, 'evaluation_label': label}, event_id='evt_same') for label in (None, *R.labels_for('upi'))]
    feats = [s['features_provider']() for s in specs]
    assert all(f == feats[0] for f in feats) and all('true_cause' not in f and 'evaluation_label' not in f for f in feats)
    assert all('evaluation_label' not in s['payload'] and 'true_cause' not in json.dumps(s['payload']) for s in specs)
    decided = [svc.decide(s) for s in specs]
    assert all(d == decided[0] for d in decided)                                          # same diagnosis and plan whatever the label
    assert [s['hidden_cause'] for s in specs] == [None, *R.labels_for('upi')]             # the label rides along for the simulator only
    assert 'true_cause' not in inspect.getsource(R.allowed_values) and 'true_cause' not in inspect.getsource(R.explain)


def test_the_label_and_the_observables_do_not_filter_each_other():
    plain = opts({'method': 'upi', 'gw_error_source': 'customer'})
    for label in R.labels_for('upi'):
        labelled = opts({'method': 'upi', 'gw_error_source': 'customer', 'true_cause': label})
        for f in R.OBSERVABLE:
            assert values(labelled, f) == values(plain, f)                                                # label never filters observables
    narrow = opts({'method': 'upi', 'gw_error_source': 'customer', 'gw_error_step': 'payment_authentication', 'error_reason': 'payment_cancelled'})
    assert values(narrow, 'true_cause') == values(plain, 'true_cause') == R.labels_for('upi')            # observables never filter the label


# ------------------------------------------------------------------------------------------------- 8. nothing else changed
def test_dataset_and_csv_paths_are_not_subject_to_the_custom_rules():
    from services.app import revenue_api
    assert 'rules.' not in inspect.getsource(revenue_api.create_dataset_event) and 'validate_custom' not in inspect.getsource(revenue_api.create_dataset_event)
    assert 'rules.' not in inspect.getsource(test_api.csv_validate) and 'rules.' not in inspect.getsource(revenue_api.start_run)


def test_frontend_has_no_free_text_scenario_fields():
    js = (Path(__file__).resolve().parents[1] / 'services' / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')
    assert 'dl-code' not in js and 'dl-source' not in js and 'dl-step' not in js and "tests/custom-options" in js
    for sel in ('c-method', 'c-hint', 'c-code', 'c-source', 'c-step', 'c-reason', 'c-truth'):
        assert f"'{sel}'" in js
    assert "h('input', { id: 'c-reason'" not in js and "F('c-reason'" not in js


def test_rules_explain_the_training_data():
    """The rule set must contain nearly all of the synthetic training rows (the rest is the generator's declared noise)."""
    import glob
    import pandas as pd
    files = sorted(glob.glob('eval/data/phase3_run/cache/*train*.pkl'))
    if not files:
        pytest.skip('Phase 3 training worlds are not present')
    df = pd.concat([pd.read_pickle(f)['failures'] for f in files], ignore_index=True)
    inside = 0
    for r in df.itertuples():
        reason = r.error_reason if isinstance(r.error_reason, str) and r.error_reason != 'missing' else None
        inside += (r.instrument_hint, r.gw_error_code, r.gw_error_source, r.gw_error_step, reason) in R.scenarios(r.method)
    assert inside / len(df) > 0.90
