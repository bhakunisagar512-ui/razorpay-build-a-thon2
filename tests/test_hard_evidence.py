from dataclasses import replace
from datetime import datetime,timedelta,timezone
import math
import numpy as np
import pandas as pd

from services.app.domain.features import FEATURES,CATEGORICAL_FEATURES,NUMERIC_FEATURES
from services.app.domain.hard_evidence import HardEvidence,evidence_verdict,evidence_veto,ambiguity_veto
from eval.ambiguity_analysis import analyze_pool

NOW = datetime(2026,10,29,12,tzinfo=timezone.utc)


def verdict(item):
    return evidence_verdict([item],payment_id='pay',instrument_id='inst',as_of=NOW,method='card',issuer_id='bank')


def evidence(**kw):
    return HardEvidence(payment_id='pay',instrument_id='inst',observed_at=NOW,
                        authenticated=True,source='razorpay_signed_webhook',**kw)


def test_specific_reason_is_hard_but_generic_decline_is_not():
    assert verdict(evidence(error_reason='card_expired'))['hard_cause']=='CARD_EXPIRED'
    assert verdict(evidence(error_reason='payment_risk_check_failed'))['hard_cause']=='RISK_BLOCKED'
    assert verdict(evidence(error_reason='card_declined'))['hard_cause'] is None


def test_future_stale_unsigned_or_wrong_instrument_evidence_is_ignored():
    item = evidence(error_reason='card_expired')
    for changed in (replace(item,observed_at=NOW+timedelta(seconds=1)),
                    replace(item,observed_at=NOW-timedelta(hours=1)),replace(item,authenticated=False),
                    replace(item,instrument_id='other'),replace(item,payment_id='other')):
        assert verdict(changed)['hard_cause'] is None


def test_expiry_requires_trusted_source_and_current_scope():
    item = replace(evidence(expiry_year=2026,expiry_month=9),source='merchant_vault_verified')
    assert verdict(item)['hard_cause']=='CARD_EXPIRED'
    assert verdict(replace(item,expiry_month=10))['expiry_verified_valid']
    assert not verdict(replace(item,expiry_month=13))['expiry_verified_valid']
    assert not verdict(replace(item,observed_at=NOW-timedelta(days=31)))['hard_cause']


def test_downtime_does_not_override_expired_card_or_grant_a_retry():
    item = evidence(error_reason='card_expired',downtime_status='started',method='card',issuer_id='bank')
    result = verdict(item)
    assert result['temporary_hold'] and result['hard_cause']=='CARD_EXPIRED'
    decision = {'cause':'BD_INSUFFICIENT_FUNDS','retry_allowed':True,'decision':'reliable_diagnosis'}
    assert not evidence_veto(decision,result)['retry_allowed']
    blocked = {'cause':'AMBIGUOUS','retry_allowed':False,'decision':'low_confidence'}
    assert evidence_veto(blocked,result)==blocked


def test_unknown_decline_namespace_cannot_be_assumed_nonretryable():
    assert verdict(evidence(issuer_decline_code='54',issuer_namespace='unknown'))['hard_cause'] is None


def test_verified_risk_is_not_downgraded_by_expired_instrument_metadata():
    items = [evidence(error_reason='payment_risk_check_failed'),
             replace(evidence(expiry_year=2026,expiry_month=9),source='merchant_vault_verified')]
    result = evidence_verdict(items,payment_id='pay',instrument_id='inst',as_of=NOW)
    assert result['hard_cause']=='RISK_BLOCKED'


def test_downtime_requires_all_selectors_and_a_same_instrument_retry():
    item = evidence(downtime_status='started',method='card',issuer_id='bank',psp_id='other')
    assert not evidence_verdict([item],payment_id='pay',instrument_id='inst',as_of=NOW,method='card',issuer_id='bank',psp_id='psp')['temporary_hold']
    result = verdict(evidence(downtime_status='started',method='card',issuer_id='bank'))
    decision = {'cause':'TD_ISSUER_DOWN','retry_allowed':True,'decision':'reliable_diagnosis'}
    assert evidence_veto(decision,result)['retry_allowed']  # May switch to an unaffected rail.
    assert not evidence_veto(decision,result,retry_on_affected_instrument=True)['retry_allowed']


def test_ambiguity_requires_abstention_without_negative_evidence():
    decision = {'cause':'BD_INSUFFICIENT_FUNDS','retry_allowed':True,'decision':'reliable_diagnosis'}
    empty = {'hard_cause':None,'expiry_verified_valid':False,'temporary_hold':False}
    kwargs = dict(method='card',instrument_hint='unknown',error_tuple=('GATEWAY_ERROR','gateway','authorization'))
    assert not ambiguity_veto(decision,verdict=empty,**kwargs)['retry_allowed']
    assert ambiguity_veto(decision,verdict={**empty,'expiry_verified_valid':True},**kwargs)['retry_allowed']
    assert not ambiguity_veto(decision,verdict=empty,**{**kwargs,'instrument_hint':'valid'})['retry_allowed']


def test_fixture_does_not_read_true_cause_or_change_past_observation_time():
    from eval.evidence_fixture import EvidenceProfile,make_evidence
    profile = EvidenceProfile('complete',1.,1.,1.)
    row = dict(payment_id='pay',instrument_id='inst',observed_at=NOW,method='card',
               issuer_id='bank',psp_id='psp',fixture_expired=True,fixture_risk_blocked=False,
               fixture_downtime=False,fixture_issuer_down=False,fixture_psp_down=False,true_cause='CARD_EXPIRED')
    first = make_evidence(row,profile)
    assert make_evidence({**row,'true_cause':'TD_PSP_TIMEOUT'},profile)==first
    assert all(item.observed_at==NOW for item in first)


def test_evidence_comparison_preserves_binary_results_and_uses_available_scope():
    from eval.phase26 import evidence_policies
    from eval.evidence_fixture import EvidenceProfile
    row = frame_row('pay','CARD_EXPIRED',method='card',instrument_hint='unknown',observed_at=NOW)
    frame = pd.DataFrame([row])
    data = {'frame':frame,'baseline':(np.array(['BD_INSUFFICIENT_FUNDS'],dtype=object),
            np.array(['reliable_diagnosis'],dtype=object),np.array([True]))}
    raw = frame.assign(instrument_id='inst',fixture_expired=True,fixture_risk_blocked=False,
                       fixture_downtime=False,fixture_issuer_down=False,fixture_psp_down=False)
    results,_ = evidence_policies(data,raw,EvidenceProfile('complete',1.,1.),'unknown_expiry')
    assert results['binary'][2][0]
    assert not results['evidence'][2][0] and not results['ambiguity'][2][0]
    assert results['evidence'][0][0]=='CARD_EXPIRED'


def frame_row(pid,cause,**changes):
    row = {name:'x' for name in CATEGORICAL_FEATURES}
    row.update({name:0. for name in NUMERIC_FEATURES})
    row.update(payment_id=pid,true_cause=cause,world_family='fixture',world_seed=1)
    row.update(changes)
    return row


def test_exact_observable_conflict_is_detected_without_a_classifier():
    pool = pd.DataFrame([frame_row('a','CARD_EXPIRED'),frame_row('b','BD_INSUFFICIENT_FUNDS')])
    report = analyze_pool(pool,[{'payment_id':'a','true_cause':'CARD_EXPIRED','predicted_cause':'BD_INSUFFICIENT_FUNDS'}])
    assert report['exact_conflicting_cells']==1
    assert report['unsafe_cases_with_exact_retryable_match']==1


def test_near_conflict_is_not_reported_as_exact():
    pool = pd.DataFrame([frame_row('a','CARD_EXPIRED'),frame_row('b','BD_INSUFFICIENT_FUNDS',customer_last_success_age_s=30.)])
    report = analyze_pool(pool,[{'payment_id':'a','true_cause':'CARD_EXPIRED','predicted_cause':'BD_INSUFFICIENT_FUNDS'}])
    assert report['exact_conflicting_cells']==0
    assert report['unsafe_cases_with_near_retryable_match']==1
