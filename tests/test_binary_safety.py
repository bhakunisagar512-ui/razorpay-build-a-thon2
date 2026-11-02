import numpy as np
import pandas as pd
import pytest

from eval.binary_safety import (
    BinarySafetyClassifier, FAMILIES, apply_binary, assess_candidate, choose_candidate,
    one_sided_upper, required_zero_failure_worlds,
)
from services.app.domain.binary_safety import veto_retry
from services.app.domain.features import FEATURES


def test_binary_targets_group_both_hard_classes_without_changing_features():
    frame = pd.DataFrame({'true_cause':['CARD_EXPIRED','RISK_BLOCKED','TD_PSP_TIMEOUT','BD_INSUFFICIENT_FUNDS']})
    assert BinarySafetyClassifier.target(frame).tolist()==['HARD_DECLINE','HARD_DECLINE','OTHER','OTHER']
    assert BinarySafetyClassifier().features==FEATURES


def test_guard_cannot_reenable_a_retry_and_does_not_change_raw_input():
    blocked = {'cause':'AMBIGUOUS','retry_allowed':False,'decision':'hard_decline_mass_veto'}
    assert veto_retry(blocked,0.,.1)==blocked
    allowed = {'cause':'TD_ISSUER_DOWN','retry_allowed':True,'decision':'reliable_diagnosis'}
    guarded = veto_retry(allowed,.9,.1)
    assert not guarded['retry_allowed'] and guarded['cause']=='AMBIGUOUS'
    assert allowed['retry_allowed']


@pytest.mark.parametrize('probability',[None,float('nan'),float('inf'),-.1,1.1,'bad',object()])
def test_invalid_binary_score_fails_closed(probability):
    result = veto_retry({'cause':'TD_ISSUER_DOWN','retry_allowed':True,'decision':'reliable_diagnosis'},probability,.1)
    assert not result['retry_allowed'] and result['decision']=='invalid_binary_safety_score'


def test_vector_guard_matches_pure_guard():
    scores = np.array([.01,.1,.11,np.nan,.9])
    causes = np.array(['TD_ISSUER_DOWN','TD_ISSUER_DOWN','TD_ISSUER_DOWN','TD_ISSUER_DOWN','CARD_EXPIRED'],dtype=object)
    reasons = np.array(['reliable_diagnosis']*4+['hard_rule_veto'],dtype=object)
    retry = np.array([True,True,True,True,False])
    result = apply_binary((causes,reasons,retry),scores,.1)
    for i in range(len(scores)):
        pure = veto_retry({'cause':causes[i],'decision':reasons[i],'retry_allowed':bool(retry[i])},float(scores[i]),.1)
        assert (result[0][i],result[1][i],bool(result[2][i]))==(pure['cause'],pure['decision'],pure['retry_allowed'])
    assert retry.tolist()==[True,True,True,True,False]


@pytest.mark.parametrize('threshold',[float('nan'),-.1,1.1])
def test_vector_guard_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        apply_binary((np.array(['TD_ISSUER_DOWN'],dtype=object),np.array(['reliable_diagnosis'],dtype=object),np.array([True])),np.array([.1]),threshold)


def test_sample_planning_uses_worlds_and_familywise_error_control():
    n = required_zero_failure_worlds()
    assert n==94
    assert one_sided_upper(0,n,alpha=.05/len(FAMILIES))<=.05
    assert one_sided_upper(0,n-1,alpha=.05/len(FAMILIES))>.05
    assert one_sided_upper(0,0) is None
    assert one_sided_upper(10,10)==1
    assert one_sided_upper(1,100)>one_sided_upper(0,100)


def row(f1=.7,recall=.5,unsafe=0,blocked=0.,abstention=0.):
    return {'macro_f1_all':f1,'accuracy':.8,'true_retryable_recall':recall,'unsafe_retry_count':unsafe,
            'additional_legitimate_blocked_fraction':blocked,'additional_abstention':abstention}


def test_safety_and_utility_are_both_required_for_eligibility():
    reference = {'a':row(f1=.5)}
    control = {'a':row(f1=.7)}
    assert assess_candidate({'a':row(f1=.68)},reference,control)['eligible']
    assert not assess_candidate({'a':row(f1=.69,unsafe=1)},reference,control)['eligible']
    assert not assess_candidate({'a':row(f1=.69,blocked=.06)},reference,control)['eligible']
    assert not assess_candidate({'a':row(f1=.60)},reference,control)['eligible']


def test_binary_is_not_selected_when_it_adds_no_utility_over_multiclass():
    assessment = {'eligible':True,'mean_legitimate_retry_recall':.5,'mean_f1':.7}
    rows = [{'family':'binary','assessment':assessment},{'family':'multiclass','assessment':assessment}]
    assert choose_candidate(rows)['family']=='multiclass'
    assert choose_candidate([{'family':'binary','assessment':{'eligible':False}}]) is None
