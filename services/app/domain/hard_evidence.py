"""Deterministic vetoes for authenticated, scoped, point-in-time evidence.

Unknown/generic declines are not reinterpreted as confirmed permanent failures.
Neither valid expiry metadata nor downtime clears an existing safety veto.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

EXPLICIT_REASONS = {'card_expired':'CARD_EXPIRED', 'payment_risk_check_failed':'RISK_BLOCKED'}


@dataclass(frozen=True)
class HardEvidence:
    payment_id: str
    instrument_id: str
    observed_at: datetime
    authenticated: bool = False
    source: str = 'unknown'
    error_reason: str | None = None
    expiry_year: int | None = None
    expiry_month: int | None = None
    issuer_decline_code: str | None = None
    issuer_namespace: str | None = None
    confirmed_non_retryable: bool | None = None
    downtime_status: str | None = None
    method: str | None = None
    issuer_id: str | None = None
    psp_id: str | None = None


def evidence_verdict(evidence, *, payment_id, instrument_id, as_of, method=None, issuer_id=None, psp_id=None,
                     issuer_code_registry=None):
    result = {'hard_cause':None,'expiry_verified_valid':False,'temporary_hold':False,'reasons':[]}
    def mark_hard(cause):
        # A risk decision cannot be downgraded to a card-update diagnosis.
        if result['hard_cause']=='RISK_BLOCKED':
            return
        if cause=='RISK_BLOCKED' or result['hard_cause'] in (None,'AMBIGUOUS'):
            result['hard_cause'] = cause
    for item in evidence:
        if (not item.authenticated or item.payment_id!=payment_id or item.instrument_id!=instrument_id
            or item.observed_at.tzinfo is None or as_of.tzinfo is None or item.observed_at>as_of):
            continue
        # Provider failure/incident evidence must be recent. Expiry facts can be
        # cached longer, but must still identify this instrument and observation.
        age = as_of-item.observed_at
        recent = age<=timedelta(minutes=15)
        if recent and item.source in ('razorpay_signed_webhook','provider_server_verified'):
            cause = EXPLICIT_REASONS.get(item.error_reason)
            if cause:
                mark_hard(cause)
                result['reasons'].append(f'provider_reason:{item.error_reason}')
            mapping = (issuer_code_registry or {}).get(item.issuer_namespace,{})
            if mapping.get(item.issuer_decline_code) in ('CARD_EXPIRED','RISK_BLOCKED','AMBIGUOUS'):
                mark_hard(mapping[item.issuer_decline_code])
                result['reasons'].append(f'issuer_code:{item.issuer_namespace}:{item.issuer_decline_code}')
            if item.confirmed_non_retryable is True:
                mark_hard('AMBIGUOUS')
                result['reasons'].append('provider_confirmed_non_retryable')
        if (instrument_id not in ('unknown','na','',None) and
            item.source in ('merchant_vault_verified','provider_server_verified') and age<=timedelta(days=30)):
            year,month = item.expiry_year,item.expiry_month
            if isinstance(year,int) and isinstance(month,int) and 1<=month<=12 and 2000<=year<=2200:
                local = as_of.astimezone(timezone(timedelta(hours=5,minutes=30)))
                expired = (year,month)<(local.year,local.month)
                if expired:
                    mark_hard('CARD_EXPIRED')
                    result['reasons'].append('verified_instrument_expired')
                else:
                    result['expiry_verified_valid'] = True
        if (recent and item.source in ('razorpay_signed_webhook','provider_server_verified')
            and item.downtime_status=='started' and item.method==method
            and (item.issuer_id is not None or item.psp_id is not None)
            and (item.issuer_id is None or item.issuer_id==issuer_id)
            and (item.psp_id is None or item.psp_id==psp_id)):
            result['temporary_hold'] = True
            result['reasons'].append('matched_active_downtime')
    return result


def evidence_veto(decision, verdict, *, retry_on_affected_instrument=False):
    if not decision['retry_allowed']:
        return dict(decision)
    if verdict['hard_cause']:
        return {**decision,'cause':verdict['hard_cause'],'retry_allowed':False,'decision':'hard_evidence_veto'}
    if verdict['temporary_hold'] and retry_on_affected_instrument:
        return {**decision,'cause':'AMBIGUOUS','retry_allowed':False,'decision':'temporary_downtime_hold'}
    return dict(decision)


def ambiguity_veto(decision, *, method, instrument_hint, verdict, error_tuple, mode='unknown_expiry'):
    if mode not in ('unknown_expiry','collision_only'):
        raise ValueError('Unknown ambiguity policy')
    output = evidence_veto(decision,verdict)
    if not output['retry_allowed']:
        return output
    # A valid expiry fact rules out only expiration; it does not clear a risk
    # verdict or any upstream block. Absence of evidence never means valid.
    # The model's bare "valid" hint is not an independently authenticated fact.
    unknown = method=='card' and not verdict['expiry_verified_valid']
    collisions = {('GATEWAY_ERROR','gateway','authorization'), ('BAD_REQUEST_ERROR','bank','authorization'),
                  ('GATEWAY_ERROR','issuer','authorization')}
    ambiguous = unknown and (mode=='unknown_expiry' or mode=='collision_only' and error_tuple in collisions)
    if ambiguous:
        return {**output,'cause':'AMBIGUOUS','retry_allowed':False,'decision':'unresolved_card_expiry_ambiguity'}
    return output
