"""What-if evidence channels derived from detector state, never class labels.

Coverage and integrity are explicit synthetic assumptions. These fixtures are NOT
real Razorpay observations and cannot establish production evidence quality.
"""
from dataclasses import dataclass
from datetime import timedelta
import hashlib

from services.app.domain.hard_evidence import HardEvidence


@dataclass(frozen=True)
class EvidenceProfile:
    name: str
    reason_coverage: float
    expiry_coverage: float
    issuer_coverage: float = 0.
    stale_fraction: float = 0.
    corrupt_expiry_fraction: float = 0.
    downtime_coverage: float = .5

    def __post_init__(self):
        for key,value in vars(self).items():
            if key!='name' and not 0<=value<=1:
                raise ValueError(f'Invalid evidence probability: {key}')


PROFILES = (
    EvidenceProfile('no_additional_evidence',0.,0.,downtime_coverage=0.),
    EvidenceProfile('partial_verified',.5,.5,.1),
    EvidenceProfile('high_verified',.9,.9,.1),
    EvidenceProfile('partial_stale',.5,.5,.1,stale_fraction=.5),
    EvidenceProfile('high_with_1pct_expiry_corruption',.9,.9,.1,corrupt_expiry_fraction=.01),
)
# This is an explicitly synthetic issuer-code contract, not an ISO/Razorpay map.
FIXTURE_CODES = {'fixture-issuer-v1':{'EXPIRED':'CARD_EXPIRED','RISK':'RISK_BLOCKED'}}


def draw(payment_id, channel):
    return int(hashlib.sha256(f'evidence-v1:{payment_id}:{channel}'.encode()).hexdigest()[:16],16)/2**64


def make_evidence(row, profile):
    """Reads observable scope/times and simulator detector flags, never true_cause.

    Same channel draws across coverage profiles give matched availability sweeps.
    Adds no RNG consumption to the payment generator or classifier inputs.
    """
    pid, instrument, at = row['payment_id'],row['instrument_id'],row['observed_at']
    # Current provider response is available at diagnosis, never before its event.
    common = dict(payment_id=pid,instrument_id=instrument,observed_at=at,authenticated=True)
    expired, risk = bool(row['fixture_expired']),bool(row['fixture_risk_blocked'])
    result = []
    if draw(pid,'reason')<profile.reason_coverage:
        reason = 'payment_risk_check_failed' if risk else 'card_expired' if expired else 'payment_failed'
        item = HardEvidence(**common,source='razorpay_signed_webhook',error_reason=reason)
        if draw(pid,'reason_stale')<profile.stale_fraction:
            item = HardEvidence(**{**common,'observed_at':at-timedelta(hours=1)},source='razorpay_signed_webhook',error_reason=reason)
        result.append(item)
    if row['method']=='card' and draw(pid,'expiry')<profile.expiry_coverage:
        year,month = at.year,at.month
        expired_sensor = expired
        if draw(pid,'expiry_corruption')<profile.corrupt_expiry_fraction:
            expired_sensor = not expired_sensor
        if expired_sensor:
            year,month = (year-1,12) if month==1 else (year,month-1)
        else:
            year += 1
        observed = at-timedelta(days=31) if draw(pid,'expiry_stale')<profile.stale_fraction else common['observed_at']
        result.append(HardEvidence(**{**common,'observed_at':observed},source='merchant_vault_verified',
                                   expiry_year=year,expiry_month=month))
    if draw(pid,'issuer')<profile.issuer_coverage and (expired or risk):
        result.append(HardEvidence(**common,source='provider_server_verified',issuer_namespace='fixture-issuer-v1',
                                   issuer_decline_code='RISK' if risk else 'EXPIRED'))
    if row['fixture_downtime'] and draw(pid,'downtime')<profile.downtime_coverage:
        result.append(HardEvidence(**common,source='provider_server_verified',downtime_status='started',method=row['method'],
            issuer_id=row['issuer_id'] if row['fixture_issuer_down'] else None,
            psp_id=row['psp_id'] if row['fixture_psp_down'] else None))
    return result
