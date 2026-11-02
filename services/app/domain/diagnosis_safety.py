"""Pure revised diagnosis policy. Rules may support or veto, never force a retry.

The legacy serving classifier is retained as a reference until offline confirmation
supports promotion of a calibrated artifact and these validation-selected settings.
"""
from dataclasses import dataclass
import math
from services.app.domain.taxonomy import Cause, RETRYABLE

HARD = (str(Cause.CARD_EXPIRED), str(Cause.RISK_BLOCKED))


@dataclass(frozen=True)
class SafetyThresholds:
    min_confidence: float = .95
    max_hard_mass: float = .03
    retry_enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.max_hard_mass <= 1:
            raise ValueError('Safety thresholds must be probabilities')


def decide(probabilities, *, rule_result=None, instrument_hint='na', known_tuple=True,
           thresholds=SafetyThresholds()):
    if not probabilities or not set(probabilities) <= {str(cause) for cause in Cause} or any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()) or abs(sum(probabilities.values())-1) > 1e-6:
        return {'cause': 'AMBIGUOUS', 'retry_allowed': False, 'decision': 'invalid_probabilities'}
    if instrument_hint == 'expired':
        return {'cause': str(Cause.CARD_EXPIRED), 'retry_allowed': False, 'decision': 'expired_instrument_veto'}
    if rule_result in HARD:
        return {'cause': rule_result, 'retry_allowed': False, 'decision': 'hard_rule_veto'}
    cause = max(probabilities, key=probabilities.get)
    confidence = probabilities[cause]
    hard_mass = sum(probabilities.get(label, 0) for label in HARD)
    reason = 'reliable_diagnosis'
    if confidence < thresholds.min_confidence:
        cause, reason = 'AMBIGUOUS', 'low_confidence'
    elif RETRYABLE.get(Cause(cause), False):
        if not thresholds.retry_enabled:
            cause, reason = 'AMBIGUOUS', 'retry_disabled_by_validation'
        elif not known_tuple:
            cause, reason = 'AMBIGUOUS', 'unknown_error_tuple'
        elif hard_mass > thresholds.max_hard_mass:
            cause, reason = 'AMBIGUOUS', 'hard_decline_mass_veto'
        elif rule_result is not None and rule_result != cause:
            cause, reason = 'AMBIGUOUS', 'rule_model_disagreement'
    return {'cause': cause, 'retry_allowed': RETRYABLE.get(Cause(cause), False), 'decision': reason}
