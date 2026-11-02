"""An optional binary hard-decline veto can only reduce allowed retries."""
import math
from numbers import Real


def veto_retry(decision, hard_probability, threshold):
    if not isinstance(threshold,Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Binary safety threshold must be a probability')
    if not decision['retry_allowed']:
        return dict(decision)
    valid = isinstance(hard_probability,Real) and math.isfinite(hard_probability) and 0 <= hard_probability <= 1
    if not valid or hard_probability > threshold:
        return {**decision, 'cause': 'AMBIGUOUS', 'retry_allowed': False,
                'decision': 'binary_hard_decline_veto' if valid else 'invalid_binary_safety_score'}
    return dict(decision)
