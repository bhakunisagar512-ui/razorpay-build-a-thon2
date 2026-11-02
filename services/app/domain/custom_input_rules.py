"""Compatibility rules for the Individual Payment Testing form (custom payments). One source of truth for the form AND the API.

Goal: give Model C in-distribution, meaningful observable inputs. Nothing here changes the model, its 43 features, a threshold,
a gate, hypothesis selection or a plan.

WHERE THE RULES COME FROM (provenance of every rule is one of these):
  [vocab]     services.app.domain.failure_vocab: documented error codes, per-method error sources and flow steps, the
              BAD_REQUEST / GATEWAY reason sets.
  [world]     eval.world_config_v4: the emission tables of the synthetic world Model C was trained on (STEP_SOURCE,
              SPECIFIC_REASONS, GENERIC_BY_METHOD). These declare which steps, sources and reasons each failure cause
              produces for each method. The generator adds only uniform noise on top (5% random source, 5% random step,
              1-6% random reason), so a combination that no declared profile produces is noise, not a business rule.
  [proposed]  Rules this module adds. Each is marked below and listed in PROPOSED_RULES.

THE MODEL OF A VALID SCENARIO
  A scenario is (method, instrument_hint, error_code, error_source, error_step, error_reason). It is VALID for a method when
  some failure-cause profile of that method contains the hint, source, step and reason together ([world]), and the error code
  is one the reason allows ([vocab]; a missing reason allows BAD_REQUEST_ERROR or GATEWAY_ERROR). The latent cause used for
  this test is existential ("some cause"), never the tester's evaluation label.

EVALUATION LABEL
  The label is hidden ground truth for offline testing. It is checked against the METHOD only (the causes the method's
  profile defines; CARD_EXPIRED exists for card only). It never filters or changes the observable fields, and the observable
  fields never filter the label, so a tester can still label a scenario with a cause its observables do not point to.

Valid scenarios are enumerated once as exact tuples, so every option list is exact (no dead ends) and the same function
validates the API request.
"""
from __future__ import annotations

from functools import lru_cache

from eval.world_config_v4 import GENERIC_BY_METHOD, SPECIFIC_REASONS, STEP_SOURCE
from services.app.domain.failure_vocab import BAD_REQUEST_REASONS, ERROR_CODES, GATEWAY_REASONS, REASON_FAMILY, SOURCES, STEPS

METHODS = ('upi', 'card', 'netbanking')
OBSERVABLE = ('instrument_hint', 'gw_error_code', 'gw_error_source', 'gw_error_step', 'error_reason')
FORM_ORDER = ('method',) + OBSERVABLE + ('true_cause',)           # the order in which selections are preserved when others change
CARD_HINTS = ('valid', 'unknown', 'expired')
DEFAULT_HINT = {'upi': 'na', 'netbanking': 'na', 'card': 'unknown'}
# [proposed, user-specified] reasons that describe a card instrument problem. Never offered for UPI or netbanking.
CARD_ONLY_REASONS = frozenset({'card_expired', 'incorrect_cvv', 'incorrect_card_expiry_date', 'card_not_enrolled'})
ALL_REASONS = tuple(sorted(BAD_REQUEST_REASONS | GATEWAY_REASONS))
ALL_STEPS = tuple(dict.fromkeys(s for m in METHODS for s in STEPS[m]))
ALL_SOURCES = tuple(dict.fromkeys(s for m in METHODS for s in SOURCES[m]))
ALL_HINTS = ('na',) + CARD_HINTS
_UNSET = object()

PROPOSED_RULES = (
    'Card instrument hint is valid / unknown / expired and defaults to unknown; UPI and netbanking are na only. (hint values from the '
    'generator; expired only occurs with the CARD_EXPIRED profile: 66 of 66 training rows)',
    'card_expired, incorrect_cvv, incorrect_card_expiry_date, card_not_enrolled are card-only reasons (owner decision).',
    'CARD_EXPIRED is an available evaluation label for card only (it is the only method with a CARD_EXPIRED profile).',
    'An error reason that no declared profile of the method produces is not offered, even though Razorpay documents it and a few '
    'synthetic noise rows contain it. This also hides incorrect_card_expiry_date and card_not_enrolled for card (no profile uses them).',
    'SERVER_ERROR is offered only together with the server_error reason, which only card and netbanking profiles produce.',
    'Selecting "no reason" (reason missing) is always allowed and fixes nothing else; it allows BAD_REQUEST_ERROR or GATEWAY_ERROR.',
)


# ------------------------------------------------------------------------------------------------- descriptions (UI text only)
METHOD_INFO = {'upi': 'UPI', 'card': 'Card', 'netbanking': 'Netbanking'}
HINT_INFO = {'na': 'no instrument information (UPI and netbanking)', 'valid': 'card on file, not known to be expired',
             'unknown': 'card expiry not known', 'expired': 'merchant knows the card has expired'}
CODE_INFO = {'BAD_REQUEST_ERROR': 'request or customer-side problem', 'GATEWAY_ERROR': 'bank, network or gateway-side problem',
             'SERVER_ERROR': 'Razorpay-side error (reason server_error)'}
CAUSE_INFO = {'BD_INSUFFICIENT_FUNDS': 'Insufficient funds', 'BD_USER_ABANDONED_OTP': 'OTP abandoned by customer', 'CARD_EXPIRED': 'Card expired',
              'RISK_BLOCKED': 'Risk blocked', 'TD_ISSUER_DOWN': 'Issuer downtime', 'TD_PSP_TIMEOUT': 'PSP timeout'}
SOURCE_INFO = {'customer': 'the customer (their action or authentication)', 'business': 'the merchant', 'internal': 'Razorpay internal',
               'gateway': 'the payment gateway', 'issuer_bank': "the customer's issuing bank", 'customer_psp': "the customer's UPI app / PSP",
               'network': 'the payment network (for example NPCI)', 'beneficiary_bank': "the merchant's (beneficiary) bank"}
STEP_INFO = {
    'payment_initiation': 'payment initiated', 'payment_creation': 'payment request created', 'card_enrollment_check': 'card enrollment check',
    'payment_authentication': 'customer authentication (OTP, PIN or 3DS)', 'payment_authentication_request': 'authentication request sent',
    'payment_authentication_response': 'authentication response received', 'payment_authorization': 'bank authorisation',
    'payment_request': 'collect / intent request sent', 'payment_request_beneficiary_details': 'beneficiary details requested',
    'payment_response_beneficiary_details': 'beneficiary details received', 'payment_debit_request': "debit requested from the payer's bank",
    'payment_debit_response': "debit response from the payer's bank", 'payment_credit_request': 'credit requested',
    'payment_credit_response': 'credit response received', 'payment_status_request': 'status check requested',
    'payment_status_response': 'status check response', 'payment_response': 'final payment response', 'payment_capture': 'capture',
}
REASON_INFO = {
    'authentication_failed': 'authentication did not succeed', 'bank_cutoff_in_progress': 'bank cut-off window in progress',
    'bank_not_available': 'bank not available', 'bank_technical_error': 'technical error at the bank', 'card_declined': 'card declined (generic)',
    'card_expired': 'the card has expired', 'card_not_enrolled': 'card not enrolled for this authentication',
    'credit_limit_exceeded': 'credit limit exceeded', 'debit_declined': 'debit declined (generic)',
    'debit_instrument_blocked': 'instrument blocked for debits', 'debit_instrument_inactive': 'instrument inactive',
    'gateway_technical_error': 'technical error at the gateway', 'incorrect_card_expiry_date': 'wrong card expiry date entered',
    'incorrect_cvv': 'wrong CVV entered', 'incorrect_otp': 'wrong OTP entered', 'incorrect_pin': 'wrong PIN entered',
    'insufficient_funds': 'insufficient balance', 'invalid_response_from_gateway': 'invalid response from the gateway',
    'issuer_technical_error': 'technical error at the issuer', 'otp_attempts_exceeded': 'too many OTP attempts', 'otp_expired': 'OTP expired',
    'payment_cancelled': 'payment cancelled by the customer', 'payment_collect_request_expired': 'UPI collect request expired',
    'payment_declined': 'payment declined (generic)', 'payment_declined_due_to_high_traffic': 'declined due to high traffic',
    'payment_failed': 'payment failed (generic)', 'payment_risk_check_failed': 'failed a risk check',
    'payment_session_expired': 'payment session expired', 'payment_timed_out': 'payment timed out', 'pin_attempts_exceeded': 'too many PIN attempts',
    'psp_app_not_available': 'UPI app not available', 'psp_not_available': 'PSP not available', 'request_timed_out': 'request timed out',
    'server_error': 'Razorpay server error', 'transaction_daily_count_exceeded': 'daily transaction count exceeded',
    'transaction_daily_limit_exceeded': 'daily transaction limit exceeded', 'transaction_frequency_limit_exceeded': 'transaction frequency limit exceeded',
    'transaction_limit_exceeded': 'transaction limit exceeded', 'transaction_on_vpa_restricted': 'transactions restricted on this VPA',
    'upi_app_technical_error': 'technical error in the UPI app',
}
INFO = {'method': METHOD_INFO, 'instrument_hint': HINT_INFO, 'gw_error_code': CODE_INFO, 'gw_error_source': SOURCE_INFO,
        'gw_error_step': STEP_INFO, 'error_reason': REASON_INFO, 'true_cause': CAUSE_INFO}
FIELD_LABEL = {'method': 'Payment method', 'instrument_hint': 'Instrument hint', 'gw_error_code': 'Gateway error code',
               'gw_error_source': 'Error source', 'gw_error_step': 'Error step', 'error_reason': 'Error reason', 'true_cause': 'Evaluation label'}
_REASON_ORDER = ('customer_auth', 'customer_session', 'funds_limits', 'bank_technical', 'gateway_technical', 'instrument_state', 'risk', 'generic', 'other')


# ------------------------------------------------------------------------------------------------- the rule tables
def codes_for_reason(reason):
    """[vocab + world] Error codes a reason can carry: the generator derives the code from the documented reason sets."""
    if reason is None:
        return ('BAD_REQUEST_ERROR', 'GATEWAY_ERROR')
    if reason == 'server_error':
        return ('SERVER_ERROR',)
    both = reason in BAD_REQUEST_REASONS and reason in GATEWAY_REASONS
    if both:
        return ('BAD_REQUEST_ERROR', 'GATEWAY_ERROR')
    return ('GATEWAY_ERROR',) if reason in GATEWAY_REASONS else ('BAD_REQUEST_ERROR',)


def causes_of(method):
    return tuple(STEP_SOURCE[method])


def cause_profiles(method):
    """{cause: {'hints', 'sources', 'steps', 'reasons'}} for one method, from the world tables."""
    out = {}
    for cause, (steps, sources) in STEP_SOURCE[method].items():
        hints = {'valid', 'unknown'} | ({'expired'} if cause == 'CARD_EXPIRED' else set()) if method == 'card' else {'na'}
        out[cause] = {'hints': hints, 'sources': set(sources), 'steps': set(steps),
                      'reasons': set(SPECIFIC_REASONS[cause][method]) | set(GENERIC_BY_METHOD[method])}
    return out


@lru_cache(maxsize=None)
def specific_reason_causes(method):
    """{reason: frozenset(causes)}: the causes whose declared profile for this method produces the reason as a SPECIFIC reason.
    Generic catch-alls (payment_failed, card_declined, ...) say nothing about the cause and are left out."""
    from services.app.domain.failure_vocab import GENERIC_REASONS
    out = {}
    for cause in causes_of(method):
        for reason in SPECIFIC_REASONS[cause][method]:
            if reason not in GENERIC_REASONS:
                out.setdefault(reason, set()).add(cause)
    return {r: frozenset(c) for r, c in out.items()}


def clear_reason_cause(method, reason):
    """The one cause a reason identifies for this method, or None (missing, generic, or shared by several causes)."""
    causes = specific_reason_causes(method).get(reason) if reason else None
    return next(iter(causes)) if causes and len(causes) == 1 else None


@lru_cache(maxsize=None)
def scenarios(method, scope='all'):
    """Every valid (hint, code, source, step, reason) for the method, as a frozenset of tuples (reason None = not provided).
    scope 'clear' keeps only scenarios whose reason identifies a single cause: the inputs Custom Testing accepts (see revenue/routing.py)."""
    if scope == 'clear':
        return frozenset(r for r in scenarios(method) if clear_reason_cause(method, r[4]) is not None)
    if scope == 'generic':          # consistent inputs that carry no clear evidence: a missing / generic / shared reason and no 'expired' hint (those the router sends to Model C)
        return frozenset(r for r in scenarios(method) if clear_reason_cause(method, r[4]) is None and r[0] != 'expired')
    if scope == 'conflict':         # not enumerated: every vocabulary value is selectable, so the tester can build evidence that contradicts itself
        return frozenset()
    rows = set()
    for p in cause_profiles(method).values():
        for hint in p['hints']:
            for source in p['sources']:
                for step in p['steps']:
                    for reason in (*p['reasons'], None):
                        for code in codes_for_reason(reason):
                            rows.add((hint, code, source, step, reason))
    return frozenset(rows)


_INDEX = {f: i for i, f in enumerate(OBSERVABLE)}


def _matching(method, selected, scope='all'):
    rows = scenarios(method, scope)
    for field, value in selected.items():
        if field in _INDEX:
            i = _INDEX[field]
            rows = [r for r in rows if r[i] == value]
    return rows


def default_hint(method):
    return DEFAULT_HINT[method]


def ordered(field, values):
    """Display order: flow order for steps, documented vocabulary order for sources, reason families for reasons."""
    vals = set(values)
    if field == 'gw_error_step':
        return [s for s in ALL_STEPS if s in vals]
    if field == 'gw_error_source':
        return [s for s in ALL_SOURCES if s in vals]
    if field == 'gw_error_code':
        return [c for c in ERROR_CODES if c in vals]
    if field == 'instrument_hint':
        return [h for h in ALL_HINTS if h in vals]
    if field == 'error_reason':
        rank = {f: i for i, f in enumerate(_REASON_ORDER)}
        return sorted((v for v in vals if v is not None), key=lambda r: (rank[REASON_FAMILY.get(r, 'other')], r))
    return sorted(vals, key=str)


# ------------------------------------------------------------------------------------------------- queries
def allowed_values(method, field, selected, scope='all'):
    """Values of `field` that keep the scenario valid given the OTHER selected observable fields (the field's own selection is ignored)."""
    if scope == 'conflict':
        return set(universe(field)) | ({None} if field == 'error_reason' else set())
    others = {k: v for k, v in selected.items() if k != field and k in _INDEX}
    i = _INDEX[field]
    return {r[i] for r in _matching(method, others, scope)}


def explain(method, field, value, selected, scope='all'):
    """Why `value` is not offered for `field`, in words a tester can act on."""
    domain = {r[_INDEX[field]] for r in scenarios(method, scope)}
    name = METHOD_INFO[method]
    if scope == 'conflict':
        return f'{value!r} is not a recognised gateway value for this field.'
    if scope == 'generic' and value not in domain and value in {r[_INDEX[field]] for r in scenarios(method)}:
        return (f'Clear evidence for {name}: on its own it identifies the cause, so the router would decide by rule. '
                'The Missing / generic scenario offers only inputs that need Model C.')
    if scope == 'clear' and value not in domain and value in {r[_INDEX[field]] for r in scenarios(method)}:
        return (f'Not a clear-and-consistent {FIELD_LABEL[field].lower()} for {name}: it is generic, missing or shared by several causes. '
                'Custom Testing accepts only clear inputs; Dataset Testing exercises the rest.')
    if value not in domain:
        if field == 'error_reason' and value in CARD_ONLY_REASONS and method != 'card':
            return f'Card-only reason (a card instrument problem); it cannot occur on {name}.'
        if field == 'error_reason' and value in CARD_ONLY_REASONS:
            return 'Documented for cards, but no card failure scenario in Model C\'s training world produces it (it only appears as rare synthetic noise).'
        if field == 'error_reason':
            return f'No {name} failure scenario in Model C\'s training world produces this reason (it only appears as rare synthetic noise).'
        if field == 'gw_error_code' and value == 'SERVER_ERROR':
            return f'SERVER_ERROR goes with the server_error reason, which no {name} scenario produces.'
        if field == 'instrument_hint':
            return f'Not a {name} value: {name} uses ' + ' / '.join(sorted({r[0] for r in scenarios(method)})) + '.'
        return f'Not part of the documented {name} flow.'
    for other, other_value in selected.items():
        if other == field or other not in _INDEX:
            continue
        if value not in allowed_values(method, field, {other: other_value}, scope):
            shown = 'not provided' if other_value is None else other_value
            return f'Not seen together with {FIELD_LABEL[other].lower()} = {shown} for {name}.'
    return 'Not supported together with the combination of your current selections.'


def labels_for(method):
    return list(causes_of(method))


def universe(field):
    return {'instrument_hint': ALL_HINTS, 'gw_error_code': ERROR_CODES, 'gw_error_source': ALL_SOURCES, 'gw_error_step': ALL_STEPS,
            'error_reason': ALL_REASONS}[field]


def reconcile(method, selected, changed=None, scope='all'):
    """Keep every selection that is still compatible, in a fixed priority: the field the user just changed first, then form order.
    Returns (kept, cleared) where cleared = [{'field', 'value', 'why'}]. Nothing is ever replaced by another value."""
    kept, cleared = {}, []
    order = ([changed] if changed in OBSERVABLE and changed in selected else []) + [f for f in OBSERVABLE if f != changed and f in selected]
    for field in order:
        value = selected[field]
        if value in allowed_values(method, field, kept, scope):
            kept[field] = value
        else:
            cleared.append({'field': field, 'value': value, 'why': explain(method, field, value, kept, scope)})
    return kept, cleared


def build_options(selected, changed=None, scope='all'):
    """Everything the form needs after any change: the reconciled selections and, per field, the offered and withheld options.
    `selected` holds only fields the tester has set; error_reason None means 'no reason provided' (set), absence means not chosen."""
    method = selected.get('method')
    if method not in METHODS:
        method = None
    fields, cleared = [], []
    obs = {f: selected[f] for f in OBSERVABLE if f in selected}
    label = selected.get('true_cause')
    applied = []
    if method:
        obs, cleared = reconcile(method, obs, changed, scope)
        if 'instrument_hint' not in obs and default_hint(method) in allowed_values(method, 'instrument_hint', obs, scope):
            obs = {**obs, 'instrument_hint': default_hint(method)}                 # the specified default; reported, never a different failure value
            applied.append({'field': 'instrument_hint', 'value': default_hint(method), 'why': f'default for {METHOD_INFO[method]}'})
        if label is not None and label not in labels_for(method):
            cleared.append({'field': 'true_cause', 'value': label,
                            'why': f'{CAUSE_INFO.get(label, label)} cannot occur on {METHOD_INFO[method]}; choose an evaluation label again.'})
            label = None
    for field in ('method',) + OBSERVABLE + ('true_cause',):
        spec = {'field': field, 'label': FIELD_LABEL[field], 'required': field != 'true_cause', 'enabled': True, 'options': [], 'withheld': [],
                'placeholder': f'Select {FIELD_LABEL[field].lower()}'}
        if field == 'method':
            spec['options'] = [{'value': m, 'label': METHOD_INFO[m], 'description': ''} for m in METHODS]
        elif method is None:
            spec.update(enabled=False, note='Select the payment method first.')
        elif field == 'true_cause':
            spec['placeholder'] = 'None: diagnosis preview only'
            spec['options'] = [{'value': c, 'label': CAUSE_INFO[c], 'description': ''} for c in labels_for(method)]
            spec['withheld'] = [{'value': c, 'label': CAUSE_INFO[c], 'why': f'{CAUSE_INFO[c]} occurs only on card payments.'}
                                for c in CAUSE_INFO if c not in labels_for(method)]
            spec['note'] = 'Hidden ground truth for offline testing only. It never changes the inputs above or the diagnosis.'
        else:
            others = {k: v for k, v in obs.items() if k != field}
            ok = allowed_values(method, field, others, scope)
            values = ordered(field, ok)
            spec['options'] = [{'value': v, 'label': v, 'description': INFO[field].get(v, '')} for v in values]
            if field == 'error_reason' and scope in ('all', 'generic', 'conflict'):
                spec['options'].append({'value': None, 'label': 'No reason provided', 'description': 'the payload carries no error reason'})
                spec['placeholder'] = 'Select error reason (or "No reason provided")'
            spec['withheld'] = [{'value': v, 'label': v, 'why': explain(method, field, v, others, scope)} for v in universe(field) if v not in ok]
            spec['note'] = f'{len(values)} of {len(universe(field))} available'
            if len(values) == 1 and field != 'error_reason':
                spec['note'] += ' (only one option applies)'
        fields.append(spec)
    chosen = ({'method': method} if method else {}) | (obs if method else {})
    if label is not None and method:
        chosen['true_cause'] = label
    return {'selected': chosen, 'cleared': cleared, 'applied_defaults': applied, 'fields': fields,
            'complete': bool(method) and all(f in obs for f in OBSERVABLE)}


def validate(fields, scope='all'):
    """API-side check of a custom-payment request. `fields` must contain the observable fields (error_reason may be None) and
    optionally true_cause. Returns [{'field', 'message'}]; empty = compatible."""
    method = fields.get('method')
    if method not in METHODS:
        return [{'field': 'method', 'message': f'method must be one of {", ".join(METHODS)}.'}]
    errors, selected = [], {}
    for f in OBSERVABLE:
        if f not in fields:
            errors.append({'field': f, 'message': f'{FIELD_LABEL[f]} is required.'})
        else:
            selected[f] = fields[f]
    kept, cleared = reconcile(method, selected, None, scope)
    errors += [{'field': c['field'], 'message': f"{FIELD_LABEL[c['field']]} '{c['value']}' is not supported: {c['why']}"} for c in cleared]
    label = fields.get('true_cause')
    if label is not None and label not in labels_for(method):
        errors.append({'field': 'true_cause', 'message': f'Evaluation label {label} cannot occur on {METHOD_INFO[method]}.'})
    return errors
