"""Phase 4 / 10 static guarantees about the testing interfaces (no browser needed): no free-text input creates a revenue-risk event, the payment
page has no input at all, and the wording follows the simulated terminology."""
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / 'services' / 'app' / 'static'
JS = (STATIC / 'app.js').read_text(encoding='utf-8')
CONSOLE = JS[JS.index('Revenue-risk console (Individual tab)'):]          # event creation, journey, dataset card, intervention view
INDIVIDUAL = JS[JS.index('function initIndividual() {'):JS.index('/* ------------------------------------------------------------------ boot */')]


def test_event_creation_uses_dropdowns_only():
    assert not re.findall(r"h\('textarea'", CONSOLE), 'the revenue console creates no textarea'
    inputs = re.findall(r"h\('input', \{[^}]*\}", CONSOLE)
    assert inputs and all("type: 'file'" in i for i in inputs), 'the only input is the CSV file chooser, never a text field'
    assert 'type="text"' not in CONSOLE and "type: 'text'" not in CONSOLE and "type: 'search'" not in INDIVIDUAL and "id: 'i-search'" not in JS
    assert 'datalist' not in CONSOLE and "list: 'dl-" not in JS                                         # no suggestion lists over free text either
    for field in ('c-method', 'c-hint', 'c-code', 'c-source', 'c-step', 'c-reason', 'c-truth', 'c-customer', 'c-issuer', 'c-amount', 'c-time'):
        assert f"'{field}'" in CONSOLE
    assert "h('select', { id, name: id" in CONSOLE                                                      # every scenario field is a select
    for field in ('s-customer', 's-amount', 's-stage', 's-plan', 's-method', 's-kind', 's-invoice', 's-time'):
        assert f"'{field}'" in CONSOLE


def test_the_payment_page_has_no_input_of_any_kind_and_uses_the_agreed_wording():
    html, js = (STATIC / 'pay.html').read_text(encoding='utf-8'), (STATIC / 'pay.js').read_text(encoding='utf-8')
    assert not re.search(r'<(input|textarea|select)\b', html) and not re.search(r"el\('(input|textarea|select)'", js)
    assert 'ACCEPT' in js and 'REJECT' in js and 'SIMULATED' in html
    assert 'successful payment' in js and 'not merely opening or reading' in js                          # ACCEPT is a payment, not an acknowledgement
    for field in ('Synthetic payment ID', 'Amount', 'Payment method', 'Recovery action', 'Request ID', 'Workflow ID', 'Current status', 'Created'):
        assert field in js


def test_terminology_is_explicit_and_never_claims_real_revenue():
    for label in ('PENDING', 'SIMULATED_RECOVERED', 'SIMULATED_FAILED', 'HUMAN_INTERVENTION', 'WORKFLOW_COMPLETED'):
        assert label in CONSOLE
    text = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'SIMULATED_RECOVERED means the simulated customer completed the requested payment' in text and 'max 3 cycles' not in text
    for claim in ('recovered revenue of', 'actual revenue was recovered', 'real revenue recovered'):
        assert claim not in JS.lower() and claim not in text.lower()


def test_the_three_major_sections_and_four_event_types_are_present():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    for tab in ('Individual Payment Testing', 'Dataset Testing', 'Human Intervention'):
        assert tab in html
    for etype in ('CHECKOUT_ABANDONMENT', 'SUBSCRIPTION_FAILURE', 'OVERDUE_RECEIVABLE', 'PAYMENT_FAILURE'):
        assert etype in CONSOLE


def test_the_dropdown_logic_stays_server_side():
    assert '/tests/custom-options' in CONSOLE and 'scope: SCOPE_OF[CF.scenario]' in CONSOLE
    assert "SCOPE_OF = { clear: 'clear', generic: 'generic', contradictory: 'conflict' }" in CONSOLE
    assert 'CARD_ONLY' not in JS and "'incorrect_cvv'" not in JS and "'payment_cancelled'" not in CONSOLE.replace("error_reason: 'payment_cancelled'", '')   # no second copy of the compatibility rules in the browser


def test_the_diagnosis_scenario_selector_has_three_choices_three_examples_and_never_sends_a_route():
    for needle in ('Clear and consistent (deterministic rules)', 'Missing / generic gateway information (Model C)', 'Contradictory gateway information (human intervention)',
                   "'c-scenario'", 'c-example-${kind}', 'c-preview', 'diagnosis_scenario: CF.scenario'):
        assert needle in CONSOLE, needle
    assert CONSOLE.count("clear: ['Clear example'") == 1 and "generic: ['Missing / generic example'" in CONSOLE and "contradictory: ['Contradictory example'" in CONSOLE
    submit = CONSOLE[CONSOLE.index('async function submitCustom'):CONSOLE.index('/* ---- the other three event types')]
    assert 'route' not in submit.replace('diagnosis_scenario', '') and 'model_c' not in submit.lower()                      # the browser never supplies a route, a prediction or a call count
    for shown in ('Route taken', 'Decided by', 'Predicted cause', 'Model C calls', 'Safety gate', 'Temporal execution', 'Recovery plan'):
        assert shown in CONSOLE, shown
