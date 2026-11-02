"""Browser test of the testing console (/ui) against the RUNNING local stack. Dev tooling, runs on the host:

    python tests/frontend/e2e_ui.py            # needs Chrome or Edge installed; uses the websocket-client package

Drives a real headless Chrome through the Chrome DevTools Protocol: clicks, typing, file upload, screenshots (saved under
eval/data/phase4_ui/screenshots/). It uses small synthetic cases only. Individual payments create a few synthetic test
payments in the live database (that is what the feature does); dataset runs are limited to the first 40-60 payments.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import time
import urllib.error
import urllib.request
from pathlib import Path

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # the repo root, so the test can read the deterministic fixture list

BASE = os.environ.get('UI_BASE', 'http://localhost:8000')
CHROME = next((p for p in (r'C:\Program Files\Google\Chrome\Application\chrome.exe', r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                           r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe') if Path(p).exists()), None)
SHOTS = Path('eval/data/phase4_ui/screenshots')
PORT = 9333
results = []


def api(path, body=None, raw=None):
    data = raw.encode() if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(BASE + path, data=data, headers={'content-type': 'text/csv' if raw is not None else 'application/json'},
                                 method='POST' if data is not None else 'GET')
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class Browser:
    def __init__(self):
        self.profile = tempfile.mkdtemp(prefix='ui-e2e-')
        self.proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={PORT}', f'--user-data-dir={self.profile}', '--no-first-run',
                                      '--no-default-browser-check', '--disable-extensions', '--disable-gpu', '--window-size=1366,1100', 'about:blank'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                targets = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json/list', timeout=2).read())
                page = next(t for t in targets if t['type'] == 'page')
                break
            except Exception:                                                # noqa: BLE001
                time.sleep(0.5)
        else:
            raise RuntimeError('Chrome did not start')
        self.ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=120, suppress_origin=True)
        self.n = 0
        self.errors = []
        for m in ('Page.enable', 'Runtime.enable', 'Log.enable', 'DOM.enable'):
            self.send(m)

    def send(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({'id': self.n, 'method': method, 'params': params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('method') == 'Runtime.exceptionThrown':
                self.errors.append(msg['params']['exceptionDetails'].get('exception', {}).get('description', 'exception'))
            elif msg.get('method') == 'Log.entryAdded' and msg['params']['entry']['level'] == 'error':
                e = msg['params']['entry']
                if 'favicon' not in e.get('url', ''):
                    self.errors.append(e['text'] + ' ' + e.get('url', ''))
            if msg.get('id') == self.n:
                if 'error' in msg:
                    raise RuntimeError(msg['error'])
                return msg.get('result', {})

    def js(self, expr, timeout=60):
        r = self.send('Runtime.evaluate', expression=expr, awaitPromise=True, returnByValue=True, timeout=timeout * 1000)
        if 'exceptionDetails' in r:
            raise RuntimeError(r['exceptionDetails'].get('exception', {}).get('description', str(r['exceptionDetails'])))
        return r['result'].get('value')

    def wait(self, cond, timeout=60, label=''):
        value = self.js(f"(async()=>{{const end=Date.now()+{timeout * 1000};while(Date.now()<end){{try{{const v=({cond});if(v)return true;}}catch(e){{}}await new Promise(r=>setTimeout(r,150));}}return false;}})()", timeout + 5)
        if not value:
            raise AssertionError(f'timed out waiting for: {label or cond}')
        return value

    def goto(self, url, ready='[role=tablist]'):
        self.send('Page.navigate', url='about:blank')          # force a real reload even when only the #fragment differs
        self.send('Page.navigate', url=url)
        self.wait(f"document.readyState==='complete' && !!document.querySelector({json.dumps(ready)})", 20, 'page load')

    def shot(self, name):
        SHOTS.mkdir(parents=True, exist_ok=True)
        import base64
        data = self.send('Page.captureScreenshot', format='png', captureBeyondViewport=True)['data']
        (SHOTS / f'{name}.png').write_bytes(base64.b64decode(data))

    def viewport(self, w, h, mobile=False):
        self.send('Emulation.setDeviceMetricsOverride', width=w, height=h, deviceScaleFactor=1, mobile=mobile)

    def close(self):
        try:
            self.send('Browser.close')
        except Exception:                                                    # noqa: BLE001
            pass
        self.proc.terminate()


def test(name):
    def deco(fn):
        fn.test_name = name
        results.append(fn)
        return fn
    return deco


TEXT = "document.body.innerText"
FORBIDDEN_DATASET_WORDS = ['acceptance', 'threshold', 'benchmark', 'compliance', 'upper bound', 'wilson', 'exposure', 'unsafe', 'pass/fail', 'passed', 'meets the']


def click(b, selector, nth=0):
    b.js(f"document.querySelectorAll({json.dumps(selector)})[{nth}].click()")


def set_value(b, selector, value, event='change'):
    b.js(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.value={json.dumps(value)};e.dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}));}})()")


def journey_text(b):
    return b.js("document.querySelector('#i-journey').innerText")


@test('tabs load, navigate (click, hash, arrow keys) and the simulated banner is present')
def t_tabs(b):
    b.goto(BASE + '/ui')
    assert b.js("document.querySelectorAll('[role=tab]').length") == 3
    names = b.js("[...document.querySelectorAll('[role=tab]')].map(t=>t.textContent)")
    assert 'Individual Payment Testing' in names[0] and 'Dataset Testing' in names[1] and 'Human Intervention' in names[2], names
    assert 'SIMULATED' in b.js(TEXT) and 'SIMULATED_RECOVERED means' in b.js(TEXT)
    for i, t in enumerate(['individual', 'dataset', 'intervention']):
        click(b, '[role=tab]', i)
        b.wait(f"!document.querySelector('#panel-{t}').hidden && document.querySelector('#tab-{t}').getAttribute('aria-selected')==='true'", 10, f'tab {t}')
        assert b.js("[...document.querySelectorAll('.panel')].filter(p=>!p.hidden).length") == 1
    b.js("window.location.hash='#dataset'")
    b.wait("!document.querySelector('#panel-dataset').hidden", 5)
    b.js("document.querySelector('#tab-dataset').focus()")
    b.send('Input.dispatchKeyEvent', type='keyDown', key='ArrowRight', code='ArrowRight', windowsVirtualKeyCode=39)
    b.wait("!document.querySelector('#panel-intervention').hidden", 5, 'arrow key moves to the next tab')
    assert 'online' in b.js("document.querySelector('#chip-api').textContent")


OTP_BODY = {'event_type': 'PAYMENT_FAILURE', 'customer_id': 'cust_syn_01', 'amount_rupees': 499, 'issuer_id': 'issuer_01', 'failure_time': 't_1200', 'method': 'upi',
            'instrument_hint': 'na', 'gw_error_code': 'BAD_REQUEST_ERROR', 'gw_error_source': 'issuer_bank', 'gw_error_step': 'payment_debit_response', 'error_reason': 'insufficient_funds'}
OTHER_FORM_TYPES = (('CHECKOUT_ABANDONMENT', 'Checkout abandonment'), ('SUBSCRIPTION_FAILURE', 'Subscription payment failure'), ('OVERDUE_RECEIVABLE', 'Overdue receivable'))


def respond(rid, decision):
    return api(f'/revenue/requests/{rid}/respond', {'decision': decision})


def wait_api(fn, timeout=40, what='api condition'):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.5)
    raise AssertionError('timed out waiting for ' + what)


def pending_link(b, timeout=40):
    b.wait("document.querySelector('#i-journey a[href^=\"/pay/\"]')", timeout, 'a pending payment request link')
    return b.js("document.querySelector('#i-journey a[href^=\"/pay/\"]').getAttribute('href')").split('/')[-1]


@test('individual: a custom payment failure is diagnosed once by a rule, planned, executed by Temporal, and ACCEPT recovers it (simulated)')
def t_individual_custom(b):
    b.goto(BASE + '/ui')
    click(b, '.seg[aria-label="Source of the payment failure"] button', 1)
    b.wait("document.querySelector('#c-form')", 10, 'custom form')
    assert b.js("document.querySelector('#c-submit').disabled"), 'the button stays disabled until the scenario is complete'
    assert b.js("document.querySelectorAll('#c-form input, #c-form textarea').length") == 0, 'no free-text field in the custom form'
    b.wait("document.querySelector('#c-method option[value=upi]')", 20, 'method options')
    click(b, '#c-example-clear')
    b.wait("!document.querySelector('#c-submit').disabled", 20, 'example scenario complete')
    set_value(b, '#c-truth', 'BD_USER_ABANDONED_OTP')
    b.wait("document.querySelector('#c-truth').value==='BD_USER_ABANDONED_OTP'", 10, 'label chosen')
    click(b, '#c-submit')
    b.wait("document.querySelector('#i-journey').innerText.includes('Diagnosis (made once)')", 30, 'journey')
    rid = pending_link(b)
    txt = journey_text(b)
    assert 'Deterministic rule' in txt and 'Model C was not called' in txt and 'CLEAR_AND_CONSISTENT' in txt and 'carries a payment opportunity' in txt
    assert 'matches' in txt and b.js("document.querySelector('#c-truth').value") == '', 'the label is compared afterwards and cleared from the form'
    b.shot('individual-custom-pending')
    assert respond(rid, 'ACCEPT')['status'] == 'ACCEPTED'                         # what the payment page does
    b.wait("document.querySelector('#i-journey').innerText.includes('Simulated recovered amount')", 40, 'recovered')
    txt = journey_text(b)
    assert 'SIMULATED_RECOVERED' in txt and 'WORKFLOW_COMPLETED' in txt and '₹499' in txt and 'not real revenue' in txt and 'CANCELLED' in txt
    b.shot('individual-custom-recovered')


@test('payment page: shows the request, REJECT continues to the next action, a later ACCEPT stops the workflow, answered pages are closed')
def t_payment_page(b):
    ev = api('/revenue/events', OTP_BODY)
    first = wait_api(lambda: (lambda r: r[0] if r else None)(api(f"/revenue/events/{ev['event_id']}")['requests']), what='first request')
    b.goto(BASE + f"/pay/{first['request_id']}", ready='#pay-root')
    b.wait("document.querySelector('#reject')", 15, 'payment page')
    txt = b.js(TEXT)
    for needle in ('Payment request', 'Your payment is pending. Complete payment using the available option.', '₹499', 'Pay with another method', ev['event_id'], first['request_id'],
                   f"revenue-{ev['event_id']}", 'PENDING', 'SIMULATED'):
        assert needle in txt, needle
    assert b.js("document.querySelectorAll('input, textarea, select').length") == 0, 'the payment page has no input field'
    b.shot('payment-page')
    click(b, '#reject')
    b.wait("document.body.innerText.includes('No payment was made')", 15, 'reject outcome')
    assert 'REJECTED' in b.js(TEXT) and not b.js("document.querySelector('#accept')")
    second = wait_api(lambda: (lambda r: r[1] if len(r) > 1 else None)(api(f"/revenue/events/{ev['event_id']}")['requests']), what='second request after the reject')
    mid = api(f"/revenue/events/{ev['event_id']}")
    assert mid['status'] == 'PENDING' and mid['requests'][0]['status'] == 'REJECTED' and mid['diagnosis']['model_c_calls'] == 0
    b.goto(BASE + f"/pay/{second['request_id']}", ready='#pay-root')
    b.wait("document.querySelector('#accept')", 15, 'second payment page')
    click(b, '#accept')
    b.wait("document.body.innerText.includes('Simulated payment completed')", 15, 'accept outcome')
    done = wait_api(lambda: (lambda v: v if v['workflow_state'] == 'WORKFLOW_COMPLETED' else None)(api(f"/revenue/events/{ev['event_id']}")), what='workflow completion')
    assert done['status'] == 'SIMULATED_RECOVERED' and [r['status'] for r in done['requests']] == ['REJECTED', 'ACCEPTED'] and done['recovered_amount_minor'] == 49900
    b.goto(BASE + f"/pay/{first['request_id']}", ready='#pay-root')
    b.wait("document.body.innerText.includes('REJECTED')", 15, 'closed first page')
    assert not b.js("document.querySelector('#accept')"), 'an answered request offers no buttons'
    b.goto(BASE + '/pay/rq_missing', ready='#pay-root')
    b.wait("document.body.innerText.includes('does not exist')", 10, 'unknown request')
    b.errors.clear()                                                              # the 404 for an unknown id is the expected response, not a script error


@test('individual: checkout, subscription and overdue events come from fixed selectors and each creates a simulated payment request')
def t_other_event_types(b):
    b.goto(BASE + '/ui')
    for etype, label in OTHER_FORM_TYPES:
        b.js(f"document.querySelector('#type-{etype}').click()")
        b.wait("document.querySelector('#s-form')", 15, f'{etype} form')
        assert b.js("document.querySelectorAll('#s-form input, #s-form textarea').length") == 0, 'fixed selectors only'
        assert b.js("document.querySelectorAll('#s-form select').length") >= 3
        click(b, '#s-submit')
        b.wait(f"document.querySelector('#i-journey').innerText.includes({json.dumps(label)})", 30, f'{etype} journey')
        rid = pending_link(b)
        txt = journey_text(b)
        assert 'PENDING' in txt and 'Diagnosis (made once)' in txt and 'Event policy' in txt or 'reported by the subscription system' in txt.lower()
        assert respond(rid, 'ACCEPT')['status'] == 'ACCEPTED'
        b.wait("document.querySelector('#i-journey').innerText.includes('SIMULATED_RECOVERED')", 40, f'{etype} recovered')
    b.shot('individual-other-types')


@test('individual: custom dropdowns depend on each other, offer clear inputs only, clear what no longer fits and never offer card-only values for UPI')
def t_custom_dependencies(b):
    b.goto(BASE + '/ui')
    click(b, '.seg[aria-label="Source of the payment failure"] button', 1)
    b.wait("document.querySelector('#c-form')", 10, 'custom form')
    opts = lambda sel: b.js(f"[...document.querySelectorAll('{sel} option')].map(o=>o.value)")
    assert b.js("document.querySelector('#c-step').disabled") and b.js("document.querySelector('#c-submit').disabled"), 'the method comes first'
    b.wait("document.querySelector('#c-method option[value=upi]')", 20, 'method options loaded')
    set_value(b, '#c-method', 'upi')
    b.wait("!document.querySelector('#c-step').disabled", 20, 'fields enabled')
    assert b.js("document.querySelector('#c-hint').value") == 'na' and 'card_expired' not in opts('#c-reason') and 'incorrect_cvv' not in opts('#c-reason')
    assert '__none__' not in opts('#c-reason') and 'payment_failed' not in opts('#c-reason'), 'missing and generic reasons are not offered in Custom Testing'
    assert 'payment_authorization' not in opts('#c-step') and 'customer_psp' in opts('#c-source')
    set_value(b, '#c-source', 'customer')
    b.wait("![...document.querySelectorAll('#c-step option')].some(o=>o.value==='payment_credit_response')", 20, 'steps narrowed by the source')
    set_value(b, '#c-step', 'payment_authentication')
    b.wait("![...document.querySelectorAll('#c-reason option')].some(o=>o.value==='insufficient_funds')", 20, 'reasons narrowed by source and step')
    set_value(b, '#c-reason', 'payment_cancelled')
    set_value(b, '#c-truth', 'BD_USER_ABANDONED_OTP')
    b.wait("document.querySelector('#c-truth').value==='BD_USER_ABANDONED_OTP'", 10, 'label kept')
    assert 'CARD_EXPIRED' not in opts('#c-truth')
    set_value(b, '#c-method', 'card')
    b.wait("document.querySelector('#c-hint').value==='unknown'", 20, 'card default hint')
    assert b.js("document.querySelector('#c-truth').value") == 'BD_USER_ABANDONED_OTP' and 'CARD_EXPIRED' in opts('#c-truth'), 'the label is kept and widened'
    assert 'incorrect_cvv' in opts('#c-reason') and 'card_expired' not in opts('#c-reason'), 'card-only reasons appear for card, still filtered by source and step'
    set_value(b, '#c-reason', 'incorrect_cvv')
    set_value(b, '#c-truth', 'CARD_EXPIRED')
    b.wait("document.querySelector('#c-truth').value==='CARD_EXPIRED'", 10, 'card-only label chosen')
    set_value(b, '#c-method', 'upi')
    b.wait("document.querySelector('#c-reason').value===''", 20, 'card-only reason cleared on UPI')
    assert b.js("document.querySelector('#c-truth').value") == '', 'the label is cleared, never replaced by another cause'
    assert b.js("document.querySelector('#c-step').value") == 'payment_authentication', 'compatible selections are preserved'
    shown = b.js("document.querySelector('#c-cleared').innerText")
    assert 'cleared' in shown, f'the form explains what it cleared (got: {shown!r})'
    assert b.js("document.querySelector('#c-submit').disabled"), 'the button is disabled until the cleared fields are chosen again'
    b.shot('individual-custom-dependencies')


PREVIEW_MARK = {'clear': 'decided by a rule', 'generic': 'was asked once', 'contradictory': 'sent to a human'}


@test('individual: the diagnosis-scenario selector builds inputs for all three routes and shows what the system actually did')
def t_custom_scenarios(b):
    b.goto(BASE + '/ui')
    click(b, '.seg[aria-label="Source of the payment failure"] button', 1)
    b.wait("document.querySelector('#c-scenario') && document.querySelector('#c-method option[value=upi]')", 20, 'scenario selector')
    assert b.js("[...document.querySelectorAll('#c-scenario option')].map(o=>o.textContent)") == [
        'Clear and consistent (deterministic rules)', 'Missing / generic gateway information (Model C)', 'Contradictory gateway information (human intervention)']
    assert b.js("document.querySelectorAll('#c-form input, #c-form textarea').length") == 0, 'still no free-text field'
    cases = (('clear', 'CLEAR_AND_CONSISTENT', ('Deterministic rule', 'Model C was not called'), '0'),
             ('generic', 'MISSING_OR_GENERIC', ('Frozen Model C (called once)', 'the frozen Model C was asked once', 'Safety gate'), '1'),
             ('contradictory', 'CONTRADICTORY', ('No automatic diagnosis', 'Model C was not called', 'No workflow was started', 'cannot occur on UPI'), '0'))
    for kind, route, needles, model_calls in cases:
        click(b, f'#c-example-{kind}')
        b.wait(f"document.querySelector('#c-scenario').value==={json.dumps(kind)} && !document.querySelector('#c-submit').disabled && document.querySelector('#c-preview').innerText.includes({json.dumps(PREVIEW_MARK[kind])})", 20, f'{kind} example complete')
        preview = b.js("document.querySelector('#c-preview').innerText")
        assert 'differs from the selected scenario' not in preview, preview
        if kind == 'generic':
            assert b.js("document.querySelector('#c-reason').value") == '__none__', 'the generic example carries no error reason'
            assert 'payment_cancelled' not in b.js("[...document.querySelectorAll('#c-reason option')].map(o=>o.value).join()"), 'clear reasons are not offered under the generic scenario'
        if kind == 'contradictory':
            assert 'card_expired' in b.js("[...document.querySelectorAll('#c-reason option')].map(o=>o.value).join()"), 'conflicting evidence can be built'
        set_value(b, '#c-truth', 'BD_INSUFFICIENT_FUNDS')                     # a hidden label that must change nothing
        previous = b.js("(document.querySelector('#i-journey .mono')||{}).innerText||''")          # the journey of the previous example is still on screen until the new one replaces it
        click(b, '#c-submit')
        b.wait(f"document.querySelector('#i-journey').innerText.includes('Diagnosis (made once)') && !({json.dumps(previous)} && document.querySelector('#i-journey').innerText.includes({json.dumps(previous)}))", 40, f'{kind} journey')
        txt = journey_text(b)
        assert route in txt.upper().replace(' ', '_') and 'Scenario selected' in txt, txt[:1500]
        for needle in needles:
            assert needle in txt, (kind, needle, txt[:600])
        assert 'Temporal execution' in txt and 'Recovery plan' in txt
        assert b.js("(document.querySelector('#i-journey').innerText.match(/Model C calls[\\s]*([0-9])/)||[])[1]") == model_calls, (kind, txt[:500])
        b.shot(f'individual-scenario-{kind}')
        for r in api('/revenue/requests?limit=50')['pending']:
            respond(r['request_id'], 'REJECT')
    # a selection that no longer fits the chosen scenario is cleared by the server, never silently replaced
    click(b, '#c-example-contradictory')
    b.wait("document.querySelector('#c-reason').value==='card_expired'", 20, 'conflicting reason chosen')
    set_value(b, '#c-scenario', 'clear')
    b.wait("document.querySelector('#c-cleared').innerText.includes('cleared')", 20, 'conflicting choices cleared under the clear scenario')
    assert b.js("document.querySelector('#c-submit').disabled")


@test('individual: a contradictory record goes straight to Human Intervention and a missing/generic one asks Model C exactly once')
def t_routes_in_ui(b):
    b.goto(BASE + '/ui')
    b.wait("document.querySelectorAll('.dataset-card').length===5", 20, 'five dataset cards (four prebuilt plus the routing fixtures)')
    click(b, '.dataset-card', 4)
    b.wait("document.querySelector('#i-route')", 20, 'fixture type filter')
    assert b.js("document.querySelectorAll('#panel-individual input[type=text], #panel-individual input[type=search], #panel-individual textarea').length") == 0, 'no free-text field'
    set_value(b, '#i-route', 'CONTRADICTORY')
    b.wait("document.querySelectorAll('#i-table tbody tr').length>0 && [...document.querySelectorAll('#i-table tbody tr')].every(r=>r.innerText.includes('fixture: contradictory'))", 20, 'contradictory fixtures')
    click(b, '#i-table tbody tr')
    b.wait("!document.querySelector('#i-start').disabled", 5, 'start enabled')
    click(b, '#i-start')
    b.wait("document.querySelector('#i-journey').innerText.includes('Diagnosis (made once)')", 30, 'journey')
    txt = journey_text(b)
    assert 'CONTRADICTORY' in txt and 'Model C was not called' in txt and 'HUMAN_INTERVENTION' in txt and 'No recovery action is planned' in txt and 'No payment request was created' in txt
    b.shot('individual-contradictory')
    set_value(b, '#i-route', 'MISSING_OR_GENERIC')
    b.wait("document.querySelectorAll('#i-table tbody tr').length>0 && [...document.querySelectorAll('#i-table tbody tr')].every(r=>r.innerText.includes('fixture: missing or generic'))", 20, 'generic fixtures')
    click(b, '#i-table tbody tr')
    b.wait("!document.querySelector('#i-start').disabled", 5, 'start enabled')
    click(b, '#i-start')
    b.wait("document.querySelector('#i-journey').innerText.includes('Frozen Model C (called once)')", 30, 'model c route')
    txt = journey_text(b)
    assert 'MISSING_OR_GENERIC' in txt and 'Model C calls' in txt and 'the frozen Model C was asked once' in txt
    for r in api('/revenue/requests?limit=50')['pending']:                       # leave nothing waiting on the demo timeout
        respond(r['request_id'], 'REJECT')
    b.shot('individual-model-c-route')


@test('intervention: revenue-pipeline cases (contradictory data, risk block, no consent) are listed read-only with their reasons and conflicting fields')
def t_revenue_interventions(b):
    from services.app.revenue import fixtures
    contradictory = api('/revenue/events/from-dataset', {'dataset': 'routing_fixtures', 'payment_id': next(r['payment_id'] for r in fixtures.records() if r['expected_route'] == 'CONTRADICTORY')})
    risk = api('/revenue/events', {**OTP_BODY, 'gw_error_code': 'GATEWAY_ERROR', 'error_reason': 'payment_risk_check_failed'})
    no_consent = api('/revenue/events', {**OTP_BODY, 'customer_id': 'cust_syn_05'})
    wait_api(lambda: all(api(f"/revenue/events/{e['event_id']}")['workflow_state'] in ('WORKFLOW_COMPLETED', 'NOT_REQUIRED') for e in (contradictory, risk, no_consent)), what='the three events to finish')
    wait_api(lambda: {e['event_id'] for e in (contradictory, risk, no_consent)} <= {i['id'] for i in api('/interventions?limit=200')['items']}, what='the three cases in the intervention queue')
    b.goto(BASE + '/ui#intervention')
    b.wait("document.querySelector('#f-kind') && document.querySelector('#f-ds')", 15, 'filters')
    b.wait("document.querySelectorAll('#iv-list tbody tr').length>0", 20, 'cases listed')
    txt = b.js("document.querySelector('#iv-list').innerText + document.querySelector('#iv-counts').innerText")
    assert 'Safety escalation' in txt and 'Plan intervention' in txt and 'Diagnosis intervention' in txt and not any(w in txt for w in ('Temporal runs', 'Offline'))
    set_value(b, '#f-kind', 'SAFETY_ESCALATION')
    b.wait("[...document.querySelectorAll('#iv-list tbody tr')].every(r=>r.innerText.includes('Safety escalation')) && document.querySelectorAll('#iv-list tbody tr').length>0", 20, 'category filter')
    set_value(b, '#f-kind', '')
    b.wait(f"document.querySelector('#iv-list').innerText.includes({json.dumps(contradictory['event_id'])})", 20, 'unfiltered list reloaded')
    for e, needles in ((contradictory, ('Contradictory', 'contradict each other', 'Model C was not called')), (risk, ('risk block',)), (no_consent, ('No customer-facing recovery action',))):
        found = b.js(f"(()=>{{const r=[...document.querySelectorAll('#iv-list tbody tr')].find(r=>r.innerText.includes({json.dumps(e['event_id'])})); if(r) r.click(); return !!r}})()")
        assert found, (e['event_id'], b.js("[...document.querySelectorAll('#iv-list tbody tr')].map(r=>r.cells[0].innerText).slice(0,8)"))
        b.wait(f"document.querySelector('#iv-detail').innerText.includes({json.dumps(e['event_id'])})", 20, 'case detail')
        detail = b.js("document.querySelector('#iv-detail').innerText")
        assert 'Read-only' in detail and 'Revenue pipeline' in detail and 'Audit timeline' in detail, detail[:200]
        for needle in needles:
            assert needle in detail, (needle, detail[:400])
        assert not b.js("[...document.querySelectorAll('#iv-detail button')].some(x=>/approve|reject|retry|reassess|assign|resolve|accept/i.test(x.textContent))"), 'no action buttons'
        assert not b.js("document.querySelector('#iv-detail a[href^=\"/pay/\"]')"), 'no payment page link in the read-only view'
    assert 'true cause' not in b.js("document.querySelector('#iv-detail').innerText").lower()
    b.js("document.querySelector('#iv-hist summary').click()")
    b.wait("document.querySelector('#iv-hist-body').innerText.includes('retired') || document.querySelector('#iv-hist-body').innerText.includes('earlier research')", 20, 'historical records')
    hist = b.js("document.querySelector('#iv-hist-body').innerText")
    assert 'cannot be acted on' in hist and not b.js("[...document.querySelectorAll('#iv-hist button')].length"), hist[:200]
    b.shot('intervention-revenue')


@test('dataset: the automated revenue pipeline runs all three routes without manual answers and its numbers match the API')
def t_revenue_pipeline(b):
    b.goto(BASE + '/ui#dataset')
    b.wait("document.querySelector('#rr-start') && document.querySelector('#rr-dataset option[value=routing_fixtures]')", 20, 'pipeline card')
    set_value(b, '#rr-dataset', 'routing_fixtures')
    set_value(b, '#rr-temporal', '5')
    click(b, '#rr-start')
    b.wait("document.querySelector('#rr-out').innerText.includes('Total cases') && document.querySelector('#rr-out').innerText.includes('sampled workflows')", 120, 'run results')
    txt = b.js("document.querySelector('#d-rv').innerText")
    run = api('/revenue/runs?limit=1')['runs'][0]
    m = api(f"/revenue/runs/{run['run_id']}")['metrics']
    assert m['total_cases'] == 66 and m['route_distribution'] == {'CLEAR_AND_CONSISTENT': 24, 'MISSING_OR_GENERIC': 24, 'CONTRADICTORY': 18} and m['max_model_c_calls_per_case'] == 1
    for needle in ('66', 'Clear and consistent (rule)', 'Missing or generic (Model C)', 'Contradictory (human)', f"{m['model_c_calls']} (max 1 per case)", 'SIMULATED_RECOVERED', 'SIMULATED_FAILED',
                   'HUMAN_INTERVENTION', 'Simulated recovered amount', 'Safety violations', '5 of 5 sampled workflows completed', 'matched the in-process result exactly', 'not recovered revenue'):
        assert needle.lower() in txt.lower(), needle
    assert str(m['SIMULATED_RECOVERED']) in txt and not [w for w in FORBIDDEN_DATASET_WORDS if w in txt.lower()], [w for w in FORBIDDEN_DATASET_WORDS if w in txt.lower()]
    b.shot('dataset-revenue-pipeline')


def run_card(b, dataset, limit='', temporal='0', csv_id=None):
    """Drive the automated-pipeline card and return the finished run document (from the API) once the page shows its numbers."""
    b.wait("document.querySelector('#rr-start') && document.querySelector('#rr-dataset option[value=routing_fixtures]')", 20, 'pipeline card')
    before = {r['run_id'] for r in api('/revenue/runs?limit=100')['runs']}
    if dataset:
        set_value(b, '#rr-dataset', dataset)
    set_value(b, '#rr-limit', limit)
    set_value(b, '#rr-temporal', temporal)
    click(b, '#rr-start')
    doc = wait_api(lambda: next((api(f"/revenue/runs/{r['run_id']}") for r in api('/revenue/runs?limit=100')['runs']
                                 if r['run_id'] not in before and r['status'] == 'completed'), None), 120, 'the run to finish')
    stats = lambda: {l: v for l, v in b.js("[...document.querySelectorAll('#rr-out .stat')].map(s=>[s.querySelector('.l').textContent, s.querySelector('.v').textContent])")}
    m = doc['metrics']
    expect = {'Total cases': str(m['total_cases']), 'SIMULATED_RECOVERED': str(m['SIMULATED_RECOVERED']), 'SIMULATED_FAILED': str(m['SIMULATED_FAILED']), 'HUMAN_INTERVENTION': str(m['human_interventions'])}
    b.wait("(()=>{const got=Object.fromEntries([...document.querySelectorAll('#rr-out .stat')].map(s=>[s.querySelector('.l').textContent,s.querySelector('.v').textContent]));"
           f"const want={json.dumps(expect)};return Object.keys(want).every(k=>got[k]===want[k]);}})()", 30, 'the numbers of THIS run on the page')
    return doc, stats()


@test('dataset: every prebuilt dataset and the controlled event mix run automatically and the page matches the API; no safety-limit judgments are shown')
def t_datasets(b):
    b.goto(BASE + '/ui#dataset')
    b.wait("document.querySelector('#rr-dataset')", 20, 'pipeline card')
    names = b.js("[...document.querySelectorAll('#rr-dataset option')].map(o=>o.value)")
    assert names[:4] == ['normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime'] and 'routing_fixtures' in names and 'event_type_mix' in names and '__csv__' in names
    for name in ('normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime', 'event_type_mix'):
        doc, got = run_card(b, name, '50')
        m = doc['metrics']
        want = 48 if name == 'event_type_mix' else 50                               # the mix has 48 cases in all
        assert m['total_cases'] == want and got['Total cases'] == str(want), (name, got)
        assert got['SIMULATED_RECOVERED'] == str(m['SIMULATED_RECOVERED']) and got['SIMULATED_FAILED'] == str(m['SIMULATED_FAILED']) and got['HUMAN_INTERVENTION'] == str(m['human_interventions'])
        assert m['SIMULATED_RECOVERED'] + m['SIMULATED_FAILED'] + m['human_interventions'] == want and m['max_model_c_calls_per_case'] <= 1
        assert got['Model C calls'].startswith(str(m['model_c_calls'])) and got['Safety violations'] == str(m['safety_violations'])
        text = b.js("document.querySelector('#d-rv').innerText").lower()
        assert not [w for w in FORBIDDEN_DATASET_WORDS if w in text], [w for w in FORBIDDEN_DATASET_WORDS if w in text]
    b.shot('dataset-runs')


@test('dataset: CSV upload is validated with clear messages, then runs through the same automated pipeline')
def t_csv(b):
    b.goto(BASE + '/ui#dataset')
    b.wait("document.querySelector('#rr-file')", 20, 'csv control')
    assert b.js("document.querySelector('#d-rv a[download]').getAttribute('href')") == '/tests/csv-template'
    tmp = Path(tempfile.mkdtemp(prefix='ui-csv-'))
    template = urllib.request.urlopen(BASE + '/tests/csv-template').read().decode()
    labelled = template.replace('insufficient_funds,ord_f1,card_tok_1,psp_a,na,850,card_3ds,42,', 'insufficient_funds,ord_f1,card_tok_1,psp_a,na,850,card_3ds,42,BD_INSUFFICIENT_FUNDS')\
                       .replace('payment_timed_out,ord_f2,vpa_2,psp_a,na,9000,upi_intent,95,', 'payment_timed_out,ord_f2,vpa_2,psp_a,na,9000,upi_intent,95,TD_PSP_TIMEOUT')
    files = {'template.csv': template, 'labelled.csv': labelled, 'bad.csv': template.replace(',card,', ',cheque,', 1).replace('2026-05-04T10:15:00+00:00', 'yesterday')}

    def upload(name):
        (tmp / name).write_text(files[name], encoding='utf-8')
        node = b.send('DOM.getDocument')['root']['nodeId']
        nid = b.send('DOM.querySelector', nodeId=node, selector='#rr-file')['nodeId']
        b.send('DOM.setFileInputFiles', files=[str(tmp / name)], nodeId=nid)
    upload('bad.csv')
    b.wait("document.querySelector('#rr-csv-out .notice.bad')", 20, 'validation errors')
    err = b.js("document.querySelector('#rr-csv-out').innerText")
    assert 'cannot be used' in err and 'method' in err and 'occurred_at' in err and 'cheque' in err
    b.shot('csv-errors')
    upload('template.csv')
    b.wait("document.querySelector('#rr-csv-out .notice.good')", 20, 'valid csv')
    assert 'No hidden labels' in b.js("document.querySelector('#rr-csv-out').innerText") and b.js("document.querySelector('#rr-dataset').value") == '__csv__'
    doc, got = run_card(b, None)
    assert doc['dataset'].startswith('csv:') and doc['metrics']['total_cases'] == 2 and got['Total cases'] == '2'
    upload('labelled.csv')
    b.wait("document.querySelector('#rr-csv-out .notice.good') && document.querySelector('#rr-csv-out').innerText.includes('Hidden labels were found')", 20, 'labelled csv')
    doc, got = run_card(b, None)
    assert doc['metrics']['total_cases'] == 2 and doc['label'].startswith('Automated simulation')
    assert not [w for w in FORBIDDEN_DATASET_WORDS if w in b.js("document.querySelector('#d-rv').innerText").lower()]


@test('dataset: a refused second run (HTTP 409) is explained and the button is usable again')
def t_busy(b):
    b.goto(BASE + '/ui#dataset')
    b.wait("document.querySelector('#rr-start')", 20)
    b.js("""window.__orig=window.fetch;window.fetch=(u,o)=>{if(u==='/revenue/runs'&&o&&o.method==='POST')return Promise.resolve(new Response(JSON.stringify({detail:'another revenue run is in progress (rv1); only one at a time'}),{status:409,headers:{'content-type':'application/json'}}));return window.__orig(u,o);}""")
    click(b, '#rr-start')
    b.wait("document.querySelector('#rr-msg').innerText.includes('in progress')", 10, '409 message')
    assert 'only one at a time' in b.js("document.querySelector('#rr-msg').innerText") and not b.js("document.querySelector('#rr-start').disabled")
    b.js("window.fetch=window.__orig")


@test('no hidden truth in browser state; responsive layout has no horizontal scroll on a phone-sized viewport')
def t_state_and_responsive(b):
    b.goto(BASE + '/ui')
    assert b.js("localStorage.length + sessionStorage.length") == 0 and b.js("Object.keys(window.__ui)") == ['api']
    assert not b.js("document.cookie")
    b.viewport(390, 844, mobile=True)
    for t in ['individual', 'dataset', 'intervention']:
        b.js(f"window.location.hash='#{t}'")
        b.wait(f"!document.querySelector('#panel-{t}').hidden", 5)
        time.sleep(0.8)
        overflow = b.js("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 2, f'{t}: horizontal overflow {overflow}px'
        b.shot(f'mobile-{t}')
    b.viewport(1366, 1100)


def main():
    if not CHROME:
        sys.exit('Chrome or Edge not found')
    b = Browser()
    failed = 0
    try:
        for fn in results:
            if os.environ.get('ONLY') and os.environ['ONLY'] not in fn.test_name:
                continue
            b.errors.clear()
            started = time.time()
            try:
                fn(b)
                if b.errors:
                    raise AssertionError('browser console/script errors: ' + '; '.join(b.errors[:3]))
                print(f'PASS  {fn.test_name}  ({time.time() - started:.0f}s)')
            except Exception as exc:                                         # noqa: BLE001
                failed += 1
                frame = traceback.extract_tb(exc.__traceback__)[-1]
                print(f'FAIL  {fn.test_name}\n      {type(exc).__name__}: {exc}  [line {frame.lineno}: {frame.line}]')
                try:
                    b.shot('FAIL-' + re.sub(r'\W+', '-', fn.test_name)[:40])
                except Exception:                                            # noqa: BLE001
                    pass
    finally:
        b.close()
    print(f'\n{len(results) - failed}/{len(results)} browser checks passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
