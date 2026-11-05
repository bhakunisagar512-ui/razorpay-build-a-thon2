
/* AI Revenue Recovery: testing console. Vanilla JS, no dependencies, no build step.
 * Talks only to the Stage 2 API (same origin). All text is inserted with textContent / text nodes, never as markup.
 * Deliberately NOT shown anywhere: hidden true causes of individual payments, safety limits, acceptance criteria or
 * pass/fail judgments. Explanations are generated here from structured fields so no numeric thresholds appear. */
(() => {
'use strict';

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k.startsWith('aria-') && typeof v === 'boolean') { e.setAttribute(k, String(v)); continue; }
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
const clear = (e) => { while (e.firstChild) e.removeChild(e.firstChild); return e; };

function detailText(d) {
  if (!d) return '';
  const x = d.detail;
  if (typeof x === 'string') return x;
  if (Array.isArray(x)) return x.map((i) => (i.loc ? i.loc.slice(1).join('.') + ': ' : '') + i.msg).join('; ');
  return '';
}
async function api(path, { method = 'GET', body, rawType, timeout = 20000 } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeout);
  try {
    const headers = rawType ? { 'content-type': rawType } : body !== undefined ? { 'content-type': 'application/json' } : {};
    const res = await fetch(path, { method, headers, signal: ctl.signal, body: body === undefined ? undefined : rawType ? body : JSON.stringify(body) });
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = { detail: text.slice(0, 200) }; }
    if (!res.ok) { const err = new Error(detailText(data) || `Request failed (${res.status})`); err.status = res.status; err.data = data; throw err; }
    return data;
  } catch (e) {
    if (e.name === 'AbortError') { const err = new Error('The request timed out. Is the API running?'); err.status = 0; throw err; }
    if (e.status === undefined) { e.status = 0; if (/fetch/i.test(e.message)) e.message = 'Cannot reach the API. Is the Docker stack running?'; }
    throw e;
  } finally { clearTimeout(timer); }
}

/* ------------------------------------------------------------------ vocabulary (no thresholds, no hidden truth) */
const CAUSE = {
  BD_INSUFFICIENT_FUNDS: 'Insufficient funds', BD_USER_ABANDONED_OTP: 'OTP abandoned by customer', CARD_EXPIRED: 'Card expired',
  RISK_BLOCKED: 'Risk blocked', TD_ISSUER_DOWN: 'Issuer downtime', TD_PSP_TIMEOUT: 'PSP timeout', AMBIGUOUS: 'No diagnosis (fallback)',
};
const causeName = (c) => (c ? CAUSE[c] || c : '—');
const TERMINAL = {
  PLAN_INTERVENTION: ['Plan intervention', 'bad'], DIAGNOSIS_INTERVENTION: ['Diagnosis intervention', 'failed'], SAFETY_ESCALATION: ['Safety escalation', 'escalated'],
};
const money = (minor) => (minor == null ? '—' : '₹' + (Number(minor) / 100).toLocaleString('en-IN', { maximumFractionDigits: 2 }));
const when = (iso) => { if (!iso) return '—'; const d = new Date(iso); return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString(); };
const clock = (iso) => { if (!iso) return '—'; const d = new Date(iso); return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleTimeString(); };
/* ------------------------------------------------------------------ small components */
const badge = (text, cls = 'neutral') => h('span', { class: `badge ${cls}` }, text);
function terminalBadge(kind) { const [t, c] = TERMINAL[kind] || [kind, 'neutral']; return badge(t, c); }
const simTag = () => h('span', { class: 'badge sim', title: 'Simulated under the evaluation assumption' }, 'SIMULATED');
function kv(pairs) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v] of pairs) if (v != null && v !== '') dl.append(h('dt', { text: k }), h('dd', {}, v));
  return dl;
}
const notice = (text, cls = '') => h('div', { class: `notice ${cls}`, role: cls === 'bad' ? 'alert' : 'status' }, text);
const empty = (text) => h('div', { class: 'empty' }, text);
const loading = (text = 'Loading…') => h('div', { class: 'muted' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), ' ', text);
function toast(msg, kind = 'bad') {
  const t = $('#toast');
  clear(t).append(notice(msg, kind));
  clearTimeout(toast.t);
  toast.t = setTimeout(() => clear(t), 7000);
}
function pager(state, total, onChange) {
  const { offset, limit } = state;
  return h('div', { class: 'pager' },
    h('span', { class: 'small muted' }, total ? `Showing ${offset + 1}–${Math.min(offset + limit, total)} of ${total}` : 'No rows'),
    h('span', { class: 'row' },
      h('button', { class: 'btn small', type: 'button', disabled: offset <= 0, onclick: () => onChange(Math.max(0, offset - limit)) }, '‹ Previous'),
      h('button', { class: 'btn small', type: 'button', disabled: offset + limit >= total, onclick: () => onChange(offset + limit) }, 'Next ›')));
}
/* ------------------------------------------------------------------ cycle explanation (generated here, no thresholds) */
/* ------------------------------------------------------------------ journey rendering (progress API) */
/* ------------------------------------------------------------------ tabs */
const TABS = ['individual', 'dataset', 'intervention'];
const inited = {};
function showTab(name, focus) {
  if (!TABS.includes(name)) name = 'individual';
  for (const t of TABS) {
    const sel = t === name;
    const btn = $(`#tab-${t}`);
    btn.setAttribute('aria-selected', sel ? 'true' : 'false');
    btn.tabIndex = sel ? 0 : -1;
    $(`#panel-${t}`).hidden = !sel;
  }
  if (focus) $(`#tab-${name}`).focus();
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (!inited[name]) { inited[name] = true; ({ individual: initIndividual, dataset: initDataset, intervention: initIntervention })[name](); }
  else if (name === 'intervention') loadInterventions();
}

/* ================================================================== TAB 1: individual payment testing */
const S1 = { datasets: [], ds: null, page: { items: [], total: 0, offset: 0, limit: 12 }, sel: null, mode: 'dataset', pending: false, testId: null, poll: 0, method: '', search: '', route: '' };
function initIndividual() {
  const root = $('#panel-individual');
  root.append(h('div', { class: 'grid cols-2' },
    h('div', { class: 'stack' }, h('div', { class: 'card', id: 'i-pick' }), h('div', { id: 'i-preview' }), h('div', { class: 'card', id: 'i-pending' })),
    h('div', { id: 'i-journey' }, empty('Create an event to follow its recovery: one diagnosis, the plan, Temporal execution and the simulated payment requests.'))));
  renderPick();
  api('/revenue/catalog').then((c) => { RV.catalog = c; renderPick(); }).catch((e) => clear($('#i-pick')).append(notice(`Could not load the event options: ${e.message}`, 'bad')));
  loadPending();
  loadDatasetsForIndividual();
}
const FIXTURE_DATASET = { name: 'routing_fixtures', available: true, records: 66, description: 'Controlled records: clear, missing or generic, and contradictory gateway data.' };
function renderPick() {
  const box = clear($('#i-pick'));
  const types = RV.catalog ? RV.catalog.event_types.map((t) => [t.id, t.label]) : EVENT_TYPES_FALLBACK;
  box.append(h('header', {}, h('h2', {}, 'Create a revenue-risk event'),
    h('div', { class: 'seg', role: 'group', 'aria-label': 'Event type' }, types.map(([id, text]) => h('button', { type: 'button', id: 'type-' + id, 'aria-pressed': RV.type === id, onclick: () => { RV.type = id; renderPick(); } }, text)))));
  if (RV.type === 'PAYMENT_FAILURE') {
    box.append(h('div', { class: 'seg', role: 'group', 'aria-label': 'Source of the payment failure', style: 'margin-bottom:10px' },
      h('button', { type: 'button', 'aria-pressed': S1.mode === 'dataset', onclick: () => { S1.mode = 'dataset'; renderPick(); } }, 'From a dataset'),
      h('button', { type: 'button', 'aria-pressed': S1.mode === 'custom', onclick: () => { S1.mode = 'custom'; renderPick(); } }, 'Custom payment')));
    box.append(S1.mode === 'dataset' ? pickDataset() : customForm());
  } else box.append(simpleEventForm(RV.type));
}
async function loadDatasetsForIndividual() {
  try {
    const d = await api('/tests/datasets');
    S1.datasets = [...d.datasets, FIXTURE_DATASET];
    if (!S1.ds) S1.ds = (d.datasets.find((x) => x.available) || {}).name || null;
    renderPick();
    if (S1.ds) loadPayments();
  } catch (e) { clear($('#i-pick')).append(notice(`Could not load datasets: ${e.message}`, 'bad')); }
}
function datasetCards(list, current, onPick) {
  return h('div', { class: 'grid', style: 'grid-template-columns:repeat(auto-fit,minmax(200px,1fr))' }, list.map((d) =>
    h('button', { type: 'button', class: 'dataset-card', 'aria-pressed': current === d.name, disabled: !d.available, onclick: () => onPick(d.name) },
      h('div', { class: 'name' }, d.name), h('div', { class: 'count' }, d.available ? `${d.records} payments` : 'not built'),
      d.available ? h('div', { class: 'small muted' }, d.description) : null,
      d.available && d.representative_of_natural_performance === false && d.name === 'mixed_difficult' ? badge('Deliberately selected hard cases', 'warn') : null)));
}
function pickDataset() {
  const box = h('div', { class: 'stack' });
  box.append(datasetCards(S1.datasets, S1.ds, (n) => { S1.ds = n; S1.sel = null; S1.search = ''; S1.route = ''; S1.method = ''; S1.page.offset = 0; renderPick(); loadPayments(); }));
  box.append(h('div', { class: 'row' },
    h('div', { style: 'flex:1;min-width:130px' }, h('label', { for: 'i-method' }, 'Method'),
      h('select', { id: 'i-method', onchange: (e) => { S1.method = e.target.value; S1.page.offset = 0; loadPayments(); } },
        ['', 'card', 'upi', 'netbanking'].map((m) => h('option', { value: m, selected: S1.method === m }, m || 'All methods')))),
    S1.ds === 'routing_fixtures' ? h('div', { style: 'flex:2;min-width:160px' }, h('label', { for: 'i-route' }, 'Fixture type'),
      h('select', { id: 'i-route', onchange: (e) => { S1.route = e.target.value; S1.page.offset = 0; loadPayments(); } },
        [['', 'All fixtures'], ['CLEAR_AND_CONSISTENT', 'Clear and consistent'], ['MISSING_OR_GENERIC', 'Missing or generic'], ['CONTRADICTORY', 'Contradictory']].map(([v, tx]) => h('option', { value: v, selected: S1.route === v }, tx)))) : null));
  box.append(h('div', { id: 'i-table' }, loading('Loading payments…')));
  box.append(h('div', { class: 'row' },
    h('button', { class: 'btn primary', id: 'i-start', type: 'button', disabled: !S1.sel || S1.pending, onclick: startDatasetPayment }, S1.pending ? 'Starting…' : 'Start recovery workflow'),
    h('span', { class: 'hint', id: 'i-sel' }, S1.sel ? `Selected: ${S1.sel}` : 'Select a payment from the table.')));
  box.append(h('p', { class: 'hint' }, 'The true cause of every payment stays hidden from diagnosis, planning and this screen; only the independent simulator can see it.'));
  return box;
}
let plRequest = 0;                                                           // only the newest request may update the table (responses can arrive out of order)
async function loadPayments() {
  if (!S1.ds) return;
  const mine = ++plRequest;
  const t = $('#i-table');
  if (t) clear(t).append(loading('Loading payments…'));
  try {
    const q = new URLSearchParams({ offset: S1.page.offset, limit: S1.page.limit });
    if (S1.method) q.set('method', S1.method);
    if (S1.route && S1.ds === 'routing_fixtures') q.set('route', S1.route);
    const d = await api(S1.ds === 'routing_fixtures' ? `/revenue/fixtures/payments?${q}` : `/tests/datasets/${encodeURIComponent(S1.ds)}/payments?${q}`);
    if (mine !== plRequest) return;
    S1.page = { ...S1.page, items: d.items, total: d.total, offset: d.offset };
    renderPaymentTable();
  } catch (e) { const tt = $('#i-table'); if (tt) clear(tt).append(notice(`Could not load payments: ${e.message}`, 'bad')); }
}
function renderPaymentTable() {
  const t = $('#i-table');
  if (!t) return;
  clear(t);
  if (!S1.page.items.length) { t.append(empty('No payments match these filters.')); return; }
  const body = h('tbody', {}, S1.page.items.map((r) => h('tr', { class: `clickable ${S1.sel === r.payment_id ? 'selected' : ''}`, onclick: () => selectPayment(r.payment_id), tabindex: 0,
      onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectPayment(r.payment_id); } }, 'aria-selected': S1.sel === r.payment_id },
    h('td', {}, h('input', { type: 'radio', name: 'i-pay', 'aria-label': `Select ${r.payment_id}`, checked: S1.sel === r.payment_id, style: 'width:auto' })),
    h('td', { class: 'mono' }, r.payment_id.replace(/^pay_v4_/, '…'), r.fixture_route ? h('div', { class: 'small muted' }, `fixture: ${pretty(r.fixture_route).toLowerCase()}`) : null), h('td', {}, r.method),
    h('td', {}, [r.gw_error_code, r.error_reason].filter((x) => x && x !== 'missing').join(' · ')), h('td', { class: 'num' }, money(r.amount_minor_raw)))));
  t.append(h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['', 'Payment', 'Method', 'Error', 'Amount'].map((c, i) => h('th', { class: i === 4 ? 'num' : '' }, c)))), body)),
    pager(S1.page, S1.page.total, (o) => { S1.page.offset = o; loadPayments(); }));
}
function selectPayment(id) {
  S1.sel = id;
  renderPaymentTable();
  const b = $('#i-start'); if (b) b.disabled = S1.pending;
  const s = $('#i-sel'); if (s) s.textContent = `Selected: ${id}`;
}
async function startDatasetPayment() {
  if (S1.pending || !S1.sel) return;
  S1.pending = true; renderPick(); loadPaymentsKeep();
  try {
    const view = await api('/revenue/events/from-dataset', { method: 'POST', body: { dataset: S1.ds, payment_id: S1.sel } });
    clear($('#i-preview'));
    showRevenueEvent(view);
  } catch (e) { toast(`Could not start the event: ${e.message}`); }
  finally { S1.pending = false; renderPick(); if (S1.mode === 'dataset') loadPayments(); }
}
function loadPaymentsKeep() { if (S1.mode === 'dataset') { renderPaymentTable(); } }
/* ================================================================== Revenue-risk console (Individual tab)
   Four event types, every input a dropdown (no free text; ids are generated by the server). Payment-failure scenario fields are filtered by the server
   (POST /tests/custom-options with scope "clear": Custom Testing accepts clear and consistent inputs only); the browser keeps no copy of the rules.
   Diagnosis happens once, on the server. Everything shown is SIMULATED. */
const SCENARIO_FIELDS = [['method', 'c-method'], ['instrument_hint', 'c-hint'], ['gw_error_code', 'c-code'], ['gw_error_source', 'c-source'],
  ['gw_error_step', 'c-step'], ['error_reason', 'c-reason'], ['true_cause', 'c-truth']];
const NO_REASON = '__none__';
/* The selector only chooses which inputs the dropdowns can build (the scope of the server-side rules). It never tells the server which route to take. */
const DIAGNOSIS_SCENARIOS = [['clear', 'Clear and consistent (deterministic rules)'], ['generic', 'Missing / generic gateway information (Model C)'], ['contradictory', 'Contradictory gateway information (human intervention)']];
const SCOPE_OF = { clear: 'clear', generic: 'generic', contradictory: 'conflict' };
const EXAMPLES = {
  clear: ['Clear example', 'UPI, customer cancels authentication (reason payment_cancelled).', { method: 'upi', gw_error_source: 'customer', gw_error_step: 'payment_authentication', error_reason: 'payment_cancelled', gw_error_code: 'BAD_REQUEST_ERROR' }],
  generic: ['Missing / generic example', 'UPI, GATEWAY_ERROR from the beneficiary bank and no error reason.', { method: 'upi', gw_error_code: 'GATEWAY_ERROR', gw_error_source: 'beneficiary_bank', gw_error_step: 'payment_debit_response', error_reason: null }],
  contradictory: ['Contradictory example', 'UPI payment with a card-only reason (card_expired).', { method: 'upi', gw_error_code: 'BAD_REQUEST_ERROR', gw_error_source: 'customer', gw_error_step: 'payment_authentication', error_reason: 'card_expired' }],
};
const CF = { sel: {}, seq: 0, complete: false, els: {}, scenario: 'clear' };            // sel: only fields the tester has set
const RV = { catalog: null, type: 'PAYMENT_FAILURE', poll: 0, pending: false, simple: {} };
const FIELD_NAME = { method: 'Payment method', instrument_hint: 'Instrument hint', gw_error_code: 'Gateway error code', gw_error_source: 'Error source',
  gw_error_step: 'Error step', error_reason: 'Error reason', true_cause: 'Evaluation label' };
const EVENT_TYPES_FALLBACK = [['PAYMENT_FAILURE', 'Payment failure'], ['SUBSCRIPTION_FAILURE', 'Subscription payment failure'], ['CHECKOUT_ABANDONMENT', 'Checkout abandonment'], ['OVERDUE_RECEIVABLE', 'Overdue receivable']];
const STATUS_BADGE = { PENDING: 'neutral', SIMULATED_RECOVERED: 'success', SIMULATED_FAILED: 'failed', HUMAN_INTERVENTION: 'escalated', WORKFLOW_COMPLETED: 'neutral', RUNNING: 'active', NOT_STARTED: 'neutral', NOT_REQUIRED: 'neutral' };
const STEP_BADGE = { PLANNED: 'neutral', EXECUTED: 'active', SUPPRESSED: 'warn', CANCELLED: 'neutral' };
const REQ_BADGE = { PENDING: 'warn', ACCEPTED: 'success', REJECTED: 'failed', EXPIRED: 'neutral' };
const ROUTE_TEXT = { CLEAR_AND_CONSISTENT: 'Clear and consistent: decided by a rule, Model C was not called', MISSING_OR_GENERIC: 'Missing or generic: the frozen Model C was asked once',
  CONTRADICTORY: 'Contradictory: sent to a human, Model C was not called' };
const METHOD_TEXT = { rule: 'Deterministic rule', model_c: 'Frozen Model C (called once)', none: 'No automatic diagnosis', reported_by_subscription_system: 'Reported by the subscription system', event_policy: 'Event policy (no failure diagnosis needed)' };
const REVENUE_REASON = {
  contradictory_gateway_information: 'The gateway fields contradict each other, so no automatic diagnosis or recovery was attempted.',
  no_safe_recovery_action: 'No customer-facing recovery action was permitted (consent, budgets or safety rules).',
  plan_exhausted_unresolved: 'Every planned action was used and the diagnosis was still unresolved.', plan_exhausted: 'Every planned action was used and no payment was completed.',
  risk_blocked: 'The failure was identified as a risk block; automatic recovery is not allowed.',
};
const revenueReason = (r) => (r && /^accepted_step_\d+$/.test(r) ? `The simulated customer completed the payment on step ${Number(r.slice(14)) + 1}.` : REVENUE_REASON[r] || r || '—');
const planDelay = (s) => (!s ? 'immediately' : s < 3600 ? `after ${Math.round(s / 60)} min` : s < 86400 ? `after ${Math.round(s / 3600)} h` : `after ${Math.round(s / 86400)} day(s)`);
const pretty = (s) => String(s || '').replace(/_/g, ' ');

const selectField = (id, label, options, value, onchange, required = true) => h('div', {}, h('label', { for: id }, label, required ? h('span', { class: 'req', 'aria-hidden': 'true' }, ' *') : null),
  h('select', { id, name: id, onchange }, options.map(([v, text]) => h('option', { value: v, selected: String(v) === String(value) }, text))));

function customForm() {
  CF.els = {};
  const scenarioField = ([field, id]) => {
    const select = h('select', { id, name: id, disabled: field !== 'method', onchange: (e) => onScenarioChange(field, e.target.value) });   // dependent fields start disabled until the method is chosen
    const note = h('div', { class: 'field-note', id: id + '-note' });
    const withheld = h('details', { class: 'field-note', id: id + '-why', hidden: true });
    CF.els[field] = { select, note, withheld, label: h('label', { for: id }) };
    return h('div', { class: field === 'true_cause' ? 'full' : '' }, CF.els[field].label, select, note, withheld);
  };
  const c = RV.catalog;
  if (!c) return loading('Loading the controlled options…');
  const times = c.failure_times.map((t) => [t.id, t.label]);
  const form = h('form', { class: 'stack', id: 'c-form', novalidate: true, onsubmit: submitCustom },
    h('p', { class: 'hint' }, 'Build one synthetic payment failure. The diagnosis scenario only decides which gateway inputs the dropdowns let you build; the system itself decides the route from those inputs (a rule, the frozen Model C once, or a human) and shows it below. Choose the payment method first and every other choice is limited to values that fit the scenario. * = required.'),
    h('div', { class: 'form' }, selectField('c-scenario', 'Diagnosis scenario', DIAGNOSIS_SCENARIOS, CF.scenario, (e) => onScenarioKind(e.target.value))),
    h('div', { class: 'row' }, Object.entries(EXAMPLES).map(([kind, [label]]) => h('button', { class: 'btn small', type: 'button', id: `c-example-${kind}`, onclick: () => fillExample(kind) }, label)),
      h('span', { class: 'hint' }, 'Each example fills the scenario selector and the gateway fields. The evaluation label is left for you to choose.')),
    h('div', { class: 'form' }, SCENARIO_FIELDS.map(scenarioField)),
    h('div', { id: 'c-cleared', class: 'stack', 'aria-live': 'polite' }),
    h('div', { id: 'c-preview', class: 'stack', 'aria-live': 'polite' }),
    h('div', { class: 'form' },
      selectField('c-customer', 'Customer', c.customers.map((x) => [x.id, x.label]), c.customers[0].id),
      selectField('c-issuer', 'Issuer (bank)', c.issuers.map((x) => [x, x]), c.issuers[0]),
      selectField('c-amount', 'Amount', c.amounts_rupees.map((x) => [x, '₹' + x.toLocaleString('en-IN')]), 499),
      selectField('c-time', 'Failure time', times, times[1][0])),
    h('p', { class: 'hint' }, 'Customer, issuer, amount and time come from predefined synthetic options; the event and payment ids are generated for you. The evaluation label is optional and is never used for diagnosis or planning.'),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', type: 'submit', id: 'c-submit', disabled: true }, 'Create event and start the workflow'), h('span', { class: 'hint', id: 'c-msg' })));
  refreshCustom(null);
  return form;
}
function fillExample(kind) {
  CF.scenario = kind; const s = $('#c-scenario'); if (s) s.value = kind;
  CF.sel = { ...EXAMPLES[kind][2] }; refreshCustom('method');
}
function onScenarioKind(kind) { CF.scenario = kind; refreshCustom(null); }       // the server keeps every selection that is still allowed and reports the ones it clears
function onScenarioChange(field, raw) {
  if (field === 'method' && raw === '') CF.sel = {};
  else if (raw === '') delete CF.sel[field];
  else CF.sel[field] = raw === NO_REASON ? null : raw;
  refreshCustom(field);
}
async function refreshCustom(changed) {
  const mine = ++CF.seq;
  try {
    const res = await api('/tests/custom-options', { method: 'POST', body: { ...CF.sel, changed, scope: SCOPE_OF[CF.scenario] } });
    if (mine !== CF.seq || !CF.els.method) return;                   // a newer change superseded this answer
    CF.sel = res.selected; CF.complete = res.complete;
    res.fields.forEach(paintScenarioField);
    const box = $('#c-cleared');
    if (box) {
      clear(box).append(...res.cleared.map((c) => notice(`${FIELD_NAME[c.field]} “${c.value === null ? 'no reason' : c.value}” was cleared: ${c.why} Please choose again.`, 'warn')),
        ...res.applied_defaults.map((d) => h('div', { class: 'hint' }, `${FIELD_NAME[d.field]} set to “${d.value}” (${d.why}).`)));
    }
    paintPreview(res.route_preview);
    syncSubmit();
  } catch (e) { const m = $('#c-msg'); if (m) m.textContent = `Could not load the field options: ${e.message}`; }
}
const ROUTE_OF_SCENARIO = { clear: 'CLEAR_AND_CONSISTENT', generic: 'MISSING_OR_GENERIC', contradictory: 'CONTRADICTORY' };
function paintPreview(p) {
  const box = $('#c-preview'); if (!box) return;
  clear(box);
  if (!p) return;
  const same = ROUTE_OF_SCENARIO[CF.scenario] === p.route;
  box.append(notice(`Router preview (no Model C call yet): ${ROUTE_TEXT[p.route]}.${same ? '' : ` This differs from the selected scenario; the router's verdict is what will run.`}`, same ? '' : 'warn'),
    ...(p.conflicts || []).map((c) => h('div', { class: 'hint' }, `${c.fields.map(pretty).join(' + ')}: ${c.message}`)));
}
function paintScenarioField(f) {
  const { select, note, withheld, label } = CF.els[f.field];
  clear(label).append(f.label, f.required ? h('span', { class: 'req', 'aria-hidden': 'true' }, ' *') : h('span', { class: 'muted' }, ' (optional)'));
  select.disabled = !f.enabled; select.required = f.required; select.setAttribute('aria-required', String(f.required));
  const chosen = f.field in CF.sel ? (CF.sel[f.field] === null ? NO_REASON : CF.sel[f.field]) : '';
  const alwaysSet = f.enabled && f.field === 'instrument_hint';       // the hint always has a value (its default), so no placeholder
  clear(select);
  if (!alwaysSet) select.append(h('option', { value: '' }, f.placeholder));
  f.options.forEach((o) => select.append(h('option', { value: o.value === null ? NO_REASON : o.value }, o.description ? `${o.label} — ${o.description}` : o.label)));
  select.value = chosen;
  note.textContent = f.note || '';
  clear(withheld); withheld.hidden = !f.withheld.length;
  if (f.withheld.length) withheld.append(h('summary', {}, `${f.withheld.length} not available — why?`),
    h('ul', {}, f.withheld.map((w) => h('li', {}, h('span', { class: 'mono' }, w.label), ': ', w.why))));
}
function syncSubmit() {
  const b = $('#c-submit'); if (b) b.disabled = RV.pending || !CF.complete;
  const m = $('#c-msg'); if (m && !RV.pending) m.textContent = CF.complete ? '' : 'Select every required (*) field to enable the button.';
}
async function submitCustom(ev) {
  ev.preventDefault();
  if (RV.pending || !CF.complete) return;
  const s = CF.sel, v = (id) => $('#' + id).value;
  const body = { event_type: 'PAYMENT_FAILURE', customer_id: v('c-customer'), issuer_id: v('c-issuer'), amount_rupees: Number(v('c-amount')), failure_time: v('c-time'),
                 diagnosis_scenario: CF.scenario, method: s.method, instrument_hint: s.instrument_hint, gw_error_code: s.gw_error_code, gw_error_source: s.gw_error_source, gw_error_step: s.gw_error_step, error_reason: s.error_reason };
  if (s.true_cause) body.evaluation_label = s.true_cause;
  await createEvent(body, '#c-submit', '#c-msg', () => { delete CF.sel.true_cause; const t = $('#c-truth'); if (t) t.value = ''; refreshCustom(null); });   // the hidden label is not kept after submission
}

/* ---- the other three event types: fixed selectors only */
function simpleEventForm(type) {
  const c = RV.catalog;
  if (!c) return loading('Loading the controlled options…');
  const st = RV.simple[type] = RV.simple[type] || {};
  const times = c.failure_times.map((t) => [t.id, t.label]);
  const pick = (key, fallback) => (st[key] != null ? st[key] : (st[key] = fallback));
  const rows = [selectField('s-customer', 'Customer', c.customers.map((x) => [x.id, x.label]), pick('customer_id', c.customers[0].id), (e) => { st.customer_id = e.target.value; })];
  if (type === 'CHECKOUT_ABANDONMENT') {
    rows.push(selectField('s-amount', 'Cart value', c.amounts_rupees.map((x) => [x, '₹' + x.toLocaleString('en-IN')]), pick('amount_rupees', 999), (e) => { st.amount_rupees = Number(e.target.value); }),
      selectField('s-stage', 'Where the customer left', c.checkout_stages.map((x) => [x.id, x.label]), pick('checkout_stage', 'payment_page'), (e) => { st.checkout_stage = e.target.value; }));
  } else if (type === 'SUBSCRIPTION_FAILURE') {
    const kinds = () => c.failure_kinds.filter((k) => k.id !== 'CARD_EXPIRED' || pick('method', 'card') === 'card');
    rows.push(selectField('s-plan', 'Subscription plan', c.subscription_plans.map((x) => [x.id, x.label]), pick('plan_id', c.subscription_plans[1].id), (e) => { st.plan_id = e.target.value; }),
      selectField('s-method', 'Payment method', c.methods.map((m) => [m, m]), pick('method', 'card'), (e) => { st.method = e.target.value; if (st.method !== 'card' && st.failure_kind === 'CARD_EXPIRED') st.failure_kind = null; renderPick(); }),
      selectField('s-kind', 'Why the renewal failed', kinds().map((x) => [x.id, x.label]), pick('failure_kind', 'BD_INSUFFICIENT_FUNDS'), (e) => { st.failure_kind = e.target.value; }));
  } else {
    rows.push(selectField('s-invoice', 'Overdue invoice', c.invoices.map((x) => [x.id, x.label]), pick('invoice_id', c.invoices[0].id), (e) => { st.invoice_id = e.target.value; }));
  }
  rows.push(selectField('s-time', 'Event time', times, pick('failure_time', times[1][0]), (e) => { st.failure_time = e.target.value; }));
  return h('form', { class: 'stack', id: 's-form', onsubmit: async (e) => { e.preventDefault(); await createEvent({ event_type: type, ...st }, '#s-submit', '#s-msg', () => {}); } },
    h('p', { class: 'hint' }, 'Every choice is a predefined synthetic option; the event id is generated for you. * = required.'), h('div', { class: 'form' }, rows),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', type: 'submit', id: 's-submit' }, 'Create event and start the workflow'), h('span', { class: 'hint', id: 's-msg' })));
}

async function createEvent(body, buttonSel, msgSel, after) {
  RV.pending = true;
  const b = $(buttonSel), m = $(msgSel), label = b && b.textContent;
  if (b) { b.disabled = true; b.textContent = 'Working…'; }
  if (m) m.textContent = '';
  try {
    const view = await api('/revenue/events', { method: 'POST', body });
    clear($('#i-preview')); showRevenueEvent(view);
    after();
  } catch (e) { if (m) m.textContent = e.message; toast(e.message); }
  finally { RV.pending = false; const bb = $(buttonSel); if (bb) { bb.textContent = label; bb.disabled = buttonSel === '#c-submit' ? !CF.complete : false; } }
}

/* ---- the live journey of one event */
function showRevenueEvent(view) {
  RV.poll++;
  const token = RV.poll;
  paintRevenueEvent(view);
  loadPending();
  const tick = async () => {
    if (token !== RV.poll) return;
    try {
      const v = await api(`/revenue/events/${encodeURIComponent(view.event_id)}`);
      if (token !== RV.poll) return;
      paintRevenueEvent(v); loadPending();
      if (['WORKFLOW_COMPLETED', 'NOT_REQUIRED'].includes(v.workflow_state) || v.status !== 'PENDING' && v.workflow_state !== 'RUNNING' && v.workflow_state !== 'NOT_STARTED') return;
    } catch (e) { /* keep polling */ }
    setTimeout(tick, 1500);
  };
  if (!['WORKFLOW_COMPLETED', 'NOT_REQUIRED'].includes(view.workflow_state)) setTimeout(tick, 1500);
}
function paintRevenueEvent(v) { clear($('#i-journey')).append(renderRevenueEvent(v, false)); }
function renderRevenueEvent(v, readOnly) {
  const d = v.diagnosis || {}, root = h('div', { class: 'journey' });
  root.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h2', {}, v.event_label), h('div', { class: 'small muted' }, h('span', { class: 'mono' }, v.event_id))),
      h('div', { class: 'row' }, badge(v.status, STATUS_BADGE[v.status]), badge(v.workflow_state, STATUS_BADGE[v.workflow_state] || 'neutral'), simTag())),
    notice('Everything on this page is simulated: no real message is sent and no real payment or revenue is involved.', 'warn'),
    v.workflow_warning ? notice(`The workflow could not be started (${v.workflow_warning}). The event is saved; start the worker and refresh.`, 'bad') : null,
    kv([['Customer', v.customer_id], ['Amount', money(v.amount_minor)], ...Object.entries(v.inputs || {}).filter(([k]) => !['customer_id', 'amount_rupees'].includes(k) && !k.startsWith('sim_')).map(([k, val]) => [pretty(k), val == null ? 'not provided' : String(val)])])));
  root.append(h('section', { class: 'card' }, h('header', {}, h('h3', {}, 'Diagnosis (made once)'), v.route ? badge(v.route, v.route === 'CONTRADICTORY' ? 'escalated' : 'neutral') : null),
    v.route ? h('p', { class: 'small' }, ROUTE_TEXT[v.route]) : null,
    kv([['Scenario selected', v.inputs && v.inputs.diagnosis_scenario ? pretty(v.inputs.diagnosis_scenario) : null], ['Route taken', v.route ? pretty(v.route) : null], ['Decided by', METHOD_TEXT[d.method] || d.method],
        ['Predicted cause', d.cause ? causeName(d.cause) : d.method === 'model_c' ? 'none (abstained)' : '—'], ['Rule', d.rule_id], ['Model C calls', d.method === 'model_c' || v.route ? String(d.model_c_calls ?? 0) : null],
        ['Safety gate', d.model_c ? `${pretty(d.model_c.decision)} (${pretty(d.model_c.route)})` : v.route === 'CLEAR_AND_CONSISTENT' ? 'not needed: a rule decided' : v.route ? 'not reached: no diagnosis made' : null]]),
    h('p', { class: 'small' }, d.explanation),
    (d.conflicts || []).length ? h('ul', { class: 'small' }, d.conflicts.map((c) => h('li', {}, h('strong', {}, c.fields.map(pretty).join(' + ')), ': ', c.message))) : null,
    d.model_c ? h('p', { class: 'small muted' }, `Model C decision: ${d.model_c.decision}. Top probabilities: ${d.model_c.top_probabilities.map((x) => `${causeName(x.cause)} ${(x.probability * 100).toFixed(1)}%`).join(', ')}`) : null,
    v.evaluation ? h('p', { class: 'small muted' }, `Your evaluation label (${causeName(v.evaluation.label)}): the diagnosis ${v.evaluation.diagnosis_matches_label ? 'matches' : 'does not match'} it.`) : null));
  root.append(h('section', { class: 'card' }, h('header', {}, h('h3', {}, 'Temporal execution')),
    kv([['Workflow', v.workflow_id || '—'], ['State', v.workflow_state], ['Plan version', v.plan.version || '—']]),
    v.route === 'CONTRADICTORY' ? h('p', { class: 'small muted' }, 'No workflow was started: a contradictory record goes to Human Intervention and nothing runs automatically.') : null));
  root.append(h('section', { class: 'card' }, h('header', {}, h('h3', {}, 'Recovery plan')), h('p', { class: 'small muted' }, v.plan.rationale),
    v.plan.steps.length ? h('ol', { class: 'steps' }, v.plan.steps.map((s) => h('li', {}, h('strong', {}, s.label), ' · ', planDelay(s.delay_s), s.channel ? ` · via ${s.channel}` : '', ' ',
        badge(s.status, STEP_BADGE[s.status]), s.guard ? h('span', { class: 'small muted' }, ` stopped by ${pretty(s.guard)}`) : null,
        s.payment_opportunity ? h('span', { class: 'small muted' }, ' · carries a payment opportunity') : null)))
      : h('p', { class: 'small' }, 'No recovery action is planned.')));
  const reqs = v.requests || [];
  root.append(h('section', { class: 'card' }, h('header', {}, h('h3', {}, 'Simulated payment requests')),
    reqs.length ? h('div', { class: 'stack' }, reqs.map((r) => h('div', { class: 'row' }, h('span', {}, `${r.title} · ${money(r.amount_minor)}`), badge(r.status, REQ_BADGE[r.status]),
        r.status === 'PENDING' && !readOnly ? h('a', { class: 'btn primary small', href: `/pay/${encodeURIComponent(r.request_id)}`, target: '_blank', rel: 'noopener' }, 'Open the simulated payment page') : null)))
      : h('p', { class: 'small muted' }, v.status === 'PENDING' ? 'Waiting for the workflow to create the first payment request…' : 'No payment request was created.'),
    !readOnly && reqs.some((r) => r.status === 'PENDING') ? h('p', { class: 'hint' }, 'On the payment page choose ACCEPT (the simulated customer completes the payment, the workflow stops) or REJECT (no payment, the workflow continues with the next permitted action).') : null));
  const done = v.status !== 'PENDING';
  root.append(h('section', { class: `card ${done ? '' : 'muted'}` }, h('header', {}, h('h3', {}, 'Result'), badge(v.status, STATUS_BADGE[v.status])),
    h('p', {}, done ? revenueReason(v.status_reason) : 'Not finished yet.'),
    v.status === 'SIMULATED_RECOVERED' ? h('p', {}, h('strong', {}, `Simulated recovered amount: ${money(v.recovered_amount_minor)}`), ' (not real revenue)') : null));
  root.append(h('details', { class: 'card' }, h('summary', {}, `Audit timeline (${v.timeline.length} entries)`),
    h('ol', { class: 'steps small' }, v.timeline.map((t) => h('li', {}, h('span', { class: 'mono' }, `${t.actor} · ${t.action}`), ' ', h('span', { class: 'muted' }, clock(t.at)))))));
  return root;
}
async function loadPending() {
  const box = $('#i-pending');
  if (!box) return;
  try {
    const d = await api('/revenue/requests?limit=8');
    clear(box).append(h('header', {}, h('h3', {}, 'Pending payment requests'), badge(String(d.pending.length), d.pending.length ? 'warn' : 'neutral')),
      d.pending.length ? h('div', { class: 'stack' }, d.pending.map((r) => h('div', { class: 'row small' }, h('span', { class: 'mono' }, r.event_id), h('span', {}, `${r.title} · ${money(r.amount_minor)}`),
          h('a', { href: `/pay/${encodeURIComponent(r.request_id)}`, target: '_blank', rel: 'noopener' }, 'open')))) : h('p', { class: 'small muted' }, 'Nothing is waiting for an answer.'));
  } catch (e) { /* optional */ }
}
/* ================================================================== Automated revenue pipeline (Dataset tab) and its Human Intervention view */
const RR = { dataset: 'routing_fixtures', limit: '', temporal: '0', runId: null, poll: 0, route: '', scenario: '', status: '', page: 0, open: new Set(), pageData: null, doc: null, csvId: null, eventsCsvId: null, behaviour: 'rate', pct: 80 };
const DEMO_SET = 'revenue_events_203';
const ENT = { scenario: '', offset: 0, limit: 12 };
const CASES_PER_PAGE = 20;
const entriesDataset = () => (RR.dataset === DEMO_SET ? DEMO_SET : RR.dataset === '__events_csv__' && RR.eventsCsvId ? `events:${RR.eventsCsvId}` : null);
function renderRevenueRunCard() {
  const box = clear($('#d-rv'));
  const sets = (RV.catalog && RV.catalog.datasets) || [];
  box.append(h('header', {}, h('h2', {}, 'Automated revenue pipeline'), badge('SIMULATED', 'sim')),
    h('p', { class: 'hint' }, 'Every case runs on its own, with no manual answers: the event is classified, a payment failure is routed (rule, Model C once, or a human), diagnosed once, planned and executed, and an independent simulator answers each payment opportunity. Recovery is counted only when a simulated payment succeeds; a correct diagnosis is not revenue.'),
    h('div', { class: 'form' },
      selectField('rr-dataset', 'Dataset', [...sets.map((d) => [d.name, `${d.name}${d.records ? ` (${d.records})` : ''}${d.available ? '' : ' (not built)'}`]), ['__csv__', 'Uploaded payment-failure CSV (validate a file below first)'], ['__events_csv__', 'Uploaded revenue-events CSV (validate a file below first)']], RR.dataset, (e) => onRunDataset(e.target.value)),
      selectField('rr-limit', 'Cases to run', [['', 'All'], ['50', 'First 50'], ['200', 'First 200'], ['500', 'First 500']], RR.limit, (e) => { RR.limit = e.target.value; }),
      selectField('rr-temporal', 'Also execute through Temporal (sample)', [['0', 'No (in-process only)'], ['5', 'First 5 cases'], ['10', 'First 10 cases'], ['20', 'First 20 cases'], ['50', 'Sample of 50 cases'], ['all', 'All eligible cases']], RR.temporal, (e) => { RR.temporal = e.target.value; })),
    h('div', { class: 'card', id: 'rr-rate-box' }, h('div', { class: 'row' },
      h('div', { style: 'min-width:240px' }, h('label', { for: 'rr-behaviour' }, 'Simulated customer responses'), h('select', { id: 'rr-behaviour', onchange: (e) => { RR.behaviour = e.target.value; const b = $('#rr-rate-slider'); if (b) b.hidden = RR.behaviour !== 'rate'; } },
        h('option', { value: 'rate', selected: RR.behaviour === 'rate' }, 'Configured acceptance rate'), h('option', { value: 'builtin', selected: RR.behaviour === 'builtin' }, 'Built-in cause-based simulator'))),
      h('div', { id: 'rr-rate-slider', style: 'flex:1;min-width:260px', hidden: RR.behaviour !== 'rate' }, h('label', { for: 'rr-pct' }, 'Simulated Customer Acceptance Rate (%) ', h('strong', { id: 'rr-pct-out' }, `${RR.pct}%`)),
        h('input', { id: 'rr-pct', type: 'range', min: 0, max: 100, step: 5, value: RR.pct, 'aria-valuemin': 0, 'aria-valuemax': 100, oninput: (e) => { RR.pct = Number(e.target.value); $('#rr-pct-out').textContent = `${RR.pct}%`; } }))),
      h('p', { class: 'small muted' }, 'The rate applies to the eligible cases (a payment opportunity exists after diagnosis, planning and safety checks), not to the whole dataset. It is deterministic: the same dataset and rate always give the same answers, in-process and in Temporal. An ACCEPT is only a simulated payment; safety rules still apply and a case is never marked recovered automatically.')),
    h('div', { class: 'stack' }, h('label', { for: 'rr-file' }, 'Or test your own CSV (payment failures, or revenue events with an event_type column)'), h('input', { id: 'rr-file', type: 'file', accept: '.csv,text/csv', onchange: onRevenueCsv }),
      h('span', { class: 'small' }, h('a', { href: '/tests/csv-template', download: 'payment_failures_template.csv' }, 'Payment-failure template'), ' · ', h('a', { href: '/revenue/events-csv/template', download: 'revenue_events_template.csv' }, 'Revenue-events template')), h('div', { id: 'rr-csv-out' })),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', type: 'button', id: 'rr-start', onclick: startRevenueRun }, 'Run the automated pipeline'), h('span', { class: 'hint', id: 'rr-msg' })),
    h('div', { id: 'rr-progress', style: 'margin-top:12px', 'aria-live': 'polite' }),
    h('div', { id: 'rr-entries', style: 'margin-top:12px' }),
    h('div', { id: 'rr-out', style: 'margin-top:12px' }));
  if (entriesDataset()) loadEntries();
  if (RR.doc) paintRevenueRun(RR.doc);
}
function onRunDataset(v) {
  RR.dataset = v; ENT.offset = 0; ENT.scenario = '';
  if (v === DEMO_SET && RR.temporal === '0') { RR.temporal = '50'; const s = $('#rr-temporal'); if (s) s.value = '50'; }       // the 203-entry dataset defaults to a 50-case Temporal sample (configurable)
  loadEntries();
}
async function loadEntries() {
  const host = $('#rr-entries');
  if (!host) return;
  const name = entriesDataset();
  if (!name) { clear(host); return; }
  try {
    const q = new URLSearchParams({ offset: ENT.offset, limit: ENT.limit });
    if (ENT.scenario) q.set('scenario', ENT.scenario);
    const d = await api(`/revenue/datasets/${encodeURIComponent(name)}/entries?${q}`);
    const scn = Object.entries(d.scenarios);
    clear(host).append(h('div', { class: 'card' }, h('header', {}, h('h3', {}, `Entries in ${name === DEMO_SET ? DEMO_SET : 'your revenue-events upload'} (${scn.reduce((a, [, n]) => a + n, 0)})`), badge('revenue-event format', 'neutral')),
      h('p', { class: 'small muted' }, 'These are revenue events, not payment-failure CSV rows. Each case is run on its own and an independent simulator answers its payment requests. Cases that reach Model C show its confidence; rule-based and plan-only cases have none. The Temporal sample is spread evenly over all five groups.'),
      h('div', { class: 'row' }, h('label', { for: 'rr-ent-scn' }, 'Show'), h('select', { id: 'rr-ent-scn', onchange: (e) => { ENT.scenario = e.target.value; ENT.offset = 0; loadEntries(); } },
        [['', 'All entries'], ...scn.map(([k, n]) => [k, `${pretty(k).toLowerCase()} (${n})`])].map(([v, tx]) => h('option', { value: v, selected: ENT.scenario === v }, tx)))),
      h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Entry', 'Scenario', 'Inputs', 'Expected route'].map((c) => h('th', {}, c)))),
        h('tbody', {}, d.items.map((r) => h('tr', {}, h('td', { class: 'mono' }, r.entry_id), h('td', {}, pretty(r.scenario).toLowerCase()),
          h('td', { class: 'small' }, Object.entries(r.inputs).filter(([k]) => k !== 'diagnosis_scenario').map(([k, v]) => `${pretty(k)}: ${v}`).join(' · ')), h('td', { class: 'small' }, r.expected_route ? pretty(r.expected_route).toLowerCase() : 'plan only (no diagnosis)')))))),
      pager(ENT, d.total, (o) => { ENT.offset = o; loadEntries(); })));
  } catch (e) { clear(host).append(notice(`Could not load the entries: ${e.message}`, 'bad')); }
}
async function startRevenueRun() {
  const b = $('#rr-start'), m = $('#rr-msg');
  b.disabled = true; m.textContent = 'Starting…';
  try {
    const which = RR.dataset === '__csv__' ? { csv_id: RR.csvId } : RR.dataset === '__events_csv__' ? { events_csv_id: RR.eventsCsvId } : { dataset: RR.dataset };
    const r = await api('/revenue/runs', { method: 'POST', body: { ...which, limit: RR.limit ? Number(RR.limit) : null, temporal_limit: RR.temporal === 'all' ? 0 : Number(RR.temporal), temporal_all: RR.temporal === 'all',
      acceptance_pct: RR.behaviour === 'rate' ? RR.pct : null } });
    RR.runId = r.run_id; RR.poll++; RR.page = 0; RR.open = new Set(); clear($('#rr-out'));
    const token = RR.poll;
    m.textContent = `Run ${r.run_id} started…`;
    const tick = async () => {
      if (token !== RR.poll) return;
      try {
        const d = await api(`/revenue/runs/${encodeURIComponent(r.run_id)}`);
        if (['completed', 'partial', 'failed', 'interrupted'].includes(d.status)) {
          b.disabled = false; m.textContent = d.status === 'completed' ? '' : `Run ${d.status}${d.error ? ': ' + d.error : ''}`;
          clear($('#rr-progress'));
          if (d.status === 'completed' || d.status === 'partial') { RR.doc = d; RR.route = ''; RR.scenario = ''; RR.status = ''; paintRevenueRun(d); }
          return;
        }
        paintProgress(d);
      } catch (e) { /* keep polling */ }
      setTimeout(tick, 1200);
    };
    setTimeout(tick, 800);
  } catch (e) { b.disabled = false; m.textContent = e.message; toast(e.message); }
}
const PHASE = { queued: 'Queued', evaluating: 'Evaluating cases', evaluated: 'Evaluated; preparing Temporal', temporal: 'Running Temporal workflows', done: 'Finishing' };
function paintProgress(d) {
  const box = $('#rr-progress');
  if (!box) return;
  const p = d.progress || {}, started = p.workflows_started || 0, done = p.workflows_completed || 0, want = p.workflows_requested || 0;
  const frac = p.phase === 'temporal' && want ? done / want : p.total ? (p.evaluated || 0) / p.total : 0;
  clear(box).append(h('div', { class: 'card' }, h('header', {}, h('h3', {}, 'Execution progress'), badge(PHASE[p.phase] || p.phase || 'Running', 'active'), simTag()),
    h('progress', { max: 100, value: Math.round(frac * 100), style: 'width:100%', 'aria-label': 'Run progress' }),
    h('div', { class: 'grid cols-4', style: 'margin-top:8px' }, stat(p.total || '—', 'Total cases to evaluate'), stat(p.evaluated || 0, 'Cases evaluated'), stat(p.eligible == null ? '—' : p.eligible, 'Eligible cases'),
      stat(want || '—', 'Workflows to run'), stat(started, 'Temporal workflows started'), stat(done, 'Temporal workflows completed'), stat(Math.max(0, started - done), 'Pending / running workflows'), stat(p.workflows_failed || 0, 'Failed workflow executions', p.workflows_failed ? 'bad' : '')),
    h('p', { class: 'small muted' }, 'The run is complete only when every started workflow has finished.')));
}
function paintRevenueRun(d) {
  const out = $('#rr-out');
  if (!out) return;
  const m = d.metrics, rd = m.route_distribution || {}, t = d.temporal, sim = m.simulation || {}, dm = m.diagnosis_methods || {}, rate = sim.acceptance_rate_selected_pct;
  clear(out).append(...[
    notice(d.label, 'warn'),
    d.status === 'partial' ? notice(`PARTIAL RUN: ${t && t.error ? t.error : 'some workflows did not finish'} Only workflows that actually finished are counted as executed.`, 'bad') : null,
    h('div', { class: 'grid cols-4' }, stat(m.total_cases, 'Total cases'), stat(rd.CLEAR_AND_CONSISTENT || 0, 'Clear and consistent (rule)'), stat(rd.MISSING_OR_GENERIC || 0, 'Missing or generic (Model C)'),
      stat(rd.CONTRADICTORY || 0, 'Contradictory (human)'), stat(`${m.model_c_calls} (max ${m.max_model_c_calls_per_case} per case)`, 'Model C calls'),
      stat(m.human_interventions, 'HUMAN_INTERVENTION', 'esc'), stat(m.safety_violations, 'Safety violations', m.safety_violations ? 'bad' : ''),
      stat(m.SIMULATED_RECOVERED, 'SIMULATED_RECOVERED', 'good'), stat(m.SIMULATED_FAILED, 'SIMULATED_FAILED', 'bad'),
      stat(`${money(m.simulated_recovered_amount_minor)} of ${money(m.total_amount_minor)}`, 'Simulated recovered amount'), stat(`${m.simulated_recovery_percentage}%`, 'Simulated recovery rate')),
    h('h3', { style: 'margin:14px 0 6px' }, 'How the simulated customer responded'),
    h('div', { class: 'grid cols-4', id: 'rr-sim' }, stat(rate == null ? 'Built-in simulator' : `${rate}%`, 'Acceptance rate selected'), stat(sim.eligible_cases, 'Eligible recovery cases'),
      stat(sim.recovery_opportunities, 'Recovery opportunities (requests)'), stat(sim.simulated_accepts, 'Simulated ACCEPT answers'), stat(sim.simulated_rejects, 'Simulated REJECT answers'),
      stat(sim.successful_recoveries, 'Successful recoveries'), stat(t ? `${t.started} / ${t.completed}` : 'not run', 'Temporal workflows started / completed'),
      stat(`${dm.rule || 0} / ${dm.model_c || 0}`, 'Rule-based / Model C diagnoses')),
    h('p', { class: 'small muted' }, `These are different numbers: ${sim.eligible_cases} cases were eligible (a payment opportunity exists); ${sim.recovery_opportunities} payment requests were actually made`
      + (sim.cases_assigned_to_accept != null ? `; ${sim.cases_assigned_to_accept} customers were assigned to pay at some opportunity, and ${sim.cases_with_an_accept} actually did (an assigned step may never happen: the payment state is verified first, or a safety rule suppresses it)` : '')
      + `; ${sim.successful_recoveries} cases were recovered (${sim.recoveries_by_accept} by a simulated ACCEPT, ${sim.recoveries_by_verified_original_payment} because verification found the original payment had succeeded). Contradictory cases (${sim.contradictory_cases}) and directly escalated cases (${sim.directly_escalated_cases}) have no payment opportunity and are never sent to Temporal in the "All eligible cases" mode.`),
    h('p', { class: 'small muted' }, m.diagnosis_match.note + ` Diagnosis matched the hidden cause in ${m.diagnosis_match.matched} of ${m.diagnosis_match.judged} judged cases.`),
    m.by_scenario ? h('div', { class: 'table-wrap', id: 'rr-by-scenario' }, h('table', {}, h('thead', {}, h('tr', {}, ['Scenario', 'Cases', 'Workflows expected', 'SIMULATED_RECOVERED', 'SIMULATED_FAILED', 'HUMAN_INTERVENTION', 'Model C calls', 'Model C confidence above 0.6'].map((c, i) => h('th', { class: i ? 'num' : '' }, c)))),
      h('tbody', {}, Object.entries(m.by_scenario).map(([k, v]) => h('tr', {}, h('td', {}, pretty(k).toLowerCase()), h('td', { class: 'num' }, v.cases), h('td', { class: 'num' }, v.workflows_expected), h('td', { class: 'num' }, v.SIMULATED_RECOVERED),
        h('td', { class: 'num' }, v.SIMULATED_FAILED), h('td', { class: 'num' }, v.HUMAN_INTERVENTION), h('td', { class: 'num' }, v.model_c_calls),
        h('td', { class: 'num' }, v.model_c_confidence ? `${v.model_c_confidence['above_0.6']} of ${v.model_c_confidence.cases}` : '—')))))) : null,
    m.model_c_confidence ? h('p', { class: 'small', id: 'rr-confidence' }, `Model C confidence (only the ${m.model_c_confidence.cases} cases that used Model C; rule-based and plan-only cases have none): ${m.model_c_confidence['above_0.6']} above 0.6 (${Math.round(m.model_c_confidence['share_above_0.6'] * 100)}%), median ${m.model_c_confidence.median}, range ${m.model_c_confidence.min} to ${m.model_c_confidence.max}. `
      + Object.entries(m.model_c_confidence.histogram).map(([k, v]) => `${k}: ${v}`).join(' · ')) : null,
    h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Route / group', 'Cases', 'SIMULATED_RECOVERED', 'SIMULATED_FAILED', 'HUMAN_INTERVENTION', 'Model C calls'].map((c, i) => h('th', { class: i ? 'num' : '' }, c)))),
      h('tbody', {}, Object.entries(m.by_route).map(([k, v]) => h('tr', {}, h('td', {}, pretty(k)), h('td', { class: 'num' }, v.cases), h('td', { class: 'num' }, v.SIMULATED_RECOVERED),
        h('td', { class: 'num' }, v.SIMULATED_FAILED), h('td', { class: 'num' }, v.HUMAN_INTERVENTION), h('td', { class: 'num' }, v.model_c_calls)))))),
    h('p', { class: 'small' }, 'Event types: ' + Object.entries(m.by_event_type).map(([k, v]) => `${pretty(k).toLowerCase()} ${v}`).join(' · ')
      + (Object.keys(m.human_intervention_reasons).length ? ' · Human interventions: ' + Object.entries(m.human_intervention_reasons).map(([k, v]) => `${pretty(k)} ${v}`).join(', ') : '')),
    t ? h('div', { class: t.error ? 'notice bad' : 'notice good', role: 'status', id: 'rr-temporal-note' }, h('strong', {}, 'Temporal execution: '),
      `${t.completed} of ${t.requested} ${t.mode === 'all_eligible' ? 'eligible' : 'sampled'} workflows completed on queue ${t.task_queue}${t.started != null ? ` (${t.started} started, ${t.timed_out || 0} still running or timed out, ${t.failed_to_start || 0} failed to start)` : ''}; ${t.matched_in_process} of ${t.compared != null ? t.compared : t.requested} compared cases matched the in-process result exactly.`, t.error ? ` ${t.error}` : '',
      t.responses ? ` Responses recorded by the workflows: ${t.responses.ACCEPTED} ACCEPT, ${t.responses.REJECTED} REJECT${t.responses.EXPIRED ? `, ${t.responses.EXPIRED} expired` : ''}.` : '',
      t.mismatches.length ? ` Differences: ${t.mismatches.map((x) => x.case).join(', ')}.` : '') : null,
    h('h3', { style: 'margin:16px 0 6px' }, 'Individual Payment Evaluation Results'),
    h('div', { class: 'row' }, m.by_scenario ? [h('label', { for: 'rr-scn' }, 'Scenario'), h('select', { id: 'rr-scn', onchange: (e) => { RR.scenario = e.target.value; RR.page = 0; loadRevenueCases(); } },
      [['', 'All scenarios'], ...Object.keys(m.by_scenario).map((k) => [k, pretty(k).toLowerCase()])].map(([v, tx]) => h('option', { value: v, selected: RR.scenario === v }, tx)))] : null,
      h('label', { for: 'rr-route' }, 'Route'), h('select', { id: 'rr-route', onchange: (e) => { RR.route = e.target.value; RR.page = 0; loadRevenueCases(); } },
        [['', 'All routes'], ['CLEAR_AND_CONSISTENT', 'Clear and consistent'], ['MISSING_OR_GENERIC', 'Missing or generic'], ['CONTRADICTORY', 'Contradictory']].map(([v, tx]) => h('option', { value: v, selected: RR.route === v }, tx))),
      h('label', { for: 'rr-status' }, 'Outcome'), h('select', { id: 'rr-status', onchange: (e) => { RR.status = e.target.value; RR.page = 0; loadRevenueCases(); } },
        [['', 'All outcomes'], ['SIMULATED_RECOVERED', 'SIMULATED_RECOVERED'], ['SIMULATED_FAILED', 'SIMULATED_FAILED'], ['HUMAN_INTERVENTION', 'HUMAN_INTERVENTION']].map(([v, tx]) => h('option', { value: v, selected: RR.status === v }, tx)))),
    h('div', { id: 'rr-cases' }, loading('Loading cases…'))].filter(Boolean));            // absent parts are dropped (a bare null would be printed as the word null)
  loadRevenueCases();
}
function temporalText(r) {
  const t = RR.doc && RR.doc.temporal, tr = r.temporal;
  if (tr && tr.started) return tr.finished ? `Completed · ${pretty(tr.status).toLowerCase()}` : 'Running / timed out';
  if (tr && tr.error) return 'Failed to start';
  if (r.route === 'CONTRADICTORY') return 'Not started: contradictory';
  if (!r.eligible) return 'Not started: directly escalated';
  return t ? (t.mode === 'all_eligible' ? 'Not run' : 'Not in the Temporal sample') : 'In-process only';
}
async function loadRevenueCases() {
  const host = $('#rr-cases');
  if (!host || !RR.runId) return;
  try {
    const q = new URLSearchParams({ limit: CASES_PER_PAGE, offset: RR.page * CASES_PER_PAGE });
    if (RR.route) q.set('route', RR.route);
    if (RR.scenario) q.set('scenario', RR.scenario);
    if (RR.status) q.set('status', RR.status);
    RR.pageData = await api(`/revenue/runs/${encodeURIComponent(RR.runId)}/cases?${q}`);
    paintCases();
  } catch (e) { clear(host).append(notice(`Could not load cases: ${e.message}`, 'bad')); }
}
function paintCases() {
  const host = $('#rr-cases'), d = RR.pageData;
  if (!host || !d) return;
  const pages = Math.max(1, Math.ceil(d.total / CASES_PER_PAGE));
  const toggle = (key) => { if (RR.open.has(key)) RR.open.delete(key); else RR.open.add(key); paintCases(); };
  const rows = d.items.flatMap((r) => {
    const open = RR.open.has(r.case);
    const main = h('tr', { class: `clickable case-row ${open ? 'selected' : ''}`, tabindex: 0, role: 'button', 'aria-expanded': String(open), onclick: () => toggle(r.case), onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(r.case); } } },
      h('td', { class: 'mono' }, `${open ? '▾' : '▸'} ${r.case.replace(/^routing_fixtures\|/, '')}`), h('td', {}, (r.scenario ? pretty(r.scenario) : pretty(r.event_type)).toLowerCase()),
      h('td', {}, r.route ? pretty(r.route).toLowerCase() : 'no diagnosis needed'), h('td', {}, r.cause ? causeName(r.cause) : '—'), h('td', {}, badge(r.status, STATUS_BADGE[r.status])), h('td', { class: 'small' }, temporalText(r)));
    return open ? [main, h('tr', { class: 'case-detail' }, h('td', { colspan: 6 }, renderCaseDetail(r)))] : [main];
  });
  clear(host).append(d.items.length ? h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Payment / event ID', 'Scenario', 'Diagnosis route', 'Predicted cause', 'Final outcome', 'Temporal execution'].map((c) => h('th', {}, c)))), h('tbody', {}, rows)))
    : empty('No cases for this filter.'),
    h('div', { class: 'pager' }, h('span', { class: 'small muted', id: 'rr-page-label' }, `Page ${RR.page + 1} of ${pages} · ${d.total} payment${d.total === 1 ? '' : 's'}, ${CASES_PER_PAGE} per page`),
      h('span', { class: 'row' }, h('button', { class: 'btn small', type: 'button', id: 'rr-prev', disabled: RR.page <= 0, onclick: () => { RR.page -= 1; loadRevenueCases(); } }, '‹ Previous'),
        h('button', { class: 'btn small', type: 'button', id: 'rr-next', disabled: RR.page + 1 >= pages, onclick: () => { RR.page += 1; loadRevenueCases(); } }, 'Next ›'))));
}
function renderCaseDetail(r) {
  const d = r.diagnosis || {}, pl = r.plan || {}, tr = r.temporal, mc = d.model_c, section = (title, ...kids) => h('section', { class: 'card' }, h('header', {}, h('h4', { style: 'margin:0' }, title)), ...kids);
  const skip = new Set(['customer_id', 'amount_rupees', 'dataset', 'source_payment_id']);
  const inputs = Object.entries(r.inputs || {}).filter(([k]) => !skip.has(k)).map(([k, v]) => [pretty(k), v == null ? 'not provided' : String(v)]);
  const consent = Object.entries(r.consent || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none';
  const log = {}; (r.steps || []).forEach((s) => { log[s.idx] = s; });
  const suppressed = (r.steps || []).filter((s) => s.status === 'SUPPRESSED');
  const ran = (tr && tr.steps) || [];
  const responses = (tr && tr.requests) || [];
  return h('div', { class: 'stack case-detail-body' },
    section('A. Original input', kv([['Event / payment', r.case], ['Event type', pretty(r.event_type)], ['Customer', r.customer_id], ['Amount', money(r.amount_minor)], ['Contact consent', consent], ...inputs])),
    section('B. Diagnosis', kv([['Route', d.route ? pretty(d.route) : 'none: no failure diagnosis is needed for this event'], ['Decided by', METHOD_TEXT[d.method] || d.method], ['Predicted cause', d.cause ? causeName(d.cause) : (d.method === 'model_c' ? 'none (Model C abstained)' : null)],
        ['Rule', d.rule_id], ['Model C calls', d.route ? String(d.model_c_calls ?? 0) : null], ['Model C confidence', mc ? mc.confidence.toFixed(3) : null], ['Safety gate', mc ? `${pretty(mc.decision)} (${pretty(mc.route)})` : null],
        ['Top probabilities', mc ? mc.top_probabilities.map((x) => `${causeName(x.cause)} ${(x.probability * 100).toFixed(1)}%`).join(', ') : null]]),
      d.explanation ? h('p', { class: 'small' }, d.explanation) : null,
      (d.conflicts || []).length ? h('ul', { class: 'small' }, d.conflicts.map((c) => h('li', {}, h('strong', {}, c.fields.map(pretty).join(' + ')), ': ', c.message))) : null),
    section('C. Recovery planning', pl.version ? kv([['Plan version', pl.version], ['Planned for cause', pl.cause ? causeName(pl.cause) : null], ['Eligible for automated recovery', r.eligible ? 'yes (a payment opportunity exists)' : 'no']]) : null,
      pl.rationale ? h('p', { class: 'small' }, pl.rationale) : h('p', { class: 'small' }, 'No recovery action is planned.'),
      (pl.steps || []).length ? h('ol', { class: 'steps small' }, pl.steps.map((s) => h('li', {}, h('strong', {}, s.label), ` · ${planDelay(s.delay_s)}`, s.channel ? ` · via ${s.channel}` : '', s.payment_opportunity ? ' · carries a payment opportunity' : '', s.system_retry ? ' · automatic retry' : '',
        log[s.idx] && log[s.idx].status === 'SUPPRESSED' ? h('span', { class: 'badge warn' }, ` blocked by ${pretty(log[s.idx].guard)}`) : null))) : null,
      suppressed.length ? notice(`Safety rules blocked ${suppressed.length} planned action(s): ${suppressed.map((s) => `${pretty(s.action)} (${pretty(s.guard)})`).join(', ')}.`, 'warn') : null),
    section('D. Temporal execution', tr && tr.started ? kv([['Workflow started', 'yes'], ['Workflow ID', tr.workflow_id], ['Workflow state', pretty(tr.workflow_state)], ['Final result in Temporal', `${tr.status}${tr.status === r.status ? ' (same as the in-process result)' : ` (in-process: ${r.status})`}`]])
        : kv([['Workflow started', 'no'], ['Why', temporalText(r)]]),
      ran.length ? h('ol', { class: 'steps small' }, ran.map((s) => h('li', {}, h('strong', {}, s.label), s.channel ? ` · via ${s.channel}` : '', ' ', badge(s.status, STEP_BADGE[s.status]), s.guard ? ` stopped by ${pretty(s.guard)}` : ''))) : null,
      responses.length ? h('p', { class: 'small' }, 'Simulated customer responses in the workflow: ', responses.map((x) => `${x.title}: ${x.status === 'ACCEPTED' ? 'ACCEPT' : x.status === 'REJECTED' ? 'REJECT' : x.status}`).join(' → ')) : null,
      !tr && (r.steps || []).some((s) => s.decision) ? h('p', { class: 'small' }, 'Simulated responses (in-process run): ', r.steps.filter((s) => s.decision).map((s) => `${pretty(s.action)}: ${s.decision === 'ACCEPTED' ? 'ACCEPT' : 'REJECT'}`).join(' → ')) : null,
      tr && tr.error ? notice(tr.error, 'bad') : null),
    section('E. Final outcome', h('div', { class: 'row' }, badge(r.status, STATUS_BADGE[r.status]), simTag()),
      kv([['Result', revenueReason(r.reason)], ['Simulated recovered amount', r.status === 'SIMULATED_RECOVERED' ? `${money(r.recovered_amount_minor)} (not real revenue)` : null],
        ['Escalation reason', r.status === 'HUMAN_INTERVENTION' ? pretty(r.reason) : null], ['Safety violations', (r.violations || []).length ? r.violations.map(pretty).join(', ') : 'none']])));
}
function renderRevenueCase(d) {
  const root = h('div', { class: 'journey' });
  root.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h2', {}, 'Intervention case'), h('div', { class: 'small muted' }, h('span', { class: 'mono' }, d.id))),
      h('div', { class: 'row' }, terminalBadge(d.kind), simTag(), badge('Revenue pipeline', 'active'))),
    notice('Revenue pipeline: a diagnosis or safety rule sent this event to a person. Nothing was executed automatically for contradictory gateway data.', ''),
    h('p', { style: 'margin-top:8px' }, h('strong', {}, 'Final reason: '), d.reason_text),
    h('p', { class: 'small muted' }, 'Read-only: no actions can be taken from this screen.')));
  root.append(renderRevenueEvent(d.event, true));
  return root;
}
function stat(value, label, cls = '') { return h('div', { class: `stat ${cls}` }, h('div', { class: 'v' }, value), h('div', { class: 'l' }, label)); }
/* ================================================================== TAB 2: dataset testing (automated; the card above is the whole tab) */
function initDataset() {
  const root = $('#panel-dataset');
  root.append(h('div', { class: 'card', id: 'd-rv' }, loading('Loading…')));
  (RV.catalog ? Promise.resolve(RV.catalog) : api('/revenue/catalog').then((c) => { RV.catalog = c; return c; }))
    .then(renderRevenueRunCard)
    .catch((e) => clear($('#d-rv')).append(notice(`Could not load the pipeline options: ${e.message}`, 'bad')));
}
async function onRevenueCsv(ev) {
  const out = $('#rr-csv-out'), file = ev.target.files && ev.target.files[0];
  if (!file) return;
  clear(out).append(loading('Validating…'));
  try {
    const text = await file.text();
    const first = text.split('\n', 1)[0] || '';
    const events = first.split(',').map((c) => c.trim().replace('\r', '')).includes('event_type') && !first.includes('payment_id');       // the header says which kind of file this is
    const v = await api(events ? '/revenue/events-csv/validate' : '/tests/csv/validate', { method: 'POST', body: text, rawType: 'text/csv' });
    if (!v.valid) {
      if (events) RR.eventsCsvId = null; else RR.csvId = null;
      clear(out).append(notice(`The file cannot be used (${v.errors.length} problem${v.errors.length === 1 ? '' : 's'}).`, 'bad'),
        h('ul', { class: 'small' }, v.errors.slice(0, 8).map((e) => h('li', {}, `${e.row ? 'row ' + e.row + ': ' : ''}${e.column ? e.column + ': ' : ''}${e.message}`))));
      return;
    }
    const sel = $('#rr-dataset');
    if (events) {
      RR.eventsCsvId = v.events_csv_id; RR.dataset = '__events_csv__'; ENT.offset = 0; ENT.scenario = '';
      if (sel) sel.value = '__events_csv__';
      const by = Object.entries(v.summary.by_scenario).map(([k, n]) => `${pretty(k).toLowerCase()} ${n}`).join(', ');
      clear(out).append(notice(`Valid revenue-events file: ${v.summary.rows} event(s) (${by}). ${v.summary.has_hidden_labels ? 'Hidden labels were found: the simulator will use them.' : 'No hidden labels.'} Pick the settings above and press Run.`, 'good'),
        ...v.warnings.map((w) => h('div', { class: 'small muted' }, w)));
      loadEntries();
      return;
    }
    RR.csvId = v.csv_id; RR.dataset = '__csv__';
    if (sel) sel.value = '__csv__';
    clear(out).append(notice(`Valid: ${v.summary.failed_rows} failed payment(s) to test, ${v.summary.history_rows} history row(s). ${v.summary.has_ground_truth ? 'Hidden labels were found: the simulator will use them.' : 'No hidden labels: outcomes are simulated from the payment details only.'}`, 'good'));
    loadEntries();
  } catch (e) { clear(out).append(notice(`Could not validate the file: ${e.message}`, 'bad')); }
}

/* ================================================================== TAB 3: human intervention (read-only) */
const S3 = { kind: '', dataset: '', data: null, page: { offset: 0, limit: 25 }, selected: null };
const STATUS = { SAFETY_ESCALATION: 'Escalated: mandatory review', DIAGNOSIS_INTERVENTION: 'Unresolved: needs human review', PLAN_INTERVENTION: 'Unresolved: needs human review' };
function initIntervention() {
  const root = $('#panel-intervention');
  const datasets = ['routing_fixtures', 'event_type_mix', 'normal_recoverable', 'risk_and_hard_declines', 'mixed_difficult', 'shifted_regime', 'CHECKOUT_ABANDONMENT', 'SUBSCRIPTION_FAILURE', 'OVERDUE_RECEIVABLE', 'PAYMENT_FAILURE'];
  root.append(h('div', { class: 'stack' },
    notice('Read-only review of events a person must look at: contradictory gateway data, risk blocks, cases with no safe recovery action, and unresolved diagnoses. There are no approve, reject, retry or reassign actions here by design.', ''),
    h('div', { class: 'card' }, h('div', { class: 'row' },
      h('div', { style: 'min-width:190px' }, h('label', { for: 'f-kind' }, 'Category'), h('select', { id: 'f-kind', onchange: (e) => { S3.kind = e.target.value; S3.page.offset = 0; loadInterventions(); } },
        h('option', { value: '' }, 'All categories'), ['SAFETY_ESCALATION', 'DIAGNOSIS_INTERVENTION', 'PLAN_INTERVENTION'].map((k) => h('option', { value: k }, TERMINAL[k][0])))),
      h('div', { style: 'min-width:190px' }, h('label', { for: 'f-ds' }, 'Dataset or event type'), h('select', { id: 'f-ds', onchange: (e) => { S3.dataset = e.target.value; S3.page.offset = 0; loadInterventions(); } },
        h('option', { value: '' }, 'All'), datasets.map((k) => h('option', { value: k }, pretty(k))))),
      h('span', { class: 'spacer' }), h('button', { class: 'btn', type: 'button', id: 'f-refresh', onclick: () => loadInterventions() }, 'Refresh')),
      h('div', { id: 'iv-counts', style: 'margin-top:10px' })),
    h('div', { class: 'card', id: 'iv-list' }, loading('Loading cases…')),
    h('div', { id: 'iv-detail' }),
    h('details', { class: 'card', id: 'iv-hist', ontoggle: (e) => { if (e.target.open) loadHistorical(); } }, h('summary', {}, 'Earlier research records (read-only)'), h('div', { id: 'iv-hist-body' }, loading('Loading…')))));
  loadInterventions();
}
async function loadHistorical() {
  const host = clear($('#iv-hist-body'));
  try {
    const d = await api('/interventions/historical?limit=100');
    host.append(h('p', { class: 'small muted' }, `${d.note} ${d.total} record${d.total === 1 ? '' : 's'}.`));
    if (d.items.length) host.append(h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Payment', 'Category', 'Reason', 'Recorded'].map((c) => h('th', {}, c)))),
      h('tbody', {}, d.items.map((i) => h('tr', {}, h('td', { class: 'mono' }, i.payment_id), h('td', {}, pretty(i.kind)), h('td', { class: 'small' }, pretty(i.reason)), h('td', { class: 'small' }, new Date(i.created_at).toLocaleString())))))));
  } catch (e) { host.append(notice(`Could not load the earlier records: ${e.message}`, 'bad')); }
}
let ivRequest = 0;                                                           // only the newest request may update the list (responses can arrive out of order)
async function loadInterventions() {
  const host = $('#iv-list');
  if (!host) return;
  if (!S3.data) clear(host).append(loading('Loading cases…'));
  const mine = ++ivRequest;
  try {
    const q = new URLSearchParams({ offset: S3.page.offset, limit: S3.page.limit });
    if (S3.kind) q.set('kind', S3.kind);
    if (S3.dataset) q.set('dataset', S3.dataset);
    const data = await api(`/interventions?${q}`);
    if (mine !== ivRequest) return;
    S3.data = data;
    renderInterventionList();
  } catch (e) { clear(host).append(notice(`Could not load the intervention queue: ${e.message}`, 'bad')); }
}
function renderInterventionList() {
  const d = S3.data, host = clear($('#iv-list'));
  clear($('#iv-counts')).append(h('div', { class: 'row' }, ['SAFETY_ESCALATION', 'DIAGNOSIS_INTERVENTION', 'PLAN_INTERVENTION'].map((k) => [terminalBadge(k), h('strong', {}, d.counts_by_kind[k] || 0), ' ']),
    h('span', { class: 'spacer' }), h('span', { class: 'small muted' }, `${d.total} case${d.total === 1 ? '' : 's'} match the filters`)));
  if (!d.items.length) { host.append(empty('No intervention cases for these filters. Contradictory gateway data, risk blocks and cases with no safe action appear here.')); return; }
  const rows = d.items.map((i) => h('tr', { class: `clickable ${S3.selected && S3.selected.id === i.id ? 'selected' : ''}`, tabindex: 0, onclick: () => openCase(i),
      onkeydown: (e) => { if (e.key === 'Enter') openCase(i); } },
    h('td', { class: 'mono' }, i.id), h('td', {}, pretty(i.event_type).toLowerCase()), h('td', {}, terminalBadge(i.kind)), h('td', { class: 'num' }, i.executed_steps),
    h('td', { class: 'small' }, i.cause ? causeName(i.cause) : '—'), h('td', { class: 'small' }, i.reason_text), h('td', { class: 'small' }, STATUS[i.kind])));
  host.append(h('div', { class: 'table-wrap' }, h('table', {}, h('thead', {}, h('tr', {}, ['Event', 'Type', 'Category', 'Steps run', 'Diagnosed cause', 'Final reason', 'Status'].map((c, k) => h('th', { class: k === 3 ? 'num' : '' }, c)))), h('tbody', {}, rows))),
    pager(S3.page, d.total, (o) => { S3.page.offset = o; loadInterventions(); }));
}
async function openCase(i) {
  S3.selected = i; renderInterventionList();
  const host = clear($('#iv-detail'));
  host.append(h('div', { class: 'card' }, loading('Loading case…')));
  try {
    const d = await api(`/interventions/revenue/${encodeURIComponent(i.id)}`);
    clear(host).append(renderRevenueCase(d));
    host.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) { clear(host).append(notice(`Could not load this case: ${e.message}`, 'bad')); }
}

/* ------------------------------------------------------------------ boot */
function boot() {
  const tabs = $('#tablist');
  tabs.addEventListener('click', (e) => { const b = e.target.closest('[data-tab]'); if (b) showTab(b.dataset.tab); });
  tabs.addEventListener('keydown', (e) => {
    const i = TABS.findIndex((t) => $(`#tab-${t}`).getAttribute('aria-selected') === 'true');
    if (e.key === 'ArrowRight') { e.preventDefault(); showTab(TABS[(i + 1) % 3], true); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); showTab(TABS[(i + 2) % 3], true); }
  });
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1)));
  api('/health').then((d) => { $('#chip-api').textContent = d.ok ? 'API: online' : 'API: degraded'; }).catch(() => { $('#chip-api').textContent = 'API: unreachable'; });
  showTab(location.hash.slice(1) || 'individual');
}
document.addEventListener('DOMContentLoaded', boot);
window.__ui = { api };       // exposed for the automated browser test only
})();
