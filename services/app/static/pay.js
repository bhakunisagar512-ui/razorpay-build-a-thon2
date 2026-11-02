/* Simulated payment-request page. A mock customer view, not a payment gateway. It shows one system-generated request and lets the tester
   choose ACCEPT (the simulated customer completes the payment) or REJECT (the simulated customer does not). No input fields, no free text. */
(function () {
  'use strict';
  const requestId = decodeURIComponent(location.pathname.split('/').filter(Boolean).pop() || '');
  const root = document.getElementById('pay-root');
  const money = (minor) => '₹' + (Number(minor) / 100).toLocaleString('en-IN', { maximumFractionDigits: 2 });
  const when = (iso) => { const d = new Date(iso); return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('en-IN'); };
  const el = (tag, props, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) { if (v == null || v === false) continue; if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v); }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  };
  const BADGE = { PENDING: 'warn', ACCEPTED: 'success', REJECTED: 'failed', EXPIRED: 'neutral' };
  const OUTCOME = {
    ACCEPTED: 'Simulated payment completed. The recovery workflow has been notified and stops: no further recovery actions will be taken.',
    REJECTED: 'No payment was made. The recovery workflow continues with its remaining permitted actions, if any. The diagnosis is not repeated.',
    EXPIRED: 'This request is closed (it timed out or the recovery already ended). A late answer is not accepted.',
  };
  let busy = false, message = null;

  async function load() {
    const res = await fetch(`/revenue/requests/${encodeURIComponent(requestId)}`);
    if (res.status === 404) { root.replaceChildren(el('div', { class: 'notice bad', role: 'alert' }, 'This payment request does not exist.')); return null; }
    return res.json();
  }
  async function respond(decision) {
    if (busy) return;
    busy = true; paint(current);
    try {
      const res = await fetch(`/revenue/requests/${encodeURIComponent(requestId)}/respond`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ decision }) });
      const body = await res.json().catch(() => ({}));
      message = res.ok ? { kind: 'good', text: body.duplicate ? 'This answer was already recorded; nothing changed.' : OUTCOME[body.status] } : { kind: 'bad', text: typeof body.detail === 'string' ? body.detail : 'The answer was not accepted.' };
    } catch (e) { message = { kind: 'bad', text: 'Could not reach the API. Is the stack running?' }; }
    busy = false;
    await refresh();
  }
  let current = null;
  function paint(r) {
    current = r;
    const open = r.status === 'PENDING';
    root.replaceChildren(
      el('div', { class: 'card' },
        el('header', {}, el('h1', { style: 'font-size:1.25rem;margin:0' }, 'Payment request'), el('span', { class: `badge ${BADGE[r.status] || 'neutral'}` }, r.status)),
        el('p', { style: 'font-size:1.05rem' }, r.message),
        el('dl', { class: 'kv' },
          el('dt', {}, 'Amount'), el('dd', {}, money(r.amount_minor)),
          el('dt', {}, 'Recovery action'), el('dd', {}, r.recovery_action),
          el('dt', {}, 'Payment method'), el('dd', {}, r.method || 'Customer’s choice'),
          el('dt', {}, 'Synthetic payment ID'), el('dd', { class: 'mono' }, r.event_id),
          el('dt', {}, 'Event type'), el('dd', {}, r.event_label),
          el('dt', {}, 'Request ID'), el('dd', { class: 'mono' }, r.request_id),
          el('dt', {}, 'Workflow ID'), el('dd', { class: 'mono' }, r.workflow_id || '—'),
          el('dt', {}, 'Current status'), el('dd', {}, r.status),
          el('dt', {}, 'Created'), el('dd', {}, when(r.created_at))),
        message ? el('div', { class: `notice ${message.kind}`, role: message.kind === 'bad' ? 'alert' : 'status' }, message.text) : null,
        open ? el('div', { class: 'row', style: 'margin-top:14px' },
          el('button', { class: 'btn primary', type: 'button', id: 'accept', disabled: busy, onclick: () => respond('ACCEPT') }, 'ACCEPT: the customer completes the payment'),
          el('button', { class: 'btn', type: 'button', id: 'reject', disabled: busy, onclick: () => respond('REJECT') }, 'REJECT: the customer does not pay')) : el('p', { class: 'small muted' }, OUTCOME[r.status] || ''),
        el('p', { class: 'small muted' }, 'ACCEPT means a simulated successful payment, not merely opening or reading the reminder. REJECT does not mean a bank decline or a technical failure.')));
  }
  async function refresh() { const r = await load(); if (r) paint(r); return r; }
  refresh();
  setInterval(async () => { if (!busy && current && current.status === 'PENDING') { const r = await load(); if (r) paint(r); } }, 3000);
}());
