/* Simulated customer payment page. A mock customer view, not a payment gateway. It shows one system-generated recovery request, worded for the recovery action
   and the kind of revenue event, and lets the tester choose the SIMULATED outcome: ACCEPT (the simulated customer completes the payment) or REJECT (the
   simulated customer does not). The payment methods are shown for context only: they are not selectable, nothing is charged and no method is recorded.
   No input fields, no free text. */
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
  const METHODS = [
    { id: 'upi', name: 'UPI', ico: 'UPI', hint: 'Pay with any UPI app' },
    { id: 'card', name: 'Credit / Debit Card', ico: 'CARD', hint: 'Visa, Mastercard, RuPay' },
    { id: 'netbanking', name: 'Netbanking', ico: 'NB', hint: 'Pay from your bank account' },
  ];
  const methodName = (id) => (METHODS.find((m) => m.id === id) || {}).name || null;
  const REMINDERS = new Set(['payment_reminder', 'checkout_reminder', 'invoice_reminder']);
  let busy = false, message = null, current = null, detailsOpen = true;     // the request details start open and stay as the tester left them

  /* What this request says to the customer, from the recovery action and the kind of event. Pure wording: it changes nothing the backend decides. */
  function copy(r) {
    const c = r.context || {}, amt = money(r.amount_minor), orig = methodName(c.original_method);
    const sub = r.event_type === 'SUBSCRIPTION_FAILURE', chk = r.event_type === 'CHECKOUT_ABANDONMENT', inv = r.event_type === 'OVERDUE_RECEIVABLE';
    const base = { title: 'Complete your payment', lead: `A payment of ${amt} is waiting to be completed.`, offered: METHODS.map((m) => m.id), unavailable: {}, optionsNote: 'Choose how you would like to pay.', facts: [] };
    if (inv) base.facts = [['Invoice', c.invoice_number || '—'], ['Days overdue', c.days_overdue != null ? `${c.days_overdue} days` : '—'], ['Outstanding amount', amt]];
    if (sub) base.facts = [['Subscription', c.subscription_plan || '—'], ['Renewal amount', amt]];
    if (chk) base.facts = [['Order total', amt], ['Where you stopped', c.checkout_stage || 'Before paying']];
    switch (r.action) {
      case 'checkout_reminder':
        return { ...base, reminder: 'You left items behind.', title: 'Your checkout is incomplete', lead: `You stopped${c.checkout_stage ? ` (${c.checkout_stage.toLowerCase()})` : ''} before paying. Your order of ${amt} is not placed until the payment is completed.` };
      case 'invoice_reminder':
        return { ...base, reminder: 'An invoice is overdue.', title: 'Your invoice is overdue', lead: `${c.invoice_number ? `Invoice ${c.invoice_number}` : 'An invoice'} is ${c.days_overdue != null ? `${c.days_overdue} days ` : ''}past its due date. ${amt} is still outstanding.` };
      case 'payment_reminder':
        return { ...base, reminder: 'Your payment has not been completed.', title: sub ? 'Your subscription renewal failed' : 'Your payment is pending',
                 lead: sub ? `Your ${c.subscription_plan || 'subscription'} renewal of ${amt} could not be charged. Pay now to keep it active.` : `Your payment of ${amt} did not go through${orig ? ` with ${orig}` : ''}. You can complete it now.` };
      case 'alternative_payment_method': {
        const others = METHODS.map((m) => m.id).filter((id) => id !== c.original_method);
        return { ...base, title: 'Try another payment method', lead: `${orig ? `Your ${orig} payment` : 'Your earlier payment attempt'} of ${amt} did not go through. You can pay with a different method.`,
                 offered: others, unavailable: c.original_method ? { [c.original_method]: 'did not work, so it is not offered again' } : {}, optionsNote: 'Other methods you could use:' };
      }
      case 'card_update_request':
        return { ...base, title: 'Your card needs updating', lead: `The card on file has expired and cannot be charged for ${amt}. Update your card, or pay another way.`, offered: ['card'],
                 unavailable: { upi: 'Not part of this request', netbanking: 'Not part of this request' }, optionsNote: 'Update your card to continue (simulated: no card details are collected).' };
      case 'delayed_retry':
        return { ...base, title: 'Retrying your original payment method', lead: `We are retrying ${orig ? `your ${orig}` : 'your original'} payment of ${amt}. No new method is needed from you.`,
                 offered: c.original_method ? [c.original_method] : [], unavailable: {}, optionsNote: 'This retry uses the method from the original attempt.' };
      default:
        return { ...base, title: r.recovery_action || base.title };
    }
  }

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
  function methodCards(r, c) {
    const attempted = (r.context || {}).original_method;
    const rows = METHODS.filter((m) => c.offered.includes(m.id) || c.unavailable[m.id] || m.id === attempted).map((m) => {
      const isOffered = c.offered.includes(m.id), isAttempted = m.id === attempted;
      const state = isOffered ? (isAttempted ? 'attempted' : 'offered') : 'unavailable';
      const tag = !isOffered ? ((isAttempted ? 'Originally attempted · ' : '') + (c.unavailable[m.id] || 'Not offered')) : isAttempted ? 'Originally attempted' : (r.action === 'alternative_payment_method' ? 'Available' : 'Eligible');
      return el('li', { class: `pp-method ${state}` }, el('div', { class: 'ico', 'aria-hidden': 'true' }, m.ico), el('div', { class: 't' }, el('b', {}, m.name), el('span', {}, m.hint)), el('span', { class: 'tag' }, tag));
    });
    return rows.length ? el('ul', { class: 'pp-methods', 'aria-label': 'Payment methods' }, rows) : el('p', { class: 'small muted' }, 'No payment method is needed for this step.');
  }
  function paint(r) {
    current = r;
    const open = r.status === 'PENDING', c = copy(r), attempted = methodName((r.context || {}).original_method);
    root.replaceChildren(
      el('div', { class: 'pp-shop' }, el('strong', {}, 'Demo Store (synthetic merchant)'), el('span', { class: `badge ${BADGE[r.status] || 'neutral'}` }, r.status)),
      el('div', { class: 'pp-card' },
        c.reminder && REMINDERS.has(r.action) ? el('div', { class: 'pp-reminder', role: 'note' }, el('b', {}, 'Reminder'), el('span', {}, c.reminder)) : null,
        el('div', { class: 'pp-head' }, el('h1', {}, 'Payment request'), el('p', { style: 'font-weight:600' }, c.title), el('p', { class: 'small' }, c.lead), el('p', { class: 'small muted' }, r.message)),
        el('div', { class: 'pp-amount' }, el('div', { class: 'v' }, money(r.amount_minor)), el('div', { class: 'l' }, `${r.event_label} · ${r.recovery_action}`)),
        c.facts.length ? el('dl', { class: 'kv pp-facts' }, c.facts.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)])) : null,
        el('div', { class: 'pp-sec' }, el('h2', {}, 'Payment options'), el('p', { class: 'sub' }, c.optionsNote + ' Shown for illustration only: choosing a method is not simulated, charged or recorded.'), methodCards(r, c)),
        el('div', { class: 'pp-sec pp-sim' }, el('h2', {}, 'Simulated customer outcome'),
          el('p', { class: 'sub' }, 'You are playing the customer. These controls set the outcome of this request; they do not process a real payment.'),
          message ? el('div', { class: `notice ${message.kind}`, role: message.kind === 'bad' ? 'alert' : 'status' }, message.text) : null,
          open ? el('div', { class: 'row' },
            el('button', { class: 'btn primary', type: 'button', id: 'accept', disabled: busy, onclick: () => respond('ACCEPT') }, 'ACCEPT: the customer completes the payment'),
            el('button', { class: 'btn', type: 'button', id: 'reject', disabled: busy, onclick: () => respond('REJECT') }, 'REJECT: the customer does not pay')) : el('p', { class: 'small muted' }, OUTCOME[r.status] || ''),
          el('p', { class: 'small muted', style: 'margin-top:10px' }, 'ACCEPT means a simulated successful payment, not merely opening or reading the reminder. REJECT does not mean a bank decline or a technical failure.')),
        el('details', { class: 'pp-sec pp-details', open: detailsOpen, ontoggle: (e) => { detailsOpen = e.target.open; } }, el('summary', {}, 'Request details (synthetic)'),
          el('dl', { class: 'kv' },
            el('dt', {}, 'Amount'), el('dd', {}, money(r.amount_minor)),
            el('dt', {}, 'Recovery action'), el('dd', {}, r.recovery_action),
            el('dt', {}, 'Payment method'), el('dd', {}, attempted ? `${attempted} (originally attempted)` : 'Not recorded: the customer’s choice'),
            el('dt', {}, 'Synthetic payment ID'), el('dd', { class: 'mono' }, r.event_id),
            el('dt', {}, 'Event type'), el('dd', {}, r.event_label),
            el('dt', {}, 'Request ID'), el('dd', { class: 'mono' }, r.request_id),
            el('dt', {}, 'Workflow ID'), el('dd', { class: 'mono' }, r.workflow_id || '—'),
            el('dt', {}, 'Current status'), el('dd', {}, r.status),
            el('dt', {}, 'Created'), el('dd', {}, when(r.created_at))))),
      el('p', { class: 'pp-foot' }, 'Synthetic data only. No real payment gateway, message or money is involved, and no payment method is charged or stored.'));
  }
  async function refresh() { const r = await load(); if (r) paint(r); return r; }
  refresh();
  setInterval(async () => { if (!busy && current && current.status === 'PENDING') { const r = await load(); if (r) paint(r); } }, 3000);
}());
