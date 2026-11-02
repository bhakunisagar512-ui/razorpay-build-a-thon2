"""Phase 2 shadow-mode validation: synthetic failures -> actual FastAPI app (/shadow/event) -> Model C -> metrics.

    docker compose exec -T api python -m eval.phase2_shadow --orders 3000 --seed 777001

The world replays in arrival order through the HTTP endpoints exactly as a gateway would send it: downtime notices
and every attempt (warm-up and successes feed history only; measurement failures are scored). The true cause is
attached afterwards through a separate label endpoint; it is never in the request that is scored. No recovery action
is executed: "unsafe" is computed from the plan the existing policy WOULD build for the shadow cause.

Also scores the same failures offline (assemble_v4 features straight into the artifact) to separate serving
mismatches from sample-size effects. Seed 777001 is outside every training/validation/final-test range.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from eval.classifier import HARD_CAUSES
from eval.sequences_v4 import assemble_v4, generate_world, _observation
from services.app import main
from services.app import shadow
from services.app.adapters import diagnosis_v4
from eval.oracle_plan import Ctx, build_plan
from services.app.domain.taxonomy import Cause

OBS_FIELDS = ('payment_id', 'order_id', 'customer_id', 'issuer_id', 'psp_id', 'instrument_id', 'amount_minor', 'method',
              'occurred_at', 'observed_at', 'status', 'instrument_hint', 'gw_error_code', 'gw_error_source',
              'gw_error_step', 'latency_ms', 'error_reason', 'payment_flow', 'attempt_duration_s')
OFFLINE = {'accuracy': 0.912, 'macro_f1': 0.878}


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value.item() if hasattr(value, 'item') else value


def event_payload(row, *, diagnose, observe):
    body = {k: _clean(row.get(k)) for k in OBS_FIELDS}
    return {**{k: v for k, v in body.items() if v is not None}, 'diagnose': diagnose, 'observe': observe}


def replay_through_api(world, client):
    rows = sorted(world.rows, key=lambda r: (r['observed_at'], r['payment_id']))
    notices = sorted(world.notices, key=lambda n: n.at)
    cursor, truth, responses = 0, {}, {}
    for row in rows:
        while cursor < len(notices) and notices[cursor].at <= row['observed_at']:
            n = notices[cursor]
            r = client.post('/shadow/downtime', json={'event_id': n.event_id, 'kind': n.kind, 'at': n.at.isoformat(),
                                                      'method': n.method, 'severity': n.severity, 'issuer_id': n.issuer_id,
                                                      'psp_id': n.psp_id})
            assert r.status_code == 200 and r.json()['status'] == 'ok', r.text
            cursor += 1
        scored = row['status'] == 'failed' and not row['is_warmup']
        r = client.post('/shadow/event', json=event_payload(row, diagnose=scored, observe=bool(row['telemetry_visible'])))
        assert r.status_code == 200, r.text
        if scored:
            truth[row['payment_id']] = row['true_cause']
            responses[row['payment_id']] = r.json()
            assert client.put(f"/shadow/label/{row['payment_id']}", params={'actual_cause': row['true_cause']}).json()['attached']
    return truth, responses


def offline_predictions(dataset):
    art = diagnosis_v4.bundle()
    model = art['model']
    p = art['calibrator'].transform(model.predict_proba(dataset.failures))
    return dict(zip(dataset.failures.payment_id, np.asarray(model.classes_)[p.argmax(axis=1)])), list(model.classes_)


def would_retry(cause, row):
    ctx = Ctx(amount_minor=int(row['amount_minor_raw']), method=row.get('method', 'card'), today=date(2026, 10, 29),
              available_rails=frozenset({'upi_intent', 'card', 'netbanking'}), consent_whatsapp=True)
    return build_plan(Cause(cause), ctx).retry_count > 0


def evaluate(truth, responses, classes, dataset, offline):
    ids = [p for p in truth if responses[p].get('status') == 'scored']
    y = np.array([truth[p] for p in ids])
    pred = np.array([responses[p]['predicted_cause'] for p in ids])
    conf = np.array([responses[p]['confidence'] for p in ids])
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    failures = dataset.failures.set_index('payment_id')
    unsafe = hard = 0
    for p, t in zip(ids, y):
        if t in HARD_CAUSES:
            hard += 1
            unsafe += int(would_retry(responses[p]['policy_cause'], failures.loc[p]))
    offline_pred = np.array([offline[p] for p in ids])
    return {
        'rows_total_failures': len(truth), 'rows_scored': len(ids), 'rows_not_scored': len(truth) - len(ids),
        'status_counts': pd.Series([r.get('status') for r in responses.values()]).value_counts().to_dict(),
        'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
        'per_class': {c: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(sup[i])}
                      for i, c in enumerate(classes)},
        'confusion_matrix': {'labels': classes, 'rows_true_cols_predicted': confusion_matrix(y, pred, labels=classes).tolist()},
        'confidence': {'mean': float(conf.mean()), 'median': float(np.median(conf)), 'p10': float(np.quantile(conf, .1)),
                       'below_0.60': int((conf < .60).sum()), 'share_below_0.60': float((conf < .60).mean()),
                       'between_0.60_0.80': int(((conf >= .60) & (conf < .80)).sum()), 'above_0.80': int((conf >= .80).sum()),
                       'histogram_0.1_bins': np.histogram(conf, bins=np.linspace(0, 1, 11))[0].tolist()},
        'safety_without_execution': {
            'hard_declines': hard, 'unsafe_retries_under_existing_policy': unsafe,
            'unsafe_rate_of_hard_declines': unsafe / hard if hard else None,
            'abstained_to_ambiguous': int(sum(responses[p]['policy_cause'] == str(Cause.AMBIGUOUS) for p in ids)),
            'definition': 'true cause in {CARD_EXPIRED, RISK_BLOCKED} and build_plan(shadow policy cause).retry_count > 0'},
        'serving_vs_offline': {'prediction_agreement': float((offline_pred == pred).mean()),
                               'offline_same_rows_accuracy': float(accuracy_score(y, offline_pred)),
                               'offline_same_rows_macro_f1': float(f1_score(y, offline_pred, labels=classes, average='macro', zero_division=0))},
        'latency_ms_per_scored_failure': {'mean': float(np.mean([responses[p]['latency_ms'] for p in ids])),
                                          'p95': float(np.quantile([responses[p]['latency_ms'] for p in ids], .95))},
        'previous_offline_benchmark': OFFLINE,
    }


def main_cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=777001)
    ap.add_argument('--scenario', default='typical')
    ap.add_argument('--out', default='eval/data/phase2_shadow')
    a = ap.parse_args(argv)
    started = time.time()
    world = generate_world(a.orders, a.seed, scenario=a.scenario)
    dataset = assemble_v4(world)
    offline, classes = offline_predictions(dataset)
    shadow.reset()
    with TestClient(main.app) as client:
        truth, responses = replay_through_api(world, client)
        sample = client.get('/shadow/diagnoses', params={'limit': 1}).json()
    metrics = evaluate(truth, responses, classes, dataset, offline)
    metrics['meta'] = {'scenario': a.scenario, 'seed': a.seed, 'orders': a.orders, 'attempts_replayed': len(world.rows),
                       'model_id': diagnosis_v4.bundle()['schema']['model_id'], 'shadow_only': True, 'runtime_s': round(time.time() - started, 1),
                       'example_record': sample[-1] if sample else None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('confusion_matrix', 'per_class')}, indent=2, default=str))
    return metrics


if __name__ == '__main__':
    main_cli()
