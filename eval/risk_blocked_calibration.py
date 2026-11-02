"""How reliably does Model C's own P(RISK_BLOCKED) identify genuine RISK_BLOCKED payments? (offline, held-out only)

    docker compose exec -T api python -m eval.risk_blocked_calibration

Uses only the individual RISK_BLOCKED probability: not the combined hard-decline mass, not the safety budget. The hidden cause
is used strictly as the evaluation label. No threshold is selected or applied; candidates are listed for the owner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval import phase3_heldout as h

OUT = Path('eval/data/phase3_heldout/risk_blocked_calibration.json')
BIN_EDGES = [0, .05, .10, .20, .40, .60, .80, 1.0]
THRESHOLDS = [.05, .10, .15, .20, .30, .40, .50, .60, .70]


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    centre, half = p + z * z / (2 * n), z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5
    return max(0., (centre - half) / (1 + z * z / n)), min(1., (centre + half) / (1 + z * z / n))


def confusion(pred, y):
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'precision': precision, 'recall': recall,
            'false_positive_rate': fp / (fp + tn) if fp + tn else None, 'F1': f1, 'flagged': tp + fp}


def reliability(p, y, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n, k = int(m.sum()), int(y[m].sum())
        lo95, hi95 = wilson(k, n)
        rows.append({'bin': f'{lo:.0%}-{hi:.0%}', 'payments': n, 'actual_risk_blocked': k,
                     'mean_predicted': float(p[m].mean()) if n else None, 'observed_rate': k / n if n else None,
                     'wilson95': [lo95, hi95]})
    return rows


def ece(p, y, edges):
    total = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.sum() / len(p) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def main():
    frame = h.load_final()
    art, classes, probs = h.probabilities(frame)
    classes = list(classes)
    idx = classes.index('RISK_BLOCKED')
    p = probs[:, idx].astype(float)
    y = (frame.true_cause.to_numpy() == 'RISK_BLOCKED')
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    sweep = {f'{t:.0%}': confusion(p >= t, y) | {'threshold': t} for t in THRESHOLDS}
    top1 = confusion(probs.argmax(axis=1) == idx, y)
    sweep_rows = list(sweep.values())
    best_f1 = max(sweep_rows, key=lambda r: r['F1'])

    def lowest_with(key, floor):
        ok = [r for r in sweep_rows if r[key] is not None and r[key] >= floor]
        return min(ok, key=lambda r: r['threshold'])['threshold'] if ok else None
    recall90 = [r for r in sweep_rows if r['recall'] is not None and r['recall'] >= .9]
    out = {
        'payments': len(y), 'actual_risk_blocked': int(y.sum()), 'base_rate': base,
        'brier_score': brier, 'brier_of_always_predicting_base_rate': base * (1 - base),
        'brier_skill_vs_base_rate': 1 - brier / (base * (1 - base)),
        'ece_equal_width_10_bins': ece(p, y, list(np.linspace(0, 1, 11))), 'ece_requested_bins': ece(p, y, BIN_EDGES),
        'reliability_requested_bins': reliability(p, y, BIN_EDGES), 'reliability_equal_width_10_bins': reliability(p, y, list(np.linspace(0, 1, 11))),
        'threshold_sweep': sweep, 'top1_prediction': top1,
        'candidates': {
            'max_F1_threshold': best_f1['threshold'], 'max_F1': best_f1['F1'],
            'lowest_threshold_with_precision_at_least_0.5': lowest_with('precision', .5),
            'lowest_threshold_with_precision_at_least_0.7': lowest_with('precision', .7),
            'lowest_threshold_with_precision_at_least_0.9': lowest_with('precision', .9),
            'lowest_false_positive_rate_with_recall_at_least_0.9': (min(recall90, key=lambda r: r['false_positive_rate'])['threshold'] if recall90 else None),
        },
        'note': 'Individual P(RISK_BLOCKED) only; the 10% hard-decline budget and combined mass are not used. Nothing is selected.',
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    return out


def fmt(v, pct=True):
    return '-' if v is None else (f'{v:.1%}' if pct else f'{v:.4f}')


if __name__ == '__main__':
    r = main()
    print(f"payments {r['payments']}, actual RISK_BLOCKED {r['actual_risk_blocked']} ({r['base_rate']:.2%})")
    print(f"Brier {r['brier_score']:.5f} (base-rate Brier {r['brier_of_always_predicting_base_rate']:.5f}, skill {r['brier_skill_vs_base_rate']:.3f})  "
          f"ECE 10 equal bins {r['ece_equal_width_10_bins']:.5f}  ECE requested bins {r['ece_requested_bins']:.5f}")
    print('\nbin        payments  actual  mean_pred  observed   wilson95')
    for b in r['reliability_requested_bins']:
        print(f"{b['bin']:9s} {b['payments']:8d} {b['actual_risk_blocked']:7d} {fmt(b['mean_predicted']):>9s} {fmt(b['observed_rate']):>9s}   "
              f"[{fmt(b['wilson95'][0])}, {fmt(b['wilson95'][1])}]")
    print('\nthreshold    TP     FP     TN     FN  precision recall   FPR     F1   flagged')
    for k, v in r['threshold_sweep'].items():
        print(f"{k:>8s} {v['TP']:6d} {v['FP']:6d} {v['TN']:6d} {v['FN']:6d} {fmt(v['precision']):>9s} {fmt(v['recall']):>7s} {fmt(v['false_positive_rate']):>6s} {v['F1']:.3f} {v['flagged']:8d}")
    t = r['top1_prediction']
    print(f"{'top-1':>8s} {t['TP']:6d} {t['FP']:6d} {t['TN']:6d} {t['FN']:6d} {fmt(t['precision']):>9s} {fmt(t['recall']):>7s} {fmt(t['false_positive_rate']):>6s} {t['F1']:.3f} {t['flagged']:8d}")
    print('\ncandidates:', json.dumps(r['candidates']))
