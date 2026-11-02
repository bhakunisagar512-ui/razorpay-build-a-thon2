"""Confidence, calibration and recovery-safety metrics for a probability matrix (Phase 4).

Pure functions over (probabilities, truth frame); no model or file access, so they are unit-testable.
The recovery gate is the frozen Phase 2.5 control (confidence .60, hard-decline mass .03), applied
unchanged through eval.safety_pipeline.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support

from eval.probability_calibration import expected_calibration_error
from eval.safety_pipeline import apply_policy, metrics as gate_metrics, prepare
from services.app.domain.diagnosis_safety import SafetyThresholds
from services.app.domain.failure_vocab import GENERIC_REASONS

GATE = SafetyThresholds(.60, .03)      # frozen; never retuned in Phase 4
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
RETRYABLE = ('BD_INSUFFICIENT_FUNDS', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
LOW, HIGH = .60, .80
THRESHOLDS = (.5, .6, .7, .8, .9)


def upper95(k, n):
    return None if n == 0 else float(1.0 if k == n else beta.ppf(.95, k + 1, n - k))


def _q(values, qs=(.1, .25, .5, .75, .9)):
    if not len(values):
        return None
    return {f'p{int(q * 100)}': float(np.quantile(values, q)) for q in qs}


def reliability_bins(confidence, correct, bins=10):
    index = np.minimum((np.asarray(confidence) * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = index == b
        rows.append({'lo': b / bins, 'hi': (b + 1) / bins, 'n': int(m.sum()),
                     'mean_confidence': float(confidence[m].mean()) if m.any() else None,
                     'accuracy': float(correct[m].mean()) if m.any() else None})
    return rows


def brier(p, y, labels):
    truth = (np.asarray(y)[:, None] == np.asarray(labels)[None, :]).astype(float)
    return float(np.mean(np.sum((p - truth) ** 2, axis=1)))


def confidence_profile(p, y, labels):
    """How sure is the model, and is that sureness earned? Everything keyed on max predicted probability."""
    labels = np.asarray(labels)
    pred = labels[p.argmax(axis=1)]
    conf = p.max(axis=1)
    correct = pred == y
    above, mid, below = conf >= HIGH, (conf >= LOW) & (conf < HIGH), conf < LOW
    answered = conf >= LOW
    out = {
        'n': int(len(y)),
        'share_below_60': float(below.mean()), 'share_60_to_80': float(mid.mean()), 'share_above_80': float(above.mean()),
        'accuracy_by_band': {'below_60': float(correct[below].mean()) if below.any() else None,
                             'from_60_to_80': float(correct[mid].mean()) if mid.any() else None,
                             'above_80': float(correct[above].mean()) if above.any() else None},
        'coverage_at_60': float(answered.mean()),
        'accuracy_when_confident_ge_60': float(correct[answered].mean()) if answered.any() else None,
        'error_rate_when_confident_ge_60': float(1 - correct[answered].mean()) if answered.any() else None,
        'confident_errors_per_1000': float(1000 * (answered & ~correct).mean()),
        'mean_confidence': float(conf.mean()), 'mean_accuracy': float(correct.mean()),
        'confidence_correct': {'n': int(correct.sum()), 'mean': float(conf[correct].mean()) if correct.any() else None,
                               'quantiles': _q(conf[correct]), 'share_below_60': float((conf[correct] < LOW).mean()) if correct.any() else None},
        'confidence_incorrect': {'n': int((~correct).sum()), 'mean': float(conf[~correct].mean()) if (~correct).any() else None,
                                 'quantiles': _q(conf[~correct]), 'share_below_60': float((conf[~correct] < LOW).mean()) if (~correct).any() else None},
        'histogram_10_bins': {'correct': np.histogram(conf[correct], bins=10, range=(0, 1))[0].tolist(),
                              'incorrect': np.histogram(conf[~correct], bins=10, range=(0, 1))[0].tolist()},
        'risk_coverage': {str(t): {'coverage': float((conf >= t).mean()),
                                   'accuracy': float(correct[conf >= t].mean()) if (conf >= t).any() else None}
                          for t in THRESHOLDS},
        'reliability_bins': reliability_bins(conf, correct),
    }
    return out


def calibration_block(p, y, labels):
    labels = np.asarray(labels)
    pred = labels[p.argmax(axis=1)]
    return {'ece_top_label_10_bins': expected_calibration_error(p.max(axis=1), pred == y),
            'logloss': float(log_loss(y, np.clip(p, 1e-12, 1), labels=labels)),
            'brier_multiclass': brier(p, y, labels)}


def classification_block(p, y, labels):
    labels = np.asarray(labels)
    pred = labels[p.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    matrix = pd.crosstab(pd.Categorical(y, categories=labels), pd.Categorical(pred, categories=labels), dropna=False)
    return {'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'accuracy': float((pred == y).mean()),
            'per_class': {str(l): {'precision': float(a), 'recall': float(b), 'f1': float(c), 'support': int(d)}
                          for l, a, b, c, d in zip(labels, precision, recall, f1, support)},
            'confusion_matrix': {'labels': [str(l) for l in labels], 'rows_true_cols_predicted': matrix.to_numpy().tolist()}}


def safety_block(frame, model, p):
    """Frozen gate over this probability matrix. Counts carry denominators; nothing is retuned."""
    labels = np.asarray(model.classes_)
    y = frame.true_cause.to_numpy()
    data = prepare(frame, model, p)
    gm = gate_metrics(data, apply_policy(data, GATE))
    hard = np.isin(y, HARD)
    retryable = np.isin(y, RETRYABLE)
    pred = labels[p.argmax(axis=1)]
    raw_unsafe = int((hard & np.isin(pred, RETRYABLE)).sum())
    n_retryable = int(retryable.sum())
    allowed_true = round(gm['true_retryable_recall'] * n_retryable) if n_retryable else 0
    return {'n': int(len(y)), 'hard_declines': int(hard.sum()), 'retryable_payments': n_retryable,
            'unsafe_retry_gated': {'count': gm['unsafe_retry_count'], 'rate': gm['unsafe_retry_rate'],
                                   'upper95': upper95(gm['unsafe_retry_count'], int(hard.sum()))},
            'unsafe_retry_ungated_argmax': {'count': raw_unsafe, 'rate': raw_unsafe / hard.sum() if hard.any() else None,
                                            'upper95': upper95(raw_unsafe, int(hard.sum()))},
            'retryable_recall_gated': gm['true_retryable_recall'],
            'retryable_recall_ungated_argmax': float(np.isin(pred, RETRYABLE)[retryable].mean()) if n_retryable else None,
            'abstention_rate': gm['abstain_rate'],
            'recovery_eligibility': {'retry_allowed_share': gm['retry_coverage'],
                                     'retry_allowed_count': int(round(gm['retry_coverage'] * len(y))),
                                     'true_retryable_allowed_count': int(allowed_true),
                                     'false_retry_fraction_of_allowed': gm['false_retry_fraction_of_allowed']},
            'gated_macro_f1': gm['macro_f1_all'], 'gated_accuracy': gm['accuracy'],
            'gate_decisions': {k: int(v) for k, v in gm['decisions'].items()}}


def subsets(frame):
    """Masks for the cases the user asked about. All defined from observable columns or simulator metadata."""
    reason = frame['error_reason'].astype(str)
    return {
        'generic_error_reason': reason.isin(GENERIC_REASONS).to_numpy(),
        'missing_error_reason': (reason == 'missing').to_numpy(),
        'missing_timing_telemetry': frame['attempt_duration_s'].isna().to_numpy(),
        'ambiguous_overlapping_causes': (frame['n_causes_fired'] >= 2).to_numpy(),
        'all_failures': np.ones(len(frame), dtype=bool),
    }


def subset_profiles(frame, p, labels):
    y = frame.true_cause.to_numpy()
    out = {}
    for name, mask in subsets(frame).items():
        if mask.sum() < 20:
            out[name] = {'n': int(mask.sum()), 'note': 'too few rows'}
            continue
        present = sorted(set(y[mask]))
        pred = np.asarray(labels)[p[mask].argmax(axis=1)]
        profile = confidence_profile(p[mask], y[mask], labels)
        out[name] = {'n': int(mask.sum()), 'accuracy': float((pred == y[mask]).mean()),
                     'macro_f1_present_classes': float(f1_score(y[mask], pred, labels=present, average='macro', zero_division=0)),
                     'share_below_60': profile['share_below_60'],
                     'accuracy_when_confident_ge_60': profile['accuracy_when_confident_ge_60'],
                     'confident_errors_per_1000': profile['confident_errors_per_1000'],
                     'ece': calibration_block(p[mask], y[mask], labels)['ece_top_label_10_bins']}
    return out


def full_report(frame, model, p, per_world=None):
    """Complete Step 3 + Step 4 block for one (model, calibration method, partition)."""
    labels = np.asarray(model.classes_)
    y = frame.true_cause.to_numpy()
    report = {**classification_block(p, y, labels), 'calibration': calibration_block(p, y, labels),
              'confidence': confidence_profile(p, y, labels), 'safety': safety_block(frame, model, p),
              'subsets': subset_profiles(frame, p, labels)}
    if per_world is not None:
        scores = [classification_block(p[idx], y[idx], labels)['macro_f1'] for idx in per_world]
        report['macro_f1_over_worlds'] = {'mean': float(np.mean(scores)),
                                          'sd': float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                                          'n_worlds': len(scores)}
    return report


def world_indices(frame):
    """Row positions of each world within a pooled frame (frame.world_key is set when loading)."""
    keys = frame['world_key'].to_numpy()
    return [np.flatnonzero(keys == w) for w in sorted(set(keys))]
