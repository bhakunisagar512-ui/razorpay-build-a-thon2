"""Before/after evaluation of the v2 deterministic safety layer on the saved Phase 4 checkpoints.

    python -m eval.phase4_safety_cases            # once: rebuild cases.pkl (no retraining)
    python -m eval.phase4_safety --out eval/data/phase4_safety

Definitions are the benchmark's, unchanged:
  unsafe retry          true cause in {CARD_EXPIRED, RISK_BLOCKED} AND the policy allows an automated retry
  retryable recall      share of true {FUNDS, ISSUER_DOWN, PSP_TIMEOUT} failures for which a retry is allowed
  abstention            final cause == AMBIGUOUS
Added (secondary, never a replacement): doomed-retry share = unsafe / automated retries, share of the frozen gate's
retries that v2 now holds for review, and retries
aimed at BD_USER_ABANDONED_OTP failures (non-retryable by taxonomy but not a hard decline).

v2 only layers extra vetoes on the stored frozen-gate outputs, so it can never allow a retry that v1 blocked.
tests/test_safety_policy_v2.py proves this vectorised layering equals the pure decide_v2 row by row.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eval.phase4_confidence import upper95
from eval.phase4_ledger import atomic_write_json
from services.app.domain.failure_vocab import reason_family
from services.app.domain.safety_policy_v2 import (AUTO_RETRY_MAX_HARD_MASS, HARD_REASON_CAUSE, HARD_REASON_FAMILIES,
                                                  MIN_CONFIDENCE, REVIEW_MAX_HARD_MASS)

RETRYABLE = ('BD_INSUFFICIENT_FUNDS', 'TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
VARIANTS = {'v1_frozen_gate': None,
            'v2_signals_only': (True, .03),
            'v2': (True, AUTO_RETRY_MAX_HARD_MASS),
            'v2_budget_0.02': (True, .02),
            'v2_budget_0.005': (True, .005),
            'budget_only_0.01': (False, .01)}
DOOMED_TARGET = .001     # at most 1 in 1000 automated retries aimed at a hard decline (validation, winning model)


def signal_masks(df):
    explicit = df.error_reason.isin(HARD_REASON_CAUSE).to_numpy()
    family = df.error_reason.map(lambda r: reason_family(None if r == 'missing' else r) in HARD_REASON_FAMILIES).to_numpy()
    prev = df.instrument_prev_reason.isin(HARD_REASON_CAUSE).to_numpy()
    return explicit, family, prev


def apply_variant(df, variant):
    """-> (retry, cause) arrays after layering the variant on the stored frozen-gate outputs."""
    retry, cause = df.retry_allowed.to_numpy().copy(), df.gate_cause.to_numpy().astype(object).copy()
    if VARIANTS[variant] is None:
        return retry, cause
    use_signals, budget = VARIANTS[variant]
    mass = df.hard_mass.to_numpy()
    veto_mass = retry & (mass > budget)
    cause[veto_mass] = 'AMBIGUOUS'
    if use_signals:
        explicit, family, prev = signal_masks(df)
        veto_sig = retry & ~veto_mass & (explicit | family | prev)
        asserted = veto_sig & explicit
        cause[veto_sig] = 'AMBIGUOUS'
        cause[asserted] = df.error_reason[asserted].map(HARD_REASON_CAUSE).to_numpy()
        retry = retry & ~veto_sig
    return retry & ~veto_mass, cause


def metrics(df, retry, cause):
    hard = df.hard.to_numpy()
    truth = df.true_cause.to_numpy()
    retryable = np.isin(truth, RETRYABLE)
    unsafe = int((retry & hard).sum())
    allowed = int(retry.sum())
    held = df.retry_allowed.to_numpy() & ~retry          # v1 would have retried; v2 withholds (review / no retry)
    return {'rows': int(len(df)), 'hard_declines': int(hard.sum()), 'unsafe_retries': unsafe,
            'unsafe_per_hard_decline': unsafe / hard.sum() if hard.any() else None,
            'unsafe_upper95': upper95(unsafe, int(hard.sum())),
            'automated_retries': allowed, 'doomed_share_of_retries': unsafe / allowed if allowed else None,
            'retryable_recall': float((retry & retryable).sum() / retryable.sum()) if retryable.any() else None,
            'abstention_rate': float((cause == 'AMBIGUOUS').mean()), 'v1_retries_now_held': int(held.sum()),
            'v1_retries_now_held_share': float(held.sum() / max(int(df.retry_allowed.sum()), 1)),
            'otp_abandon_retries': int((retry & (truth == 'BD_USER_ABANDONED_OTP')).sum())}


def evaluate(cases):
    out = {}
    for (split, config, family), df in cases.groupby(['split', 'config', 'family']):
        for variant in VARIANTS:
            retry, cause = apply_variant(df, variant)
            out.setdefault(split, {}).setdefault(config, {}).setdefault(family, {})[variant] = metrics(df, retry, cause)
    return out


def residual_buckets(cases):
    """What the unsafe retries that survive v2 look like (same taxonomy as the investigation)."""
    from services.app.domain.failure_vocab import GENERIC_REASONS
    rows = {}
    for (split, config, family), df in cases.groupby(['split', 'config', 'family']):
        retry, _ = apply_variant(df, 'v2')
        u = df[retry & df.hard.to_numpy()]
        bucket = np.where(u.error_reason == 'missing', 'missing_reason',
                          np.where(u.error_reason.isin(GENERIC_REASONS), 'generic_reason', 'specific_reason_contradicting'))
        rows[f'{split}/{config}/{family}'] = {'by_bucket': pd.Series(bucket).value_counts().to_dict(),
                                              'by_true_cause': u.true_cause.value_counts().to_dict(),
                                              'overlapping_causes': int((u.n_causes_fired >= 2).sum()),
                                              'by_scenario': u.scenario.value_counts().to_dict()}
    return rows


def budget_choice(cases):
    """Loosest budget whose validation doomed-share for the frozen winner (HGB, deployment_core) meets the target."""
    df = cases[(cases.split == 'validation') & (cases.config == 'deployment_core') & (cases.family == 'hgb')]
    table = {}
    for budget in (.03, .02, .01, .005):
        VARIANTS['_probe'] = (True, budget)
        retry, cause = apply_variant(df, '_probe')
        table[budget] = metrics(df, retry, cause)
    del VARIANTS['_probe']
    chosen = next((b for b in (.03, .02, .01, .005) if table[b]['doomed_share_of_retries'] <= DOOMED_TARGET), .005)
    return {'target_doomed_share': DOOMED_TARGET, 'chosen': chosen,
            'validation_hgb_by_budget': {str(b): {k: table[b][k] for k in ('unsafe_retries', 'automated_retries',
                                                                         'doomed_share_of_retries', 'retryable_recall')} for b in table}}


def md(report):
    lines = []
    for split in ('validation', 'test', 'shifted'):
        for config in ('deployment_core', 'deployment_with_timing'):
            lines += [f'\n#### {split} / {config}', '',
                      '| Model | Policy | Unsafe / hard | Unsafe upper 95% | Doomed share of retries | Retryable recall | Abstain | v1 retries now held | OTP-abandon retries |',
                      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for family, variants in report['metrics'][split][config].items():
                for variant, m in variants.items():
                    if variant in ('v2_budget_0.02', 'v2_budget_0.005', 'budget_only_0.01') and config != 'deployment_core':
                        continue
                    lines.append(f'| {family} | {variant} | {m["unsafe_retries"]}/{m["hard_declines"]} | {100 * m["unsafe_upper95"]:.2f}% | '
                                 f'{100 * m["doomed_share_of_retries"]:.3f}% | {100 * m["retryable_recall"]:.1f}% | '
                                 f'{100 * m["abstention_rate"]:.1f}% | {100 * m["v1_retries_now_held_share"]:.1f}% | {m["otp_abandon_retries"]} |')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='eval/data/phase4_safety')
    args = parser.parse_args()
    out = Path(args.out)
    cases = pickle.loads((out / 'cases.pkl').read_bytes())
    report = {'definitions': __doc__, 'policy': {'min_confidence': MIN_CONFIDENCE, 'auto_retry_max_hard_mass': AUTO_RETRY_MAX_HARD_MASS,
                                                  'review_max_hard_mass': REVIEW_MAX_HARD_MASS},
              'budget_choice_on_validation': budget_choice(cases), 'metrics': evaluate(cases),
              'residual_after_v2': residual_buckets(cases)}
    atomic_write_json(out / 'safety_v2_report.json', report)
    (out / 'safety_v2_tables.md').write_text(md(report), encoding='utf-8')
    print(json.dumps(report['budget_choice_on_validation'], indent=1))
    for split in ('validation', 'test'):
        for family in ('hgb', 'xgboost', 'catboost', 'random_forest'):
            m = report['metrics'][split]['deployment_core'][family]
            print(split, family, {v: (m[v]['unsafe_retries'], round(m[v]['retryable_recall'], 3), round(m[v]['abstention_rate'], 3)) for v in ('v1_frozen_gate', 'v2_signals_only', 'v2')})
