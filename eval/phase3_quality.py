"""Data-quality and distribution checks for sequences@v4 worlds.

world_quality() runs inside the world-generation worker so the large transaction frame
never has to be pickled. aggregate_quality() combines worlds into per-scenario checks
with explicit pass/fail flags. Thresholds are stated here, not tuned after the fact.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.world_config_v4 import UPI_ANCHORS
from services.app.domain.failure_vocab import DOCUMENTED_REASONS, ERROR_CODES, SOURCES, STEPS
from services.app.domain.features_v4 import FEATURES_V4

BD = ('BD_INSUFFICIENT_FUNDS', 'BD_USER_ABANDONED_OTP', 'RISK_BLOCKED')
TD = ('TD_ISSUER_DOWN', 'TD_PSP_TIMEOUT')
LABEL_LIKE = {'true_cause', 'n_causes_fired', 'secondary_causes', 'world_id', 'scenario', 'payment_id',
              'order_id', 'customer_id', 'instrument_id'}
# Minimum failures per class by split, across all worlds of the split (rare-class adequacy).
MIN_CLASS_COUNT = {'train': 200, 'validation': 50, 'test': 60, 'shifted_test': 40}


def world_quality(dataset):
    tx, f = dataset.transactions, dataset.failures
    m = tx[~tx.is_warmup]
    failed = m[m.status == 'failed']
    upi = m[m.method == 'upi']
    reasons = failed.error_reason.dropna()
    steps_ok = failed.apply(lambda r: (r.gw_error_step == '' or r.gw_error_step in STEPS[r.method]), axis=1)
    sources_ok = failed.apply(lambda r: (r.gw_error_source == '' or r.gw_error_source in SOURCES[r.method]), axis=1)
    cards = f[f.method == 'card'] if 'method' in f else None
    expired = failed[(failed.true_cause == 'CARD_EXPIRED')]
    risk_instruments = set(failed[failed.true_cause == 'RISK_BLOCKED'].instrument_id)
    later = m[m.instrument_id.isin(risk_instruments)]
    out = dict(
        measurement_attempts=int(len(m)), failures=int(len(f)),
        failure_rate=float((m.status == 'failed').mean()),
        failure_rate_by_method={k: float((g.status == 'failed').mean()) for k, g in m.groupby('method')},
        attempts_by_method={k: int(len(g)) for k, g in m.groupby('method')},
        upi_bd_pct=float(100 * upi.true_cause.isin(BD).mean()), upi_td_pct=float(100 * upi.true_cause.isin(TD).mean()),
        cause_counts=f.true_cause.value_counts().to_dict(),
        retry_failure_share=float((failed.retry_index > 0).mean()),
        generic_or_missing_share=float(f.is_generic_reason.mean()),
        missing_reason_share=float((f.error_reason == 'missing').mean()),
        undocumented_reason_count=int((~reasons.isin(DOCUMENTED_REASONS)).sum()),
        invalid_error_code_count=int((~failed.gw_error_code.isin(ERROR_CODES)).sum()),
        invalid_step_count=int((~steps_ok).sum()), invalid_source_count=int((~sources_ok).sum()),
        hint_expired_given_card_expired=float((expired.instrument_hint == 'expired').mean()) if len(expired) else None,
        hint_unknown_share_by_cause={k: float((g.instrument_hint == 'unknown').mean())
                                     for k, g in failed[failed.method == 'card'].groupby('true_cause')},
        risk_instrument_success_share=float((later.status == 'success').mean()) if len(later) else None,
        failure_overlap_fraction=dataset.metadata['failure_overlap_fraction'],
        n_episodes=dataset.metadata['n_episodes'], n_notices=dataset.metadata['n_notices'],
        outage_signature=sorted((r.kind, r.entity, round(float(r.start_s), 0)) for r in dataset.outages.itertuples()),
        duplicate_payment_ids=int(tx.payment_id.duplicated().sum()),
        warmup_failures_in_feature_frame=int(f.payment_id.isin(tx[tx.is_warmup].payment_id).sum()),
        feature_build_us_per_failure=dataset.metadata['feature_build_us_per_failure'],
        unknown_instrument_share=float((m.instrument_id == 'unknown').mean()),
    )
    cols = [c for c in FEATURES_V4]
    out['feature_missing_share'] = {c: float(f[c].isna().mean()) for c in cols}
    out['feature_constant'] = [c for c in cols if f[c].nunique(dropna=False) <= 1]
    out['label_like_columns_in_features'] = sorted(set(FEATURES_V4) & LABEL_LIKE)
    return out


def _lookup_accuracy(train, test, keys):
    """Held-out majority-vote lookup: how far a plain dictionary on `keys` gets."""
    table = (train.groupby(keys, dropna=False).true_cause.agg(lambda s: s.value_counts().index[0]))
    fallback = train.true_cause.value_counts().index[0]
    merged = test[keys + ['true_cause']].merge(table.rename('guess').reset_index(), on=keys, how='left')
    guess = merged.guess.fillna(fallback)
    return float((guess.to_numpy() == merged.true_cause.to_numpy()).mean())


def lookup_ceilings(train, test):
    tuple_cols = ['gw_error_code', 'gw_error_source', 'gw_error_step']
    sets = {'error_tuple': tuple_cols, 'error_tuple+method+hint': tuple_cols + ['method', 'instrument_hint'],
            'error_reason': ['error_reason'], 'error_reason+method': ['error_reason', 'method'],
            'error_reason+tuple+method': ['error_reason', *tuple_cols, 'method']}
    return {name: _lookup_accuracy(train, test, cols) for name, cols in sets.items()}


def aggregate_quality(scenario, per_role_quality):
    """per_role_quality: {role: [world_quality dicts]} for one scenario."""
    out = {'scenario': scenario, 'roles': {}}
    flags = {}
    for role, worlds in per_role_quality.items():
        counts = pd.DataFrame([w['cause_counts'] for w in worlds]).fillna(0).sum()
        rates = pd.DataFrame([w['failure_rate_by_method'] for w in worlds]).mean().to_dict()
        role_out = dict(
            worlds=len(worlds), failures=int(sum(w['failures'] for w in worlds)),
            attempts=int(sum(w['measurement_attempts'] for w in worlds)),
            overall_failure_rate=float(np.mean([w['failure_rate'] for w in worlds])),
            failure_rate_by_method=rates,
            upi_bd_pct=float(np.mean([w['upi_bd_pct'] for w in worlds])),
            upi_td_pct=float(np.mean([w['upi_td_pct'] for w in worlds])),
            class_counts={k: int(v) for k, v in counts.items()},
            class_share={k: float(v / counts.sum()) for k, v in counts.items()},
            min_class_count=int(counts.min()), n_classes=int(len(counts)),
            generic_or_missing_share=float(np.mean([w['generic_or_missing_share'] for w in worlds])),
            missing_reason_share=float(np.mean([w['missing_reason_share'] for w in worlds])),
            failure_overlap_fraction=float(np.mean([w['failure_overlap_fraction'] for w in worlds])),
            retry_failure_share=float(np.mean([w['retry_failure_share'] for w in worlds])),
            hint_expired_given_card_expired=float(np.nanmean([w['hint_expired_given_card_expired'] or np.nan
                                                              for w in worlds])),
            risk_instrument_success_share=float(np.nanmean([w['risk_instrument_success_share'] or np.nan
                                                            for w in worlds])),
            feature_build_us_per_failure=float(np.mean([w['feature_build_us_per_failure'] for w in worlds])),
            feature_missing_share_mean={c: float(np.mean([w['feature_missing_share'][c] for w in worlds]))
                                        for c in FEATURES_V4},
            episodes_per_world=[w['n_episodes'] for w in worlds])
        out['roles'][role] = role_out
        flags[f'{role}:rare_classes_adequate'] = (role_out['n_classes'] == 6
                                                  and role_out['min_class_count'] >= MIN_CLASS_COUNT[role])
        flags[f'{role}:vocabulary_valid'] = all(
            w['undocumented_reason_count'] == 0 and w['invalid_error_code_count'] == 0 and w['invalid_step_count'] == 0
            and w['invalid_source_count'] == 0 for w in worlds)
        flags[f'{role}:no_duplicates_or_warmup_leak'] = all(
            w['duplicate_payment_ids'] == 0 and w['warmup_failures_in_feature_frame'] == 0 for w in worlds)
        flags[f'{role}:no_label_like_features'] = all(not w['label_like_columns_in_features'] for w in worlds)
        # A rare-event feature (e.g. a matched downtime) can be constant inside ONE world; the check is that
        # no feature is constant across the whole split. World-level constants are reported for information.
        always_constant = set.intersection(*[set(w['feature_constant']) for w in worlds])
        role_out['features_constant_across_split'] = sorted(always_constant)
        role_out['features_constant_in_some_world'] = sorted({f for w in worlds for f in w['feature_constant']})
        flags[f'{role}:no_constant_features'] = not always_constant
    signatures = [tuple(w['outage_signature']) for worlds in per_role_quality.values() for w in worlds]
    out['outage_calendars_distinct'] = len(set(signatures)) == len(signatures)
    flags['outage_calendars_distinct'] = out['outage_calendars_distinct']
    train = out['roles'].get('train')
    if train:
        # Documented/NPCI anchors apply to UPI only. Stressed intentionally exceeds the NPCI TD band.
        lo, hi = UPI_ANCHORS['bd_pct_range']
        flags['train:upi_bd_within_npci_band'] = lo <= train['upi_bd_pct'] <= hi
        lo, hi = UPI_ANCHORS['td_pct_range']
        flags['train:upi_td_within_npci_band'] = lo <= train['upi_td_pct'] <= hi
    out['flags'] = flags
    out['all_flags_pass_excluding_intentional_stress'] = all(
        v for k, v in flags.items() if not (scenario == 'stressed' and k.endswith('upi_td_within_npci_band')))
    return out


def single_feature_probe(train, test, features, limit_rows=6000):
    """Held-out accuracy of a tiny model on ONE feature: flags unrealistic shortcuts."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OneHotEncoder
    from services.app.domain.features_v4 import CATEGORICAL_FEATURES_V4
    train = train.sample(min(len(train), limit_rows), random_state=0)
    result = {}
    for name in features:
        if name in CATEGORICAL_FEATURES_V4:
            encoder = ColumnTransformer([('c', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [name])])
            x_train, x_test = encoder.fit_transform(train[[name]].astype(str)), encoder.transform(test[[name]].astype(str))
        else:
            x_train = train[[name]].astype(float).to_numpy()
            x_test = test[[name]].astype(float).to_numpy()
        model = HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=0).fit(x_train, train.true_cause)
        result[name] = float((model.predict(x_test) == test.true_cause.to_numpy()).mean())
    return result


def _clean(value):
    """JSON-safe copy: numpy scalars become Python, NaN/inf become None, tuples/sets become lists."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(_clean(payload), indent=2, allow_nan=False))
