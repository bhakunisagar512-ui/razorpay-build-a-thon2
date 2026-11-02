"""Shared replay/packaging only; training and stress worlds define their own dynamics."""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from services.app.domain.features import FEATURE_SCHEMA_VERSION, FeatureBuilder, PaymentObservation


@dataclass
class SyntheticDataset:
    transactions: pd.DataFrame  # successes + failures; one row per payment attempt
    failures: pd.DataFrame     # unique failure snapshots + labels
    deliveries: pd.DataFrame   # duplicates + delayed webhook deliveries
    outcomes: pd.DataFrame    # later observed success on the same order; not causal
    infrastructure: pd.DataFrame  # latent episode truth; never a model input
    metadata: dict


def assemble(transactions, infrastructure, *, name, seed, duplicate_rate=.04, include_candidates=False):
    frame = pd.DataFrame(transactions).sort_values(['observed_at', 'payment_id']).reset_index(drop=True)
    if 'is_warmup' not in frame:
        frame['is_warmup'] = False
    if 'instrument_id' not in frame:
        frame['instrument_id'] = 'unknown'
    builder = FeatureBuilder(include_candidates=include_candidates)
    observation_fields = [f.name for f in fields(PaymentObservation)]
    failures = []
    for row in frame.to_dict('records'):
        values = {f: row[f] for f in observation_fields}
        # Pandas represents mixed numeric/null columns as NaN; normalize back
        # to the shared observation contract rather than invent a latency.
        if pd.isna(values['latency_ms']):
            values['latency_ms'] = None
        current = PaymentObservation(**values)
        if row['status'] == 'failed' and not row['is_warmup']:
            failures.append({**builder.build(current), 'payment_id': row['payment_id'],
                             'order_id': row['order_id'], 'customer_id': row['customer_id'],
                             'amount_minor': row['amount_minor'],
                             'failed_at': row['occurred_at'], 'observed_at': row['observed_at'],
                             'true_cause': row['true_cause'], 'available_rails': row['available_rails'],
                             'consent_whatsapp': row['consent_whatsapp']})
        if row['telemetry_visible']:
            builder.observe(current)
    failures = pd.DataFrame(failures)
    if failures.empty:
        raise ValueError('No failures generated; increase n to build an evaluation dataset')
    deliveries = failures.copy()
    deliveries['is_duplicate'] = False
    rng = np.random.default_rng(seed + 1009)
    duplicates = deliveries.sample(n=int(len(deliveries) * duplicate_rate), random_state=seed).copy()
    duplicates['is_duplicate'] = True
    deliveries = pd.concat([deliveries, duplicates], ignore_index=True)
    deliveries['received_at'] = deliveries['observed_at'] + pd.to_timedelta(rng.exponential(20, len(deliveries)), unit='s')
    deliveries = deliveries.sort_values(['received_at', 'payment_id']).reset_index(drop=True)
    outcomes = []
    observation_end = frame.observed_at.max()
    by_order = {}
    for row in frame[~frame.is_warmup].sort_values(['occurred_at', 'payment_id']).to_dict('records'):
        by_order.setdefault(row['order_id'], []).append(row)
    for order_id, group in by_order.items():
        failed = [row for row in group if row['status'] == 'failed']
        if not failed:
            continue
        first = failed[0]
        successes = [row for row in group if row['status'] == 'success' and row['occurred_at'] > first['occurred_at']]
        success = successes[0] if successes else None
        outcomes.append({'order_id': order_id, 'first_failed_payment_id': first['payment_id'],
                         'later_success_observed': bool(success),
                         'recovered_minor': int(first['amount_minor']) if success else 0,
                         'success_payment_id': success['payment_id'] if success else None,
                         'outcome_available_at': success['observed_at'] if success else None,
                         'observation_end': observation_end,
                         'right_censored': not bool(success), 'action_source': 'customer_retry'})
    return SyntheticDataset(frame, failures, deliveries, pd.DataFrame(outcomes),
                            pd.DataFrame(infrastructure),
                            {'generator': name, 'seed': seed, 'feature_schema': FEATURE_SCHEMA_VERSION,
                             'model_features': list(builder.feature_names),
                             'n_attempts': len(frame), 'n_failures': len(failures),
                             'n_warmup_attempts': int(frame.is_warmup.sum()),
                             'n_measurement_attempts': int((~frame.is_warmup).sum()),
                             'n_successes': int((frame.status == 'success').sum()),
                             'duplicate_deliveries': len(duplicates), 'units': 'INR minor units (paise)',
                             'synthetic_only': True})


def temporal_split(dataset: SyntheticDataset, embargo_s=600):
    """60/20/20 arrival-time boundaries, with an embargo and whole-order exclusion.

    An order crossing a boundary is dropped rather than share correlated attempts
    across partitions. Never use payment-ID hash splits for sequential datasets.
    """
    frame = dataset.failures
    measurement = dataset.transactions[~dataset.transactions.is_warmup]
    clock = measurement.observed_at
    first, second = clock.quantile(.6), clock.quantile(.8)
    gap = pd.Timedelta(seconds=embargo_s)
    bounds = measurement.groupby('order_id').observed_at.agg(['min', 'max'])
    orders = {
        'train': bounds.index[bounds['max'] < first - gap],
        'validation': bounds.index[(bounds['min'] >= first + gap) & (bounds['max'] < second - gap)],
        'test': bounds.index[bounds['min'] >= second + gap],
    }
    return {name: frame[frame.order_id.isin(ids)].copy() for name, ids in orders.items()}


def safety_split(dataset, embargo_s=600):
    """Disjoint 45/15/20/20 fit/calibration/policy/confirmation time windows."""
    start = pd.Timestamp(dataset.metadata['measurement_start'])
    end = pd.Timestamp(dataset.metadata['measurement_end'])
    clock = [start, start + (end-start)*.45, start + (end-start)*.60,
             start + (end-start)*.80, end]
    measurement = dataset.transactions[~dataset.transactions.is_warmup]
    bounds = measurement.groupby('order_id').observed_at.agg(['min', 'max'])
    result = {}
    for i, name in enumerate(('fit', 'calibration', 'policy', 'confirmation')):
        lo = clock[i] + (pd.Timedelta(seconds=embargo_s) if i else pd.Timedelta(0))
        hi = clock[i+1] - (pd.Timedelta(seconds=embargo_s) if i < 3 else pd.Timedelta(0))
        orders = bounds.index[(bounds['min'] >= lo) & (bounds['max'] < hi)]
        result[name] = dataset.failures[dataset.failures.order_id.isin(orders)].copy()
    return result
