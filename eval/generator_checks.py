"""Fail-fast coverage and history checks for the corrected training world."""
import pandas as pd
from eval.datasets import safety_split
from eval.investigate_features import audit_generator


def verify_generator(dataset):
    audit = audit_generator(dataset)
    start = pd.Timestamp(dataset.metadata['measurement_start'])
    end = pd.Timestamp(dataset.metadata['measurement_end'])
    events = dataset.transactions
    measured = events[~events.is_warmup]
    historical = events[events.is_warmup & events.telemetry_visible & events.observed_at.lt(start)]
    instrument_counts = historical.groupby('instrument_id').size().reindex(measured.instrument_id.unique(), fill_value=0)
    customer_counts = historical.groupby('customer_id').size().reindex(measured.customer_id.unique(), fill_value=0)
    assert measured.occurred_at.max() >= end-pd.Timedelta(seconds=1), 'Incomplete horizon'
    assert all(episode['inside']['n'] > 0 for episode in audit['episodes']), 'Unrepresented episode'
    assert instrument_counts.min() > 0, 'Instrument missing warm-up history'
    assert customer_counts.min() > 0, 'Customer missing warm-up history'
    assert set(historical.payment_id).isdisjoint(dataset.failures.payment_id), 'Warm-up label leakage'
    coverage = {}
    for name, frame in safety_split(dataset).items():
        counts = frame.true_cause.value_counts().to_dict()
        assert counts.get('TD_ISSUER_DOWN', 0) > 0 and counts.get('TD_PSP_TIMEOUT', 0) > 0, f'Missing technical scenario in {name}'
        coverage[name] = counts
    return {'passed': True, **audit, 'class_counts_by_split': coverage,
            'minimum_customer_history_attempts': int(customer_counts.min()),
            'median_customer_history_attempts': float(customer_counts.median()),
            'minimum_instrument_history_attempts': int(instrument_counts.min()),
            'median_instrument_history_attempts': float(instrument_counts.median()),
            'measurement_customer_count': int(len(customer_counts)),
            'measurement_instrument_count': int(len(instrument_counts))}
