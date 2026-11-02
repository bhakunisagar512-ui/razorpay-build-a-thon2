"""Summarise NPCI UPI remitter-bank decline statistics (BD = business, TD = technical).

Input: raw NPCI "Top 50 Member Performance" API responses as republished by the
third-party upi-pulse project (eval/data/upi_pulse_raw/). NPCI's own site blocked
automated download (HTTP 403), so this is NOT a direct NPCI download.
UPI remitter banks only: nothing here describes cards, netbanking or other UPI sides.

Run (inside the container; read-only on the raw files):
    python -I eval/npci_stats.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path('eval/data/upi_pulse_raw')
OUT = Path('eval/data/npci_upi_summary.json')
MONTHS = {m: i for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}


def pct(value):
    return float(str(value).replace('%', '').replace(',', ''))


def load():
    rows = []
    batch = json.loads((RAW / 'top50_batch1_remitter_2022-01_to_2024-10.json').read_text())
    for item in batch['items']:
        if item.get('http') != 200 or item['type'] != 'remitter':
            continue
        for r in item['body']['data']['results']:
            rows.append((int(item['year']), item['month'], r))
    single = json.loads((RAW / 'top50_remitter_2026_Sep.json').read_text())
    for r in single['data']['results']:
        rows.append((int(r['year']), r['month'], r))
    frame = pd.DataFrame([{
        'year': y, 'month': MONTHS[m], 'bank': r['upi_remitter_banks'],
        'volume_mn': pct(r['total_volume_in_mn']), 'approved': pct(r['approved_percent']),
        'bd': pct(r['bd_percent']), 'td': pct(r['td_percent'])} for y, m, r in rows])
    frame['period'] = pd.PeriodIndex.from_fields(year=frame.year, month=frame.month, freq='M')
    return frame.sort_values(['bank', 'period']).reset_index(drop=True)


def weighted(frame, column):
    return float((frame[column] * frame.volume_mn).sum() / frame.volume_mn.sum())


def summarize(frame):
    frame = frame.copy()
    total = frame[['approved', 'bd', 'td']].sum(axis=1)
    # Data check carried over from the source project: the three shares add to ~100%.
    assert (total.between(98, 102)).mean() > .99, 'approved+BD+TD should sum to ~100%'
    yearly = {int(y): {'rows': int(len(g)), 'periods': int(g.period.nunique()),
                       'volume_weighted_approved_pct': round(weighted(g, 'approved'), 3),
                       'volume_weighted_bd_pct': round(weighted(g, 'bd'), 3),
                       'volume_weighted_td_pct': round(weighted(g, 'td'), 3)}
              for y, g in frame.groupby('year')}
    recent = frame[frame.period >= frame.period.max() - 11] if len(frame) else frame
    latest = frame[frame.period == frame.period.max()]
    quantiles = [.05, .25, .5, .75, .95]
    dispersion = {name: {'bd_pct': {str(q): round(float(sub.bd.quantile(q)), 3) for q in quantiles},
                         'td_pct': {str(q): round(float(sub.td.quantile(q)), 3) for q in quantiles},
                         'bank_months': int(len(sub))}
                  for name, sub in (('all_periods', frame), ('latest_month', latest))}
    # Spikes: a bank's TD in a month relative to its own median over earlier months.
    ratios, absolute = [], []
    for _, g in frame.groupby('bank'):
        g = g.sort_values('period')
        history = g.td.expanding().median().shift(1)
        keep = history.notna() & (history > 0)
        ratios += (g.td[keep] / history[keep]).tolist()
        absolute += (g.td[keep] - history[keep]).tolist()
    ratios, absolute = np.array(ratios), np.array(absolute)
    spikes = {'bank_months': int(len(ratios)),
              'td_vs_own_prior_median_ratio_quantiles': {str(q): round(float(np.quantile(ratios, q)), 3)
                                                         for q in (.5, .9, .95, .99)},
              'fraction_ratio_above_2': round(float((ratios > 2).mean()), 4),
              'fraction_abs_increase_above_1pt': round(float((absolute > 1).mean()), 4)}
    concentration = frame.groupby('period').apply(
        lambda g: g.nlargest(5, 'volume_mn').volume_mn.sum() / g.volume_mn.sum()).mean()
    return {
        'source': 'NPCI UPI Ecosystem Statistics, Top 50 Member Performance (remitter banks), via third-party '
                  'upi-pulse raw NPCI API responses; NPCI site returned HTTP 403 to automated download.',
        'scope': 'UPI remitter (payer) banks only. Not applicable to cards, netbanking, or UPI beneficiary side.',
        'periods': {'first': str(frame.period.min()), 'last': str(frame.period.max()),
                    'count': int(frame.period.nunique()),
                    'missing_between': [str(p) for p in pd.period_range(frame.period.min(), frame.period.max(), freq='M')
                                        if p not in set(frame.period.unique())]},
        'by_year': yearly, 'bank_dispersion': dispersion, 'technical_decline_spikes': spikes,
        'top5_volume_share_mean': round(float(concentration), 3),
        'caveats': ['Percentages are of all payer-bank transaction attempts, not of Razorpay merchant traffic.',
                    'Top 50 banks only; excludes the long tail.',
                    'Monthly aggregates hide intra-day outages, so outage duration/severity remain assumptions.',
                    'Business declines include user errors, balance and limit failures; the split into our six '
                    'root causes is an assumption.']}


if __name__ == '__main__':
    summary = summarize(load())
    OUT.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
