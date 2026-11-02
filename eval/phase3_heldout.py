"""Held-out (Phase 4 final-test) data access shared by the dataset builder and the calibration analyses.

Held-out data = the Phase 4 final-test worlds (seeds 500000+, never used for training, tuning or model selection). The three-cycle
journey, offline and Temporal stages that used to live here belonged to the retired recovery architecture and were removed; their
results stay in eval/data/phase3_heldout and docs/phase3*.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from services.app.adapters import diagnosis_v4
from services.app.domain import guards
from services.app.domain.safety_policy_v2 import decide_v2
from services.app.domain.taxonomy import Cause

CACHE = Path('eval/data/phase4_run/cache')
TRAIN_CACHE = Path('eval/data/phase3_run/cache')
HARD = ('CARD_EXPIRED', 'RISK_BLOCKED')
RETRY = ('silent_retry', 'rail_switch')
DEFAULT_CONSENT = {'whatsapp': True, 'sms': False, 'email': False}      # same defaults as ModelCFailure
RAILS = 'upi_intent,card,netbanking'
KINDS = ('SIMULATED_RECOVERY_SUCCESS', 'PLAN_INTERVENTION', 'DIAGNOSIS_INTERVENTION', 'SAFETY_ESCALATION')


def load_final():
    parts = []
    for path in sorted(CACHE.glob('*_final_test_*.pkl')):
        d = pickle.load(path.open('rb'))
        parts.append(d['failures'].assign(world_file=path.stem, world_seed=d['seed']))
    return pd.concat(parts, ignore_index=True)


def verify_split(frame):
    """Held-out evidence: no payment id and no world seed shared with any Phase 3 train/validation/test cache."""
    ids, seeds = set(), set()
    for path in sorted(TRAIN_CACHE.glob('*.pkl')):
        d = pickle.load(path.open('rb'))
        ids |= set(d['failures'].payment_id)
        seeds.add(d.get('seed'))
    return {'final_test_failures': len(frame), 'final_test_worlds': int(frame.world_file.nunique()),
            'final_test_seeds': sorted(int(s) for s in frame.world_seed.unique()),
            'earlier_cache_files': len(list(TRAIN_CACHE.glob('*.pkl'))), 'payment_id_overlap_with_train_validation_test': len(ids & set(frame.payment_id)),
            'world_seed_overlap': sorted(set(int(s) for s in frame.world_seed.unique()) & {s for s in seeds if s is not None})}


def probabilities(frame):
    art = diagnosis_v4.bundle()
    model = art['model']
    return art, np.asarray(model.classes_), art['calibrator'].transform(model.predict_proba(frame))


def first_cycle_cause(art, row, p, classes):
    from eval.classifier import RULES
    code, src, step = row['gw_error_code'], row['gw_error_source'], row['gw_error_step']
    rule = RULES.get((code, src, step))
    verdict = decide_v2(dict(zip(classes, p)), rule_result=str(rule) if rule is not None else None, instrument_hint=row['instrument_hint'],
                        known_tuple=(code, src, step) in art['model'].known_tuples, error_reason=_none(row['error_reason']),
                        instrument_prev_reason=_none(row['instrument_prev_reason']))
    return verdict['cause']


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v in ('missing', 'none') else v
