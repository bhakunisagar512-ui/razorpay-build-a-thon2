"""Phase 4 safety investigation: rebuild every hard-decline failure with the model's probabilities and the
frozen gate's decision, from the saved Phase 4 checkpoints (nothing is retrained, nothing is overwritten).

    python -m eval.phase4_safety_cases --run-dir eval/data/phase4_run --out eval/data/phase4_safety

Writes cases.pkl (one row per failure x split x config x family; `hard` marks hard declines) used by the analysis and by
the fixed-policy evaluation. "Unsafe retry" is exactly the benchmark's definition (eval.safety_pipeline.metrics):
true cause in {CARD_EXPIRED, RISK_BLOCKED} and the gated decision allows a retry (RETRYABLE cause).
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eval import phase4
from eval.phase4_confidence import GATE, HARD
from eval.safety_pipeline import apply_policy, prepare

CONFIGS = ('deployment_core', 'deployment_with_timing')


def split_frames(run_dir):
    """validation = Phase 3 validation worlds (model selection data); test = the fresh Phase 4 final-test worlds."""
    data = phase4.load_phase3_roles()
    validation = phase4.pooled(data, 'validation')
    fresh, _, _ = phase4.load_final(Path(run_dir) / 'cache', 4, phase4.FULL)
    return {'validation': validation, 'test': phase4.pooled(fresh, 'final_test'),
            'shifted': phase4.pooled(fresh, 'final_shift')}


def build(run_dir, out_dir):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((run_dir / 'selection.json').read_text())
    method_of = selection['calibration_method_by_family']
    frames = split_frames(run_dir)
    rows = []
    for config in CONFIGS:
        for family in phase4.FAMILIES:
            bundle = joblib.load(run_dir / 'models' / f'{config}__{family}.joblib')
            model = bundle['model']
            calibrator = bundle['calibrators'][method_of[family]]
            for split, frame in frames.items():
                p = calibrator.transform(model.predict_proba(frame))
                data = prepare(frame, model, p)
                cause, reason, retry = apply_policy(data, GATE)
                classes = np.asarray(model.classes_)
                hard = frame.true_cause.isin(HARD).to_numpy()
                keep = np.ones(len(frame), dtype=bool)   # all failures: recall and abstention need the retryable ones too
                part = frame.loc[keep, ['payment_id', 'scenario', 'world_key', 'true_cause', 'n_causes_fired', 'method',
                                         'error_reason', 'reason_family', 'is_generic_reason', 'gw_error_code', 'gw_error_source',
                                         'gw_error_step', 'instrument_hint', 'issuer_id', 'instrument_consecutive_failures',
                                         'instrument_prev_reason', 'attempt_duration_s']].copy()
                part['hard'] = hard[keep]
                part['split'], part['config'], part['family'] = split, config, family
                part['top'] = data['top'][keep]
                part['confidence'] = data['confidence'][keep]
                part['hard_mass'] = data['hard_mass'][keep]
                part['gate_reason'] = np.asarray(reason)[keep]
                part['gate_cause'] = np.asarray(cause)[keep]
                part['retry_allowed'] = retry[keep]
                for i, c in enumerate(classes):
                    part[f'p_{c}'] = p[keep][:, i]
                rows.append(part)
    cases = pd.concat(rows, ignore_index=True)
    cases['unsafe'] = cases.retry_allowed & cases.hard
    with (out_dir / 'cases.pkl').open('wb') as f:
        pickle.dump(cases, f)
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='eval/data/phase4_run')
    parser.add_argument('--out', default='eval/data/phase4_safety')
    args = parser.parse_args()
    c = build(args.run_dir, args.out)
    print(c.groupby(['split', 'config', 'family']).agg(rows=('hard', 'size'), hard=('hard', 'sum'), unsafe=('unsafe', 'sum')).to_string())
