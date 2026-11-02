"""Package the frozen Phase 4 winner as a deployable-but-inactive artifact.

    python -m eval.phase4_package --run-dir eval/data/phase4_run --out eval/artifacts/phase4

Writes model.joblib (preprocessing + model + chosen calibrator), schema.json (feature schema, class order,
missing/unseen-category policy, safety gate, provenance, integrity hash). Does not touch the serving path.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn

from services.app.domain.deployment_features import AVAILABILITY, DEPLOYMENT_CORE
from services.app.domain.features_v4 import CATEGORICAL_FEATURES_V4


def package(run_dir, out_dir, family=None):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    selection = json.loads((run_dir / 'selection.json').read_text())
    family = family or selection['winner']
    method = selection['calibration_method_by_family'][family]
    bundle = joblib.load(run_dir / 'models' / f'deployment_core__{family}.joblib')
    model, calibrator = bundle['model'], bundle['calibrators'][method]
    if tuple(model.features) != tuple(DEPLOYMENT_CORE):
        raise ValueError('Packaged model is not the deployment-compatible feature set')
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'calibrator': calibrator}, out_dir / 'model.joblib', compress=3)
    digest = hashlib.sha256((out_dir / 'model.joblib').read_bytes()).hexdigest()
    libs = {'scikit-learn': sklearn.__version__, 'numpy': np.__version__, 'python': platform.python_version()}
    for name in ('xgboost', 'catboost'):
        if family == name:
            libs[name] = __import__(name).__version__
    schema = {
        'model_id': f'phase4-{family}-deployment_core-{method}-{digest[:8]}',
        'family': family, 'calibration': method, 'activated': False, 'synthetic_only': True,
        'created_unix': time.time(), 'libraries': libs, 'model_sha256': digest,
        'features': list(model.features), 'categorical': list(model.categorical), 'numeric': list(model.numeric),
        'classes': list(model.classes_), 'hyperparameters': model.params,
        'feature_availability': {f: AVAILABILITY[f]['class'] for f in model.features},
        'missing_policy': {'categorical': 'absent or null -> "missing"', 'numeric': 'absent, null, non-numeric or infinite -> NaN '
                           '(native handling, no imputation)'},
        'unseen_category_policy': 'one-hot ignores unseen levels (all-zero block); CatBoost treats them as new categories',
        'safety_gate': {'min_confidence': .60, 'max_hard_mass': .03, 'source': 'frozen Phase 2.5 control, unchanged'},
        'selection': {'winner': selection['winner'], 'rule': selection['rule'], 'trace': selection['selection_trace']},
        'training': 'Phase 3 sequences@v4 train worlds (4 per scenario x calm/typical/stressed, pooled); calibrated on '
                    'separate calibration worlds',
        'categorical_set': list(CATEGORICAL_FEATURES_V4),
        'note': 'Trained on simulated data only. Do not activate without validation on real Razorpay traffic.'}
    (out_dir / 'schema.json').write_text(json.dumps(schema, indent=2))
    return schema


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='eval/data/phase4_run')
    parser.add_argument('--out', default='eval/artifacts/phase4')
    parser.add_argument('--family', default=None, help='default: the frozen winner')
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.out, args.family)['model_id']))
