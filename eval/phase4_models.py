"""Phase 4 model families behind one interface: HistGradientBoosting, XGBoost, CatBoost, Random Forest.

Shared preprocessing rules (identical for every family):
  * categoricals are strings; a missing value or a column that is absent from the request is the
    explicit level "missing"; a level never seen in training is tolerated (one-hot ignores it,
    CatBoost treats it as a new category);
  * numerics are coerced to float; unparsable and infinite values become NaN; NaN is passed to the
    model unchanged (no imputation), so each family uses its own native missing-value handling;
  * labels are the six root causes, classes sorted alphabetically.
Per-family encoding: HGB / XGBoost / Random Forest use one-hot (the existing recipe); CatBoost uses
its native ordered target statistics for categoricals, which is how it is meant to be used.
No class weighting anywhere: probabilities stay faithful to the training prevalence.

Nothing here is imported by the serving path.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from services.app.domain.features_v4 import CATEGORICAL_FEATURES_V4

FAMILIES = ('hgb', 'xgboost', 'catboost', 'random_forest')
FAMILY_LABEL = {'hgb': 'HistGradientBoosting', 'xgboost': 'XGBoost', 'catboost': 'CatBoost',
                'random_forest': 'Random Forest'}
# Simplicity tie-break used by the frozen selection rule (existing dependency first).
SIMPLICITY_RANK = {'hgb': 0, 'random_forest': 1, 'xgboost': 2, 'catboost': 3}
SEED = 0
MAX_THREADS = 6   # CPU cap for fitting and inference (laptop heat); also enforced by a Docker quota, see scripts/run_phase4.ps1

SEARCH_SPACE = {
    'hgb': {'learning_rate': [.03, .05, .1, .2], 'max_iter': [100, 200, 400], 'max_leaf_nodes': [15, 31, 63],
            'min_samples_leaf': [10, 20, 50, 100], 'l2_regularization': [0., .1, 1., 10.]},
    'xgboost': {'n_estimators': [100, 200, 400], 'learning_rate': [.03, .05, .1, .2], 'max_depth': [3, 4, 6, 8],
                'min_child_weight': [1, 5, 10], 'subsample': [.7, .85, 1.], 'colsample_bytree': [.5, .7, 1.],
                'reg_lambda': [1., 5., 20.]},
    'catboost': {'iterations': [200, 400, 800], 'learning_rate': [.03, .05, .1, .2], 'depth': [4, 6, 8],
                 'l2_leaf_reg': [1., 3., 10., 30.]},
    'random_forest': {'n_estimators': [200, 400], 'max_depth': [None, 12, 20], 'min_samples_leaf': [1, 3, 5, 10],
                      'max_features': ['sqrt', .3, .5]},
}
# Trial 0 of every family is its conventional starting point; for HGB it is exactly the existing recipe.
DEFAULTS = {
    'hgb': {'max_iter': 200},
    'xgboost': {'n_estimators': 200, 'learning_rate': .1, 'max_depth': 6},
    'catboost': {'iterations': 500, 'learning_rate': .1, 'depth': 6},
    'random_forest': {'n_estimators': 300},
}
TRIALS_PER_FAMILY = 10


def trial_params(family, n_trials=TRIALS_PER_FAMILY, seed=SEED):
    """Trial 0 = default; trials 1.. = random search drawn from a fixed stream (same budget per family)."""
    rng = np.random.default_rng(seed)
    space = SEARCH_SPACE[family]
    trials, seen = [dict(DEFAULTS[family])], {tuple(sorted(DEFAULTS[family].items(), key=str))}
    while len(trials) < n_trials:
        draw = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(sorted(draw.items(), key=str))
        if key not in seen:
            seen.add(key)
            trials.append(draw)
    return trials


class BenchModel:
    """fit(frame) / predict_proba(frame) over a feature tuple; frame must carry `true_cause` to fit."""

    def __init__(self, family, features, params=None, threads=MAX_THREADS):
        if family not in FAMILIES:
            raise ValueError(f'Unknown model family {family!r}')
        self.family, self.features = family, tuple(features)
        self.params = dict(DEFAULTS[family] if params is None else params)
        self.threads = threads
        self.categorical = [f for f in self.features if f in CATEGORICAL_FEATURES_V4]
        self.numeric = [f for f in self.features if f not in CATEGORICAL_FEATURES_V4]
        self.classes_ = []
        self.known_tuples = set()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.set_inference_threads(MAX_THREADS)   # models reloaded from checkpoints also honour the CPU cap

    # ---- shared preprocessing ------------------------------------------------
    def _prep(self, frame):
        data = pd.DataFrame(index=frame.index)
        for name in self.categorical:
            column = frame[name] if name in frame.columns else pd.Series('missing', index=frame.index)
            data[name] = column.astype(object).where(column.notna(), 'missing').astype(str)
        for name in self.numeric:
            column = frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)
            data[name] = pd.to_numeric(column, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)
        return data[list(self.features)]

    def _build(self):
        p, t = self.params, self.threads
        if self.family == 'hgb':
            return HistGradientBoostingClassifier(random_state=SEED, **p)
        if self.family == 'random_forest':
            return RandomForestClassifier(random_state=SEED, n_jobs=t, **p)
        if self.family == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(objective='multi:softprob', tree_method='hist', random_state=SEED, n_jobs=t,
                                 verbosity=0, **p)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function='MultiClass', random_seed=SEED, thread_count=t, verbose=0,
                                  allow_writing_files=False, **p)

    def _encoder(self):
        parts = []
        if self.categorical:
            parts.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical))
        if self.numeric:
            parts.append(('numeric', 'passthrough', self.numeric))
        return ColumnTransformer(parts)

    # ---- fit / predict -------------------------------------------------------
    def fit(self, frame):
        y = frame.true_cause.astype(str).to_numpy()
        self.classes_ = sorted(set(y))
        codes = np.searchsorted(self.classes_, y)
        data = self._prep(frame)
        self.model = self._build()
        if self.family == 'catboost':
            self.encoder = None
            self.model.fit(data, codes, cat_features=self.categorical)
        else:
            self.encoder = self._encoder()
            self.model.fit(self.encoder.fit_transform(data), codes)
        self.known_tuples = set(frame[['gw_error_code', 'gw_error_source', 'gw_error_step']]
                                .itertuples(index=False, name=None))
        return self

    def predict_proba(self, frame):
        data = self._prep(frame)
        if self.family == 'catboost':
            p = self.model.predict_proba(data, thread_count=self.threads)
        else:
            p = self.model.predict_proba(self.encoder.transform(data))
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 0., None)
        return p / p.sum(axis=1, keepdims=True)

    def set_inference_threads(self, n):
        """Single-request serving runs on one thread; used by the latency profile."""
        self.threads = n
        if self.family in ('xgboost', 'random_forest'):
            self.model.set_params(n_jobs=n)   # CatBoost reads self.threads at predict time; HGB via threadpoolctl
        return self
