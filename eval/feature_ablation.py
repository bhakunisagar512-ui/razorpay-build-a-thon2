"""Offline-only feature experiments. The serving CauseClassifier is unchanged.

Same boosting hyperparameters, rule precedence, and safety thresholds as baseline;
only the input columns/preprocessing vary. Numeric missing values remain NaN for
HistGradientBoosting, and unseen categorical values are ignored by the encoder.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

from eval.classifier import CONF_THRESHOLD, HARD_CAUSES, HARD_DECLINE_MASS_LIMIT, RULES
from services.app.domain.features import BASELINE_FEATURES, CATEGORICAL_FEATURES, FEATURES
from services.app.domain.taxonomy import Cause

INFRASTRUCTURE = tuple(f for f in FEATURES if f.startswith(('issuer_', 'psp_', 'method_')) and f not in CATEGORICAL_FEATURES) + ('current_latency_ms',)
HISTORY = tuple(f for f in FEATURES if f.startswith(('customer_', 'order_')) or f == 'amount_to_customer_mean_7d')
CONTEXT = ('amount_minor', 'log_amount_minor', 'hour_ist', 'weekday_ist')
FEATURE_GROUPS = {
    'baseline_5': BASELINE_FEATURES,
    'plus_infrastructure': (*BASELINE_FEATURES, *INFRASTRUCTURE),
    'plus_history': (*BASELINE_FEATURES, *HISTORY),
    'plus_context': (*BASELINE_FEATURES, *CONTEXT),
    'all_without_identifiers': tuple(f for f in FEATURES if f not in ('issuer_id', 'psp_id')),
    'all_26': FEATURES,
}


class FeatureAblationClassifier:
    def __init__(self, features=None):
        if features is None:
            from services.app.domain.selected_features import MODEL_FEATURES
            features = MODEL_FEATURES
        self.features = tuple(features)
        categorical = [f for f in features if f in CATEGORICAL_FEATURES]
        numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
        self.encoder = ColumnTransformer([
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('numeric', 'passthrough', numeric),
        ])
        self.model = HistGradientBoostingClassifier(max_iter=200, random_state=0)

    def fit(self, train):
        self.model.fit(self.encoder.fit_transform(train[list(self.features)]), train['true_cause'])
        self.classes_ = list(self.model.classes_)
        return self

    def predict_df(self, frame):
        probabilities = self.model.predict_proba(self.encoder.transform(frame[list(self.features)]))
        hard_idx = [i for i, cause in enumerate(self.classes_) if cause in HARD_CAUSES]
        hard_mass = probabilities[:, hard_idx].sum(axis=1)
        predicted, confidence, stage = [], [], []
        for i, row in enumerate(frame.itertuples(index=False)):
            hit = RULES.get((row.gw_error_code, row.gw_error_source, row.gw_error_step))
            hit = str(hit) if hit is not None else None
            top = int(np.argmax(probabilities[i]))
            cause, conf = self.classes_[top], float(probabilities[i, top])
            if hit in HARD_CAUSES:
                result = hit, 1., 'rules'
            elif hit is not None:
                result = ((str(Cause.AMBIGUOUS), 1 - hard_mass[i], 'gated:rules_hard_risk')
                          if hard_mass[i] > HARD_DECLINE_MASS_LIMIT else (hit, 1 - hard_mass[i], 'rules'))
            elif conf < CONF_THRESHOLD:
                result = str(Cause.AMBIGUOUS), conf, 'gated:low_conf'
            elif cause not in HARD_CAUSES and hard_mass[i] > HARD_DECLINE_MASS_LIMIT:
                result = str(Cause.AMBIGUOUS), conf, 'gated:hard_decline_risk'
            else:
                result = cause, conf, 'model'
            predicted.append(result[0]); confidence.append(result[1]); stage.append(result[2])
        out = frame.copy()
        out['pred_cause'], out['pred_conf'], out['pred_stage'] = predicted, confidence, stage
        return out
