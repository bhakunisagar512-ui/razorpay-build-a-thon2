"""Held-out per-class sigmoid/isotonic calibration with probability normalization."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method, classes):
        if method not in ('uncalibrated', 'sigmoid', 'isotonic'):
            raise ValueError(method)
        self.method, self.classes = method, tuple(classes)
        self.models = []

    def fit(self, probabilities, labels):
        self.models = []
        if self.method == 'uncalibrated':
            return self
        for i, cause in enumerate(self.classes):
            target = (np.asarray(labels) == cause).astype(int)
            if len(set(target)) != 2:
                raise ValueError(f'Calibration partition lacks positive/negative examples for {cause}')
            if self.method == 'sigmoid':
                model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
                model.fit(self._logit(probabilities[:, i]).reshape(-1, 1), target)
            else:
                model = IsotonicRegression(out_of_bounds='clip').fit(probabilities[:, i], target)
            self.models.append(model)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-8, 1-1e-8)
        return np.log(p/(1-p))

    def transform(self, probabilities):
        if self.method == 'uncalibrated':
            return probabilities.copy()
        if len(self.models) != len(self.classes):
            raise ValueError('Calibrator must be fitted before inference')
        columns = []
        for i, model in enumerate(self.models):
            columns.append(model.predict_proba(self._logit(probabilities[:, i]).reshape(-1, 1))[:, 1]
                           if self.method == 'sigmoid' else model.predict(probabilities[:, i]))
        result = np.column_stack(columns)
        totals = result.sum(axis=1, keepdims=True)
        # Preserve input probabilities when every OVR calibrator predicts zero.
        return np.divide(result, totals, out=probabilities.copy(), where=totals > 0)


def expected_calibration_error(probability, correct, bins=10):
    probability, correct = np.asarray(probability), np.asarray(correct)
    if not len(probability):
        return None
    indices = np.minimum((probability*bins).astype(int), bins-1)
    error = 0.
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean()-correct[mask].mean())
    return float(error)


def calibration_metrics(probabilities, labels, classes):
    labels = np.asarray(labels)
    truth = np.column_stack([labels == cause for cause in classes]).astype(float)
    hard_indices = [i for i, cause in enumerate(classes) if cause in ('CARD_EXPIRED', 'RISK_BLOCKED')]
    hard_probability = probabilities[:, hard_indices].sum(axis=1)
    hard_truth = np.isin(labels, ['CARD_EXPIRED', 'RISK_BLOCKED']).astype(float)
    return {'multiclass_brier': float(np.mean(np.sum((probabilities-truth)**2, axis=1))),
            'hard_decline_brier': float(np.mean((hard_probability-hard_truth)**2)),
            'top_label_ece_10_bins': expected_calibration_error(probabilities.max(axis=1),
                                                              np.asarray(classes)[probabilities.argmax(axis=1)] == labels),
            'hard_decline_ece_10_bins': expected_calibration_error(hard_probability, hard_truth)}
