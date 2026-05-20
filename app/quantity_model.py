"""Reusable sklearn estimators for construction quantity prediction."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class QuantityRatioRegressor(BaseEstimator, RegressorMixin):
    """
    Predict material intensity first, then scale by total built area.

    Construction material quantities are mostly linear with total area, while
    floors, quality, building type, and city explain the per-sqft intensity.
    Training on ratios keeps large projects from dominating the loss.
    """

    def __init__(self, estimator=None, total_area_index: int = 4, min_total_area: float = 1.0):
        self.estimator = estimator
        self.total_area_index = total_area_index
        self.min_total_area = min_total_area

    def fit(self, X, y):
        total_area = self._total_area_column(X)
        ratio_targets = np.asarray(y, dtype=float) / total_area
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, ratio_targets)
        return self

    def predict(self, X):
        total_area = self._total_area_column(X)
        ratios = self.estimator_.predict(X)
        return ratios * total_area

    def _total_area_column(self, X):
        values = np.asarray(X)[:, self.total_area_index].reshape(-1, 1)
        return np.maximum(values.astype(float), self.min_total_area)
