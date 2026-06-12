from __future__ import annotations


class BaseEstimator:
    """Base class for all estimators in minilearn."""

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
