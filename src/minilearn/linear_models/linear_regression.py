from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from minilearn.base import BaseEstimator


class LinearRegression(BaseEstimator):
    def __init__(self, lr: float = 0.01, epochs: int = 1000) -> None:
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape

        self.weights_: NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)
        self.bias_: float = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights_ + self.bias_
            error = y_pred - y

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.weights_ -= self.lr * dw
            self.bias_ -= self.lr * db

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights_ + self.bias_
