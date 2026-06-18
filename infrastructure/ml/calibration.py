"""Probability calibration using isotonic regression.

Ensures that when the model predicts "60% probability", it wins ~60%
of the time in practice. One calibrator per class (one-vs-rest).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import joblib
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrates 1X2 probabilities with isotonic regression (Platt scaling variant).

    Fits one IsotonicRegression per outcome class using a one-vs-rest
    approach, then normalises calibrated probabilities to sum to 1.0.
    """

    def __init__(self) -> None:
        self._calibrators: dict[int, IsotonicRegression] = {}
        self._is_fitted = False
        self._fitted_at: str | None = None
        self._n_samples_fitted: int = 0

    def fit(
        self,
        y_true: list[int],
        y_prob_matrix: list[list[float]],
    ) -> dict:
        for c in (0, 1, 2):
            y_binary = [1 if y == c else 0 for y in y_true]
            probs_c = [row[c] for row in y_prob_matrix]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(probs_c, y_binary)
            self._calibrators[c] = cal

        self._is_fitted = True
        self._fitted_at = datetime.now().isoformat()
        self._n_samples_fitted = len(y_true)

        logger.info(
            "Calibrator fitted on %d samples (%s)",
            self._n_samples_fitted,
            self._fitted_at,
        )
        return {
            "status": "fitted",
            "n_samples": self._n_samples_fitted,
            "fitted_at": self._fitted_at,
        }

    def transform(self, prob_matrix: list[list[float]]) -> list[list[float]]:
        if not self._is_fitted:
            return prob_matrix

        result_rows: list[list[float]] = []
        for row in prob_matrix:
            calibrated = [float(self._calibrators[c].predict([row[c]])[0]) for c in (0, 1, 2)]
            total = sum(calibrated)
            if total > 0:
                calibrated = [p / total for p in calibrated]
            result_rows.append(calibrated)
        return result_rows

    def transform_single(
        self,
        prob_home: float,
        prob_draw: float,
        prob_away: float,
    ) -> tuple[float, float, float]:
        result = self.transform([[prob_home, prob_draw, prob_away]])
        row = result[0]
        return row[0], row[1], row[2]

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "calibrators": self._calibrators,
                "is_fitted": self._is_fitted,
                "fitted_at": self._fitted_at,
                "n_samples": self._n_samples_fitted,
            },
            path,
        )

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibrator file not found: {path}")
        data = joblib.load(path)
        self._calibrators = data["calibrators"]
        self._is_fitted = data["is_fitted"]
        self._fitted_at = data["fitted_at"]
        self._n_samples_fitted = data["n_samples"]

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
