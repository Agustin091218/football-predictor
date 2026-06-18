"""XGBoost predictor for football match outcome classification.

Trains on flat feature vectors built by FeatureEngineer.
Uses TimeSeriesSplit for honest temporal cross-validation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


class XGBoostPredictor:
    """Trains and evaluates an XGBoost classifier for 1X2 match prediction.

    Stores feature names after training to ensure consistent predict() columns.
    Supports model persistence via JSON + UBJSON formats.
    """

    def __init__(self, model_params: dict[str, Any] | None = None) -> None:
        self._params = {**_DEFAULT_PARAMS, **(model_params or {})}
        self._model: XGBClassifier | None = None
        self._feature_names: list[str] | None = None
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Train the model with TimeSeriesSplit cross-validation.

        First runs 5-fold temporal CV for honest evaluation metrics,
        then trains a final model on the complete dataset.

        Args:
            X: Feature matrix (shape: n_samples × n_features).
            y: Target labels (0=home_win, 1=draw, 2=away_win).

        Returns:
            Dictionary with cv_log_loss_mean/std, cv_accuracy_mean/std,
            and n_samples / n_features.
        """
        self._feature_names = list(X.columns)

        tscv = TimeSeriesSplit(n_splits=5)
        fold_log_losses: list[float] = []
        fold_accuracies: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model_fold = XGBClassifier(**self._params)
            model_fold.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred_proba = model_fold.predict_proba(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)

            ll = log_loss(y_val, y_pred_proba, labels=[0, 1, 2])
            acc = accuracy_score(y_val, y_pred)
            fold_log_losses.append(ll)
            fold_accuracies.append(acc)

            logger.debug("Fold %d: log_loss=%.4f accuracy=%.4f", fold_idx + 1, ll, acc)

        # Final model on full dataset
        self._model = XGBClassifier(**self._params)
        self._model.fit(X, y, verbose=False)
        self._is_trained = True

        metrics = {
            "cv_log_loss_mean": float(np.mean(fold_log_losses)),
            "cv_log_loss_std": float(np.std(fold_log_losses)),
            "cv_accuracy_mean": float(np.mean(fold_accuracies)),
            "cv_accuracy_std": float(np.std(fold_accuracies)),
            "n_samples": len(X),
            "n_features": len(X.columns),
        }

        logger.info(
            "Trained XGBoost: acc=%.4f±%.4f, log_loss=%.4f±%.4f (%d samples)",
            metrics["cv_accuracy_mean"],
            metrics["cv_accuracy_std"],
            metrics["cv_log_loss_mean"],
            metrics["cv_log_loss_std"],
            metrics["n_samples"],
        )

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0, 1, or 2).

        Args:
            X: Feature matrix with same columns as training data.

        Returns:
            Array of predicted class indices.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        self._check_trained()
        if self._feature_names:
            X = X[self._feature_names]
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(home_win), P(draw), P(away_win)].

        Args:
            X: Feature matrix with same columns as training data.

        Returns:
            Array of shape (n_samples, 3) with row-wise probabilities.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        self._check_trained()
        if self._feature_names:
            X = X[self._feature_names]
        return self._model.predict_proba(X)  # type: ignore[union-attr]

    def predict_single(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict a single match from a feature dict.

        Args:
            features: Flat feature dict from FeatureEngineer.

        Returns:
            Dict with probabilities, predicted class, and confidence.
        """
        self._check_trained()
        row = pd.DataFrame([features])
        if self._feature_names:
            row = row[self._feature_names]
        row = row.fillna(0.0)

        proba = self._model.predict_proba(row)[0]  # type: ignore[union-attr]
        pred_class = int(np.argmax(proba))
        confidence = float(np.max(proba))

        result_labels = {0: "home_win", 1: "draw", 2: "away_win"}

        return {
            "prob_home_win": float(proba[0]),
            "prob_draw": float(proba[1]),
            "prob_away_win": float(proba[2]),
            "predicted_result": result_labels.get(pred_class, "draw"),
            "predicted_class": pred_class,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate the trained model on a test set.

        Args:
            X: Test feature matrix.
            y: Test target labels.

        Returns:
            Dict with accuracy and log_loss.
        """
        self._check_trained()
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X)

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "log_loss": float(log_loss(y, y_pred_proba, labels=[0, 1, 2])),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores (gain-based)."""
        self._check_trained()
        if self._feature_names is None:
            return {}
        scores = self._model.feature_importances_  # type: ignore[union-attr]
        return dict(zip(self._feature_names, scores, strict=False))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save the trained model to disk (JSON format).

        Args:
            path: File path for the model artifact.
        """
        self._check_trained()
        self._model.save_model(path)  # type: ignore[union-attr]

        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "feature_names": self._feature_names,
                    "params": {k: v for k, v in self._params.items() if k != "use_label_encoder"},
                },
                f,
            )
        logger.info("Model saved to %s (meta: %s)", path, meta_path)

    def load_model(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: File path to the model artifact.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._model = XGBClassifier()
        self._model.load_model(path)

        meta_path = path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self._feature_names = meta.get("feature_names")
            self._params.update(meta.get("params", {}))

        self._is_trained = True
        logger.info(
            "Model loaded from %s (%d features)",
            path,
            len(self._feature_names) if self._feature_names else 0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names
