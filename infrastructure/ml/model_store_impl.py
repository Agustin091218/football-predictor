"""Local filesystem ModelStore implementation using joblib.

Implements the ModelStore ABC from domain/repositories.py.
Stores model artifacts as .joblib files in a configurable directory.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime

import joblib

from football_predictor.domain.repositories import ModelStore

logger = logging.getLogger(__name__)


class LocalModelStore(ModelStore):
    """Filesystem-backed model persistence using joblib serialization.

    Each model is stored as {name}_v{version}.joblib with metadata.
    Supports listing versions and loading the latest.
    """

    def __init__(self, models_dir: str = "models") -> None:
        os.makedirs(models_dir, exist_ok=True)
        self._dir = models_dir

    def save_model(self, model: object, version: str, name: str = "xgboost_model") -> str:
        filename = f"{name}_v{version}.joblib"
        path = os.path.join(self._dir, filename)
        joblib.dump(
            {
                "model": model,
                "name": name,
                "version": version,
                "saved_at": datetime.now().isoformat(),
            },
            path,
        )
        logger.info("Modelo guardado: %s", path)
        return path

    def load_model(self, version: str, name: str = "xgboost_model") -> object:
        if version == "latest":
            pattern = os.path.join(self._dir, f"{name}_v*.joblib")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(f"No model found for '{name}' in {self._dir}")
            files.sort(key=os.path.getmtime, reverse=True)
            path = files[0]
        else:
            path = os.path.join(self._dir, f"{name}_v{version}.joblib")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)
        logger.info("Modelo cargado: %s", path)
        return data["model"]

    def list_versions(self, name: str = "xgboost_model") -> list[str]:
        pattern = os.path.join(self._dir, f"{name}_v*.joblib")
        files = glob.glob(pattern)
        if not files:
            return []

        files.sort(key=os.path.getmtime, reverse=True)
        versions: list[str] = []
        for f in files:
            base = os.path.basename(f)
            ver = base.replace(f"{name}_v", "").replace(".joblib", "")
            versions.append(ver)
        return versions

    def delete_model(self, version: str, name: str = "xgboost_model") -> None:
        path = os.path.join(self._dir, f"{name}_v{version}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        os.unlink(path)
        logger.info("Modelo eliminado: %s", path)
