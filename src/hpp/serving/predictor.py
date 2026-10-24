"""Loads the deployed bundle (deployments/current or the copy baked into the image) and scores requests.

Kept free of web-framework code so it can be unit-tested anywhere.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self.model: Any = None
        self.manifest: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        """(Re)load model + manifest atomically; the old model keeps serving if loading fails."""
        manifest = json.loads((self.model_dir / "manifest.json").read_text())
        model = joblib.load(self.model_dir / "model.joblib")
        with self._lock:
            self.model, self.manifest = model, manifest
        return manifest

    @property
    def feature_columns(self) -> list[str]:
        return list(self.manifest["feature_columns"])

    @property
    def version(self) -> str:
        return str(self.manifest.get("model_version", "unknown"))

    def predict(self, records: list[dict[str, Any]]) -> np.ndarray:
        df = pd.DataFrame.from_records(records)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        with self._lock:
            preds = self.model.predict(df[self.feature_columns])
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)
