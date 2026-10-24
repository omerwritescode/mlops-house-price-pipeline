"""Shared fixtures. Tests never touch the real artifacts/ or mlflow.db: everything runs in tmp dirs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from hpp.config import load_config
from hpp.context import StageContext

ROOT = Path(__file__).resolve().parents[1]
FAST = {
    "orchestration": {"engine": "local"},
    "training": {"cv_folds": 2, "n_jobs": 1, "models": {"gradient_boosting": {
        "params": {"loss": "squared_error", "subsample": 0.8, "min_samples_leaf": 5, "random_state": 42},
        "search": {"n_estimators": [150], "learning_rate": [0.1], "max_depth": [3]}}}},
}


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs/config.yaml", extra=FAST)


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/capstone_house_prices_train.csv")


@pytest.fixture
def incoming_df() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw/capstone_incoming_batches.csv")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throw-away copy of the project (data + configs) with its own MLflow store."""
    for d in ("data/raw", "configs"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\n")
    return tmp_path


def make_ctx(project: Path, extra: dict | None = None, **kw) -> StageContext:
    merged = {**FAST, "tracking": {"uri": f"sqlite:///{project}/mlflow.db", "experiment": "test",
                                   "artifact_location": "mlartifacts"}}
    if extra:
        from hpp.config import deep_merge

        merged = deep_merge(merged, extra)
    cfg = load_config(project / "configs/config.yaml", extra=merged)
    return StageContext(cfg=cfg, root=project, pipeline_run_id=kw.pop("run_id", "test-run"), **kw)


@pytest.fixture
def ctx_factory():
    return make_ctx
