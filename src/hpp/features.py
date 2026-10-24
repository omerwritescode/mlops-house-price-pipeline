"""Stage 3 - build features.

Inputs : artifacts/ingest/dataset.csv (validated), artifacts/validate/report.json
Outputs: artifacts/features/{train,test}.csv           cleaned, split raw-feature tables
         artifacts/features/train_features.csv          engineered view (for inspection)
         artifacts/features/reference_profile.json      distribution profile used for drift
         artifacts/features/reference_sample.csv        sample used for KS / chi-square tests
         artifacts/features/meta.json

Feature engineering lives in a scikit-learn transformer (``FeatureEngineer``) that is
part of the served model pipeline, so training and serving can never disagree.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from hpp.config import Config
from hpp.context import StageContext
from hpp.utils import read_json, utcnow, write_json
from hpp.validation import assert_validated

log = logging.getLogger("hpp.features")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain features to the raw listing columns. Stateless, so safe to refit anywhere."""

    def __init__(self, engineered: tuple[str, ...] = (), categorical: tuple[str, ...] = (),
                 numeric: tuple[str, ...] = ()):
        self.engineered = engineered
        self.categorical = categorical
        self.numeric = numeric

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = pd.DataFrame(X).copy()
        for col in self.numeric:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        for col in self.categorical:  # plain object dtype with np.nan for missing (encoder friendly)
            if col in X.columns:
                X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        rooms = X["Bedrooms"] + X["Bathrooms"]
        if "Rooms_Total" in self.engineered:
            X["Rooms_Total"] = rooms
        if "Area_per_Room" in self.engineered:
            X["Area_per_Room"] = X["Area_sqft"] / rooms.where(rooms > 0)
        if "Log_Distance" in self.engineered:
            X["Log_Distance"] = np.log1p(X["Distance_to_Center_km"].clip(lower=0))
        if "Log_Area" in self.engineered:
            X["Log_Area"] = np.log(X["Area_sqft"].clip(lower=1))
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + [e for e in self.engineered if e not in base], dtype=object)


def build_preprocessor(cfg: Config) -> Pipeline:
    d, f = cfg.data, cfg.features
    numeric = list(d.numeric_features) + list(f.engineered)
    ordinal = [c for c in d.categorical_features if c in d.ordinal_features]
    nominal = [c for c in d.categorical_features if c not in d.ordinal_features]
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy=f.imputation_strategy, add_indicator=f.add_missing_indicators)),
            ("scale", StandardScaler()),
        ]), numeric),
    ]
    if ordinal:
        transformers.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=[d.categorical_features[c] for c in ordinal],
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ordinal))
    if nominal:
        transformers.append(("nom", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(categories=[d.categorical_features[c] for c in nominal],
                                     handle_unknown="ignore")),
        ]), nominal))
    return Pipeline([
        ("engineer", FeatureEngineer(tuple(f.engineered), tuple(d.categorical_features), tuple(d.numeric_features))),
        ("columns", ColumnTransformer(transformers, remainder="drop")),
    ])


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Type-coerce, drop unusable rows (no target) and exact duplicates."""
    d = cfg.data
    df = df.copy()
    for col in list(d.numeric_features) + [d.target]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df[df[d.target].notna()]
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "_source"])
    if len(df) != before:
        log.info("cleaning removed %d rows (missing target / duplicates)", before - len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- profiling (drift reference)
def numeric_profile(s: pd.Series, bins: int) -> dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, bins + 1))) if len(valid) else np.array([0.0, 1.0])
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    inner = edges[1:-1]
    counts = np.histogram(valid, bins=np.concatenate([[-np.inf], inner, [np.inf]]))[0]
    return {
        "type": "numeric",
        "mean": float(valid.mean()) if len(valid) else None,
        "std": float(valid.std()) if len(valid) > 1 else None,
        "min": float(valid.min()) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "missing_fraction": float(s.isna().mean()),
        "bin_edges": inner.tolist(),          # interior edges; outer bins are open-ended
        "bin_fractions": (counts / max(len(valid), 1)).tolist(),
    }


def categorical_profile(s: pd.Series, levels: list[str]) -> dict[str, Any]:
    vc = s.dropna().astype(str).value_counts(normalize=True)
    return {
        "type": "categorical",
        "missing_fraction": float(s.isna().mean()),
        "levels": list(levels),
        "fractions": {lvl: float(vc.get(lvl, 0.0)) for lvl in levels},
    }


def build_profile(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    bins = cfg.monitoring.psi_bins
    prof: dict[str, Any] = {"n_rows": int(len(df)), "features": {}}
    for col in cfg.data.numeric_features:
        prof["features"][col] = numeric_profile(df[col], bins)
    for col, levels in cfg.data.categorical_features.items():
        prof["features"][col] = categorical_profile(df[col], levels)
    prof["target"] = numeric_profile(df[cfg.data.target], bins)
    return prof


def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/test split by hashing the listing ID.

    A listing always falls in the same split, across retrains and data versions. So when a
    retrained candidate is compared with the production model on the test set, production is
    never scored on rows it was trained on (a random re-split would leak and favour it).
    """
    idc = cfg.data.id_column
    if idc in df.columns and df[idc].notna().all():
        salt = str(cfg.project.seed)
        bucket = df[idc].astype(str).map(
            lambda s: int(hashlib.sha256((salt + s).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
        is_test = (bucket < cfg.features.test_size).to_numpy()
        return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    train, test = train_test_split(df, test_size=cfg.features.test_size, random_state=cfg.project.seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def features(ctx: StageContext) -> dict:
    ctx.require_inputs("features")
    report = assert_validated(ctx)
    cfg = ctx.cfg
    out = ctx.stage_dir("features")
    meta_in = read_json(ctx.artifacts / "ingest" / "meta.json")

    df = clean(pd.read_csv(ctx.artifacts / "ingest" / "dataset.csv"), cfg)
    keep = [c for c in [cfg.data.id_column, "_source", cfg.data.batch_column] if c in df.columns]
    df = df[keep + cfg.data.feature_columns + [cfg.data.target]]

    train, test = split(df, cfg)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    engineer = build_preprocessor(cfg).named_steps["engineer"]
    engineer.transform(train[cfg.data.feature_columns]).to_csv(out / "train_features.csv", index=False)

    profile = build_profile(train, cfg)
    write_json(out / "reference_profile.json", profile)
    sample = train.sample(n=min(cfg.features.reference_sample_rows, len(train)), random_state=cfg.project.seed)
    sample[cfg.data.feature_columns + [cfg.data.target]].to_csv(out / "reference_sample.csv", index=False)

    meta = {
        "stage": "features",
        "created_at": utcnow(),
        "pipeline_run_id": ctx.pipeline_run_id,
        "data_version": meta_in["data_version"],
        "validation_message": report["message"],
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_columns": cfg.data.feature_columns,
        "engineered_features": list(cfg.features.engineered),
        "target": cfg.data.target,
        "split": "sha256(seed + House_ID) < test_size",
        "split_seed": cfg.project.seed,
        "test_size": cfg.features.test_size,
        "target_mean_train": float(train[cfg.data.target].mean()),
    }
    write_json(out / "meta.json", meta)
    log.info("features built: train=%d test=%d, %d raw + %d engineered features",
             len(train), len(test), len(cfg.data.feature_columns), len(cfg.features.engineered))
    return {"n_train": len(train), "n_test": len(test)}
