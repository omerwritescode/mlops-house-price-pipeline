"""Stage 4 - train (with hyperparameter search).

Inputs : artifacts/features/train.csv, artifacts/features/meta.json
Outputs: artifacts/train/model.joblib   full scikit-learn Pipeline (feature engineering + preprocessing + model)
         artifacts/train/cv_results.csv grid-search results
         artifacts/train/meta.json      best params, CV scores, MLflow run id + model URI

Starts the MLflow run for this model. The run carries every lineage tag (git commit,
data version, config hash, pipeline run id) and the effective config as an artifact.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from hpp.config import Config
from hpp.context import StageContext
from hpp.features import build_preprocessor
from hpp.tracking import Tracker, lineage_tags
from hpp.utils import read_json, utcnow, write_json

log = logging.getLogger("hpp.train")

ESTIMATORS = {
    "gradient_boosting": GradientBoostingRegressor,
    "random_forest": RandomForestRegressor,
    "ridge": Ridge,
}


def build_pipeline(cfg: Config) -> Pipeline:
    spec = cfg.training.spec
    estimator = ESTIMATORS[cfg.training.model_type](**spec.params)
    return Pipeline([("preprocess", build_preprocessor(cfg)), ("model", estimator)])


def fit_model(cfg: Config, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:  # noqa: N803
    pipe = build_pipeline(cfg)
    grid = {f"model__{k}": v for k, v in cfg.training.spec.search.items()}
    cv = KFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.project.seed)
    started = time.time()
    if grid:
        search = GridSearchCV(pipe, grid, cv=cv, scoring=cfg.training.scoring, n_jobs=cfg.training.n_jobs,
                              refit=True, return_train_score=True)
        search.fit(X, y)
        model = search.best_estimator_
        results = pd.DataFrame(search.cv_results_)
        best_idx = int(search.best_index_)
        cv_mean = -float(results.loc[best_idx, "mean_test_score"])
        cv_std = float(results.loc[best_idx, "std_test_score"])
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415

        scores = cross_val_score(pipe, X, y, cv=cv, scoring=cfg.training.scoring, n_jobs=cfg.training.n_jobs)
        model = pipe.fit(X, y)
        results = pd.DataFrame({"fold": range(len(scores)), "test_score": scores})
        cv_mean, cv_std, best_params = -float(np.mean(scores)), float(np.std(scores)), {}
    info = {
        "best_params": best_params,
        "cv_rmse_mean": cv_mean,
        "cv_rmse_std": cv_std,
        "n_candidates": int(len(results)) if grid else 1,
        "fit_seconds": round(time.time() - started, 2),
    }
    return model, info, results


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def train(ctx: StageContext) -> dict:
    ctx.require_inputs("train")
    cfg = ctx.cfg
    out = ctx.stage_dir("train")
    fmeta = read_json(ctx.artifacts / "features" / "meta.json")
    train_df = pd.read_csv(ctx.artifacts / "features" / "train.csv")
    X, y = train_df[cfg.data.feature_columns], train_df[cfg.data.target]  # noqa: N806

    log.info("training %s on %d rows; grid=%s", cfg.training.model_type, len(X), cfg.training.spec.search)
    model, info, results = fit_model(cfg, X, y)
    log.info("best params %s | CV RMSE %.3f +/- %.3f (%d candidates, %.1fs)", info["best_params"],
             info["cv_rmse_mean"], info["cv_rmse_std"], info["n_candidates"], info["fit_seconds"])

    joblib.dump(model, out / "model.joblib")
    results.to_csv(out / "cv_results.csv", index=False)
    (out / "effective_config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    # ---------------------------------------------------------------- MLflow
    tracker = Tracker(cfg, ctx.root)
    tags = lineage_tags(ctx.root, ctx.pipeline_run_id, fmeta["data_version"], cfg.config_hash(), ctx.orchestrator)
    tags.update({"model_type": cfg.training.model_type, "stage": "train"})
    run_id = tracker.start_run(f"{cfg.training.model_type}-{ctx.pipeline_run_id}", tags)
    params = {
        "model_type": cfg.training.model_type,
        **{f"best.{k}": v for k, v in info["best_params"].items()},
        **{f"fixed.{k}": v for k, v in cfg.training.spec.params.items()},
        "search_space": cfg.training.spec.search,
        "cv_folds": cfg.training.cv_folds,
        "n_train": fmeta["n_train"],
        "n_test": fmeta["n_test"],
        "test_size": cfg.features.test_size,
        "seed": cfg.project.seed,
        "data_version": fmeta["data_version"],
        "config_hash": cfg.config_hash(),
        "engineered_features": cfg.features.engineered,
        "train_with_batches": cfg.data.train_with_batches,
    }
    tracker.log_params(run_id, params)
    tracker.log_metrics(run_id, {"cv_rmse_mean": info["cv_rmse_mean"], "cv_rmse_std": info["cv_rmse_std"],
                                 "fit_seconds": info["fit_seconds"]})
    for rel, dest in [("train/effective_config.yaml", "config"), ("train/cv_results.csv", "training"),
                      ("ingest/meta.json", "data"), ("validate/report.json", "data"),
                      ("features/meta.json", "data"), ("features/reference_profile.json", "data")]:
        tracker.log_artifact(run_id, ctx.artifacts / rel, dest)
    example = X.dropna().head(5)
    model_uri = tracker.log_model(run_id, model, example)
    log.info("MLflow run %s logged (model %s)", run_id, model_uri)

    meta = {
        "stage": "train",
        "created_at": utcnow(),
        "pipeline_run_id": ctx.pipeline_run_id,
        "model_type": cfg.training.model_type,
        **info,
        "data_version": fmeta["data_version"],
        "config_hash": cfg.config_hash(),
        "mlflow_run_id": run_id,
        "mlflow_model_uri": model_uri,
        "lineage": tags,
    }
    write_json(out / "meta.json", meta)
    return {"mlflow_run_id": run_id, "cv_rmse": round(info["cv_rmse_mean"], 3), "best_params": info["best_params"]}
