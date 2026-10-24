"""Stage 5 - evaluate (the MODEL GATE).

Inputs : artifacts/train/model.joblib, artifacts/train/meta.json, artifacts/features/test.csv
Outputs: artifacts/evaluate/metrics.json, artifacts/evaluate/gate.json, evaluation plots
Fails  : raises ModelGateError if the candidate misses an absolute threshold OR is worse
         than the current production model on the *same* test set by more than
         ``max_regression_pct``. Nothing is registered or deployed; production keeps serving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from hpp.config import Config
from hpp.context import StageContext
from hpp.exceptions import ModelGateError
from hpp.tracking import Tracker
from hpp.utils import read_json, utcnow, write_json

log = logging.getLogger("hpp.evaluate")


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "bias": float(err.mean()),                       # + means the model under-prices
        "p90_abs_error": float(np.quantile(np.abs(err), 0.9)),
        "within_10pct": float(np.mean(np.abs(err) <= 0.10 * np.abs(y_true))),
        "n": int(len(y_true)),
    }


def check_thresholds(metrics: dict[str, float], cfg: Config) -> list[dict[str, Any]]:
    t = cfg.evaluation.thresholds
    rules = [
        ("threshold.rmse", metrics["rmse"] <= t.rmse_max, metrics["rmse"], t.rmse_max, "<="),
        ("threshold.mae", metrics["mae"] <= t.mae_max, metrics["mae"], t.mae_max, "<="),
        ("threshold.r2", metrics["r2"] >= t.r2_min, metrics["r2"], t.r2_min, ">="),
        ("threshold.mape", metrics["mape"] <= t.mape_max, metrics["mape"], t.mape_max, "<="),
    ]
    return [{"check": n, "passed": bool(ok), "value": round(v, 4), "threshold": th,
             "detail": f"{n.split('.')[1].upper()} {v:.3f} (required {op} {th})"} for n, ok, v, th, op in rules]


def compare_to_production(cand: dict[str, float], prod: dict[str, float] | None, cfg: Config,
                          prod_label: str) -> dict[str, Any]:
    metric = cfg.evaluation.primary_metric
    limit = cfg.evaluation.max_regression_pct
    if prod is None:
        return {"check": "compare.production", "passed": True, "value": None, "threshold": limit,
                "detail": "no production model yet - first deployment, comparison skipped"}
    pct = (cand[metric] - prod[metric]) / prod[metric] * 100.0
    return {"check": "compare.production", "passed": bool(pct <= limit), "value": round(pct, 3),
            "threshold": limit,
            "detail": (f"candidate {metric.upper()} {cand[metric]:.3f} vs production {prod_label} "
                       f"{prod[metric]:.3f} ({pct:+.2f}%, allowed <= +{limit:g}%)")}


def _load_production(ctx: StageContext, tracker: Tracker | None) -> tuple[Any, str] | tuple[None, None]:
    if tracker is not None:
        try:
            mv = tracker.version_by_alias(ctx.cfg.registry.production_alias)
            if mv is not None:
                return tracker.load_by_alias(ctx.cfg.registry.production_alias), f"v{mv.version}"
        except Exception as exc:  # registry unreachable -> fall back to the deployed bundle
            log.warning("could not load production model from registry (%s); trying deployed bundle", exc)
    bundle = ctx.current_bundle / "model.joblib"
    if bundle.exists():
        manifest = read_json(ctx.current_bundle / "manifest.json")
        return joblib.load(bundle), f"v{manifest.get('model_version', '?')} (bundle)"
    return None, None


def _plots(out: Path, y: pd.Series, pred: np.ndarray) -> list[Path]:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return []
    paths = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y, pred, s=8, alpha=0.5, color="#2a6fdb")
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax[0].plot(lim, lim, color="#888", lw=1, ls="--")
    ax[0].set(xlabel="Actual price (lakhs)", ylabel="Predicted price (lakhs)", title="Holdout: predicted vs actual")
    ax[1].hist(y - pred, bins=40, color="#2a6fdb", alpha=0.8)
    ax[1].axvline(0, color="#888", lw=1)
    ax[1].set(xlabel="Residual = actual - predicted (lakhs)", ylabel="Listings", title="Holdout residuals")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out / "holdout_diagnostics.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def evaluate(ctx: StageContext) -> dict:
    ctx.require_inputs("evaluate")
    cfg = ctx.cfg
    out = ctx.stage_dir("evaluate")
    tmeta = read_json(ctx.artifacts / "train" / "meta.json")
    test = pd.read_csv(ctx.artifacts / "features" / "test.csv")
    train_df = pd.read_csv(ctx.artifacts / "features" / "train.csv")
    X, y = test[cfg.data.feature_columns], test[cfg.data.target]  # noqa: N806

    model = joblib.load(ctx.artifacts / "train" / "model.joblib")
    pred = model.predict(X)
    cand = regression_metrics(y, pred)
    train_metrics = regression_metrics(train_df[cfg.data.target], model.predict(train_df[cfg.data.feature_columns]))
    reference = {"holdout_rmse": cand["rmse"], "holdout_mae": cand["mae"],
                 "mean_price": float(train_df[cfg.data.target].mean()),
                 "holdout_pred_mean": float(np.mean(pred))}
    np.save(out / "holdout_predictions.npy", pred)

    tracker = Tracker(cfg, ctx.root)
    checks = check_thresholds(cand, cfg)
    prod_metrics, prod_label = None, None
    if cfg.evaluation.compare_to_production:
        prod_model, prod_label = _load_production(ctx, tracker)
        if prod_model is not None:
            prod_metrics = regression_metrics(y, prod_model.predict(X))
        checks.append(compare_to_production(cand, prod_metrics, cfg, prod_label or ""))

    failed = [c for c in checks if not c["passed"]]
    passed = not failed
    message = (f"model gate passed - RMSE {cand['rmse']:.3f}, MAE {cand['mae']:.3f}, R2 {cand['r2']:.3f}"
               if passed else
               "MODEL GATE FAILED - " + " | ".join(c["detail"] for c in failed))
    metrics = {"stage": "evaluate", "created_at": utcnow(), "pipeline_run_id": ctx.pipeline_run_id,
               "mlflow_run_id": tmeta["mlflow_run_id"], "holdout": cand, "train": train_metrics,
               "reference": reference, "production_on_same_test": prod_metrics,
               "production_label": prod_label}
    write_json(out / "metrics.json", metrics)
    gate = {"passed": passed, "message": message, "checks": checks, "candidate": cand,
            "production": {"label": prod_label, "metrics": prod_metrics},
            "mlflow_run_id": tmeta["mlflow_run_id"], "data_version": tmeta["data_version"],
            "created_at": utcnow(), "pipeline_run_id": ctx.pipeline_run_id}
    write_json(out / "gate.json", gate)
    plots = _plots(out, y, pred)

    run_id = tmeta["mlflow_run_id"]
    tracker.log_metrics(run_id, {f"test_{k}": v for k, v in cand.items() if k != "n"}
                        | {f"train_{k}": v for k, v in train_metrics.items() if k in ("rmse", "mae", "r2")}
                        | ({"production_test_rmse": prod_metrics["rmse"]} if prod_metrics else {}))
    tracker.set_tags(run_id, {"model_gate": "passed" if passed else "failed",
                              "model_gate_message": message[:4900]})
    for p in [out / "gate.json", out / "metrics.json", *plots]:
        tracker.log_artifact(run_id, p, "evaluation")

    if not passed:
        log.error(message)
        raise ModelGateError(message, gate)
    log.info(message)
    return {"passed": True, "rmse": round(cand["rmse"], 3), "mae": round(cand["mae"], 3),
            "r2": round(cand["r2"], 4), "vs_production": prod_label}
