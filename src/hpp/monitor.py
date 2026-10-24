"""Stage 8 - monitor.

Inputs : deployments/current/ (the model that is actually serving + its reference profile/sample)
         the incoming batches (config.paths.incoming_batches, batches config.monitoring.batches)
Outputs: reports/monitoring/summary.json   per-batch drift, performance, operational metrics, alerts
         reports/monitoring/report.html    human-readable dashboard (self-contained)
         reports/monitoring/batch_<n>_predictions.csv
         (optional) metrics pushed to a Prometheus Pushgateway for Grafana

What is measured for every batch
  data drift         PSI per feature vs the training reference (+ KS / chi-square p-values)
  target drift       PSI of the actual prices (labels are supplied with the batches)
  prediction drift   PSI of the model's predictions vs its holdout predictions
  data quality       data-gate checks + change in missing values
  performance drift  RMSE / MAE / bias vs the holdout reference
  operational        latency p50/p95/p99, throughput, error rate (in-process or via the live API)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from hpp.config import Config
from hpp.context import StageContext
from hpp.data import load_incoming, read_csv
from hpp.evaluate import regression_metrics
from hpp.features import numeric_profile
from hpp.utils import read_json, utcnow, write_json
from hpp.validation import validate_frame

log = logging.getLogger("hpp.monitor")

SEVERITY = {"OK": 0, "INFO": 0, "WARNING": 1, "CRITICAL": 2}
LEVELS = ["OK", "WARNING", "CRITICAL"]
EPS = 1e-4

ACTIONS = {
    "OK": "No action - keep serving the production model.",
    "WARNING": ("Investigate the drifted features and keep collecting labels; schedule a retrain that includes "
                "this batch. Production keeps serving; nothing is promoted automatically."),
    "CRITICAL": ("Trigger the retraining pipeline on training data + the labelled recent batches "
                 "(`hpp retrain --batches <n>`). The new model must pass the data gate and the model gate "
                 "(including the comparison against production) before it replaces production. If the alert is "
                 "data-quality related, quarantine the batch and exclude it from training."),
}


# ---------------------------------------------------------------- drift statistics
def psi_from_fractions(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(profile: dict[str, Any], current: pd.Series) -> float:
    vals = pd.to_numeric(current, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    edges = np.concatenate([[-np.inf], profile["bin_edges"], [np.inf]])
    counts = np.histogram(vals, bins=edges)[0]
    return psi_from_fractions(np.asarray(profile["bin_fractions"]), counts / len(vals))


def psi_categorical(profile: dict[str, Any], current: pd.Series) -> float:
    vc = current.dropna().astype(str).value_counts(normalize=True)
    levels = list(profile["levels"])
    extra = [v for v in vc.index if v not in levels]
    exp = np.array([profile["fractions"][lvl] for lvl in levels] + [0.0] * len(extra))
    act = np.array([vc.get(lvl, 0.0) for lvl in levels] + [vc[v] for v in extra])
    return psi_from_fractions(exp, act)


def chi2_pvalue(profile: dict[str, Any], current: pd.Series) -> float:
    s = current.dropna().astype(str)
    levels = list(profile["levels"])
    obs = np.array([(s == lvl).sum() for lvl in levels], float)
    if obs.sum() == 0:
        return float("nan")
    exp = np.clip(np.array([profile["fractions"][lvl] for lvl in levels]), EPS, None)
    exp = exp / exp.sum() * obs.sum()
    return float(chisquare(obs, exp).pvalue)


def _alert(batch: str, severity: str, rule: str, metric: str, value: float, threshold: float,
           message: str) -> dict[str, Any]:
    return {"batch": batch, "severity": severity, "rule": rule, "metric": metric,
            "value": None if value is None or (isinstance(value, float) and np.isnan(value)) else round(value, 4),
            "threshold": threshold, "message": message}


def drift_report(df: pd.DataFrame, profile: dict[str, Any], ref_sample: pd.DataFrame, cfg: Config,
                 batch: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    m = cfg.monitoring
    rows, alerts = [], []
    for col, prof in profile["features"].items():
        if col not in df.columns:
            continue
        if prof["type"] == "numeric":
            psi = psi_numeric(prof, df[col])
            cur = pd.to_numeric(df[col], errors="coerce")
            p = float(ks_2samp(pd.to_numeric(ref_sample[col], errors="coerce").dropna(), cur.dropna()).pvalue) \
                if cur.notna().sum() > 1 else float("nan")
            test, cur_mean = "ks", float(cur.mean())
        else:
            psi, p, test, cur_mean = psi_categorical(prof, df[col]), chi2_pvalue(prof, df[col]), "chi2", None
            cur = df[col]
        miss = float(df[col].isna().mean())
        severity = "OK"
        if psi >= m.psi_critical:
            severity = "CRITICAL"
        elif psi >= m.psi_warning and (np.isnan(p) or p < m.ks_pvalue):
            severity = "WARNING"
        rows.append({"feature": col, "psi": psi, "p_value": p, "test": test, "severity": severity,
                     "ref_mean": prof.get("mean"), "cur_mean": cur_mean,
                     "ref_missing": prof["missing_fraction"], "cur_missing": miss})
        if severity != "OK":
            thr = m.psi_critical if severity == "CRITICAL" else m.psi_warning
            alerts.append(_alert(batch, severity, "data_drift.feature", f"psi.{col}", psi, thr,
                                 f"{col} distribution shifted: PSI {psi:.3f} (>= {thr:g}), {test} p={p:.2g}"))
        if miss - prof["missing_fraction"] > m.missing_increase_warning:
            alerts.append(_alert(batch, "WARNING", "data_quality.missing", f"missing.{col}", miss,
                                 prof["missing_fraction"] + m.missing_increase_warning,
                                 f"{col} missing share rose to {miss:.1%} (reference {prof['missing_fraction']:.1%})"))
    drifted = [r for r in rows if r["severity"] != "OK"]
    share = len(drifted) / max(len(rows), 1)
    if share >= m.drift_share_critical:
        alerts.append(_alert(batch, "CRITICAL", "data_drift.dataset", "drift_share", share, m.drift_share_critical,
                             f"dataset drift: {len(drifted)}/{len(rows)} features drifted ({share:.0%})"))
    return rows, alerts


def performance_report(y: pd.Series, pred: np.ndarray, reference: dict[str, Any], cfg: Config,
                       batch: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    m = cfg.monitoring
    perf = regression_metrics(y, pred)
    ratio = perf["rmse"] / reference["holdout_rmse"]
    bias_pct = abs(perf["bias"]) / reference["mean_price"]
    perf.update({"rmse_ratio": ratio, "bias_pct": bias_pct, "reference_rmse": reference["holdout_rmse"]})
    alerts = []
    if ratio >= m.rmse_ratio_critical or ratio >= m.rmse_ratio_warning:
        sev = "CRITICAL" if ratio >= m.rmse_ratio_critical else "WARNING"
        thr = m.rmse_ratio_critical if sev == "CRITICAL" else m.rmse_ratio_warning
        alerts.append(_alert(batch, sev, "performance.rmse_ratio", "rmse_ratio", ratio, thr,
                             f"RMSE {perf['rmse']:.2f} is {ratio:.2f}x the holdout RMSE "
                             f"{reference['holdout_rmse']:.2f} (>= {thr:g}x)"))
    if bias_pct >= m.bias_warning_pct:
        sev = "CRITICAL" if bias_pct >= m.bias_critical_pct else "WARNING"
        thr = m.bias_critical_pct if sev == "CRITICAL" else m.bias_warning_pct
        direction = "under-pricing" if perf["bias"] > 0 else "over-pricing"
        alerts.append(_alert(batch, sev, "performance.bias", "bias_pct", bias_pct, thr,
                             f"systematic {direction}: mean error {perf['bias']:+.2f} lakhs = {bias_pct:.1%} of the "
                             f"reference mean price (>= {thr:.0%})"))
    return perf, alerts


# ---------------------------------------------------------------- serving (operational metrics)
def predict_in_process(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: N803
    """Score row by row, like the API does, and time each call."""
    preds, lat, errors = np.full(len(X), np.nan), [], 0
    started = time.perf_counter()
    for i in range(len(X)):
        t0 = time.perf_counter()
        try:
            preds[i] = float(model.predict(X.iloc[[i]])[0])
        except Exception:
            errors += 1
        lat.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - started
    return preds, _ops_summary(lat, errors, len(X), total, "in-process (row-by-row model.predict)")


def predict_via_api(api_url: str, X: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:  # noqa: N803
    import requests  # noqa: PLC0415

    preds, lat, errors = np.full(len(X), np.nan), [], 0
    session = requests.Session()
    records = X.astype(object).where(X.notna(), None).to_dict(orient="records")
    started = time.perf_counter()
    for i, rec in enumerate(records):
        t0 = time.perf_counter()
        try:
            r = session.post(api_url.rstrip("/") + "/predict", json=rec, timeout=10)
            if r.status_code == 200:
                preds[i] = r.json()["predicted_price_lakhs"]
            else:
                errors += 1
        except Exception:
            errors += 1
        lat.append((time.perf_counter() - t0) * 1000)
    total = time.perf_counter() - started
    return preds, _ops_summary(lat, errors, len(X), total, f"live API {api_url}")


def _ops_summary(lat: list[float], errors: int, n: int, total_s: float, mode: str) -> dict[str, Any]:
    arr = np.asarray(lat) if lat else np.asarray([np.nan])
    return {"mode": mode, "requests": n, "errors": errors, "error_rate": errors / max(n, 1),
            "latency_ms_p50": float(np.nanpercentile(arr, 50)), "latency_ms_p95": float(np.nanpercentile(arr, 95)),
            "latency_ms_p99": float(np.nanpercentile(arr, 99)), "throughput_rps": n / total_s if total_s else None}


def ops_alerts(ops: dict[str, Any], cfg: Config, batch: str) -> list[dict[str, Any]]:
    m, alerts = cfg.monitoring, []
    if ops["error_rate"] > m.error_rate_max:
        alerts.append(_alert(batch, "CRITICAL", "operational.error_rate", "error_rate", ops["error_rate"],
                             m.error_rate_max, f"{ops['error_rate']:.1%} of prediction requests failed"))
    if ops["latency_ms_p95"] > m.latency_p95_ms_max:
        alerts.append(_alert(batch, "WARNING", "operational.latency_p95", "latency_ms_p95", ops["latency_ms_p95"],
                             m.latency_p95_ms_max, f"p95 latency {ops['latency_ms_p95']:.0f} ms"))
    return alerts


# ---------------------------------------------------------------- orchestration of one batch
def monitor_batch(df: pd.DataFrame, batch: str, model: Any, manifest: dict, profile: dict,
                  ref_sample: pd.DataFrame, holdout_pred: np.ndarray | None, cfg: Config,
                  api_url: str = "") -> dict[str, Any]:
    d = cfg.data
    has_target = d.target in df.columns and df[d.target].notna().any()
    quality = validate_frame(df, cfg, require_target=has_target)
    alerts: list[dict[str, Any]] = []
    if not quality["passed"]:
        alerts.append(_alert(batch, "CRITICAL", "data_quality.gate", "failed_checks",
                             float(len(quality["failed_checks"])), 0, quality["message"]))

    drift_rows, drift_alerts = drift_report(df, profile, ref_sample, cfg, batch)
    alerts += drift_alerts

    X = df[d.feature_columns]  # noqa: N806
    preds, ops = predict_via_api(api_url, X) if api_url else predict_in_process(model, X)
    alerts += ops_alerts(ops, cfg, batch)

    pred_psi = None
    if holdout_pred is not None and np.isfinite(preds).any():
        pprof = numeric_profile(pd.Series(holdout_pred), cfg.monitoring.psi_bins)
        pred_psi = psi_numeric(pprof, pd.Series(preds))
        if pred_psi >= cfg.monitoring.psi_warning:
            sev = "CRITICAL" if pred_psi >= cfg.monitoring.psi_critical else "WARNING"
            alerts.append(_alert(batch, sev, "prediction_drift", "psi.prediction", pred_psi,
                                 cfg.monitoring.psi_critical if sev == "CRITICAL" else cfg.monitoring.psi_warning,
                                 f"distribution of predicted prices shifted: PSI {pred_psi:.3f}"))

    perf, target_psi = None, None
    if has_target:
        ok = np.isfinite(preds) & df[d.target].notna().to_numpy()
        perf, perf_alerts = performance_report(df.loc[ok, d.target], preds[ok], manifest["reference"], cfg, batch)
        alerts += perf_alerts
        target_psi = psi_numeric(profile["target"], df[d.target])
        if target_psi >= cfg.monitoring.psi_warning:
            sev = "CRITICAL" if target_psi >= cfg.monitoring.psi_critical else "WARNING"
            alerts.append(_alert(batch, sev, "target_drift", "psi.target", target_psi,
                                 cfg.monitoring.psi_critical if sev == "CRITICAL" else cfg.monitoring.psi_warning,
                                 f"actual price distribution shifted: PSI {target_psi:.3f}, mean "
                                 f"{df[d.target].mean():.1f} vs reference {profile['target']['mean']:.1f}"))

    level = max((a["severity"] for a in alerts), key=lambda s: SEVERITY[s], default="OK")
    return {
        "batch": batch, "n_rows": int(len(df)), "level": level, "recommended_action": ACTIONS[level],
        "data_quality": {k: quality[k] for k in ("passed", "failed_checks", "message")},
        "drift": drift_rows, "drift_share": sum(r["severity"] != "OK" for r in drift_rows) / max(len(drift_rows), 1),
        "target_psi": target_psi, "prediction_psi": pred_psi,
        "prediction_summary": {"mean": float(np.nanmean(preds)), "p10": float(np.nanpercentile(preds, 10)),
                               "p50": float(np.nanpercentile(preds, 50)), "p90": float(np.nanpercentile(preds, 90))},
        "performance": perf, "operational": ops, "alerts": alerts, "_preds": preds,
    }


def push_metrics(results: list[dict[str, Any]], manifest: dict, url: str) -> str:
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway  # noqa: PLC0415
    except ImportError:
        return "prometheus_client not installed"
    reg = CollectorRegistry()
    g_psi = Gauge("hpp_batch_feature_psi", "PSI per feature vs training reference", ["batch", "feature"], registry=reg)
    g_rmse = Gauge("hpp_batch_rmse_lakhs", "RMSE on a labelled incoming batch", ["batch"], registry=reg)
    g_ratio = Gauge("hpp_batch_rmse_ratio", "Batch RMSE / holdout RMSE", ["batch"], registry=reg)
    g_bias = Gauge("hpp_batch_bias_lakhs", "Mean(actual - predicted) on a batch", ["batch"], registry=reg)
    g_level = Gauge("hpp_batch_alert_level", "0=OK 1=WARNING 2=CRITICAL", ["batch"], registry=reg)
    g_pred = Gauge("hpp_batch_prediction_mean_lakhs", "Mean predicted price on a batch", ["batch"], registry=reg)
    for r in results:
        b = r["batch"]
        for row in r["drift"]:
            if row["psi"] == row["psi"]:
                g_psi.labels(b, row["feature"]).set(row["psi"])
        if r["performance"]:
            g_rmse.labels(b).set(r["performance"]["rmse"])
            g_ratio.labels(b).set(r["performance"]["rmse_ratio"])
            g_bias.labels(b).set(r["performance"]["bias"])
        g_level.labels(b).set(SEVERITY[r["level"]])
        g_pred.labels(b).set(r["prediction_summary"]["mean"])
    try:
        push_to_gateway(url, job="hpp_monitor", registry=reg,
                        grouping_key={"model_version": str(manifest.get("model_version"))})
        return "pushed"
    except Exception as exc:
        log.warning("could not push to Pushgateway at %s: %s", url, exc)
        return f"failed: {exc}"


def monitor(ctx: StageContext, batches: list[int] | None = None) -> dict:
    ctx.require_inputs("monitor")
    cfg = ctx.cfg
    bundle = ctx.current_bundle
    manifest = read_json(bundle / "manifest.json")
    model = joblib.load(bundle / "model.joblib")
    profile = read_json(bundle / "reference_profile.json")
    ref_sample = pd.read_csv(bundle / "reference_sample.csv")
    hp = bundle / "holdout_predictions.npy"
    holdout_pred = np.load(hp) if hp.exists() else None
    out_dir = ctx.reports / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[str, pd.DataFrame]] = []
    if ctx.monitor_batch_file:
        p = ctx.path(ctx.monitor_batch_file)
        frames.append((f"file:{p.name}", read_csv(p)))
    else:
        inc = load_incoming(ctx, batches or list(cfg.monitoring.batches))
        for b, sub in inc.groupby(cfg.data.batch_column, sort=True):
            frames.append((str(int(b)), sub.reset_index(drop=True)))

    results = []
    for label, df in frames:
        r = monitor_batch(df, label, model, manifest, profile, ref_sample, holdout_pred, cfg, cfg.monitoring.api_url)
        preds = r.pop("_preds")
        keep = [c for c in [cfg.data.id_column, cfg.data.target] if c in df.columns]
        pd.DataFrame({**{c: df[c] for c in keep}, "predicted_price_lakhs": preds}).to_csv(
            out_dir / f"batch_{label.replace(':', '_')}_predictions.csv", index=False)
        perf = r["performance"] or {}
        log.info("batch %s: %s | RMSE %s (x%s) | %d alerts", label, r["level"],
                 f"{perf.get('rmse', float('nan')):.2f}", f"{perf.get('rmse_ratio', float('nan')):.2f}",
                 len(r["alerts"]))
        results.append(r)

    overall = max((r["level"] for r in results), key=lambda s: SEVERITY[s], default="OK")
    summary = {
        "stage": "monitor", "created_at": utcnow(), "pipeline_run_id": ctx.pipeline_run_id,
        "model": {k: manifest.get(k) for k in ("model_name", "model_version", "mlflow_run_id", "git_commit",
                                               "data_version", "config_hash", "image_tag")},
        "reference": manifest["reference"],
        "thresholds": cfg.monitoring.model_dump(),
        "overall_level": overall,
        "retrain_recommended": any(a["severity"] == "CRITICAL" and a["rule"].startswith(("performance", "target"))
                                   for r in results for a in r["alerts"]),
        "batches": results,
        "alerts": [a for r in results for a in r["alerts"]],
    }
    if cfg.monitoring.pushgateway_url:
        summary["pushgateway"] = push_metrics(results, manifest, cfg.monitoring.pushgateway_url)
    write_json(out_dir / "summary.json", summary)

    from hpp.report import render_report  # noqa: PLC0415  (matplotlib is optional at import time)

    frames_by_label = dict(frames)
    preds_by_label = {r["batch"]: pd.read_csv(out_dir / f"batch_{r['batch'].replace(':', '_')}_predictions.csv")
                      for r in results}
    render_report(summary, frames_by_label, preds_by_label, ref_sample, profile, cfg, out_dir)
    log.info("monitoring overall level: %s; report: %s", overall, out_dir / "report.html")
    return {"overall_level": overall, "batches": {r["batch"]: r["level"] for r in results},
            "alerts": len(summary["alerts"]), "retrain_recommended": summary["retrain_recommended"]}
