"""Prediction API (FastAPI) with Prometheus metrics.

Endpoints
  GET  /health        liveness + whether a model is loaded
  GET  /model-info    full deployment manifest (model version, MLflow run, git commit, data version, config hash)
  GET  /version       image tag + git commit + model version (links a running container to a commit)
  POST /predict       one listing -> predicted price (lakhs)
  POST /predict/batch up to 1000 listings
  POST /admin/reload  hot-reload the bundle after a deploy (header X-Admin-Token)
  GET  /metrics       Prometheus exposition

Operational metrics (scraped by Prometheus, plotted in Grafana)
  hpp_http_requests_total{method,endpoint,status}  -> throughput + error rate
  hpp_http_request_duration_seconds{endpoint}      -> latency histogram (p50/p95/p99)
  hpp_predictions_total                            -> listings scored
  hpp_predicted_price_lakhs                        -> distribution of predicted prices
  hpp_prediction_errors_total{reason}              -> model/validation failures
  hpp_model_info{model_version,git_commit,image_tag,data_version} = 1
  hpp_model_reloads_total{result}

Run:  MODEL_DIR=deployments/current uvicorn hpp.serving.app:app --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

from hpp import __version__
from hpp.serving.predictor import Predictor
from hpp.serving.schemas import BatchPrediction, BatchRequest, Listing, Prediction

log = logging.getLogger("hpp.api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "deployments/current"))
ADMIN_TOKEN = os.environ.get("HPP_ADMIN_TOKEN", "change-me")
IMAGE_TAG = os.environ.get("IMAGE_TAG", "local")
GIT_SHA = os.environ.get("GIT_SHA", "unknown")

# A registry owned by this module (not prometheus_client's global one), so re-importing the app - as the
# tests do, or a dev-server reload - never fails with "Duplicated timeseries in CollectorRegistry".
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
REQUESTS = Counter("hpp_http_requests_total", "HTTP requests", ["method", "endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("hpp_http_request_duration_seconds", "Request latency", ["endpoint"], registry=REGISTRY,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0))
PREDICTIONS = Counter("hpp_predictions_total", "Listings scored", registry=REGISTRY)
PRED_PRICE = Histogram("hpp_predicted_price_lakhs", "Predicted price (lakhs)", registry=REGISTRY,
                       buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 200, 300))
PRED_ERRORS = Counter("hpp_prediction_errors_total", "Failed predictions", ["reason"], registry=REGISTRY)
MODEL_INFO = Gauge("hpp_model_info", "Currently loaded model", ["model_version", "git_commit", "image_tag",
                                                                 "data_version"], registry=REGISTRY)
RELOADS = Counter("hpp_model_reloads_total", "Model reloads", ["result"], registry=REGISTRY)

app = FastAPI(title="House price API", version=__version__,
              description="Predicts asking prices (₹ lakhs) - MLOps Project 4, Group 10")
predictor: Predictor | None = None


def _set_model_info() -> None:
    MODEL_INFO.clear()
    if predictor is not None:
        m = predictor.manifest
        MODEL_INFO.labels(predictor.version, str(m.get("git_commit", ""))[:12], IMAGE_TAG,
                          str(m.get("data_version", ""))).set(1)


@app.on_event("startup")
def _startup() -> None:
    global predictor
    try:
        predictor = Predictor(MODEL_DIR)
        _set_model_info()
        log.info("loaded model v%s from %s", predictor.version, MODEL_DIR)
    except Exception as exc:  # start anyway; /health reports it and /predict returns 503
        log.error("could not load model from %s: %s", MODEL_DIR, exc)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint != "/metrics":
            LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        REQUESTS.labels(request.method, endpoint, str(status)).inc()


def _require_model() -> Predictor:
    if predictor is None:
        PRED_ERRORS.labels("no_model").inc()
        raise HTTPException(status_code=503, detail="no model loaded")
    return predictor


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if predictor is not None else "degraded", "model_loaded": predictor is not None,
            "model_version": predictor.version if predictor else None}


@app.get("/model-info")
def model_info() -> dict:
    return {**_require_model().manifest, "image_tag": IMAGE_TAG, "serving_git_sha": GIT_SHA}


@app.get("/version")
def version() -> dict:
    p = predictor
    return {"api_version": __version__, "image_tag": IMAGE_TAG, "git_sha": GIT_SHA,
            "model_version": p.version if p else None,
            "model_git_commit": p.manifest.get("git_commit") if p else None}


@app.post("/predict", response_model=Prediction)
def predict(listing: Listing) -> Prediction:
    p = _require_model()
    try:
        value = float(p.predict([listing.model_dump()])[0])
    except Exception as exc:
        PRED_ERRORS.labels("model_error").inc()
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    PREDICTIONS.inc()
    PRED_PRICE.observe(value)
    return Prediction(predicted_price_lakhs=round(value, 2), model_version=p.version)


@app.post("/predict/batch", response_model=BatchPrediction)
def predict_batch(req: BatchRequest) -> BatchPrediction:
    p = _require_model()
    try:
        values = p.predict([x.model_dump() for x in req.listings])
    except Exception as exc:
        PRED_ERRORS.labels("model_error").inc()
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
    PREDICTIONS.inc(len(values))
    for v in values:
        PRED_PRICE.observe(float(v))
    return BatchPrediction(predictions=[round(float(v), 2) for v in values], model_version=p.version)


@app.post("/admin/reload")
def reload_model(x_admin_token: str = Header(default="")) -> dict:
    global predictor
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="bad admin token")
    old = predictor.version if predictor else None
    try:
        if predictor is None:
            predictor = Predictor(MODEL_DIR)
        else:
            predictor.load()
        _set_model_info()
        RELOADS.labels("success").inc()
        return {"reloaded": True, "from_version": old, "to_version": predictor.version}
    except Exception as exc:  # keep serving the old model
        RELOADS.labels("failure").inc()
        raise HTTPException(status_code=500, detail=f"reload failed, still serving v{old}: {exc}") from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
