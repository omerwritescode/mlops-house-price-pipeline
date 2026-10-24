"""API contract + metrics. Builds a small bundle without MLflow so it runs anywhere FastAPI is installed."""

import importlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402

from hpp.features import split  # noqa: E402
from hpp.train import build_pipeline  # noqa: E402

LISTING = {"Area_sqft": 1150, "Bedrooms": 2, "Bathrooms": 2, "Age_Years": 12, "Distance_to_Center_km": 9.5,
           "Neighborhood_Score": 6.5, "Has_Parking": 1, "Property_Condition": "Good", "City_Zone": "Central"}


@pytest.fixture
def client(tmp_path, cfg, train_df, monkeypatch):
    import joblib

    tr, _ = split(train_df, cfg)
    pipe = build_pipeline(cfg).set_params(model__n_estimators=50).fit(tr[cfg.data.feature_columns],
                                                                        tr[cfg.data.target])
    joblib.dump(pipe, tmp_path / "model.joblib")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_name": "house-price-model", "model_version": "7", "git_commit": "abc1234", "data_version": "d1",
        "feature_columns": cfg.data.feature_columns}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("HPP_ADMIN_TOKEN", "secret")
    import hpp.serving.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def test_health_and_version(client):
    assert client.get("/health").json()["model_version"] == "7"
    assert client.get("/model-info").json()["git_commit"] == "abc1234"
    assert "image_tag" in client.get("/version").json()


def test_predict_single_and_batch(client):
    r = client.post("/predict", json=LISTING)
    assert r.status_code == 200
    body = r.json()
    assert 10 < body["predicted_price_lakhs"] < 200 and body["model_version"] == "7"
    missing = {k: v for k, v in LISTING.items() if k != "Neighborhood_Score"}  # optional field
    assert client.post("/predict", json=missing).status_code == 200
    rb = client.post("/predict/batch", json={"listings": [LISTING, LISTING]})
    assert len(rb.json()["predictions"]) == 2


@pytest.mark.parametrize("bad", [{"Area_sqft": -5}, {"City_Zone": "Downtown"}, {"Bedrooms": 0}])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json={**LISTING, **bad}).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json=LISTING)
    client.post("/predict", json={**LISTING, "Area_sqft": -1})
    text = client.get("/metrics").text
    assert 'hpp_http_requests_total{method="POST",endpoint="/predict",status="200"}' in text
    assert 'status="422"' in text
    assert "hpp_http_request_duration_seconds_bucket" in text
    assert "hpp_predicted_price_lakhs_bucket" in text
    assert 'hpp_model_info{' in text and 'model_version="7"' in text


def test_reload_requires_token(client):
    assert client.post("/admin/reload").status_code == 401
    r = client.post("/admin/reload", headers={"X-Admin-Token": "secret"})
    assert r.status_code == 200 and r.json()["to_version"] == "7"
