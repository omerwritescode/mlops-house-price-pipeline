import numpy as np
import pandas as pd

from hpp.features import build_profile, numeric_profile
from hpp.monitor import drift_report, performance_report, psi_categorical, psi_numeric


def test_psi_zero_for_same_distribution(cfg, train_df):
    prof = numeric_profile(train_df["Area_sqft"], 10)
    assert psi_numeric(prof, train_df["Area_sqft"]) < 1e-6


def test_psi_large_for_shifted_distribution(cfg, train_df):
    prof = numeric_profile(train_df["Area_sqft"], 10)
    assert psi_numeric(prof, train_df["Area_sqft"] * 0.7) > 0.2


def test_categorical_psi_detects_new_mix(cfg, train_df):
    prof = build_profile(train_df, cfg)["features"]["City_Zone"]
    shifted = pd.Series(["Central"] * 60 + ["North"] * 40)
    assert psi_categorical(prof, shifted) > 0.2
    assert psi_categorical(prof, train_df["City_Zone"]) < 1e-6


def test_batches_get_expected_drift_levels(cfg, train_df, incoming_df):
    prof = build_profile(train_df, cfg)
    ref = train_df.sample(2000, random_state=0)
    levels = {}
    for b, part in incoming_df.groupby("Batch"):
        rows, alerts = drift_report(part, prof, ref, cfg, str(b))
        levels[b] = max((a["severity"] for a in alerts), key={"WARNING": 1, "CRITICAL": 2}.get, default="OK")
    assert levels[1] == "OK"
    assert levels[3] == "CRITICAL"   # smaller homes, Central share doubles


def test_performance_alerts(cfg):
    y = pd.Series(np.full(100, 60.0))
    ref = {"holdout_rmse": 7.5, "mean_price": 62.0}
    perf, alerts = performance_report(y, y - np.random.default_rng(0).normal(0, 7.0, 100), ref, cfg, "1")
    assert not alerts
    perf, alerts = performance_report(y, y - 13.0, ref, cfg, "3")
    rules = {a["rule"]: a["severity"] for a in alerts}
    assert rules == {"performance.rmse_ratio": "CRITICAL", "performance.bias": "CRITICAL"}
