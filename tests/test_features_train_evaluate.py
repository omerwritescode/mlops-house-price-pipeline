import numpy as np
import pandas as pd

from hpp.evaluate import check_thresholds, compare_to_production, regression_metrics
from hpp.features import FeatureEngineer, build_preprocessor, build_profile, split
from hpp.train import build_pipeline, fit_model


def test_feature_engineer_adds_columns_and_handles_missing(cfg, train_df):
    fe = FeatureEngineer(tuple(cfg.features.engineered), tuple(cfg.data.categorical_features),
                         tuple(cfg.data.numeric_features))
    out = fe.transform(train_df[cfg.data.feature_columns])
    for col in cfg.features.engineered:
        assert col in out.columns
    assert (out["Rooms_Total"] == train_df["Bedrooms"] + train_df["Bathrooms"]).all()
    assert out["Log_Distance"].isna().sum() == train_df["Distance_to_Center_km"].isna().sum()


def test_preprocessor_imputes_and_ignores_unseen_category(cfg, train_df):
    pre = build_preprocessor(cfg).fit(train_df[cfg.data.feature_columns])
    odd = train_df[cfg.data.feature_columns].head(3).copy().astype({"City_Zone": object})
    odd.loc[0, "City_Zone"] = "Downtown"
    odd.loc[1, ["Age_Years", "Neighborhood_Score"]] = np.nan
    X = pre.transform(odd)
    assert not np.isnan(np.asarray(X, dtype=float)).any()


def test_split_is_deterministic_and_stable_across_data_versions(cfg, train_df, incoming_df):
    tr1, te1 = split(train_df, cfg)
    tr2, te2 = split(pd.concat([train_df, incoming_df.drop(columns="Batch")]), cfg)
    assert set(te1.House_ID) <= set(te2.House_ID)          # old test rows stay in test
    assert not set(tr2.House_ID) & set(te1.House_ID)        # ...and never move into train
    assert 0.17 < len(te1) / len(train_df) < 0.23


def test_profile_fractions_sum_to_one(cfg, train_df):
    prof = build_profile(train_df, cfg)
    for col, p in prof["features"].items():
        fr = p["bin_fractions"] if p["type"] == "numeric" else list(p["fractions"].values())
        assert abs(sum(fr) - 1) < 1e-9, col


def test_training_beats_naive_baseline(cfg, train_df):
    tr, te = split(train_df, cfg)
    model, info, _ = fit_model(cfg, tr[cfg.data.feature_columns], tr[cfg.data.target])
    m = regression_metrics(te[cfg.data.target], model.predict(te[cfg.data.feature_columns]))
    naive = regression_metrics(te[cfg.data.target], np.full(len(te), tr[cfg.data.target].mean()))
    assert m["rmse"] < 0.5 * naive["rmse"]
    assert info["cv_rmse_mean"] > 0 and "n_estimators" in info["best_params"]


def test_pipeline_uses_only_configured_features(cfg, train_df):
    pipe = build_pipeline(cfg).fit(train_df[cfg.data.feature_columns], train_df[cfg.data.target])
    shuffled = train_df.head(20).copy()
    shuffled["House_ID"] = "X"
    a = pipe.predict(train_df.head(20))
    b = pipe.predict(shuffled)
    assert np.allclose(a, b)  # identifiers have no influence


def test_threshold_checks(cfg):
    good = {"rmse": 7.5, "mae": 5.8, "r2": 0.88, "mape": 0.10}
    bad = {"rmse": 12.0, "mae": 9.0, "r2": 0.7, "mape": 0.2}
    assert all(c["passed"] for c in check_thresholds(good, cfg))
    assert not any(c["passed"] for c in check_thresholds(bad, cfg))


def test_production_comparison(cfg):
    assert compare_to_production({"rmse": 7.6}, {"rmse": 7.5}, cfg, "v1")["passed"]       # +1.3%
    assert not compare_to_production({"rmse": 8.0}, {"rmse": 7.5}, cfg, "v1")["passed"]   # +6.7%
    assert compare_to_production({"rmse": 9.0}, None, cfg, "")["passed"]                  # first deploy
