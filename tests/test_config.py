import pytest

from hpp.config import deep_merge, load_config
from hpp.exceptions import ConfigError


def test_default_config_is_valid(root):
    cfg = load_config(root / "configs/config.yaml")
    assert cfg.training.model_type == "gradient_boosting"
    assert cfg.data.target not in cfg.data.feature_columns
    assert "House_ID" not in cfg.data.feature_columns and "Batch" not in cfg.data.feature_columns


def test_all_overlays_are_valid(root):
    for ov in ["configs/ci.yaml", "configs/scenarios/weak_model.yaml", "configs/scenarios/regressed_model.yaml"]:
        load_config(root / "configs/config.yaml", [root / ov])


def test_hash_is_stable_and_sensitive(root):
    a = load_config(root / "configs/config.yaml").config_hash()
    b = load_config(root / "configs/config.yaml").config_hash()
    c = load_config(root / "configs/config.yaml", extra={"evaluation": {"thresholds": {"rmse_max": 8.5}}}).config_hash()
    assert a == b and a != c


@pytest.mark.parametrize("extra, fragment", [
    ({"evaluation": {"thresholds": {"rmse_max": -1}}}, "rmse_max"),
    ({"features": {"test_size": 1.5}}, "test_size"),
    ({"training": {"model_type": "xgboost"}}, "model_type"),
    ({"monitoring": {"psi_warning": 0.5}}, "psi_warning"),
    ({"validation": {"ranges": {"Area_sqft": [500, 100]}}}, "low < high"),
    ({"data": {"numeric_features": ["Area_sqft", "House_ID"]}}, "non-predictive"),
    ({"unknown_section": {"x": 1}}, "unknown_section"),
])
def test_invalid_configs_are_rejected(root, extra, fragment):
    with pytest.raises(ConfigError) as exc:
        load_config(root / "configs/config.yaml", extra=extra)
    assert fragment in str(exc.value)


def test_env_interpolation(root, tmp_path):
    import os

    os.environ["HPP_TEST_URI"] = "sqlite:///from-env.db"
    p = tmp_path / "c.yaml"
    p.write_text((root / "configs/config.yaml").read_text().replace(
        "${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}", "${HPP_TEST_URI:-nope}"))
    assert load_config(p).tracking.uri == "sqlite:///from-env.db"
    del os.environ["HPP_TEST_URI"]


def test_deep_merge_replaces_lists_and_merges_dicts():
    out = deep_merge({"a": {"b": 1, "c": [1, 2]}, "d": 1}, {"a": {"c": [3]}, "e": 2})
    assert out == {"a": {"b": 1, "c": [3]}, "d": 1, "e": 2}
