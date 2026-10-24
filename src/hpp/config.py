"""Load, merge, interpolate and validate the pipeline configuration.

The YAML file is the single source of truth. Nothing in the code base carries a
path, threshold or hyperparameter of its own: every stage receives a validated
``Config`` object built here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hpp.exceptions import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    # protected_namespaces=(): fields such as model_type / model_name are ours, not pydantic internals
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ProjectCfg(_Strict):
    name: str
    seed: int = 42


class PathsCfg(_Strict):
    raw_train: str
    incoming_batches: str
    artifacts_dir: str
    logs_dir: str
    deployments_dir: str
    reports_dir: str


class DataCfg(_Strict):
    id_column: str
    batch_column: str
    target: str
    numeric_features: list[str] = Field(min_length=1)
    categorical_features: dict[str, list[str]]
    ordinal_features: list[str] = []
    non_predictive: list[str] = []
    train_with_batches: list[int] = []
    extra_training_files: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> DataCfg:
        feats = set(self.numeric_features) | set(self.categorical_features)
        if self.target in feats:
            raise ValueError("target must not be listed as a feature")
        leaked = feats & set(self.non_predictive)
        if leaked:
            raise ValueError(f"non-predictive columns used as features: {sorted(leaked)}")
        for col in self.ordinal_features:
            if col not in self.categorical_features:
                raise ValueError(f"ordinal feature {col!r} must be in categorical_features")
        for col, levels in self.categorical_features.items():
            if len(levels) < 2 or len(set(levels)) != len(levels):
                raise ValueError(f"categorical feature {col!r} needs >=2 unique levels")
        return self

    @property
    def feature_columns(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)


class ValidationCfg(_Strict):
    min_rows: int = Field(ge=1)
    max_missing_fraction: float = Field(ge=0, le=1)
    max_missing_fraction_target: float = Field(ge=0, le=1)
    max_invalid_row_fraction: float = Field(ge=0, le=1)
    max_duplicate_id_fraction: float = Field(ge=0, le=1)
    ranges: dict[str, tuple[float, float]]

    @field_validator("ranges")
    @classmethod
    def _ranges_ordered(cls, v: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        for col, (lo, hi) in v.items():
            if lo >= hi:
                raise ValueError(f"range for {col} must be [low, high] with low < high")
        return v


class FeaturesCfg(_Strict):
    test_size: float = Field(gt=0, lt=1)
    imputation_strategy: Literal["median", "mean", "most_frequent"]
    add_missing_indicators: bool = True
    engineered: list[Literal["Rooms_Total", "Area_per_Room", "Log_Distance", "Log_Area"]] = []
    reference_sample_rows: int = Field(ge=100)


class ModelSpec(_Strict):
    params: dict[str, Any] = {}
    search: dict[str, list[Any]] = {}

    @field_validator("search")
    @classmethod
    def _non_empty(cls, v: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for k, vals in v.items():
            if not vals:
                raise ValueError(f"search grid for {k!r} is empty")
        return v


class TrainingCfg(_Strict):
    model_type: Literal["gradient_boosting", "random_forest", "ridge"]
    cv_folds: int = Field(ge=2, le=20)
    scoring: str
    n_jobs: int = -1
    models: dict[str, ModelSpec]

    @model_validator(mode="after")
    def _model_present(self) -> TrainingCfg:
        if self.model_type not in self.models:
            raise ValueError(f"training.models has no entry for model_type={self.model_type!r}")
        return self

    @property
    def spec(self) -> ModelSpec:
        return self.models[self.model_type]


class ThresholdsCfg(_Strict):
    rmse_max: float = Field(gt=0)
    mae_max: float = Field(gt=0)
    r2_min: float = Field(ge=-1, le=1)
    mape_max: float = Field(gt=0)


class EvaluationCfg(_Strict):
    primary_metric: Literal["rmse", "mae"]
    thresholds: ThresholdsCfg
    compare_to_production: bool = True
    max_regression_pct: float = Field(ge=0, le=100)
    reproducibility_tolerance: float = Field(gt=0)


class RegistryCfg(_Strict):
    model_name: str
    candidate_alias: str
    production_alias: str
    previous_alias: str

    @model_validator(mode="after")
    def _distinct(self) -> RegistryCfg:
        aliases = [self.candidate_alias, self.production_alias, self.previous_alias]
        if len(set(aliases)) != 3:
            raise ValueError("registry aliases must be distinct")
        return self


class TrackingCfg(_Strict):
    uri: str
    experiment: str
    artifact_location: str


class OrchestrationCfg(_Strict):
    engine: Literal["prefect", "local"]
    flow_name: str


class DeploymentCfg(_Strict):
    api_reload_url: str = ""
    api_admin_token: str = ""


class MonitoringCfg(_Strict):
    batches: list[int] = Field(min_length=1)
    psi_bins: int = Field(ge=4, le=50)
    psi_warning: float = Field(gt=0)
    psi_critical: float = Field(gt=0)
    ks_pvalue: float = Field(gt=0, lt=1)
    drift_share_critical: float = Field(gt=0, le=1)
    rmse_ratio_warning: float = Field(gt=1)
    rmse_ratio_critical: float = Field(gt=1)
    bias_warning_pct: float = Field(gt=0)
    bias_critical_pct: float = Field(gt=0)
    missing_increase_warning: float = Field(gt=0, le=1)
    latency_p95_ms_max: float = Field(gt=0)
    error_rate_max: float = Field(ge=0, le=1)
    pushgateway_url: str = ""
    api_url: str = ""
    auto_retrain: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> MonitoringCfg:
        if self.psi_warning >= self.psi_critical:
            raise ValueError("psi_warning must be < psi_critical")
        if self.rmse_ratio_warning >= self.rmse_ratio_critical:
            raise ValueError("rmse_ratio_warning must be < rmse_ratio_critical")
        if self.bias_warning_pct >= self.bias_critical_pct:
            raise ValueError("bias_warning_pct must be < bias_critical_pct")
        return self


class Config(_Strict):
    project: ProjectCfg
    paths: PathsCfg
    data: DataCfg
    validation: ValidationCfg
    features: FeaturesCfg
    training: TrainingCfg
    evaluation: EvaluationCfg
    registry: RegistryCfg
    tracking: TrackingCfg
    orchestration: OrchestrationCfg
    deployment: DeploymentCfg
    monitoring: MonitoringCfg

    @model_validator(mode="after")
    def _cross_checks(self) -> Config:
        expected = set(self.data.numeric_features) | {self.data.target}
        missing = expected - set(self.validation.ranges)
        if missing:
            raise ValueError(f"validation.ranges missing bounds for: {sorted(missing)}")
        return self

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Stable hash of the effective configuration (used for lineage)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------- loading
def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into a copy of ``base`` (lists are replaced)."""
    out = copy.deepcopy(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return data


def load_config(path: str | Path = "configs/config.yaml", overrides: list[str | Path] | None = None,
                extra: dict[str, Any] | None = None) -> Config:
    """Load ``path``, deep-merge each overlay file then ``extra``, interpolate env vars, validate."""
    raw = _read_yaml(Path(path))
    for ov in overrides or []:
        raw = deep_merge(raw, _read_yaml(Path(ov)))
    if extra:
        raw = deep_merge(raw, extra)
    raw = _interpolate(raw)
    try:
        return Config.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> readable ConfigError
        raise ConfigError(f"configuration is invalid:\n{exc}") from exc
