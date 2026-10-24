"""Thin wrapper around MLflow tracking + model registry.

All MLflow access goes through this module so the rest of the code base talks in
pipeline terms (lineage tags, candidate/production aliases) rather than MLflow calls.

Registry lifecycle (MLflow aliases replace the deprecated "stages"):
    register  -> version gets alias  ``candidate``          tag lifecycle=candidate
    deploy    -> version gets alias  ``production``         tag lifecycle=production
                 old production gets ``previous-production`` tag lifecycle=archived
    rollback  -> the two production aliases are swapped
Every transition writes ``transitioned_by=pipeline`` + the pipeline run id onto the
version, so the lineage audit can prove no transition happened by hand.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hpp.config import Config
from hpp.utils import utcnow

log = logging.getLogger("hpp.tracking")


def _mlflow():
    try:
        import mlflow  # noqa: PLC0415

        return mlflow
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError("MLflow is required for this stage: pip install -r requirements.txt") from exc


class Tracker:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = root
        self.mlflow = _mlflow()
        uri = cfg.tracking.uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            # make relative sqlite paths relative to the project root, not the cwd
            uri = "sqlite:///" + str((root / uri[len("sqlite:///"):]).resolve())
        self.uri = uri
        self.mlflow.set_tracking_uri(uri)
        from mlflow.tracking import MlflowClient  # noqa: PLC0415

        self.client = MlflowClient(tracking_uri=uri)
        self.experiment_id = self._ensure_experiment()
        self.model_name = cfg.registry.model_name

    # ------------------------------------------------------------- experiment / runs
    def _ensure_experiment(self) -> str:
        name = self.cfg.tracking.experiment
        exp = self.mlflow.get_experiment_by_name(name)
        if exp is not None:
            self.mlflow.set_experiment(name)
            return exp.experiment_id
        loc = self.cfg.tracking.artifact_location
        if "://" not in loc:
            loc = (self.root / loc).resolve().as_uri()
        exp_id = self.mlflow.create_experiment(name, artifact_location=loc)
        self.mlflow.set_experiment(name)
        return exp_id

    def start_run(self, run_name: str, tags: dict[str, Any]) -> str:
        with self.mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name,
                                   tags={k: str(v) for k, v in tags.items()}) as run:
            return run.info.run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        clean = {k[:250]: str(v)[:500] for k, v in params.items()}
        with self.mlflow.start_run(run_id=run_id):
            self.mlflow.log_params(clean)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        with self.mlflow.start_run(run_id=run_id):
            self.mlflow.log_metrics(clean)

    def set_tags(self, run_id: str, tags: dict[str, Any]) -> None:
        with self.mlflow.start_run(run_id=run_id):
            self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, run_id: str, path: str | Path, artifact_path: str | None = None) -> None:
        if Path(path).exists():
            self.client.log_artifact(run_id, str(path), artifact_path)

    def log_model(self, run_id: str, model: Any, input_example: Any) -> str:
        from mlflow.models import infer_signature  # noqa: PLC0415

        signature = infer_signature(input_example, model.predict(input_example))
        with self.mlflow.start_run(run_id=run_id):
            info = self.mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                                 input_example=input_example)
        return info.model_uri

    def get_run(self, run_id: str) -> Any:
        return self.client.get_run(run_id)

    def artifact_uri(self, run_id: str) -> str:
        return self.client.get_run(run_id).info.artifact_uri

    # ------------------------------------------------------------- registry
    def register(self, model_uri: str, tags: dict[str, Any]) -> str:
        mv = self.mlflow.register_model(model_uri, self.model_name)
        for k, v in tags.items():
            self.client.set_model_version_tag(self.model_name, mv.version, k, str(v))
        return str(mv.version)

    def version_by_alias(self, alias: str) -> Any | None:
        try:
            return self.client.get_model_version_by_alias(self.model_name, alias)
        except Exception:  # MlflowException when alias / model does not exist
            return None

    def get_version(self, version: str) -> Any:
        return self.client.get_model_version(self.model_name, str(version))

    def transition(self, version: str, alias: str, lifecycle: str, pipeline_run_id: str,
                   reason: str) -> None:
        """Point ``alias`` at ``version`` and record who/why on the version (pipeline only)."""
        self.client.set_registered_model_alias(self.model_name, alias, str(version))
        stamp = {
            "lifecycle": lifecycle,
            "transitioned_by": "pipeline",
            f"transition.{alias}.at": utcnow(),
            f"transition.{alias}.pipeline_run_id": pipeline_run_id,
            f"transition.{alias}.reason": reason,
        }
        for k, v in stamp.items():
            self.client.set_model_version_tag(self.model_name, str(version), k, str(v))
        log.info("registry: %s v%s -> alias '%s' (%s)", self.model_name, version, alias, reason)

    def tag_version(self, version: str, tags: dict[str, Any]) -> None:
        for k, v in tags.items():
            self.client.set_model_version_tag(self.model_name, str(version), k, str(v))

    def load_by_alias(self, alias: str) -> Any | None:
        mv = self.version_by_alias(alias)
        if mv is None:
            return None
        return self.mlflow.sklearn.load_model(f"models:/{self.model_name}@{alias}")

    def all_versions(self) -> list[Any]:
        try:
            return list(self.client.search_model_versions(f"name='{self.model_name}'"))
        except Exception:
            return []


def lineage_tags(ctx_root: Path, pipeline_run_id: str, data_version: str, config_hash: str,
                 orchestrator: str) -> dict[str, Any]:
    from hpp import __version__  # noqa: PLC0415
    from hpp.utils import git_info  # noqa: PLC0415

    tags = {
        "pipeline_run_id": pipeline_run_id,
        "data_version": data_version,
        "config_hash": config_hash,
        "orchestrator": orchestrator,
        "hpp_version": __version__,
        **git_info(ctx_root),
    }
    if os.environ.get("GITHUB_RUN_ID"):
        tags["ci_run"] = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/" \
                         f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    return tags
