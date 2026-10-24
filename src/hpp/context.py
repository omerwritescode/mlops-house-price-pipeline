"""Shared run context and the declared inputs/outputs of every stage.

Every stage reads its inputs from, and writes its outputs to, ``artifacts/<stage>/``.
Because the contract is on disk, any single stage can be re-run on its own
(``hpp stage train``) without repeating upstream work - as long as the upstream
outputs exist. Missing inputs raise ``StageInputError`` naming the stage to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hpp.config import Config
from hpp.exceptions import StageInputError

STAGES: list[str] = ["ingest", "validate", "features", "train", "evaluate", "register", "deploy", "monitor"]

# Inputs are relative to the artifacts dir unless prefixed with "@deploy/" (deployments dir).
STAGE_IO: dict[str, dict[str, list[str]]] = {
    "ingest": {"inputs": [], "outputs": ["ingest/dataset.csv", "ingest/meta.json"]},
    "validate": {"inputs": ["ingest/dataset.csv", "ingest/meta.json"], "outputs": ["validate/report.json"]},
    "features": {
        "inputs": ["ingest/dataset.csv", "ingest/meta.json", "validate/report.json"],
        "outputs": ["features/train.csv", "features/test.csv", "features/reference_profile.json",
                    "features/reference_sample.csv", "features/meta.json"],
    },
    "train": {
        "inputs": ["features/train.csv", "features/meta.json"],
        "outputs": ["train/model.joblib", "train/meta.json", "train/cv_results.csv"],
    },
    "evaluate": {
        "inputs": ["train/model.joblib", "train/meta.json", "features/test.csv"],
        "outputs": ["evaluate/metrics.json", "evaluate/gate.json"],
    },
    "register": {"inputs": ["train/meta.json", "evaluate/gate.json"], "outputs": ["register/registered.json"]},
    "deploy": {
        "inputs": ["register/registered.json", "train/model.joblib", "evaluate/metrics.json",
                   "features/reference_profile.json", "features/reference_sample.csv"],
        "outputs": ["deploy/deployed.json", "@deploy/current/manifest.json"],
    },
    "monitor": {"inputs": ["@deploy/current/manifest.json", "@deploy/current/model.joblib"],
                "outputs": ["@reports/monitoring/summary.json", "@reports/monitoring/report.html"]},
}

PRODUCER: dict[str, str] = {}
for _stage, _io in STAGE_IO.items():
    for _out in _io["outputs"]:
        PRODUCER[_out] = _stage


@dataclass
class StageContext:
    cfg: Config
    root: Path
    pipeline_run_id: str
    config_path: str = "configs/config.yaml"
    overrides: list[str] = field(default_factory=list)
    batch_file: str | None = None          # newly arrived labelled batch to add to training (ingest)
    monitor_batch_file: str | None = None  # alternative file to monitor instead of the incoming batches
    orchestrator: str = "local"

    # ------------------------------------------------------------ paths
    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts(self) -> Path:
        return self.path(self.cfg.paths.artifacts_dir)

    @property
    def deployments(self) -> Path:
        return self.path(self.cfg.paths.deployments_dir)

    @property
    def reports(self) -> Path:
        return self.path(self.cfg.paths.reports_dir)

    @property
    def logs(self) -> Path:
        return self.path(self.cfg.paths.logs_dir)

    @property
    def current_bundle(self) -> Path:
        return self.deployments / "current"

    @property
    def previous_bundle(self) -> Path:
        return self.deployments / "previous"

    def stage_dir(self, stage: str) -> Path:
        d = self.artifacts / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve_io(self, rel: str) -> Path:
        if rel.startswith("@deploy/"):
            return self.deployments / rel[len("@deploy/"):]
        if rel.startswith("@reports/"):
            return self.reports / rel[len("@reports/"):]
        return self.artifacts / rel

    def require_inputs(self, stage: str) -> None:
        missing = [rel for rel in STAGE_IO[stage]["inputs"] if not self.resolve_io(rel).exists()]
        if missing:
            needed = sorted({PRODUCER.get(m, "?") for m in missing}, key=STAGES.index)
            raise StageInputError(
                f"stage '{stage}' cannot run: missing inputs {missing}. "
                f"Run upstream stage(s) first: {', '.join(needed)}  (e.g. `hpp stage {needed[0]}`)"
            )
