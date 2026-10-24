"""Stage 6 - register.

Inputs : artifacts/train/meta.json, artifacts/evaluate/gate.json (must have passed)
Outputs: artifacts/register/registered.json
Effect : creates a new version of the registered model, tagged with full lineage,
         and points the ``candidate`` alias at it. Production is NOT touched here.
"""

from __future__ import annotations

import logging

from hpp.context import StageContext
from hpp.exceptions import ModelGateError
from hpp.tracking import Tracker
from hpp.utils import read_json, utcnow, write_json

log = logging.getLogger("hpp.register")


def register(ctx: StageContext) -> dict:
    ctx.require_inputs("register")
    cfg = ctx.cfg
    tmeta = read_json(ctx.artifacts / "train" / "meta.json")
    gate = read_json(ctx.artifacts / "evaluate" / "gate.json")
    if not gate.get("passed"):
        raise ModelGateError("refusing to register: the model gate did not pass - " + gate.get("message", ""), gate)
    if gate.get("mlflow_run_id") != tmeta["mlflow_run_id"]:
        raise ModelGateError("refusing to register: gate result belongs to a different training run. "
                             "Run `hpp stage evaluate` first.", gate)

    tracker = Tracker(cfg, ctx.root)
    cand = gate["candidate"]
    tags = {
        **tmeta["lineage"],
        "mlflow_run_id": tmeta["mlflow_run_id"],
        "model_type": tmeta["model_type"],
        "test_rmse": round(cand["rmse"], 4),
        "test_mae": round(cand["mae"], 4),
        "test_r2": round(cand["r2"], 4),
        "registered_at": utcnow(),
        "registered_by_pipeline_run": ctx.pipeline_run_id,
    }
    version = tracker.register(tmeta["mlflow_model_uri"], tags)
    tracker.transition(version, cfg.registry.candidate_alias, "candidate", ctx.pipeline_run_id,
                       "passed data gate and model gate")
    tracker.set_tags(tmeta["mlflow_run_id"], {"registered_version": version})
    out = {
        "stage": "register",
        "created_at": utcnow(),
        "pipeline_run_id": ctx.pipeline_run_id,
        "model_name": cfg.registry.model_name,
        "model_version": version,
        "alias": cfg.registry.candidate_alias,
        "mlflow_run_id": tmeta["mlflow_run_id"],
        "mlflow_model_uri": tmeta["mlflow_model_uri"],
    }
    write_json(ctx.stage_dir("register") / "registered.json", out)
    log.info("registered %s version %s as '%s'", cfg.registry.model_name, version, cfg.registry.candidate_alias)
    return {"model_version": version}
