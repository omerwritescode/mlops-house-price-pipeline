"""Stage 7 - deploy, plus rollback.

Inputs : artifacts/register/registered.json, the trained model and its evaluation/reference files
Outputs: deployments/current/   (model.joblib, manifest.json, reference_profile.json, reference_sample.csv, metrics.json)
         deployments/previous/  (the bundle that was serving before - used for rollback)
         artifacts/deploy/deployed.json
Effect : registry alias ``production`` -> new version, ``previous-production`` -> old version;
         the serving API is told to hot-reload (if configured) - a container restart also picks it up.

The bundle is built in a staging directory and swapped in with an atomic rename, so the
API never sees a half-written model. If any earlier gate failed this stage never runs and
``deployments/current`` stays byte-for-byte identical (see scripts/demo_failures.py).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
from typing import Any

import sklearn

from hpp import __version__
from hpp.context import StageContext
from hpp.exceptions import PipelineError
from hpp.tracking import Tracker
from hpp.utils import dir_fingerprint, read_json, replace_dir_atomically, utcnow, write_json

log = logging.getLogger("hpp.deploy")


def notify_api(ctx: StageContext) -> str:
    url = ctx.cfg.deployment.api_reload_url
    if not url:
        return "not configured (API picks up the bundle on restart)"
    try:
        import requests  # noqa: PLC0415

        r = requests.post(url, headers={"X-Admin-Token": ctx.cfg.deployment.api_admin_token}, timeout=10)
        return f"{r.status_code} {r.text[:120]}"
    except Exception as exc:  # the deploy itself succeeded; reload failure is a warning
        log.warning("API reload call failed (%s) - restart the container to load the new model", exc)
        return f"failed: {exc}"


def build_manifest(ctx: StageContext, reg: dict, tmeta: dict, metrics: dict) -> dict[str, Any]:
    ingest_meta = read_json(ctx.artifacts / "ingest" / "meta.json")
    return {
        "model_name": reg["model_name"],
        "model_version": reg["model_version"],
        "mlflow_run_id": reg["mlflow_run_id"],
        "mlflow_model_uri": reg["mlflow_model_uri"],
        "model_type": tmeta["model_type"],
        "best_params": tmeta["best_params"],
        "git_commit": tmeta["lineage"].get("git_commit"),
        "git_branch": tmeta["lineage"].get("git_branch"),
        "git_dirty": tmeta["lineage"].get("git_dirty"),
        "data_version": tmeta["data_version"],
        "data_sources": ingest_meta["sources"],
        "train_with_batches": ingest_meta.get("train_with_batches", []),
        "config_hash": tmeta["config_hash"],
        "trained_by_pipeline_run": tmeta["pipeline_run_id"],
        "deployed_by_pipeline_run": ctx.pipeline_run_id,
        "deployed_at": utcnow(),
        "deployed_by": "pipeline",
        "image_tag": os.environ.get("IMAGE_TAG", "local"),
        "feature_columns": ctx.cfg.data.feature_columns,
        "target": ctx.cfg.data.target,
        "holdout_metrics": metrics["holdout"],
        "reference": metrics["reference"],
        "runtime": {"hpp": __version__, "sklearn": sklearn.__version__, "python": platform.python_version()},
    }


def deploy(ctx: StageContext) -> dict:
    ctx.require_inputs("deploy")
    cfg = ctx.cfg
    reg = read_json(ctx.artifacts / "register" / "registered.json")
    tmeta = read_json(ctx.artifacts / "train" / "meta.json")
    metrics = read_json(ctx.artifacts / "evaluate" / "metrics.json")
    if reg["mlflow_run_id"] != tmeta["mlflow_run_id"]:
        raise PipelineError("refusing to deploy: registered version does not belong to the current training run")

    tracker = Tracker(cfg, ctx.root)
    cand = tracker.version_by_alias(cfg.registry.candidate_alias)
    if cand is None or str(cand.version) != str(reg["model_version"]):
        raise PipelineError(f"refusing to deploy: alias '{cfg.registry.candidate_alias}' does not point at "
                            f"version {reg['model_version']} any more")

    # ---------------------------------------------------- build bundle in staging, then swap
    staging = ctx.deployments / f".staging-{ctx.pipeline_run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(ctx.artifacts / "train" / "model.joblib", staging / "model.joblib")
    shutil.copy2(ctx.artifacts / "features" / "reference_profile.json", staging / "reference_profile.json")
    shutil.copy2(ctx.artifacts / "features" / "reference_sample.csv", staging / "reference_sample.csv")
    shutil.copy2(ctx.artifacts / "evaluate" / "metrics.json", staging / "metrics.json")
    hp = ctx.artifacts / "evaluate" / "holdout_predictions.npy"
    if hp.exists():
        shutil.copy2(hp, staging / "holdout_predictions.npy")
    manifest = build_manifest(ctx, reg, tmeta, metrics)
    write_json(staging / "manifest.json", manifest)

    previous = tracker.version_by_alias(cfg.registry.production_alias)
    tracker.transition(reg["model_version"], cfg.registry.production_alias, "production", ctx.pipeline_run_id,
                       "deployed by pipeline")
    if previous is not None and str(previous.version) != str(reg["model_version"]):
        tracker.transition(previous.version, cfg.registry.previous_alias, "archived", ctx.pipeline_run_id,
                           f"replaced by v{reg['model_version']}")
    replace_dir_atomically(staging, ctx.current_bundle, ctx.previous_bundle)
    reload_result = notify_api(ctx)

    out = {
        "stage": "deploy",
        "created_at": utcnow(),
        "pipeline_run_id": ctx.pipeline_run_id,
        "model_version": reg["model_version"],
        "previous_version": str(previous.version) if previous is not None else None,
        "bundle": str(ctx.current_bundle),
        "bundle_fingerprint": dir_fingerprint(ctx.current_bundle),
        "api_reload": reload_result,
    }
    write_json(ctx.stage_dir("deploy") / "deployed.json", out)
    tracker.set_tags(reg["mlflow_run_id"], {"deployed": "true", "deployed_at": out["created_at"]})
    log.info("deployed %s v%s (previous: %s); bundle fingerprint %s; API reload: %s", reg["model_name"],
             reg["model_version"], out["previous_version"], out["bundle_fingerprint"], reload_result)
    return {"model_version": reg["model_version"], "previous_version": out["previous_version"]}


def rollback(ctx: StageContext) -> dict:
    """Restore the previous deployment (bundle + registry aliases). Tested in tests/test_deploy.py."""
    cfg = ctx.cfg
    if not ctx.previous_bundle.exists():
        raise PipelineError("nothing to roll back to: deployments/previous does not exist")
    cur_manifest = read_json(ctx.current_bundle / "manifest.json") if ctx.current_bundle.exists() else {}
    prev_manifest = read_json(ctx.previous_bundle / "manifest.json")

    # swap directories: previous -> current, current -> previous
    tmp = ctx.deployments / f".rollback-{ctx.pipeline_run_id}"
    shutil.copytree(ctx.previous_bundle, tmp)
    replace_dir_atomically(tmp, ctx.current_bundle, ctx.previous_bundle)

    registry_note = "skipped"
    try:
        tracker = Tracker(cfg, ctx.root)
        tracker.transition(prev_manifest["model_version"], cfg.registry.production_alias, "production",
                           ctx.pipeline_run_id, f"rollback from v{cur_manifest.get('model_version')}")
        if cur_manifest.get("model_version"):
            tracker.transition(cur_manifest["model_version"], cfg.registry.previous_alias, "archived",
                               ctx.pipeline_run_id, "rolled back")
        registry_note = "aliases swapped"
    except Exception as exc:
        log.warning("bundle rolled back but registry update failed: %s", exc)
        registry_note = f"failed: {exc}"

    reload_result = notify_api(ctx)
    out = {"rolled_back_from": cur_manifest.get("model_version"), "rolled_back_to": prev_manifest["model_version"],
           "registry": registry_note, "api_reload": reload_result, "at": utcnow(),
           "bundle_fingerprint": dir_fingerprint(ctx.current_bundle)}
    write_json(ctx.stage_dir("deploy") / "rollback.json", out)
    log.info("rolled back v%s -> v%s (%s)", out["rolled_back_from"], out["rolled_back_to"], registry_note)
    return out
