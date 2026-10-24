"""Lineage: start from the model that is *deployed* and walk back to everything that produced it.

    deployed bundle / live API /model-info
      -> registry version (aliases, tags, who transitioned it)
      -> MLflow run (params, metrics, lineage tags, logged config)
      -> code version (git commit, verified against the local repository)
      -> data version (content hashes, re-computed from the files on disk)
      -> configuration (config hash, re-computed from the config logged with the run)
      -> pipeline run (stage history from logs/run_history.jsonl)

``hpp lineage`` prints this as a report; ``hpp lineage --json`` returns the same as JSON.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from hpp.context import StageContext
from hpp.exceptions import PipelineError
from hpp.utils import combined_hash, file_sha256, read_json


def _manifest_from_api(url: str) -> dict:
    import requests  # noqa: PLC0415

    r = requests.get(url.rstrip("/") + "/model-info", timeout=10)
    r.raise_for_status()
    return r.json()


def _git_commit_exists(commit: str, root: Path) -> bool | None:
    if not commit or commit == "unknown":
        return None
    try:
        subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"], check=True, capture_output=True, timeout=5)
        return True
    except Exception:
        return False


def _git_subject(commit: str, root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%h %an %ad %s", "--date=short", commit],
                              capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        return None


def trace(ctx: StageContext, api_url: str | None = None) -> dict[str, Any]:
    if api_url:
        manifest, source = _manifest_from_api(api_url), f"live API {api_url}/model-info"
    else:
        path = ctx.current_bundle / "manifest.json"
        if not path.exists():
            raise PipelineError("no deployed model found (deployments/current/manifest.json). Run the pipeline first.")
        manifest, source = read_json(path), str(path)

    out: dict[str, Any] = {"start": source, "deployed": {k: manifest.get(k) for k in (
        "model_name", "model_version", "image_tag", "deployed_at", "deployed_by", "deployed_by_pipeline_run")}}

    # ---------------- data: recompute hashes of the recorded sources
    data_checks, hashes = [], []
    for src in manifest.get("data_sources", []):
        p = ctx.path(src["path"]) if not Path(src["path"]).is_absolute() else Path(src["path"])
        now = file_sha256(p) if p.exists() else None
        hashes.append(src["sha256"])
        data_checks.append({"path": src["path"], "rows": src["rows"], "note": src.get("note"),
                            "recorded_sha256": src["sha256"][:16], "current_sha256": now[:16] if now else None,
                            "matches": now == src["sha256"], "dvc_md5": src.get("dvc_md5")})
    recomputed = combined_hash(hashes + [f"batches={sorted(manifest.get('train_with_batches', []))}"])[:16]
    out["data"] = {"data_version": manifest.get("data_version"), "recomputed": recomputed,
                   "matches": recomputed == manifest.get("data_version"), "sources": data_checks}

    # ---------------- code
    commit = manifest.get("git_commit")
    out["code"] = {"git_commit": commit, "git_branch": manifest.get("git_branch"),
                   "dirty_at_training": manifest.get("git_dirty"),
                   "found_in_local_repo": _git_commit_exists(commit, ctx.root), "commit": _git_subject(commit, ctx.root)}

    # ---------------- registry + run (MLflow)
    try:
        from hpp.tracking import Tracker  # noqa: PLC0415

        tracker = Tracker(ctx.cfg, ctx.root)
        mv = tracker.get_version(manifest["model_version"])
        aliases = list(getattr(mv, "aliases", []) or [])
        tags = dict(mv.tags or {})
        out["registry"] = {"model": manifest["model_name"], "version": mv.version, "aliases": aliases,
                           "source_run_id": mv.run_id, "lifecycle": tags.get("lifecycle"),
                           "transitioned_by": tags.get("transitioned_by"),
                           "transitions": {k: v for k, v in tags.items() if k.startswith("transition.")}}
        run = tracker.get_run(manifest["mlflow_run_id"])
        rtags = run.data.tags
        out["run"] = {"run_id": run.info.run_id, "run_name": rtags.get("mlflow.runName"),
                      "artifact_uri": run.info.artifact_uri, "status": run.info.status,
                      "params": dict(run.data.params), "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
                      "tags": {k: v for k, v in rtags.items() if not k.startswith("mlflow.")}}
        out["registry"]["run_matches_manifest"] = mv.run_id == manifest["mlflow_run_id"]
        # config: re-hash the config file that was logged with the run
        try:
            import yaml  # noqa: PLC0415
            from mlflow.artifacts import download_artifacts  # noqa: PLC0415

            from hpp.config import Config  # noqa: PLC0415

            local = download_artifacts(run_id=run.info.run_id, artifact_path="config/effective_config.yaml",
                                       tracking_uri=tracker.uri)
            logged_cfg = Config.model_validate(yaml.safe_load(Path(local).read_text()))
            out["config"] = {"config_hash": manifest.get("config_hash"), "logged_config_hash": logged_cfg.config_hash(),
                             "matches": logged_cfg.config_hash() == manifest.get("config_hash"),
                             "logged_config": f"{run.info.artifact_uri}/config/effective_config.yaml",
                             "model_type": logged_cfg.training.model_type}
        except Exception as exc:
            out["config"] = {"config_hash": manifest.get("config_hash"), "error": str(exc)}
        out["audit"] = audit_registry(tracker)
    except Exception as exc:
        out["registry"] = {"error": f"MLflow unavailable: {exc}"}
        out["config"] = {"config_hash": manifest.get("config_hash")}

    # ---------------- pipeline run history
    hist = ctx.logs / "run_history.jsonl"
    runs = {manifest.get("trained_by_pipeline_run"), manifest.get("deployed_by_pipeline_run")}
    if hist.exists():
        entries = [json.loads(line) for line in hist.read_text().splitlines() if line.strip()]
        out["pipeline_runs"] = [{k: e.get(k) for k in ("pipeline_run_id", "stage", "status", "duration_s",
                                                        "orchestrator", "finished_at")}
                                for e in entries if e.get("pipeline_run_id") in runs]
    out["metrics_at_deploy"] = manifest.get("holdout_metrics")
    return out


def audit_registry(tracker: Any) -> dict[str, Any]:
    """Every version that ever held an alias must have been transitioned by the pipeline."""
    offenders, checked = [], 0
    for mv in tracker.all_versions():
        tags = dict(mv.tags or {})
        if tags.get("lifecycle") or getattr(mv, "aliases", None):
            checked += 1
            if tags.get("transitioned_by") != "pipeline":
                offenders.append(str(mv.version))
    return {"versions_checked": checked, "manual_transitions": offenders, "passed": not offenders}


def render(info: dict[str, Any]) -> str:
    def ok(flag: Any) -> str:
        return "✔" if flag is True else ("✘" if flag is False else "?")

    d, c = info["deployed"], info["code"]
    lines = [
        "LINEAGE OF THE DEPLOYED MODEL",
        f"  start point        : {info['start']}",
        f"  deployed model     : {d['model_name']} v{d['model_version']}  image={d['image_tag']}  at {d['deployed_at']}",
    ]
    reg = info.get("registry", {})
    if "error" in reg:
        lines.append(f"  registry           : {reg['error']}")
    else:
        lines += [f"  registry version   : v{reg['version']} aliases={reg['aliases']} lifecycle={reg['lifecycle']} "
                  f"transitioned_by={reg['transitioned_by']}",
                  f"  -> MLflow run      : {info['run']['run_id']} ({info['run']['run_name']}) "
                  f"{ok(reg.get('run_matches_manifest'))} matches manifest",
                  f"     artifacts       : {info['run']['artifact_uri']}",
                  f"     metrics         : " + ", ".join(f"{k}={v}" for k, v in sorted(info['run']['metrics'].items())
                                                       if k.startswith(("test_", "cv_")))]
    lines += [f"  -> code version    : {c['git_commit']} (branch {c['git_branch']}, dirty={c['dirty_at_training']}) "
              f"{ok(c['found_in_local_repo'])} found in local repo",
              f"     commit          : {c.get('commit') or '-'}",
              f"  -> data version    : {info['data']['data_version']} {ok(info['data']['matches'])} re-computed from files"]
    for s in info["data"]["sources"]:
        lines.append(f"     {ok(s['matches'])} {s['path']} rows={s['rows']} sha256={s['recorded_sha256']}… ({s['note']})")
    cfgi = info.get("config", {})
    lines.append(f"  -> configuration   : hash {cfgi.get('config_hash')} "
                 + (f"{ok(cfgi.get('matches'))} matches config logged with run ({cfgi.get('logged_config')})"
                    if "logged_config_hash" in cfgi else cfgi.get("error", "")))
    if info.get("pipeline_runs"):
        lines.append("  -> pipeline runs   :")
        for e in info["pipeline_runs"]:
            lines.append(f"     {e['pipeline_run_id']}  {e['stage']:<9} {e['status']:<12} {e['duration_s']}s  "
                         f"[{e['orchestrator']}]")
    if info.get("audit"):
        a = info["audit"]
        lines.append(f"  registry audit     : {ok(a['passed'])} {a['versions_checked']} versions checked, "
                     f"manual transitions: {a['manual_transitions'] or 'none'}")
    return "\n".join(lines)
