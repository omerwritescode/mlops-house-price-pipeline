"""Orchestration: the workflow graph, per-stage logging and run history.

Workflow graph (matches docs/workflow_graph.md and the Prefect UI):

    ingest -> validate[DATA GATE] -> features -> train -> evaluate[MODEL GATE] -> register -> deploy -> monitor

Engines
  prefect  every stage is a Prefect task inside the ``house-price-pipeline`` flow, so the Prefect UI
           shows the graph, task states, logs and the full run history.
  local    the same stage functions called in order (used by CI's lightweight run and the unit tests).

Regardless of engine, every stage execution appends one line to ``logs/run_history.jsonl``
and writes its own log file to ``logs/<pipeline_run_id>/<stage>.log``.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from hpp.context import STAGES, StageContext
from hpp.exceptions import GateError, PipelineError
from hpp.utils import StageLogFile, append_jsonl, utcnow

log = logging.getLogger("hpp.orchestration")


def stage_functions() -> dict[str, Callable[[StageContext], dict]]:
    from hpp.data import ingest  # noqa: PLC0415
    from hpp.deploy import deploy  # noqa: PLC0415
    from hpp.evaluate import evaluate  # noqa: PLC0415
    from hpp.features import features  # noqa: PLC0415
    from hpp.monitor import monitor  # noqa: PLC0415
    from hpp.registry import register  # noqa: PLC0415
    from hpp.train import train  # noqa: PLC0415
    from hpp.validation import validate  # noqa: PLC0415

    return {"ingest": ingest, "validate": validate, "features": features, "train": train, "evaluate": evaluate,
            "register": register, "deploy": deploy, "monitor": monitor}


def execute_stage(ctx: StageContext, stage: str) -> dict[str, Any]:
    """Run one stage with logging, timing and a run-history record. Re-raises failures."""
    fn = stage_functions()[stage]
    record: dict[str, Any] = {"pipeline_run_id": ctx.pipeline_run_id, "stage": stage, "started_at": utcnow(),
                              "orchestrator": ctx.orchestrator, "config_hash": ctx.cfg.config_hash()}
    t0 = time.time()
    with StageLogFile(ctx.logs / ctx.pipeline_run_id / f"{stage}.log"):
        log.info("==> stage %-8s | run %s", stage, ctx.pipeline_run_id)
        try:
            result = fn(ctx) or {}
            record.update(status="success", result=result)
            log.info("<== stage %-8s | success in %.1fs | %s", stage, time.time() - t0, result)
            return result
        except GateError as exc:
            record.update(status="gate_failed", gate=exc.gate, error=str(exc))
            log.error("<== stage %-8s | %s GATE FAILED: %s", stage, exc.gate.upper(), exc)
            raise
        except PipelineError as exc:
            record.update(status="failed", error=str(exc))
            log.error("<== stage %-8s | failed: %s", stage, exc)
            raise
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
            log.exception("<== stage %-8s | unexpected error", stage)
            raise
        finally:
            record.update(finished_at=utcnow(), duration_s=round(time.time() - t0, 2))
            append_jsonl(ctx.logs / "run_history.jsonl", record)


def _select(stages: list[str] | None, until: str | None, skip_monitor: bool) -> list[str]:
    seq = list(stages) if stages else list(STAGES)
    if until:
        seq = seq[: seq.index(until) + 1]
    if skip_monitor and "monitor" in seq:
        seq.remove("monitor")
    unknown = [s for s in seq if s not in STAGES]
    if unknown:
        raise PipelineError(f"unknown stage(s): {unknown}; valid: {STAGES}")
    return seq


def run_local(ctx: StageContext, stages: list[str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in stages:
        results[stage] = execute_stage(ctx, stage)
    return results


def run_prefect(ctx: StageContext, stages: list[str]) -> dict[str, Any]:
    from prefect import flow, task  # noqa: PLC0415

    try:
        from prefect.cache_policies import NONE as NO_CACHE  # noqa: PLC0415

        task_kwargs: dict[str, Any] = {"cache_policy": NO_CACHE}
    except ImportError:  # older Prefect
        task_kwargs = {}

    tasks = {s: task(name=s, retries=0, persist_result=False, **task_kwargs)(
        lambda c, _s=s: execute_stage(c, _s)) for s in stages}

    @flow(name=ctx.cfg.orchestration.flow_name, log_prints=True, persist_result=False,
          flow_run_name=f"run-{ctx.pipeline_run_id}")
    def pipeline_flow() -> dict[str, Any]:
        results: dict[str, Any] = {}
        for s in stages:
            results[s] = tasks[s](ctx)
        return results

    return pipeline_flow()


def run_pipeline(ctx: StageContext, stages: list[str] | None = None, until: str | None = None,
                 skip_monitor: bool = False) -> dict[str, Any]:
    seq = _select(stages, until, skip_monitor)
    engine = ctx.cfg.orchestration.engine
    if engine == "prefect":
        try:
            import prefect  # noqa: F401, PLC0415
        except ImportError:
            log.warning("Prefect not installed - falling back to the local engine")
            engine = "local"
    ctx.orchestrator = engine
    log.info("pipeline run %s | engine=%s | stages=%s | config_hash=%s", ctx.pipeline_run_id, engine,
             " -> ".join(seq), ctx.cfg.config_hash())
    return run_prefect(ctx, seq) if engine == "prefect" else run_local(ctx, seq)
