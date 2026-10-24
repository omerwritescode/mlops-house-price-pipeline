"""Command-line entry point: ``hpp <command>`` (or ``python -m hpp``).

  hpp run                          full pipeline: ingest -> ... -> deploy -> monitor
  hpp run --batch-file F           retrain with a newly arrived labelled batch (data gate checks it)
  hpp retrain --batches 1,2,3      retrain on training data + labelled incoming batches
  hpp stage <name>                 re-run ONE stage from its on-disk inputs
  hpp monitor                      score + drift-check the incoming batches against production
  hpp lineage [--api-url URL]      trace the deployed model back to code, data, config and run
  hpp rollback                     restore the previous deployment (bundle + registry aliases)
  hpp history                      show the stage-level run history
  hpp validate-config              validate the YAML (and overlays) and print the config hash
  hpp make-bad-batch               derive a corrupted batch for the data-gate failure demo
  hpp verify-repro                 compare current holdout metrics with the committed reference

Exit codes: 0 ok, 2 data gate failed, 3 model gate failed, 4 bad config, 5 missing stage inputs, 1 other.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from hpp.config import load_config
from hpp.context import STAGES, StageContext
from hpp.exceptions import GateError, PipelineError
from hpp.utils import dir_fingerprint, new_run_id, read_json, setup_logging, write_json

log = logging.getLogger("hpp.cli")


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "configs").is_dir() and (p / "pyproject.toml").exists():
            return p
    return start


def _ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",") if x.strip()] if s else None


def build_context(args: argparse.Namespace) -> StageContext:
    root = Path(args.root).resolve() if args.root else _find_root(Path.cwd())
    extra: dict = {}
    if getattr(args, "engine", None):
        extra.setdefault("orchestration", {})["engine"] = args.engine
    if getattr(args, "train_with_batches", None):
        extra.setdefault("data", {})["train_with_batches"] = _ints(args.train_with_batches)
    if getattr(args, "api_url", None):
        extra.setdefault("monitoring", {})["api_url"] = args.api_url
    cfg_path = args.config if Path(args.config).is_absolute() else str(root / args.config)
    overrides = [o if Path(o).is_absolute() else str(root / o) for o in (args.override or [])]
    cfg = load_config(cfg_path, overrides, extra or None)
    return StageContext(cfg=cfg, root=root, pipeline_run_id=getattr(args, "run_id", None) or new_run_id(),
                        config_path=cfg_path, overrides=overrides,
                        batch_file=getattr(args, "batch_file", None),
                        monitor_batch_file=getattr(args, "file", None))


def _production_state(ctx: StageContext) -> str:
    man = ctx.current_bundle / "manifest.json"
    if not man.exists():
        return "no model deployed yet"
    m = read_json(man)
    return f"{m['model_name']} v{m['model_version']} (bundle fingerprint {dir_fingerprint(ctx.current_bundle)})"


def _gate_banner(ctx: StageContext, exc: GateError) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  PIPELINE STOPPED SAFELY - {exc.gate.upper()} GATE FAILED\n{bar}")
    for part in str(exc).split(" | "):
        print(f"  - {part}")
    print(f"\n  Production model unchanged : {_production_state(ctx)}")
    print("  Nothing was registered or deployed; the previous model keeps serving.")
    stage = "validate" if exc.gate == "data" else "evaluate"
    evidence = ctx.artifacts / stage / ("report.json" if exc.gate == "data" else "gate.json")
    try:
        evidence = evidence.resolve().relative_to(ctx.root.resolve())
    except ValueError:
        pass
    print(f"  Evidence                   : {evidence}")
    print(f"{bar}\n")


# ------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    from hpp.orchestration import run_pipeline  # noqa: PLC0415

    ctx = build_context(args)
    results = run_pipeline(ctx, until=args.until, skip_monitor=args.skip_monitor)
    print(json.dumps({"pipeline_run_id": ctx.pipeline_run_id, "results": results}, indent=2, default=str))
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    args.train_with_batches = args.batches
    return cmd_run(args)


def cmd_stage(args: argparse.Namespace) -> int:
    from hpp.orchestration import run_pipeline  # noqa: PLC0415

    ctx = build_context(args)
    results = run_pipeline(ctx, stages=[args.name])
    print(json.dumps(results, indent=2, default=str))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    from hpp.monitor import monitor  # noqa: PLC0415
    from hpp.orchestration import execute_stage  # noqa: PLC0415

    ctx = build_context(args)
    ctx.orchestrator = "cli"
    if args.batches:
        res = monitor(ctx, _ints(args.batches))
    else:
        res = execute_stage(ctx, "monitor")
    print(json.dumps(res, indent=2))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    from hpp.lineage import render, trace  # noqa: PLC0415

    ctx = build_context(args)
    info = trace(ctx, args.api_url)
    print(json.dumps(info, indent=2, default=str) if args.json else render(info))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from hpp.deploy import rollback  # noqa: PLC0415
    from hpp.utils import append_jsonl, utcnow  # noqa: PLC0415

    ctx = build_context(args)
    res = rollback(ctx)
    append_jsonl(ctx.logs / "run_history.jsonl", {"pipeline_run_id": ctx.pipeline_run_id, "stage": "rollback",
                                                  "status": "success", "result": res, "finished_at": utcnow()})
    print(json.dumps(res, indent=2))
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    ctx = build_context(args)
    path = ctx.logs / "run_history.jsonl"
    if not path.exists():
        print("no runs recorded yet")
        return 0
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()][-args.last:]
    print(f"{'pipeline_run_id':<24} {'stage':<9} {'status':<12} {'secs':>6}  finished_at           detail")
    for r in rows:
        detail = r.get("error") or json.dumps(r.get("result", {}), default=str)
        print(f"{r.get('pipeline_run_id', ''):<24} {r.get('stage', ''):<9} {r.get('status', ''):<12} "
              f"{r.get('duration_s', ''):>6}  {r.get('finished_at', ''):<21} {detail[:90]}")
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    ctx = build_context(args)
    print(f"configuration valid | hash {ctx.cfg.config_hash()} | model={ctx.cfg.training.model_type} "
          f"| engine={ctx.cfg.orchestration.engine} | overlays={[Path(o).name for o in ctx.overrides]}")
    return 0


def cmd_make_bad_batch(args: argparse.Namespace) -> int:
    from hpp.scenarios import make_bad_batch  # noqa: PLC0415

    ctx = build_context(args)
    path, summary = make_bad_batch(ctx, args.source_batch, args.out)
    print(f"wrote {path}\n" + json.dumps(summary, indent=2))
    return 0


def cmd_verify_repro(args: argparse.Namespace) -> int:
    ctx = build_context(args)
    current = read_json(ctx.artifacts / "evaluate" / "metrics.json")["holdout"]
    ref_path = ctx.path(args.expected)
    if args.write or not ref_path.exists():
        write_json(ref_path, {k: current[k] for k in ("rmse", "mae", "r2", "mape", "n")})
        print(f"reference metrics written to {ref_path}")
        return 0
    expected = read_json(ref_path)
    tol = ctx.cfg.evaluation.reproducibility_tolerance
    ok = True
    for k in ("rmse", "mae"):
        diff = abs(current[k] - expected[k])
        flag = "OK " if diff <= tol else "BAD"
        ok &= diff <= tol
        print(f"{flag} {k}: reproduced {current[k]:.4f} vs reported {expected[k]:.4f} (|diff| {diff:.4f} <= {tol})")
    print("REPRODUCED within tolerance" if ok else "NOT reproduced within tolerance")
    return 0 if ok else 6


# ------------------------------------------------------------------- parser
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hpp", description="House-price MLOps pipeline")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=os.environ.get("HPP_CONFIG", "configs/config.yaml"))
    common.add_argument("--override", action="append", help="YAML overlay deep-merged on top (repeatable)")
    common.add_argument("--root", default=None, help="project root (default: auto-detect)")
    common.add_argument("--engine", choices=["prefect", "local"], default=None)
    common.add_argument("--log-level", default=os.environ.get("HPP_LOG_LEVEL", "INFO"))
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", parents=[common], help="run the full pipeline")
    r.add_argument("--batch-file", help="newly arrived labelled batch to add to training")
    r.add_argument("--train-with-batches", help="comma list of incoming batch numbers to add to training")
    r.add_argument("--until", choices=STAGES, help="stop after this stage")
    r.add_argument("--skip-monitor", action="store_true")
    r.add_argument("--run-id", default=None)
    r.set_defaults(func=cmd_run)

    rt = sub.add_parser("retrain", parents=[common], help="retrain including labelled incoming batches")
    rt.add_argument("--batches", required=True, help="e.g. 1,2,3")
    rt.add_argument("--batch-file", default=None)
    rt.add_argument("--until", choices=STAGES, default=None)
    rt.add_argument("--skip-monitor", action="store_true")
    rt.add_argument("--run-id", default=None)
    rt.set_defaults(func=cmd_retrain)

    s = sub.add_parser("stage", parents=[common], help="re-run a single stage")
    s.add_argument("name", choices=STAGES)
    s.add_argument("--batch-file", default=None)
    s.add_argument("--train-with-batches", default=None)
    s.add_argument("--run-id", default=None)
    s.set_defaults(func=cmd_stage)

    m = sub.add_parser("monitor", parents=[common], help="monitor incoming batches against production")
    m.add_argument("--batches", default=None)
    m.add_argument("--file", default=None, help="monitor this CSV instead of the incoming batches")
    m.add_argument("--api-url", default=None, help="replay rows through the live API")
    m.set_defaults(func=cmd_monitor)

    li = sub.add_parser("lineage", parents=[common], help="trace the deployed model")
    li.add_argument("--api-url", default=None)
    li.add_argument("--json", action="store_true")
    li.set_defaults(func=cmd_lineage)

    sub.add_parser("rollback", parents=[common], help="restore the previous deployment").set_defaults(func=cmd_rollback)

    h = sub.add_parser("history", parents=[common], help="stage-level run history")
    h.add_argument("--last", type=int, default=30)
    h.set_defaults(func=cmd_history)

    sub.add_parser("validate-config", parents=[common]).set_defaults(func=cmd_validate_config)

    b = sub.add_parser("make-bad-batch", parents=[common], help="derive a corrupted batch (failure demo)")
    b.add_argument("--source-batch", type=int, default=1)
    b.add_argument("--out", default="data/scenarios/bad_batch.csv")
    b.set_defaults(func=cmd_make_bad_batch)

    v = sub.add_parser("verify-repro", parents=[common], help="check metrics reproduce within tolerance")
    v.add_argument("--expected", default="reports/reference_metrics.json")
    v.add_argument("--write", action="store_true", help="(re)write the reference file from the current run")
    v.set_defaults(func=cmd_verify_repro)
    return p


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except GateError as exc:
        try:
            _gate_banner(build_context(args), exc)
        except Exception:
            print(f"PIPELINE STOPPED - {exc}")
        return exc.exit_code
    except PipelineError as exc:
        print(f"\nPIPELINE STOPPED: {exc}\n", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # Prefect wraps task exceptions; unwrap gate errors if present
        cause = exc
        while cause is not None and not isinstance(cause, PipelineError):
            cause = cause.__cause__ or cause.__context__
        if isinstance(cause, GateError):
            _gate_banner(build_context(args), cause)
            return cause.exit_code
        if isinstance(cause, PipelineError):
            print(f"\nPIPELINE STOPPED: {cause}\n", file=sys.stderr)
            return cause.exit_code
        raise


if __name__ == "__main__":
    sys.exit(main())
