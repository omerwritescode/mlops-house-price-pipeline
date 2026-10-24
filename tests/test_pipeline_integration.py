"""End-to-end: full pipeline, single-stage re-run, both gates failing safely, lineage, rollback.

Needs MLflow (installed in CI); skipped otherwise.
"""

import pytest

pytest.importorskip("mlflow")

from hpp.context import StageContext  # noqa: E402
from hpp.exceptions import DataGateError, ModelGateError, StageInputError  # noqa: E402
from hpp.lineage import trace  # noqa: E402
from hpp.orchestration import run_pipeline  # noqa: E402
from hpp.scenarios import make_bad_batch  # noqa: E402
from hpp.tracking import Tracker  # noqa: E402
from hpp.utils import dir_fingerprint, read_json  # noqa: E402

WEAK = {"training": {"models": {"gradient_boosting": {"search": {
    "n_estimators": [10], "learning_rate": [0.05], "max_depth": [1]}}}}}


def _prod_version(ctx: StageContext) -> str | None:
    mv = Tracker(ctx.cfg, ctx.root).version_by_alias(ctx.cfg.registry.production_alias)
    return str(mv.version) if mv else None


def test_stage_without_inputs_explains_what_to_run(project, ctx_factory):
    ctx = ctx_factory(project)
    with pytest.raises(StageInputError, match="ingest"):
        run_pipeline(ctx, stages=["features"])


def test_full_pipeline_gates_lineage_rollback(project, ctx_factory):
    # 1) happy path -> v1 deployed, monitoring report produced
    ctx = ctx_factory(project, run_id="run-1")
    res = run_pipeline(ctx)
    assert res["evaluate"]["passed"] and res["deploy"]["model_version"] == "1"
    assert res["monitor"]["batches"]["1"] == "OK" and res["monitor"]["batches"]["3"] == "CRITICAL"
    assert (project / "reports/monitoring/report.html").exists()
    fp_v1 = dir_fingerprint(ctx.current_bundle)
    assert _prod_version(ctx) == "1"

    # 2) any single stage can be re-run on its own
    ctx_t = ctx_factory(project, run_id="run-train-only")
    out = run_pipeline(ctx_t, stages=["train"])
    assert out["train"]["mlflow_run_id"]
    history = (project / "logs/run_history.jsonl").read_text()
    assert "run-train-only" in history and (project / "logs/run-1/train.log").exists()

    # 3) bad data batch -> data gate fails, production untouched
    make_bad_batch(ctx, 1, "data/scenarios/bad_batch.csv")
    ctx_bad = ctx_factory(project, run_id="run-bad", batch_file="data/scenarios/bad_batch.csv")
    with pytest.raises(DataGateError):
        run_pipeline(ctx_bad, skip_monitor=True)
    assert dir_fingerprint(ctx.current_bundle) == fp_v1 and _prod_version(ctx) == "1"
    assert not read_json(project / "artifacts/validate/report.json")["passed"]

    # downstream stages refuse to run on unvalidated data even when called directly
    with pytest.raises(DataGateError):
        run_pipeline(ctx_bad, stages=["features"])

    # 4) weak model -> model gate fails, production untouched
    ctx_weak = ctx_factory(project, extra=WEAK, run_id="run-weak")
    with pytest.raises(ModelGateError):
        run_pipeline(ctx_weak, skip_monitor=True)
    assert dir_fingerprint(ctx.current_bundle) == fp_v1 and _prod_version(ctx) == "1"
    with pytest.raises(ModelGateError):  # register refuses a failed gate
        run_pipeline(ctx_weak, stages=["register"])

    # 5) lineage from the deployed model
    info = trace(ctx_factory(project))
    assert info["deployed"]["model_version"] == "1"
    assert info["data"]["matches"] and info["config"]["matches"]
    assert info["registry"]["run_matches_manifest"] and info["audit"]["passed"]
    assert info["run"]["tags"]["data_version"] == info["data"]["data_version"]

    # 6) retrain with labelled batches -> v2 must beat production on the same test set; then roll back
    ctx2 = ctx_factory(project, extra={"data": {"train_with_batches": [1, 2, 3]}}, run_id="run-2")
    res2 = run_pipeline(ctx2, skip_monitor=True)
    assert res2["deploy"]["model_version"] == "2" and _prod_version(ctx) == "2"
    from hpp.deploy import rollback

    rb = rollback(ctx_factory(project, run_id="run-rollback"))
    assert rb["rolled_back_to"] == "1" and _prod_version(ctx) == "1"
    assert dir_fingerprint(ctx.current_bundle) == fp_v1
