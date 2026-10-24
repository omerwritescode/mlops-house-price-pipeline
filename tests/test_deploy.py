"""Bundle swap + rollback on the filesystem (registry part is covered by the integration test)."""

import json

from hpp.deploy import rollback
from hpp.utils import dir_fingerprint, replace_dir_atomically


def _bundle(path, version):
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"model_name": "m", "model_version": version}))
    (path / "model.joblib").write_bytes(f"model-{version}".encode())


def test_atomic_replace_keeps_previous(tmp_path):
    cur, prev, stg = tmp_path / "current", tmp_path / "previous", tmp_path / ".staging"
    _bundle(cur, "1")
    _bundle(stg, "2")
    replace_dir_atomically(stg, cur, prev)
    assert json.loads((cur / "manifest.json").read_text())["model_version"] == "2"
    assert json.loads((prev / "manifest.json").read_text())["model_version"] == "1"
    assert not stg.exists()


def test_rollback_restores_previous_bundle(project, ctx_factory):
    ctx = ctx_factory(project)
    _bundle(ctx.previous_bundle, "1")
    _bundle(ctx.current_bundle, "2")
    fp_v1 = dir_fingerprint(ctx.previous_bundle)
    out = rollback(ctx)
    assert out["rolled_back_from"] == "2" and out["rolled_back_to"] == "1"
    assert dir_fingerprint(ctx.current_bundle) == fp_v1
    assert json.loads((ctx.previous_bundle / "manifest.json").read_text())["model_version"] == "2"
