"""Retrain -> deploy v(n+1) -> rollback to v(n), and record the evidence (Task 6 rollback test, Task 7 retrain).

    python scripts/demo_rollback.py [--api http://localhost:8000]

Steps
  1. record the production state (bundle fingerprint, registry alias, optional live API answer)
  2. `hpp retrain --batches 1,2,3` (must pass both gates to be deployed; monitoring re-runs on the new model)
  3. `hpp rollback`  -> deployments/previous swapped back in, registry aliases restored
  4. `hpp monitor`   -> monitoring report regenerated for the restored production model
  5. check that the restored bundle fingerprint equals the one recorded in step 1
Writes docs/evidence/rollback_test.md + .json and docs/evidence/monitoring_after_retrain.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_failures import state  # noqa: E402

from hpp.utils import read_json, utcnow, write_json  # noqa: E402


def hpp(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hpp", *args], cwd=ROOT, capture_output=True, text=True)


def _last_json(text: str) -> dict | str:
    """The CLI prints log lines first and the JSON result last."""
    start = text.rfind("\n{")
    try:
        return json.loads(text[start + 1:] if start >= 0 else text)
    except ValueError:
        return text[-2000:]


def _monitor_levels(path: Path) -> dict:
    s = read_json(path)
    return {str(b["batch"]): {"level": b["level"], "rmse": round(b["performance"]["rmse"], 2),
                              "rmse_ratio": round(b["performance"]["rmse_ratio"], 2)} for b in s["batches"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    args = ap.parse_args()
    out = ROOT / "docs/evidence"
    summary = ROOT / "reports/monitoring/summary.json"

    before = state(args.api)
    levels_before = _monitor_levels(summary) if summary.exists() else {}

    retrain = hpp("retrain", "--batches", "1,2,3")
    after_retrain = state(args.api)
    gate = read_json(ROOT / "artifacts/evaluate/gate.json")
    if summary.exists():
        shutil.copyfile(summary, out / "monitoring_after_retrain.json")
    levels_retrain = _monitor_levels(summary) if summary.exists() else {}

    rollback = hpp("rollback")
    after_rollback = state(args.api)
    monitor = hpp("monitor")
    levels_restored = _monitor_levels(summary) if summary.exists() else {}

    restored = after_rollback["bundle_fingerprint"] == before["bundle_fingerprint"]
    res = {
        "generated_at": utcnow(),
        "retrain_exit_code": retrain.returncode,
        "retrain_deployed": after_retrain["bundle_fingerprint"] != before["bundle_fingerprint"],
        "retrain_gate": {k: gate.get(k) for k in ("passed", "message", "candidate", "production")},
        "rollback_exit_code": rollback.returncode,
        "rollback_output": _last_json(rollback.stdout) if rollback.returncode == 0 else rollback.stdout[-2000:],
        "monitor_exit_code": monitor.returncode,
        "state": {"before": before, "after_retrain": after_retrain, "after_rollback": after_rollback},
        "monitoring_levels": {"before": levels_before, "after_retrain": levels_retrain,
                              "after_rollback": levels_restored},
        "restored_to_original": restored,
    }
    write_json(out / "rollback_test.json", res)

    rows = "\n".join(f"| {k} | `{before.get(k)}` | `{after_retrain.get(k)}` | `{after_rollback.get(k)}` |"
                     for k in before)
    lv = "\n".join(
        f"| {b} | {levels_before.get(b, {}).get('level')} ({levels_before.get(b, {}).get('rmse')}) "
        f"| {levels_retrain.get(b, {}).get('level')} ({levels_retrain.get(b, {}).get('rmse')}) "
        f"| {levels_restored.get(b, {}).get('level')} ({levels_restored.get(b, {}).get('rmse')}) |"
        for b in sorted(set(levels_before) | set(levels_retrain)))
    md = f"""# Evidence: retrain, then rollback

Generated {res['generated_at']} by `python scripts/demo_rollback.py`.

1. `hpp retrain --batches 1,2,3` exited **{retrain.returncode}**. Model gate: {gate.get('message', '')}
2. `hpp rollback` exited **{rollback.returncode}**.
3. `hpp monitor` exited **{monitor.returncode}**. The monitoring report now belongs to the restored model.

**Restored bundle identical to the original: {restored}**

| | before | after retrain | after rollback |
|---|---|---|---|
{rows}

Monitoring level per batch (RMSE in lakhs):

| batch | before (production model) | after retrain | after rollback |
|---|---|---|---|
{lv}

Rollback output:

```json
{json.dumps(res['rollback_output'], indent=2) if isinstance(res['rollback_output'], dict) else res['rollback_output']}
```

Note: after the retrain, batches 1-3 were part of the training data, so for most of their rows the
"after retrain" column is in-sample. Only the batch rows that went to the holdout are unseen.
"""
    (out / "rollback_test.md").write_text(md)
    print(f"retrain exit {retrain.returncode}; deployed: {res['retrain_deployed']}; rollback exit "
          f"{rollback.returncode}; restored to original: {restored}")
    if retrain.returncode != 0:
        print(retrain.stdout[-3000:], retrain.stderr[-3000:])
    return 0 if (restored and rollback.returncode == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
