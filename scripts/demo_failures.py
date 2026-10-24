"""Run the failure scenarios and record evidence that production did not change (Task 4).

    python scripts/demo_failures.py [--api http://localhost:8000]

For each scenario it captures, before and after:
  * the deployed bundle fingerprint (hash of every file in deployments/current)
  * the registry version behind the 'production' alias
  * (optional) /model-info from the live API
and writes docs/evidence/failure_scenarios.md + .json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpp.config import load_config  # noqa: E402
from hpp.utils import dir_fingerprint, read_json, utcnow, write_json  # noqa: E402

SCENARIOS = [
    ("1. Bad data batch -> DATA GATE", ["run", "--batch-file", "data/scenarios/bad_batch.csv", "--skip-monitor"], 2),
    ("2. Model below threshold -> MODEL GATE", ["run", "--override", "configs/scenarios/weak_model.yaml",
                                                "--skip-monitor"], 3),
    ("3. Model worse than production -> MODEL GATE", ["run", "--override", "configs/scenarios/regressed_model.yaml",
                                                      "--skip-monitor"], 3),
]


def state(api: str | None) -> dict:
    cfg = load_config(ROOT / "configs/config.yaml")
    cur = ROOT / cfg.paths.deployments_dir / "current"
    st = {"bundle_fingerprint": dir_fingerprint(cur),
          "bundle_model_version": read_json(cur / "manifest.json")["model_version"] if cur.exists() else None}
    try:
        from hpp.tracking import Tracker

        mv = Tracker(cfg, ROOT).version_by_alias(cfg.registry.production_alias)
        st["registry_production_version"] = mv.version if mv else None
    except Exception as exc:
        st["registry_production_version"] = f"unavailable: {exc}"
    if api:
        try:
            import requests

            info = requests.get(f"{api}/model-info", timeout=5).json()
            st["api_model_version"] = info.get("model_version")
            st["api_prediction_sample"] = requests.post(f"{api}/predict", timeout=5, json={
                "Area_sqft": 1150, "Bedrooms": 2, "Bathrooms": 2, "Age_Years": 12, "Distance_to_Center_km": 9.5,
                "Neighborhood_Score": 6.5, "Has_Parking": 1, "Property_Condition": "Good",
                "City_Zone": "Central"}).json()
        except Exception as exc:
            st["api_model_version"] = f"unreachable: {exc}"
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    args = ap.parse_args()
    subprocess.run([sys.executable, "-m", "hpp", "make-bad-batch"], cwd=ROOT, check=True)
    results = []
    for name, cmd, expected in SCENARIOS:
        before = state(args.api)
        proc = subprocess.run([sys.executable, "-m", "hpp", *cmd], cwd=ROOT, capture_output=True, text=True)
        after = state(args.api)
        banner = proc.stdout[proc.stdout.find("=" * 20):] if "=" * 20 in proc.stdout else proc.stdout[-1500:]
        res = {"scenario": name, "command": "hpp " + " ".join(cmd), "exit_code": proc.returncode,
               "expected_exit_code": expected, "before": before, "after": after,
               "production_unchanged": before == after, "output": banner.strip()}
        results.append(res)
        print(f"{name}: exit {proc.returncode} (expected {expected}); production unchanged: {before == after}")
    out = ROOT / "docs/evidence"
    write_json(out / "failure_scenarios.json", {"generated_at": utcnow(), "scenarios": results})
    md = ["# Evidence: failing-gate scenarios", "", f"Generated {utcnow()} by `python scripts/demo_failures.py`.", ""]
    for r in results:
        md += [f"## {r['scenario']}", "", f"Command: `{r['command']}`  ", f"Exit code: **{r['exit_code']}** "
               f"(expected {r['expected_exit_code']})  ", f"Production unchanged: **{r['production_unchanged']}**", "",
               "| | before | after |", "|---|---|---|"]
        for k in r["before"]:
            md.append(f"| {k} | `{json.dumps(r['before'][k])}` | `{json.dumps(r['after'].get(k))}` |")
        md += ["", "```", r["output"], "```", ""]
    (out / "failure_scenarios.md").write_text("\n".join(md))
    ok = all(r["production_unchanged"] and r["exit_code"] == r["expected_exit_code"] for r in results)
    print("ALL SCENARIOS FAILED SAFELY" if ok else "SOME SCENARIOS DID NOT BEHAVE AS EXPECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
