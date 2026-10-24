"""Stage 2 - validate (the DATA GATE).

Inputs : artifacts/ingest/dataset.csv + meta.json
Outputs: artifacts/validate/report.json  (always written, pass or fail - it is the evidence)
Fails  : raises DataGateError with an explanation; nothing downstream runs, production is untouched.

Checks (all thresholds from config.validation):
  schema        required columns present, numeric columns parse as numbers
  volume        at least ``min_rows`` rows
  missingness   per-feature missing share <= max_missing_fraction; target <= max_missing_fraction_target
  validity      share of rows with out-of-range numbers or unknown categories <= max_invalid_row_fraction
  uniqueness    duplicate-ID share <= max_duplicate_id_fraction
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from hpp.config import Config
from hpp.context import StageContext
from hpp.exceptions import DataGateError
from hpp.utils import read_json, utcnow, write_json

log = logging.getLogger("hpp.validate")


def _check(name: str, passed: bool, value: Any, threshold: Any, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "value": value, "threshold": threshold, "detail": detail}


def validate_frame(df: pd.DataFrame, cfg: Config, require_target: bool = True) -> dict[str, Any]:
    """Run every data-gate check on ``df`` and return a structured report (never raises)."""
    v = cfg.validation
    d = cfg.data
    checks: list[dict[str, Any]] = []
    n = int(len(df))

    required = d.feature_columns + ([d.target] if require_target else [])
    missing_cols = [c for c in required if c not in df.columns]
    checks.append(_check("schema.required_columns", not missing_cols, missing_cols, "none missing",
                         "all required columns present" if not missing_cols
                         else f"missing required columns: {missing_cols}"))
    if missing_cols:  # nothing else is meaningful without the schema
        return _finalise(checks, n, df)

    checks.append(_check("volume.min_rows", n >= v.min_rows, n, v.min_rows,
                         f"{n} rows (minimum {v.min_rows})"))

    invalid_rows = np.zeros(n, dtype=bool)
    violations: dict[str, str] = {}

    numeric_cols = list(d.numeric_features) + ([d.target] if require_target else [])
    numeric = {}
    for col in numeric_cols:
        raw = df[col]
        parsed = pd.to_numeric(raw, errors="coerce")
        unparseable = raw.notna() & parsed.isna()
        if unparseable.any():
            examples = sorted(map(str, raw[unparseable].unique()))[:3]
            violations[col] = f"{int(unparseable.sum())} non-numeric values e.g. {examples}"
            invalid_rows |= unparseable.to_numpy()
        numeric[col] = parsed
    bad_types = {c: m for c, m in violations.items()}
    share_unparseable = {c: float((df[c].notna() & numeric[c].isna()).mean()) for c in numeric_cols}
    worst = max(share_unparseable.values()) if share_unparseable else 0.0
    checks.append(_check("schema.numeric_types", worst <= v.max_invalid_row_fraction,
                         round(worst, 4), v.max_invalid_row_fraction,
                         "all numeric columns parse as numbers" if not bad_types
                         else "; ".join(f"{c}: {m}" for c, m in bad_types.items())))

    # missingness (computed on parsed values so junk strings are not counted twice)
    for col in d.feature_columns:
        series = numeric[col] if col in numeric else df[col]
        frac = float(series.isna().mean()) if n else 0.0
        checks.append(_check(f"missing.{col}", frac <= v.max_missing_fraction, round(frac, 4),
                             v.max_missing_fraction, f"{frac:.1%} missing in {col}"))
    if require_target:
        frac = float(numeric[d.target].isna().mean()) if n else 0.0
        checks.append(_check(f"missing.{d.target}", frac <= v.max_missing_fraction_target, round(frac, 4),
                             v.max_missing_fraction_target, f"{frac:.1%} missing target values"))

    # ranges
    for col in numeric_cols:
        lo, hi = v.ranges[col]
        vals = numeric[col]
        out = vals.notna() & ((vals < lo) | (vals > hi))
        if out.any():
            violations[f"range.{col}"] = (f"{int(out.sum())} values outside [{lo:g}, {hi:g}] "
                                          f"(min={vals.min():g}, max={vals.max():g})")
            invalid_rows |= out.to_numpy()

    # categories
    for col, levels in d.categorical_features.items():
        s = df[col]
        unknown = s.notna() & ~s.astype(str).isin(levels)
        if unknown.any():
            vals = sorted(map(str, s[unknown].unique()))[:5]
            violations[f"category.{col}"] = f"{int(unknown.sum())} unknown values {vals}"
            invalid_rows |= unknown.to_numpy()

    frac_invalid = float(invalid_rows.mean()) if n else 0.0
    detail = "no out-of-range or unknown values" if not violations else "; ".join(
        f"{k}: {m}" for k, m in violations.items())
    checks.append(_check("validity.invalid_rows", frac_invalid <= v.max_invalid_row_fraction,
                         round(frac_invalid, 4), v.max_invalid_row_fraction,
                         f"{frac_invalid:.2%} of rows invalid - {detail}"))

    if d.id_column in df.columns:
        dup = float(df[d.id_column].duplicated().mean()) if n else 0.0
        checks.append(_check("uniqueness.duplicate_ids", dup <= v.max_duplicate_id_fraction, round(dup, 4),
                             v.max_duplicate_id_fraction, f"{dup:.2%} duplicate {d.id_column}"))

    return _finalise(checks, n, df, violations)


def _finalise(checks: list[dict[str, Any]], n: int, df: pd.DataFrame,
              violations: dict[str, str] | None = None) -> dict[str, Any]:
    failed = [c for c in checks if not c["passed"]]
    if failed:
        msg = (f"DATA GATE FAILED - {len(failed)} of {len(checks)} checks failed: "
               + " | ".join(c["detail"] for c in failed))
    else:
        msg = f"data gate passed - {len(checks)} checks on {n} rows"
    return {
        "passed": not failed,
        "n_rows": n,
        "n_checks": len(checks),
        "failed_checks": [c["check"] for c in failed],
        "checks": checks,
        "violations": violations or {},
        "message": msg,
    }


def validate(ctx: StageContext) -> dict:
    ctx.require_inputs("validate")
    out_dir = ctx.stage_dir("validate")
    meta = read_json(ctx.artifacts / "ingest" / "meta.json")
    df = pd.read_csv(ctx.artifacts / "ingest" / "dataset.csv")

    report = validate_frame(df, ctx.cfg, require_target=True)
    report.update({"stage": "validate", "created_at": utcnow(), "pipeline_run_id": ctx.pipeline_run_id,
                   "data_version": meta["data_version"], "sources": meta["sources"]})
    # Each source is also judged ON ITS OWN: a corrupted new batch must not be diluted by
    # thousands of clean historical rows. Any failing source fails the gate.
    if "_source" in df.columns and df["_source"].nunique() > 1:
        report["by_source"] = {}
        for src, sub in df.groupby("_source", sort=False):
            part = validate_frame(sub, ctx.cfg, require_target=True)
            report["by_source"][str(src)] = {k: part[k] for k in ("passed", "n_rows", "failed_checks", "checks")}
            if not part["passed"]:
                report["passed"] = False
                failed = [c for c in part["checks"] if not c["passed"]]
                report["failed_checks"] = sorted(set(report["failed_checks"]) | {f"{src}:{c['check']}" for c in failed})
                report["message"] = (f"DATA GATE FAILED - source '{src}' ({part['n_rows']} rows) failed "
                                     f"{len(failed)} checks: " + " | ".join(c["detail"] for c in failed))
    write_json(out_dir / "report.json", report)

    if not report["passed"]:
        log.error(report["message"])
        raise DataGateError(report["message"], report)
    log.info(report["message"])
    return {"passed": True, "n_rows": report["n_rows"], "data_version": meta["data_version"]}


def assert_validated(ctx: StageContext) -> dict:
    """Downstream guard: the current ingest output must have passed the data gate."""
    report = read_json(ctx.artifacts / "validate" / "report.json")
    meta = read_json(ctx.artifacts / "ingest" / "meta.json")
    if not report.get("passed"):
        raise DataGateError("refusing to continue: the last data-gate run FAILED - " + report.get("message", ""),
                            report)
    if report.get("data_version") != meta.get("data_version"):
        raise DataGateError(
            "refusing to continue: ingest output changed since it was validated "
            f"({report.get('data_version')} != {meta.get('data_version')}). Run `hpp stage validate`.", report)
    return report
