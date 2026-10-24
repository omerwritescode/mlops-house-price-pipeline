"""Stage 1 - ingest.

Inputs : raw training CSV (+ optional labelled incoming batches / a newly arrived batch file)
Outputs: artifacts/ingest/dataset.csv, artifacts/ingest/meta.json (data version = content hashes)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from hpp.context import StageContext
from hpp.utils import combined_hash, dvc_md5, file_sha256, utcnow, write_json

log = logging.getLogger("hpp.ingest")

SOURCE_COL = "_source"


def read_csv(path: Path) -> pd.DataFrame:
    # keep_default_na + explicit empty-string handling: empty cells are the dataset's missing values
    return pd.read_csv(path)


def load_incoming(ctx: StageContext, batches: list[int] | None = None, path: str | None = None) -> pd.DataFrame:
    src = ctx.path(path or ctx.cfg.paths.incoming_batches)
    df = read_csv(src)
    bcol = ctx.cfg.data.batch_column
    if batches is not None and bcol in df.columns:
        df = df[df[bcol].isin(batches)]
    return df.reset_index(drop=True)


def _source_record(path: Path, rows: int, note: str = "", root: Path | None = None) -> dict:
    try:
        shown = str(path.resolve().relative_to(root.resolve())) if root else str(path)
    except ValueError:  # outside the project: keep absolute
        shown = str(path)
    return {
        "path": shown,
        "sha256": file_sha256(path),
        "dvc_md5": dvc_md5(path),
        "rows": int(rows),
        "note": note,
    }


def ingest(ctx: StageContext) -> dict:
    cfg = ctx.cfg
    out_dir = ctx.stage_dir("ingest")
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []

    train_path = ctx.path(cfg.paths.raw_train)
    train = read_csv(train_path)
    train[SOURCE_COL] = "train"
    frames.append(train)
    sources.append(_source_record(train_path, len(train), "historical training data", ctx.root))
    log.info("loaded training data %s: %d rows x %d cols", train_path.name, *train.shape)

    if cfg.data.train_with_batches:
        inc_path = ctx.path(cfg.paths.incoming_batches)
        inc = load_incoming(ctx, list(cfg.data.train_with_batches))
        inc[SOURCE_COL] = "batch_" + inc[cfg.data.batch_column].astype(str)
        frames.append(inc)
        sources.append(_source_record(inc_path, len(inc),
                                      f"labelled incoming batches {sorted(cfg.data.train_with_batches)}", ctx.root))
        log.info("added labelled incoming batches %s: %d rows", cfg.data.train_with_batches, len(inc))

    extra = list(cfg.data.extra_training_files)
    if ctx.batch_file:
        extra.append(ctx.batch_file)
    for rel in extra:
        p = ctx.path(rel)
        df = read_csv(p)
        df[SOURCE_COL] = f"file:{p.name}"
        frames.append(df)
        sources.append(_source_record(p, len(df), "newly arrived batch", ctx.root))
        log.info("added new batch file %s: %d rows", p.name, len(df))

    dataset = pd.concat(frames, ignore_index=True, sort=False)
    data_version = combined_hash(
        [s["sha256"] for s in sources] + [f"batches={sorted(cfg.data.train_with_batches)}"]
    )[:16]

    dataset.to_csv(out_dir / "dataset.csv", index=False)
    meta = {
        "stage": "ingest",
        "created_at": utcnow(),
        "pipeline_run_id": ctx.pipeline_run_id,
        "data_version": data_version,
        "rows": int(len(dataset)),
        "columns": list(dataset.columns),
        "sources": sources,
        "train_with_batches": list(cfg.data.train_with_batches),
    }
    write_json(out_dir / "meta.json", meta)
    log.info("dataset assembled: %d rows, data_version=%s", len(dataset), data_version)
    return {"rows": len(dataset), "data_version": data_version, "sources": len(sources)}
