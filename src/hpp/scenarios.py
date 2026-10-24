"""Failure-scenario helpers (Task 4).

``make_bad_batch`` derives a *corrupted copy* of one supplied incoming batch - the kind of
batch a broken upstream export would produce. It is written to ``data/scenarios/`` and is
never mixed into ``data/raw/``. The supplied datasets themselves are not modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hpp.context import StageContext
from hpp.data import load_incoming


def make_bad_batch(ctx: StageContext, source_batch: int = 1, out: str = "data/scenarios/bad_batch.csv") -> tuple[Path, dict]:
    rng = np.random.default_rng(ctx.cfg.project.seed)
    df = load_incoming(ctx, [source_batch]).drop(columns=[ctx.cfg.data.batch_column])
    df["House_ID"] = df["House_ID"].str.replace("HS-1", "HS-9", regex=False)  # a new, distinct batch
    n = len(df)
    df = df.astype({"Area_sqft": float, "Neighborhood_Score": object, "City_Zone": object})

    idx = rng.permutation(n)
    unit_bug = idx[:40]                    # sq-metres exported as negative numbers by a buggy job
    df.loc[unit_bug, "Area_sqft"] = -df.loc[unit_bug, "Area_sqft"] / 10.764
    missing = idx[40:200]                  # neighbourhood score feed dropped for a third of rows
    df.loc[missing, "Neighborhood_Score"] = np.nan
    junk = idx[200:215]                    # text leaking into a numeric column
    df.loc[junk, "Neighborhood_Score"] = "unknown"
    new_zone = idx[215:240]                # a zone the model has never seen
    df.loc[new_zone, "City_Zone"] = "Downtown"

    path = ctx.path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    summary = {"derived_from": f"incoming batch {source_batch}", "rows": n,
               "corruptions": {"negative Area_sqft (unit bug)": len(unit_bug),
                               "Neighborhood_Score set missing": len(missing),
                               "Neighborhood_Score = 'unknown' text": len(junk),
                               "City_Zone = 'Downtown' (unknown)": len(new_zone)}}
    return path, summary
