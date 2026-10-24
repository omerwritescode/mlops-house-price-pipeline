# Rollback runbook

There are two independent things that can be rolled back. In both cases nothing is rebuilt: we switch back
to an artifact that has already been tested.

| What went wrong | Roll back | Command | Time |
|---|---|---|---|
| A newly promoted **model** misbehaves (monitoring alert, complaints) | the model bundle and the registry aliases | `hpp rollback` (or `make rollback`) | seconds |
| A newly released **image** is broken (API crash, bad dependency) | the container image | GitHub → Actions → **rollback** → *Run workflow* (`image_tag`, `reason`) | about 1 min |

## A. Model rollback: `hpp rollback`

`deploy` never overwrites the serving model in place. It builds the new bundle in a staging directory, then
swaps directories: `deployments/current` → `deployments/previous`, staging → `current`. Rollback reverses
the swap:

1. `deployments/previous` becomes `current` again, and the bad bundle becomes `previous`.
2. In the MLflow registry, the `production` alias goes back to the previous version and
   `previous-production` moves to the rolled-back one (lifecycle `archived`). Both transitions are tagged
   `transitioned_by=pipeline`, and the reason is recorded as "rollback from vN".
3. If `HPP_API_RELOAD_URL` is set, the API is told to hot-reload (`POST /admin/reload`). Otherwise restart
   the `api` container: `docker compose restart api`.
4. A `rollback` line is appended to `logs/run_history.jsonl`, and the result goes to
   `artifacts/deploy/rollback.json`.

Check afterwards:

```bash
hpp lineage                          # deployed version, registry alias, run, commit, data hash all agree
curl -s localhost:8000/model-info    # the API serves the restored version
hpp monitor                          # regenerate the monitoring report for the restored model
```

### Tested

`scripts/demo_rollback.py` (`make rollback-test`) retrains, deploys the new model, rolls back, and checks
that the restored bundle is byte-identical to the original by comparing content fingerprints. Result:
[`evidence/rollback_test.md`](evidence/rollback_test.md). The fingerprint goes from `6bd3a42fea08e24e` (v1)
to `9e5bd6dadf01a876` (retrained v2) and back to `6bd3a42fea08e24e`. The registry `production` alias
returns to v1. The same flow is also covered by `tests/test_pipeline_integration.py::test_full_pipeline_gates_lineage_rollback` and `tests/test_deploy.py`.

## B. Image rollback: `rollback` workflow

Every merge to `main` runs `release.yml`. It publishes `ghcr.io/<owner>/hpp-api:sha-<short-commit>` and
`:v1.0.<run>`, moves `:latest`, and creates a GitHub release that names the commit, the model version and
the metrics. Images are immutable and carry their model inside, so going back means re-pointing a tag:

1. Find the last good tag on the **Releases** page (for example `sha-1a2b3c4`).
2. Go to **Actions → rollback → Run workflow**, enter `image_tag=sha-1a2b3c4` and a reason.
3. The workflow pulls that image, starts it, checks that `/health` and `/version` respond, re-tags it as
   `latest` and `production`, pushes both, and records a **pre-release** called `rollback-<n>` with the
   reason and the version it is now serving.
4. Hosts running `latest` pick it up with `docker compose pull api && docker compose up -d api`. To pin a
   host directly without the workflow:
   `HPP_IMAGE=ghcr.io/<owner>/hpp-api:sha-1a2b3c4 docker compose up -d api`.

To undo a bad **commit** as well as the bad image, revert it on `main` (`git revert <sha>`). The revert goes
through PR CI and `release.yml` like any other change, so the fix is tested before it ships.
