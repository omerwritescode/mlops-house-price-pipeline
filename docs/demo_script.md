# Demo script (about 8 minutes)

**Omer** covers the pipeline, the gates and the registry (about 4 min). **Arfeen** covers CI/CD, serving,
monitoring and rollback (about 4 min). Before you start, have these open:
- a terminal in the repo with the stack up (`make up`)
- tabs for Prefect (:4200), MLflow (:5000), Grafana (:3000) and the GitHub Actions page.

Run `make pipeline` once beforehand so the UIs have content.

| Time | Who | Show | Say |
|---|---|---|---|
| 0:00 | Omer | `docs/images/architecture.png` | "Problem: predict listing prices. Success means an RMSE of at most 9 lakhs, and a typical estimate is about ±₹8 lakh. No person moves a model into production; only the pipeline does." |
| 0:45 | Omer | `configs/config.yaml`, then `hpp validate-config` | "One file drives everything. It is validated strictly and hashed, and nothing is hard-coded. CI reuses it with a small overlay." |
| 1:15 | Omer | `make pipeline`, then the Prefect UI graph | "Eight stages. Each one has declared inputs and outputs, so any stage can be re-run alone." Run `hpp stage evaluate` to show it. |
| 2:00 | Omer | `make failures` → `docs/evidence/failure_scenarios.md` | "Bad batch: the data gate stops it with exit 2. Weak model and regressed model: the model gate stops them with exit 3. The production fingerprint is the same before and after each one." |
| 3:00 | Omer | MLflow UI → registered model aliases, run tags; `hpp lineage` | "Aliases only change through the pipeline, and the audit finds no manual transitions. From the deployed model we trace back to the commit, the data hash, the config hash and the run." |
| 4:00 | Arfeen | GitHub: a PR's checks, then the `release` run and Releases page | "A PR runs lint, tests, a light pipeline and an image smoke test. A merge to main runs the full pipeline, the reproducibility check, and builds `hpp-api:sha-<commit>` with a release." |
| 5:00 | Arfeen | `curl localhost:8000/predict`, `/version`; `make replay` then Grafana | "The container serves the model and its commit. Grafana shows latency p95 against the 300 ms SLO, throughput, error rate and predicted prices." |
| 6:00 | Arfeen | `reports/monitoring/report.html` | "Batch 1 is OK. Batch 2 is a WARNING: RMSE is 1.18× and prices are 7% under. Batch 3 is CRITICAL: area, bedrooms and city zone drifted, RMSE is 2×, and prices are 21% under. Each level comes with a recommended action." |
| 7:00 | Arfeen | `make rollback-test` → `docs/evidence/rollback_test.md`; `docs/rollback.md` | "The retrain passes the gates and deploys. Batch 2 recovers, but batch 3 stays critical because this is concept drift. Rolling back restores the identical bundle. For a broken image, the rollback workflow re-points `latest`." |
| 7:45 | both | `docs/reflection.md` | One lesson each: Omer on the per-source gate and the hash split, Arfeen on concept drift and the registry living per environment. |

**Fallback if the network is down:** run everything with `--engine local`. Show `hpp history` instead of
Prefect and the screenshots in `docs/screenshots/` instead of the UIs.
