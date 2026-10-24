# Deliverables: Project 4, Group 10

**Omer Raza (B052), Arfeen Khan (B050)**, MSc Data Science, NMIMS. MLOps, Dr. Santosh Kumar.

- Repository: https://github.com/omerwritescode/mlops-house-price-pipeline
- Demo video (5–10 min): `<paste the video link here>`

| # | Deliverable (brief §6) | Where to find it |
|---|---|---|
| 1 | One repository with pipeline, configuration, tests, workflows, Dockerfile and README | This repo: [`src/hpp/`](../src/hpp), [`configs/config.yaml`](../configs/config.yaml), [`tests/`](../tests), [`.github/workflows/`](../.github/workflows), [`Dockerfile`](../Dockerfile), [`README.md`](../README.md) |
| 2 | Design note (≤ 2 pages) with architecture diagram | [`design_note.pdf`](design_note.pdf) (source: [`design_note.md`](design_note.md); diagram: [`images/architecture.png`](images/architecture.png)) |
| 3 | Workflow graph, run history and registry screenshots | Graph: [`workflow_graph.md`](workflow_graph.md), [`images/workflow_graph.png`](images/workflow_graph.png). Prefect: [`screenshots/01_prefect_flow_graph.png`](screenshots/01_prefect_flow_graph.png), [`screenshots/02_prefect_run_history.png`](screenshots/02_prefect_run_history.png). Run history: [`evidence/run_history.txt`](evidence/run_history.txt). MLflow: [`screenshots/03_mlflow_runs.png`](screenshots/03_mlflow_runs.png), [`screenshots/04_mlflow_registry_aliases.png`](screenshots/04_mlflow_registry_aliases.png) |
| 4 | Evidence of two failing-gate scenarios (we show three) | [`evidence/failure_scenarios.md`](evidence/failure_scenarios.md): bad batch → data gate (exit 2); weak model → model gate (exit 3); model worse than production → model gate (exit 3). Production unchanged in all three. Screenshot: [`screenshots/05_failure_banners.png`](screenshots/05_failure_banners.png) |
| 5 | Lineage demonstration (recorded or live) | [`evidence/lineage.txt`](evidence/lineage.txt) and [`screenshots/06_lineage.png`](screenshots/06_lineage.png). The chain runs from the deployed model to its registry version, MLflow run, git commit, data hash and config hash. Shown live in the demo with `hpp lineage` |
| 6 | CI history including at least one release | [Actions](https://github.com/omerwritescode/mlops-house-price-pipeline/actions) (PR `ci` runs, `release` runs, `rollback` run) and [Releases](https://github.com/omerwritescode/mlops-house-price-pipeline/releases) (v1.0.2, v1.0.3, rollback pre-release). Rollback runbook: [`rollback.md`](rollback.md); local rollback test: [`evidence/rollback_test.md`](evidence/rollback_test.md). Reproducibility on a fresh CI checkout: the *Reproducibility check* step of every `release` run |
| 7 | Monitoring dashboard or report covering the three incoming batches | Report: [`evidence/monitoring_report.html`](evidence/monitoring_report.html) (download it and open it in a browser). Batch 1 OK, batch 2 WARNING, batch 3 CRITICAL. Grafana: [`screenshots/09_grafana_dashboard.png`](screenshots/09_grafana_dashboard.png); dashboard definition [`../monitoring/grafana/dashboards/hpp.json`](../monitoring/grafana/dashboards/hpp.json); alert rules [`../monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml) |
| 8 | 5–10 minute demonstration | Video link at the top; script with each member's parts: [`demo_script.md`](demo_script.md) |
| 9 | Short reflection (one page) | [`reflection.md`](reflection.md) |

## How to reproduce (about 10 minutes)

```bash
git clone https://github.com/omerwritescode/mlops-house-price-pipeline && cd mlops-house-price-pipeline
python3.11 -m venv .venv && source .venv/bin/activate
make setup && make pipeline-local && make verify     # reproduces the reported metrics within ±0.05 lakhs
make failures                                         # the failing-gate scenarios
hpp lineage
make up                                               # API :8000, Grafana :3000, Prometheus :9090, MLflow :5000
```
