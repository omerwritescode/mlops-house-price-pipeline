# Common tasks. On Windows run the commands inside each target manually (or use WSL / Git Bash).
PY ?= python
export HPP_ORCHESTRATOR ?= prefect

.PHONY: setup test lint pipeline pipeline-local stage monitor failures rollback-test retrain lineage rollback history evidence \
        reference verify serve up down replay prefect-ui mlflow-ui clean

setup:            ## create the environment
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install --no-deps -e .

test:             ## unit + integration + API tests
	$(PY) -m pytest -q

lint:
	ruff check src tests scripts

pipeline:         ## full pipeline via Prefect (ingest -> ... -> deploy -> monitor)
	hpp run

pipeline-local:   ## same, without Prefect
	hpp run --engine local

stage:            ## re-run one stage, e.g. make stage S=train
	hpp stage $(S)

monitor:
	hpp monitor

failures:         ## the three failing-gate scenarios + evidence in docs/evidence/
	$(PY) scripts/demo_failures.py

rollback-test:    ## retrain -> deploy -> rollback, proves the original bundle is restored (docs/evidence/)
	$(PY) scripts/demo_rollback.py

evidence:         ## regenerate every file in docs/evidence/ from a clean state
	$(MAKE) clean
	hpp run
	hpp verify-repro
	$(PY) scripts/demo_failures.py
	$(PY) scripts/demo_rollback.py
	hpp history --last 60 > docs/evidence/run_history.txt
	hpp lineage > docs/evidence/lineage.txt

retrain:          ## retrain with the labelled incoming batches
	hpp retrain --batches 1,2,3

lineage:
	hpp lineage

rollback:
	hpp rollback

history:
	hpp history

reference:        ## record the current holdout metrics as the reproducibility reference (commit the file)
	hpp verify-repro --write

verify:           ## check this checkout reproduces the reference within tolerance
	hpp verify-repro

serve:            ## run the API locally against deployments/current
	MODEL_DIR=deployments/current uvicorn hpp.serving.app:app --port 8000

up:               ## API + Prometheus + Pushgateway + Grafana + MLflow UI
	GIT_SHA=$$(git rev-parse HEAD) IMAGE_TAG=local docker compose up -d --build

down:
	docker compose down

replay:           ## send the incoming batches through the live API (fills Grafana)
	$(PY) scripts/replay_traffic.py --api http://localhost:8000

prefect-ui:
	prefect server start

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:            ## remove all pipeline outputs (keeps data + code)
	rm -rf artifacts logs deployments mlartifacts mlflow.db* reports/monitoring data/scenarios
