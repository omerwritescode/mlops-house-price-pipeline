# Reflection

**Group 10: Omer Raza (B052), Arfeen Khan (B050)**

**What worked.** Writing every stage as a function with declared inputs and outputs gave us three things
for free: re-running a single stage, the Prefect graph, and the local engine. Prefect only wraps the same
functions. The two gates make failure the *normal* path. A bad batch exits with code 2 and a weak or
regressed model exits with code 3. Production stays byte-identical (same bundle fingerprint before and
after), so "safe failure" is something we measure, not something we claim. The other piece that paid off
was a single strict config with a logged hash. Lineage can re-hash the config stored with the MLflow run
and confirm that the deployed model was trained under exactly that config.

**What surprised us.**
1. *The data gate has to run per source.* At first we validated the combined dataset. The bad batch's
   37.8% missing `Neighborhood_Score` was diluted to about 4% inside 4,500 rows and passed. We now validate
   every source on its own.
2. *A random split breaks the comparison with production.* With `train_test_split`, adding batches
   reshuffles the holdout. Houses the old model had trained on then landed in the new test set, which made
   the old model look better than it was. A deterministic hash split on `House_ID` fixed it: a house
   always stays on the same side.
3. *Retraining does not fix concept drift.* Batch 3 is CRITICAL in two ways. First, covariate shift: PSI
   0.33 for area, 0.20 for bedrooms, 0.21 for city zone. Second, a level shift in price: the mean actual
   price is 69.6 against 61.8 before, and the model under-prices by 12.8 lakhs (20.7%). Retraining on
   train data plus batches 1–3 improves batch 2 from WARNING to OK (RMSE 9.21 → 7.85). Batch 3 improves
   only from 16.24 to 13.51 and stays CRITICAL. Batch 3 is just 500 of 5,500 rows, and nothing in the
   features says "this listing is recent". `Batch` is not allowed as a feature, by design. Also, those
   post-retrain numbers are mostly in-sample, so the real improvement is smaller than it looks.
   Monitoring did its job by detecting the shift and recommending action. Fixing it needs a time or
   recency signal or sample weighting, which is a modelling decision and not a pipeline one.

**Limitations and what we would do next.**
- *The registry lives per environment.* CI and release jobs start with an empty SQLite registry, so the
  "vs production" comparison is only meaningful locally or on a long-lived host. Next step: a hosted
  MLflow server (for example DagsHub) behind `MLFLOW_TRACKING_URI`, which is already configurable.
- *Labels arrive with the batch.* In reality prices are known only after a sale. Until labels arrive,
  data drift and prediction drift are the early warning, and performance drift confirms later.
- *No automatic promotion from monitoring.* We chose `auto_retrain: false`, so a person approves a
  retrain. The new model still has to pass both gates.
- *We built our own drift checks (PSI/KS/χ²) instead of using Evidently.* They are easy to explain and
  they feed Pushgateway directly. Evidently would add richer reports.
- *Where the code was verified.* The code was written with an AI assistant in an environment where
  MLflow, Prefect, FastAPI and Docker could not be installed. The core pipeline, the gates, the
  rollback logic and the tests were run there against lightweight stand-ins, and the first real CI run
  is where the full stack is exercised. The reference metrics have to be regenerated in the pinned
  environment (`make reference`) before results are reported.

**Using AI to build this.** The assistant produced a working skeleton quickly. The design decisions that
mattered came from questioning the results: per-source validation, the hash split, and treating batch 3 as
concept drift. Each of those came from asking why a number looked wrong. We read every gate threshold
and justified it ourselves (see the design note) rather than accepting defaults.
