# Evidence: failing-gate scenarios

Generated 2026-10-24T16:51:23Z by `python scripts/demo_failures.py`.

## 1. Bad data batch -> DATA GATE

Command: `hpp run --batch-file data/scenarios/bad_batch.csv --skip-monitor`  
Exit code: **2** (expected 2)  
Production unchanged: **True**

| | before | after |
|---|---|---|
| bundle_fingerprint | `"086453c2e9e7a3b5"` | `"086453c2e9e7a3b5"` |
| bundle_model_version | `"1"` | `"1"` |
| registry_production_version | `1` | `1` |

```
==============================================================================
  PIPELINE STOPPED SAFELY - DATA GATE FAILED
==============================================================================
  - DATA GATE FAILED - source 'file:bad_batch.csv' (500 rows) failed 3 checks: Neighborhood_Score: 15 non-numeric values e.g. ['unknown']
  - 37.8% missing in Neighborhood_Score
  - 16.00% of rows invalid - Neighborhood_Score: 15 non-numeric values e.g. ['unknown']; range.Area_sqft: 40 values outside [100, 10000] (min=-181.159, max=2345); category.City_Zone: 25 unknown values ['Downtown']

  Production model unchanged : house-price-model v1 (bundle fingerprint 086453c2e9e7a3b5)
  Nothing was registered or deployed; the previous model keeps serving.
  Evidence                   : artifacts/validate/report.json
==============================================================================
```

## 2. Model below threshold -> MODEL GATE

Command: `hpp run --override configs/scenarios/weak_model.yaml --skip-monitor`  
Exit code: **3** (expected 3)  
Production unchanged: **True**

| | before | after |
|---|---|---|
| bundle_fingerprint | `"086453c2e9e7a3b5"` | `"086453c2e9e7a3b5"` |
| bundle_model_version | `"1"` | `"1"` |
| registry_production_version | `1` | `1` |

```
==============================================================================
  PIPELINE STOPPED SAFELY - MODEL GATE FAILED
==============================================================================
  - MODEL GATE FAILED - RMSE 20.409 (required <= 9.0)
  - MAE 15.809 (required <= 7.0)
  - R2 0.143 (required >= 0.8)
  - MAPE 0.329 (required <= 0.15)
  - candidate RMSE 20.409 vs production v1 7.807 (+161.41%, allowed <= +2%)

  Production model unchanged : house-price-model v1 (bundle fingerprint 086453c2e9e7a3b5)
  Nothing was registered or deployed; the previous model keeps serving.
  Evidence                   : artifacts/evaluate/gate.json
==============================================================================
```

## 3. Model worse than production -> MODEL GATE

Command: `hpp run --override configs/scenarios/regressed_model.yaml --skip-monitor`  
Exit code: **3** (expected 3)  
Production unchanged: **True**

| | before | after |
|---|---|---|
| bundle_fingerprint | `"086453c2e9e7a3b5"` | `"086453c2e9e7a3b5"` |
| bundle_model_version | `"1"` | `"1"` |
| registry_production_version | `1` | `1` |

```
==============================================================================
  PIPELINE STOPPED SAFELY - MODEL GATE FAILED
==============================================================================
  - MODEL GATE FAILED - candidate RMSE 8.516 vs production v1 7.807 (+9.08%, allowed <= +2%)

  Production model unchanged : house-price-model v1 (bundle fingerprint 086453c2e9e7a3b5)
  Nothing was registered or deployed; the previous model keeps serving.
  Evidence                   : artifacts/evaluate/gate.json
==============================================================================
```
