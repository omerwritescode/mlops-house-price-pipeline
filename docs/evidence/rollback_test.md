# Evidence: retrain, then rollback

Generated 2026-10-24T16:53:25Z by `python scripts/demo_rollback.py`.

1. `hpp retrain --batches 1,2,3` exited **0**. Model gate: model gate passed - RMSE 8.622, MAE 6.436, R2 0.851
2. `hpp rollback` exited **0**.
3. `hpp monitor` exited **0**. The monitoring report now belongs to the restored model.

**Restored bundle identical to the original: True**

| | before | after retrain | after rollback |
|---|---|---|---|
| bundle_fingerprint | `086453c2e9e7a3b5` | `3a1c049ab32b7ce0` | `086453c2e9e7a3b5` |
| bundle_model_version | `1` | `2` | `1` |
| registry_production_version | `1` | `2` | `1` |

Monitoring level per batch (RMSE in lakhs):

| batch | before (production model) | after retrain | after rollback |
|---|---|---|---|
| 1 | OK (7.18) | OK (6.88) | OK (7.18) |
| 2 | WARNING (9.21) | OK (7.85) | WARNING (9.21) |
| 3 | CRITICAL (16.24) | CRITICAL (13.51) | CRITICAL (16.24) |

Rollback output:

```json
{
  "rolled_back_from": "2",
  "rolled_back_to": "1",
  "registry": "aliases swapped",
  "api_reload": "not configured (API picks up the bundle on restart)",
  "at": "2026-10-24T16:52:55Z",
  "bundle_fingerprint": "086453c2e9e7a3b5"
}
```

Note: after the retrain, batches 1-3 were part of the training data, so for most of their rows the
"after retrain" column is in-sample. Only the batch rows that went to the holdout are unseen.
