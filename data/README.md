# Dataset Source and Availability

- Filename: `ai_worker_burnout_attrition_2026.csv`
- Original listing: https://www.kaggle.com/datasets/nudratabbas/ai-worker-burnout-and-attrition-risk-dataset
- Attributed creator/uploader: Nudrat Abbas (`nudratabbas`), based on the secondary reference below.
- Secondary attribution: https://medium.com/@himariaugm/burnout-pekerja-di-era-ai-analisis-attrition-risk-menggunakan-logistic-regression-399b9be31388
- Status checked: October 14, 2026; original listing returned 404.
- License, original release date, sampling/generation method, and observed-versus-synthetic status: unverified.
- Local CSV shape: 1,500 records, 21 columns.
- SHA-256 of the evaluated CSV: `2675fbebae40d1b6883f61567184ecf962d595cce64b049eb83b20b5b5944cf4`.

The CSV was supplied by the project author and already tracked at the repository root before the corrected workflow. This change adds no new raw dataset copies. Repository availability does not establish permission to reuse or redistribute the dataset. No license is assigned to it here.

The default script reads the existing root CSV. If using your own lawfully obtained copy, run `python code.py --data /path/to/ai_worker_burnout_attrition_2026.csv`. A fresh Kaggle download cannot currently be promised because the listing is unavailable.

## Feature Policy

- Target: `burnout_score` (continuous regression).
- Excluded: `employee_id` (identifier), target, and `attrition_risk` (potential outcome-derived leakage; construction unverified).
- Remaining 18 predictors: workplace, demographic, AI-use, productivity, and satisfaction variables.
- All valid rows are retained. Outlier status alone is not evidence of an invalid observation.
- Missing feature values are imputed within training folds. Invalid targets are dropped and counted.
- Conflicting employee IDs raise an error rather than silently choosing a record.

`job_satisfaction_1_5` and `productivity_score` could be concurrent or outcome-dependent measurements. Their use supports only exploratory same-record prediction until their definitions and measurement timing are confirmed.
