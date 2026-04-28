# Dependency Modernization — Design

**Date:** 2026-04-28
**Status:** Approved (pending implementation plan)
**Author:** Jerry Gamblin + Claude

## Background

`requirements.txt` carries lower-bound pins that have drifted several majors behind current releases. Dependabot currently has 5 open PRs against the repo (numpy, darts, xgboost, lightgbm, python-dateutil), two of which are major bumps (numpy 1→2, xgboost 2→3). Nothing is broken today, but the floors are stale enough that Dependabot will keep re-opening PRs against old minors until they are raised.

Current `requirements.txt` pins vs. latest on PyPI:

| Library         | Current floor | Latest      | Delta      |
| --------------- | ------------- | ----------- | ---------- |
| numpy           | `>=1.26.0`    | 2.4.4       | major      |
| pandas          | `>=2.2.0`     | 3.0.2       | major      |
| darts[all]      | `>=0.36.0`    | 0.43.0      | 7 minors   |
| torch           | `>=2.6.0`     | 2.11.0      | 5 minors   |
| scikit-learn    | `>=1.6.0`     | 1.8.0       | 2 minors   |
| prophet         | `>=1.1.0`     | 1.3.0       | 2 minors   |
| lightgbm        | `>=4.5.0`     | 4.6.0       | 1 minor    |
| xgboost         | `>=2.1.0`     | 3.2.0       | major      |
| catboost        | `>=1.2.0`     | 1.2.10      | 10 patches |
| python-dateutil | `>=2.8.0`     | 2.9.0.post0 | 1 minor    |

CI runs on Python 3.13 (`.github/workflows/main.yml`). The daily `main.yml` workflow runs the full production forecast end-to-end.

## Goals

1. Raise lower-bound pins so Dependabot stops opening PRs against obsolete minors.
2. Adopt NumPy 2.x and pandas 3.x to stay current with the ecosystem.
3. Pick up performance and API improvements in darts (0.43) and xgboost (3.x).
4. Ship in independently-revertible stages so a regression in one library doesn't block the rest.

## Non-goals

- Exact byte-level parity of forecast output across the bump (pipeline has stochastic elements).
- Refactoring the forecasting pipeline, tests, or CI beyond what the library bumps require.
- Moving off any library; this is a version bump, not a replacement project.

## Approach

Three sequential PRs, each independently green on CI before the next begins. Each stage raises floors in `requirements.txt`, fixes any code breakage revealed by the validation protocol, and merges as a single commit for easy revert.

### Stage 1 — NumPy 2 + ecosystem catch-up

One PR. Raises floors for every library that is already compatible with numpy 2 and has no user-impacting API breaks worth a dedicated stage.

`requirements.txt` changes:

```
numpy>=2.1
scikit-learn>=1.8
lightgbm>=4.6
catboost>=1.2.8
prophet>=1.3
torch>=2.10
python-dateutil>=2.9
```

Superseded Dependabot PRs to close: #21 (python-dateutil), #24 (lightgbm), #25 (numpy).

Known NumPy 2 concerns for this codebase:

- Removed aliases (`np.float_`, `np.complex_`, `np.Inf`, `np.NaN`) — warnings sweep catches any stragglers.
- Typed scalars (`np.int32/int64/float32/float64`) used in `code/tuner/comprehensive_tuner.py:1506` — still supported, no change needed.
- `np.isinf`, `np.mean`, `np.std`, `np.median`, `np.max`, `np.array` — unchanged semantics.
- `np.array(..., copy=False)` behavior changed — not used in our code based on grep.

### Stage 2 — pandas 3

One PR, isolated because copy-on-write (on by default in 3.0) has the largest blast radius of any bump in this project.

`requirements.txt` change:

```
pandas>=2.3,<4
```

Decision point inside the PR: if the `>=2.3` build passes the warnings-as-errors sweep, escalate the floor to `>=3.0` in the same PR. If the warnings sweep reveals chained-assignment or removed-API issues that can't be fixed within the PR scope, keep `>=2.3,<4` and open a follow-up for the 3.0 bump.

Known pandas 3 concerns:

- Copy-on-write is default. `df[col] = ...` chained-assignment patterns raise. Audit targets: `code/data_loader.py`, `code/forecast_tracker.py`, `code/cna_trend_data.py`, `code/adapters/cve_adapter.py`, `code/adapters/cna_adapter.py`.
- `inplace=True` on many methods deprecated — remove or rewrite to assignment form.
- `DataFrame.append` already removed in 2.x; `read_csv` date-parsing args tightened.
- `validate_forecast_data.py` uses pandas for output inspection — verify still works.

### Stage 3 — darts 0.43 + xgboost 3

One PR. These two libraries have the largest integration surface in our code (models, TimeSeries, metrics, hyperparameter tuner). Separating them from the numpy/pandas churn means if forecast outputs shift, we know which bump caused it.

`requirements.txt` changes:

```
darts[all]>=0.43
xgboost>=3.2
```

Superseded Dependabot PRs to close: #22 (xgboost), #23 (darts).

Known concerns:

- darts 0.37–0.43 moved some baseline imports. Verify `from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal` (used at `code/tuner/comprehensive_tuner.py:97`) and `from darts.utils.utils import SeasonalityMode` (used at `code/core/model_utils.py:64`) still resolve.
- Torch-based darts models may require `pl_trainer_kwargs` changes; smoke test via `run_production_forecast.py`.
- xgboost 3 replaces `tree_method='gpu_hist'` + `predictor` with a unified `device` parameter. Grep `code/` for xgboost usage and update calls that set the deprecated params.
- xgboost 3 changed `early_stopping_rounds` placement in some wrapper signatures.

## Validation protocol

Applied identically to each stage PR before merge:

1. **Unit tests:** `pytest -v` — must pass.
2. **Warnings-as-errors sweep:** `PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v` — any new warning from the bumped libraries becomes a hard failure; fix inline before merge.
3. **End-to-end forecast dry run:** `python code/run_production_forecast.py` against the local `~/data/cvelistV5/` mirror. Must run to completion.
4. **Output sanity check:** `python code/scripts/validate_forecast_data.py` against the generated `web/data.json`, plus eyeball top-of-file summary stats vs. the `main`-branch version. Order-of-magnitude drift blocks the merge; noise-level drift is acceptable.
5. **CI green:** push the branch and confirm `main.yml`, `lint.yml`, `test.yml` all pass.

## Rollback

Each stage is one merge commit. If the next-day scheduled `main.yml` run surfaces a regression, `git revert <merge-sha>` restores a known-good state without disturbing other stages. Dependabot PRs closed as superseded can be re-opened if needed.

## Out of scope

- Tests that pin exact forecast numbers.
- Moving Python version off 3.13.
- Changing the model selection logic, tuner, or CI schedule.
- Refactoring pandas/numpy usage beyond what copy-on-write and removed aliases require.

## Deliverables

Three PRs, each containing:

- `requirements.txt` diff for that stage.
- Any code edits required to fix breakage revealed by the validation protocol.
- PR description listing what broke (if anything) and how it was fixed, plus which superseded Dependabot PRs should be closed on merge.
