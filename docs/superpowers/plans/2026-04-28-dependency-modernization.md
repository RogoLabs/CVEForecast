# Dependency Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `requirements.txt` lower-bound pins in three independently-revertible PRs — Stage 1 (NumPy 2 + ecosystem catch-up), Stage 2 (pandas 3), Stage 3 (darts 0.43 + xgboost 3) — so Dependabot stops re-opening PRs against old minors and the project stays current with the scientific Python ecosystem.

**Architecture:** Each stage is a feature branch off `main`, validated locally and in CI before merge. Validation protocol (applied per stage): `pytest -v`, warnings-as-errors sweep (`PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning"`), end-to-end forecast dry run against `~/data/cvelistV5/`, `validate_forecast_data.py` sanity check, CI green. Breakage revealed during validation is fixed inline using `superpowers:systematic-debugging`; rollback is `git revert <merge-sha>` on any single stage.

**Tech Stack:** Python 3.13, numpy, pandas, darts (w/ torch, xgboost, lightgbm, catboost, prophet, scikit-learn), pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-04-28-dependency-modernization-design.md`

---

## Preconditions (run once before Stage 1)

### Task 0: Baseline snapshot

**Files:**

- Read: `requirements.txt`
- Read: `tests/` (entire dir)

- [ ] **Step 1: Confirm on clean main**

```bash
git status
git fetch origin
git log --oneline -5
```

Expected: working tree clean (or only the unrelated `.github/workflows/monthly_tuning.yml` formatting diff). `HEAD` matches `origin/main`.

- [ ] **Step 2: Capture a known-good pytest baseline**

```bash
pytest -v 2>&1 | tee /tmp/cveforecast-baseline-pytest.log
```

Expected: all tests pass. Save this log — each stage will diff against it.

- [ ] **Step 3: Capture a known-good forecast run (optional but recommended)**

Prereq: `~/data/cvelistV5/` mirror exists (per CLAUDE.md). If not, skip this step and rely on CI for the E2E check on each stage.

```bash
python code/run_production_forecast.py 2>&1 | tee /tmp/cveforecast-baseline-forecast.log
cp web/data.json /tmp/cveforecast-baseline-data.json
```

Expected: script exits 0, `web/data.json` is updated. **Do not commit** `web/data.json` — we only need it for eyeballing.

- [ ] **Step 4: Restore any local-only forecast output**

```bash
git checkout -- web/data.json web/cna_data.json web/forecast_history.json web/pipeline_results.json web/model_info.json
git status
```

Expected: working tree clean again. We have baselines in `/tmp` and the repo is untouched.

---

## Stage 1 — NumPy 2 + ecosystem catch-up

### Task 1.1: Create feature branch

**Files:** (none modified)

- [ ] **Step 1: Branch off main**

```bash
git checkout main
git pull --ff-only
git checkout -b deps/stage1-numpy2
```

Expected: on new branch `deps/stage1-numpy2`, tree clean.

### Task 1.2: Bump Stage 1 pins

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Apply the Stage 1 floor bumps**

Replace the contents of `requirements.txt` with:

```
darts[all]>=0.36.0
pandas>=2.2.0
numpy>=2.1
prophet>=1.3
scikit-learn>=1.8
torch>=2.10
# tensorflow>=2.15.0
lightgbm>=4.6
xgboost>=2.1.0
python-dateutil>=2.9
requests>=2.32.0
catboost>=1.2.8
```

Rationale: we do not touch `darts`, `pandas`, or `xgboost` in this stage — they're deferred to Stages 2 and 3. Every other line moves to its current latest-compatible floor.

- [ ] **Step 2: Reinstall the environment**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Expected: pip completes, no resolver errors. Confirm: `pip show numpy | grep Version` prints `Version: 2.x.x`.

- [ ] **Step 3: Commit the pin bump in isolation**

```bash
git add requirements.txt
git commit -m "deps: bump numpy to 2.x and catch up compatible libraries"
```

### Task 1.3: Run the validation gauntlet

**Files:** (none modified unless breakage is found)

- [ ] **Step 1: Run the unit tests**

```bash
pytest -v
```

Expected: all tests that passed in the Task 0 baseline still pass.

- [ ] **Step 2: Run the warnings-as-errors sweep**

```bash
PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v
```

Expected: still all-green. Any new failure here is a Stage 1 issue to fix in Task 1.4. Record the first traceback; that's the starting point for debugging.

- [ ] **Step 3: End-to-end forecast dry run**

```bash
python code/run_production_forecast.py
```

Expected: script exits 0, `web/data.json` regenerates without error.

- [ ] **Step 4: Validator + eyeball**

```bash
python code/scripts/validate_forecast_data.py
python -c "import json; d=json.load(open('web/data.json')); print({k: d.get(k) for k in list(d)[:10]})"
```

Expected: validator exits 0. Top-level keys look the same shape as the Task 0 baseline (same key names, same order-of-magnitude values).

- [ ] **Step 5: Clean up local forecast artifacts**

```bash
git checkout -- web/data.json web/cna_data.json web/forecast_history.json web/pipeline_results.json web/model_info.json
```

Expected: tree clean relative to `HEAD`.

### Task 1.4: Triage and fix breakage (conditional)

Only execute this task if Task 1.3 surfaced a failure.

**Files:**

- Modify: whichever file the traceback points to

- [ ] **Step 1: Invoke systematic debugging**

Use the `superpowers:systematic-debugging` skill with the failing traceback as the starting observation. Do **not** silence warnings with `warnings.filterwarnings(...)` — fix the root cause (removed API, deprecated kwarg, etc.).

- [ ] **Step 2: Add a regression test (if the breakage was silent)**

If the failure was "runtime error our tests didn't catch," add the smallest test that would have caught it. Location: the existing `tests/` file that most closely matches the module where the bug lived (see `tests/conftest.py` for fixtures).

- [ ] **Step 3: Re-run the full gauntlet**

Repeat all 4 steps of Task 1.3. Expected: green.

- [ ] **Step 4: Commit the fix**

```bash
git add <fixed-files> <new-test-if-any>
git commit -m "fix: <one-line description of root cause>"
```

### Task 1.5: Push and open the Stage 1 PR

**Files:** (none modified)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin deps/stage1-numpy2
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "deps: stage 1 — numpy 2 + ecosystem catch-up" --body "$(cat <<'EOF'
## Summary
- Raises `numpy>=2.1`, `scikit-learn>=1.8`, `lightgbm>=4.6`, `catboost>=1.2.8`, `prophet>=1.3`, `torch>=2.10`, `python-dateutil>=2.9`.
- Stage 1 of 3 per `docs/superpowers/specs/2026-04-28-dependency-modernization-design.md`.
- Supersedes Dependabot PRs #21, #24, #25.

## Test plan
- [x] `pytest -v` passes locally.
- [x] `PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v` passes locally.
- [x] `python code/run_production_forecast.py` runs end-to-end locally.
- [x] `python code/scripts/validate_forecast_data.py` passes on the generated output.
- [ ] CI green (`main.yml`, `lint.yml`, `test.yml`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Wait for CI, then merge**

```bash
gh pr checks --watch
# After green:
gh pr merge --squash --delete-branch
```

- [ ] **Step 4: Close superseded Dependabot PRs**

```bash
gh pr close 21 --comment "Superseded by Stage 1 dependency modernization."
gh pr close 24 --comment "Superseded by Stage 1 dependency modernization."
gh pr close 25 --comment "Superseded by Stage 1 dependency modernization."
```

Expected: `gh pr list --state open` no longer shows these three.

- [ ] **Step 5: Return to main**

```bash
git checkout main
git pull --ff-only
```

---

## Stage 2 — pandas 3

### Task 2.1: Create feature branch

**Files:** (none modified)

- [ ] **Step 1: Branch off post-Stage-1 main**

```bash
git checkout main
git pull --ff-only
git checkout -b deps/stage2-pandas3
```

### Task 2.2: Bump pandas floor (conservative first)

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Change the pandas line**

Replace `pandas>=2.2.0` with:

```
pandas>=2.3,<4
```

The `<4` cap is a safety rail — it lets pip resolve a 2.x or 3.x depending on what the other pins allow, without pulling in a hypothetical pandas 4 we haven't validated.

- [ ] **Step 2: Reinstall and confirm the pandas version pip picked**

```bash
pip install --upgrade -r requirements.txt
pip show pandas | grep Version
```

Record the resolved version. If pip picked 2.x, we'll still run the gauntlet and attempt the 3.x escalation in Task 2.5. If pip picked 3.x directly, Task 2.5 becomes a no-op.

- [ ] **Step 3: Commit the conservative bump**

```bash
git add requirements.txt
git commit -m "deps: widen pandas pin to >=2.3,<4"
```

### Task 2.3: Run the validation gauntlet on the resolved pandas

**Files:** (none modified unless breakage is found)

- [ ] **Step 1: Unit tests**

```bash
pytest -v
```

- [ ] **Step 2: Warnings-as-errors sweep**

```bash
PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v
```

Expected: any `FutureWarning` from pandas (chained assignment, `inplace=True`, removed APIs) now fails the suite. Capture each traceback.

- [ ] **Step 3: End-to-end forecast dry run**

```bash
python code/run_production_forecast.py
```

- [ ] **Step 4: Validator + eyeball**

```bash
python code/scripts/validate_forecast_data.py
python -c "import json; d=json.load(open('web/data.json')); print({k: d.get(k) for k in list(d)[:10]})"
```

- [ ] **Step 5: Clean local artifacts**

```bash
git checkout -- web/data.json web/cna_data.json web/forecast_history.json web/pipeline_results.json web/model_info.json
```

### Task 2.4: Triage and fix pandas breakage (conditional)

Only execute this task if Task 2.3 surfaced a failure. Pandas 2.2→3.0 is the highest-risk bump in this plan; expect to run it.

**Files:**

- Likely modify: `code/data_loader.py`, `code/forecast_tracker.py`, `code/cna_trend_data.py`, `code/adapters/cve_adapter.py`, `code/adapters/cna_adapter.py`
- Possibly modify: `code/scripts/validate_forecast_data.py`

- [ ] **Step 1: Audit for copy-on-write violations**

```bash
grep -rEn "inplace=True|SettingWithCopy" code/ --include="*.py"
```

Fix any `inplace=True` by rewriting to assignment form: `df = df.method(...)` instead of `df.method(inplace=True)`.

Note: A repo-wide grep done during planning found no `inplace=True` or `DataFrame.append` usage — if that still holds, this step may be a no-op. The chained-assignment cases (e.g., `df[df.col > 0]['other'] = ...`) don't grep cleanly and have to be found via the FutureWarning tracebacks from Task 2.3 step 2.

- [ ] **Step 2: Invoke systematic debugging for any remaining tracebacks**

Use `superpowers:systematic-debugging`. Do not suppress warnings — fix the root cause.

- [ ] **Step 3: Re-run the full gauntlet**

Repeat all 5 steps of Task 2.3. Expected: green.

- [ ] **Step 4: Commit the fix**

```bash
git add <fixed-files>
git commit -m "fix: adapt to pandas 3.x copy-on-write and removed APIs"
```

### Task 2.5: Decide whether to escalate to `pandas>=3`

**Files:**

- Possibly modify: `requirements.txt`

- [ ] **Step 1: Check what pip actually resolved**

```bash
pip show pandas | grep Version
```

If the resolved version is already 3.x (and the gauntlet was green), escalate the floor. If it's 2.x and we couldn't force it to 3.x within the stage's scope, keep `>=2.3,<4` and open a tracking issue for the pandas 3.x bump.

- [ ] **Step 2 (escalation path): Tighten the floor**

Edit `requirements.txt`:

```
pandas>=3.0,<4
```

- [ ] **Step 3 (escalation path): Re-run the full gauntlet**

Repeat all 5 steps of Task 2.3. Expected: green.

- [ ] **Step 4 (escalation path): Commit**

```bash
git add requirements.txt
git commit -m "deps: raise pandas floor to >=3.0"
```

- [ ] **Step 2 (deferral path): Open a tracking issue**

```bash
gh issue create --title "Raise pandas floor to >=3.0" --body "$(cat <<'EOF'
Stage 2 of the dependency modernization landed at `pandas>=2.3,<4` because [record the specific blocker here — API change we can't fix in-stage, upstream bug, etc.].

Retry in ~3 months after the blocker is resolved.
EOF
)"
```

### Task 2.6: Push and open the Stage 2 PR

**Files:** (none modified)

- [ ] **Step 1: Push**

```bash
git push -u origin deps/stage2-pandas3
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "deps: stage 2 — pandas 3" --body "$(cat <<'EOF'
## Summary
- Raises `pandas` floor to [fill in: `>=3.0,<4` or `>=2.3,<4`].
- Stage 2 of 3 per `docs/superpowers/specs/2026-04-28-dependency-modernization-design.md`.

## Breakage fixed
[fill in: list any files edited in Task 2.4, or "none — pandas 3 was a clean drop-in"]

## Test plan
- [x] `pytest -v` passes locally.
- [x] `PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v` passes locally.
- [x] `python code/run_production_forecast.py` runs end-to-end locally.
- [x] `python code/scripts/validate_forecast_data.py` passes on the generated output.
- [ ] CI green (`main.yml`, `lint.yml`, `test.yml`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Wait for CI, then merge**

```bash
gh pr checks --watch
gh pr merge --squash --delete-branch
```

- [ ] **Step 4: Return to main**

```bash
git checkout main
git pull --ff-only
```

---

## Stage 3 — darts 0.43 + xgboost 3

### Task 3.1: Create feature branch

**Files:** (none modified)

- [ ] **Step 1: Branch off post-Stage-2 main**

```bash
git checkout main
git pull --ff-only
git checkout -b deps/stage3-darts-xgboost
```

### Task 3.2: Bump Stage 3 pins

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Update the darts and xgboost lines**

Replace:

```
darts[all]>=0.36.0
```

with:

```
darts[all]>=0.43
```

Replace:

```
xgboost>=2.1.0
```

with:

```
xgboost>=3.2
```

- [ ] **Step 2: Reinstall**

```bash
pip install --upgrade -r requirements.txt
pip show darts xgboost | grep -E "^(Name|Version)"
```

Expected: darts 0.43.x, xgboost 3.2.x.

- [ ] **Step 3: Commit the pin bump**

```bash
git add requirements.txt
git commit -m "deps: bump darts to 0.43 and xgboost to 3.x"
```

### Task 3.3: Verify the most-likely-affected imports still resolve

**Files:**

- Read only: `code/tuner/comprehensive_tuner.py`, `code/core/model_utils.py`, `code/adapters/cve_adapter.py`, `code/adapters/cna_adapter.py`

- [ ] **Step 1: Import-smoke the darts symbols the code uses**

```bash
python -c "
from darts import TimeSeries
from darts.metrics import mae, mape, mase, rmsse, rmse
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from darts.utils.utils import SeasonalityMode
from darts.models import XGBModel
print('darts imports OK')
"
```

Expected: prints `darts imports OK`. If any `ImportError`, the symbol moved in 0.37–0.43 — fix the import path at the usage site (`code/tuner/comprehensive_tuner.py:73`, `:77`, `:97`; `code/core/model_utils.py:64`; `code/adapters/*.py:XGBModel`) rather than aliasing.

- [ ] **Step 2: Verify XGBModel still accepts the params our code passes**

The xgboost integration in this repo goes through `darts.models.XGBModel`, not raw xgboost — so xgboost 3.0's `device`/`tree_method` renames mostly don't affect us. Still, verify:

```bash
python -c "
from darts.models import XGBModel
from darts import TimeSeries
import numpy as np
ts = TimeSeries.from_values(np.arange(30, dtype=float))
m = XGBModel(lags=7)
m.fit(ts)
print('XGBModel fit OK')
"
```

Expected: prints `XGBModel fit OK`. If it errors with a deprecated-param warning (now fatal under warnings-as-errors), record the param name — that's the fix target.

### Task 3.4: Run the validation gauntlet

**Files:** (none modified unless breakage is found)

- [ ] **Step 1: Unit tests**

```bash
pytest -v
```

- [ ] **Step 2: Warnings-as-errors sweep**

```bash
PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v
```

- [ ] **Step 3: End-to-end forecast dry run**

```bash
python code/run_production_forecast.py
```

Expected: script exits 0. This is the single most important check for Stage 3 — darts and xgboost are where forecast numbers actually come from, so if the pipeline completes and the validator passes, we've covered the real integration surface.

- [ ] **Step 4: Validator + eyeball**

```bash
python code/scripts/validate_forecast_data.py
python -c "import json; d=json.load(open('web/data.json')); print({k: d.get(k) for k in list(d)[:10]})"
```

Compare the eyeball output to the Task 0 baseline. Noise-level drift in forecast numbers is expected (model training has stochasticity); order-of-magnitude drift or missing keys blocks the merge.

- [ ] **Step 5: Clean local artifacts**

```bash
git checkout -- web/data.json web/cna_data.json web/forecast_history.json web/pipeline_results.json web/model_info.json
```

### Task 3.5: Triage and fix breakage (conditional)

Only execute this task if Task 3.3 or 3.4 surfaced a failure.

**Files:**

- Likely modify: `code/tuner/comprehensive_tuner.py`, `code/core/model_utils.py`, `code/adapters/cve_adapter.py`, `code/adapters/cna_adapter.py`, `code/run_production_forecast.py`

- [ ] **Step 1: Invoke systematic debugging**

Use `superpowers:systematic-debugging` with the first failing traceback as the starting observation.

- [ ] **Step 2: Re-run the full gauntlet**

Repeat Task 3.4 steps 1–5. Expected: green.

- [ ] **Step 3: Commit the fix**

```bash
git add <fixed-files>
git commit -m "fix: adapt to darts 0.43 / xgboost 3 API changes"
```

### Task 3.6: Push and open the Stage 3 PR

**Files:** (none modified)

- [ ] **Step 1: Push**

```bash
git push -u origin deps/stage3-darts-xgboost
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "deps: stage 3 — darts 0.43 + xgboost 3" --body "$(cat <<'EOF'
## Summary
- Raises `darts[all]>=0.43` and `xgboost>=3.2`.
- Stage 3 of 3 per `docs/superpowers/specs/2026-04-28-dependency-modernization-design.md`.
- Supersedes Dependabot PRs #22, #23.

## Breakage fixed
[fill in: list any files edited in Task 3.5, or "none"]

## Test plan
- [x] `pytest -v` passes locally.
- [x] `PYTHONWARNINGS="error::DeprecationWarning,error::FutureWarning" pytest -v` passes locally.
- [x] `python code/run_production_forecast.py` runs end-to-end locally.
- [x] `python code/scripts/validate_forecast_data.py` passes on the generated output.
- [ ] CI green (`main.yml`, `lint.yml`, `test.yml`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Wait for CI, then merge**

```bash
gh pr checks --watch
gh pr merge --squash --delete-branch
```

- [ ] **Step 4: Close superseded Dependabot PRs**

```bash
gh pr close 22 --comment "Superseded by Stage 3 dependency modernization."
gh pr close 23 --comment "Superseded by Stage 3 dependency modernization."
```

- [ ] **Step 5: Return to main and verify the scheduled forecast still works**

```bash
git checkout main
git pull --ff-only
gh run list --workflow=main.yml --limit 3
```

Watch the next scheduled daily run (midnight UTC). If it turns red, `git revert <stage3-merge-sha>` and diagnose.

---

## Post-stage cleanup

### Task 4: Confirm the Dependabot queue is quiet

**Files:** (none modified)

- [ ] **Step 1: List open PRs**

```bash
gh pr list --state open
```

Expected: no Dependabot PRs against `requirements.txt` lines we just bumped. If Dependabot has opened a new PR (e.g., for a patch that landed while we were working), merge or close it per normal review.

- [ ] **Step 2: Verify requirements.txt matches the plan**

```bash
cat requirements.txt
```

Expected (final state — adjust pandas line per the Task 2.5 decision):

```
darts[all]>=0.43
pandas>=3.0,<4
numpy>=2.1
prophet>=1.3
scikit-learn>=1.8
torch>=2.10
# tensorflow>=2.15.0
lightgbm>=4.6
xgboost>=3.2
python-dateutil>=2.9
requests>=2.32.0
catboost>=1.2.8
```
