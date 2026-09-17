# CVEForecast — Forecasting & Site Review

**Reviewed:** 2026-09-17 · against `main` @ `f0f79ac` and the live site at cveforecast.org
**Scope:** forecasting methodology, model evaluation, the constraint layer, and the dashboard.

---

## Verdict

The engineering around this project is good — daily CI, per-CNA forecasts, a clean adapter
architecture, accessibility work, a real backtest harness. The **forecasting** underneath it is
where the problems are, and one of them is currently producing a wrong number on the front page.

Three things, in priority order:

1. **The constraint layer has overwritten the models.** Every one of the 12 models now reports a
   2026 total inside a 5-CVE band (112,811–112,816). That is not consensus; it is a formula.
   The published number is `2025_actual × 1.141`, and the same code path produces a 2027 forecast
   (52,563) that is **less than half** of 2026. This is the highest-severity item on the list.
2. **There is no benchmark, so nobody can tell whether the models work.** On rolling-origin
   evaluation the current model class scores **MASE ≈ 1.26** — *worse than a naive forecast*.
   The dashboard reports MAPE only, which hides this completely.
3. **The series changed regime in 2026 and the pipeline has no way to notice.** Month-over-month
   growth has been **+14.7%/month** this year (≈ +481% annualised) against +12%/yr for 2017–2023.
   Every model trains on the full 2017→present history with equal weight, which is empirically the
   *worst* window choice available (MASE 2.13 vs 1.10 for a 48-month window).

Fixing #1 is a day. Fixing #2 is a day. #3 is the interesting work.

> **On the evidence below.** `darts` isn't installed on this machine, so the quantitative
> experiments were run against a series reconstructed by summing `web/cna_data.json` — 117 months,
> 2017-01 → 2026-09, covering **95.3%** of published CVEs. Shapes and growth rates match the real
> series; absolute levels are ~5% low. 2026-09 is partial and was excluded from every fit.
> Reproduction scripts are described inline.

---

## 1. The constraint layer is the forecast

### What the site publishes

| Year | all-models | spread across 12 models |
|---|---|---|
| 2026 | 113,267 | **5 CVEs** (112,811 → 112,816) |
| 2027 | 52,563 | 4 CVEs |

Twelve model families — Prophet, TBATS, AutoARIMA, XGBoost, LightGBM, CatBoost, Kalman, Croston,
Theta — do not independently agree to within 5 CVEs on a 55,000-CVE quantity. Something downstream
is collapsing them.

### What actually happens

`CVEForecaster.apply_constraints` ([cve_adapter.py:196](../code/adapters/cve_adapter.py#L196)) sums
each model's monthly forecasts into **yearly** totals and hands them to
[`ForecastConstraints`](../code/forecast_constraints.py). But on 2026-09-17 the forecast only covers
**Sep–Dec 2026** — four months. `ForecastConstraints.apply_constraints` compares that four-month sum
against the **full previous year's actual** (48,153) and applies an annual growth floor:

```
apply_growth_floor(4-month sum, 48,153)     → max(48,153 × 1.05, …)  = 50,560
trend_adjusted_forecast(50,560, 48,153)     → 5.0% < 13.5% threshold
                                            → blend: 0.05×0.3 + 0.18×0.7 = 14.1%
                                            → 48,153 × 1.141           = 54,942
```

And `112,812 − 57,872 (Jan–Aug actual) = 54,940`. That is the number, to the rounding error.
Every model gets clamped to the same floor, so every model emerges with the same answer.
I reproduced this exactly by calling `ForecastConstraints` directly with three different raw inputs
(21,600 / 21,000 / 22,000) — all three came out at **54,942**.

### The 2027 number is worse

For 2027 the previous year *is* in `yearly_totals`, so the baseline becomes the mean of the
**four-month** 2026 forecasts. The max-growth cap then applies:

```
2027 = 1.40 × (Sep–Dec 2026 raw sum) = 52,563
```

A year-long forecast capped at 1.4× a four-month baseline. The site consequently implies a **−54%
year-over-year collapse** in 2027, from a module whose stated purpose is to enforce a *growth floor*.

### Fixes

- **Never compare a partial year to a full year.** Constrain the annualised rate, or the
  *remaining-months* total against the same months last year. Cleanest: compute
  `year_total = YTD_actual + forecast(remaining months)` and constrain *that*.
- **Constrain the ensemble, not each model.** Applying a floor per-model destroys exactly the
  disagreement that makes a multi-model dashboard worth looking at. If you want a floor, apply it
  once, to the published headline, and show the unconstrained model spread alongside it.
- **Wire the config.** `ForecastConstraints(config=self.config, …)`
  ([cve_adapter.py:101](../code/adapters/cve_adapter.py#L101)) passes the *whole* config, but the
  class reads flat keys (`config.get('min_annual_growth_rate')`). Everything under
  `forecast_constraints` in `config.json` is silently ignored and the hard-coded defaults are used —
  which is why `trend_adjustment_confidence` is 0.7 in the arithmetic above and 0.8 in your config.
  Pass `self.config['forecast_constraints']`.
- **Consider deleting the layer.** A growth floor is a patch for models that under-forecast
  exponential growth. §3 shows that modelling in log space fixes the under-forecasting properly, at
  which point the floor stops earning its keep. Every method I tested in levels space showed a
  **−7% to −18% downward bias**; in log space that drops to −1.4%.

---

## 2. No benchmark, so no way to know if any of this works

### MAPE alone can't answer "is this better than doing nothing?"

Rolling-origin backtest, 45 origins (2022-01 → 2025-09), h = 1…12, MASE scaled by in-sample
seasonal-naive MAE:

| Method | MASE | MAPE | bias | h=1 | h=6 | h=12 |
|---|---:|---:|---:|---:|---:|---:|
| LogTrend36 + seasonality + business-day | **0.98** | **11.5%** | −6.9% | 0.77 | 0.83 | 1.63 |
| LogTrend24 damped (φ=0.9) | 1.09 | 12.8% | −5.3% | 0.78 | 0.93 | 1.88 |
| LogTrend24 + seasonality + business-day | 1.10 | 13.5% | −1.4% | 0.78 | 0.97 | 1.78 |
| Lags-12, log + business-day normalised | 1.24 | 15.5% | −10.4% | 0.82 | 0.99 | 2.29 |
| **Lags-24 on levels — your current best class** | **1.26** | **16.5%** | +5.2% | 0.90 | 1.12 | 1.95 |
| Naive (last value) | 1.41 | 16.4% | −10.9% | 0.78 | 1.17 | 2.55 |
| Drift | 1.45 | 17.6% | −2.3% | 0.79 | 1.21 | 2.43 |
| Seasonal naive | 1.69 | 19.9% | −18.5% | 1.34 | 1.49 | 2.55 |

**MASE > 1 means worse than the naive benchmark.** Your current model class sits at 1.26. It beats
the *naive* forecast on MAPE but loses on scaled error, and a ~40-line log-linear model with month
dummies and a business-day offset beats it by **30% on MAPE** with zero hyperparameter tuning, no
gradient boosting, and no 20,000-trial monthly Optuna run.

### Concretely

- **Add `NaiveSeasonal` and `NaiveDrift` to the enabled model list and rank them.** They're already
  wired in `create_model` and disabled in `config.json`. A dashboard where a naive baseline
  sometimes wins is *more* credible, not less — it tells readers what the models are worth.
- **Make MASE the primary ranking metric.** MAPE is asymmetric — it penalises over-forecasting
  harder than under-forecasting, which biases model selection toward the low side on a growing
  series, which is precisely the bias you then bolt a growth floor on to correct. The
  `config.json` `metrics` list already names `mase` and `rmsse`; the tuner already imports them
  ([comprehensive_tuner.py:74](../code/tuner/comprehensive_tuner.py#L74)). They just never reach
  the output. The dashboard even has MASE and RMSSE *column headers* today, with no data behind them.
- **Report MASE by horizon.** The h=1 / h=6 / h=12 columns above are the honest picture: h=1 is a
  solved problem (0.77–0.90), h=12 is not (1.6–2.5). Publishing one number for a 16-month forecast
  hides all of that.

### Rank on more than one origin

`_calculate_forecast_vs_published` ([cve_adapter.py:645](../code/adapters/cve_adapter.py#L645))
trains through 31-Dec and forecasts the current year. That's a genuine out-of-sample backtest —
good — but it is **one origin**, and this year it's 8 points. LinearRegression currently ranks #1 at
25.55% while LightGBM — the README's headline model at 6.22% — ranks #10 at 35.75%. That ordering
will reshuffle next month.

Use rolling origins: expanding-window backtest at 24+ origins, report mean ± std. You have
`RobustTimeSeriesValidator` ([time_series_cv.py](../code/validation/time_series_cv.py)) already
written for exactly this. See §5 — it never runs.

---

## 3. Modelling: what I'd actually change

### 3a. Model in log space

The series is multiplicative. 2017 → 2026 is 17,950 → ~63,500 with compounding growth, occasional
step changes when a large CNA onboards, and variance that scales with level. Levels-space models
fight this on every axis: they under-forecast trend, their residuals are heteroscedastic, and their
prediction intervals (if you had any) would be symmetric when the truth is skewed.

Forecast `log(count)`, forecast in that space, exponentiate. In darts this is one line:

```python
from darts.dataprocessing.transformers import BoxCox   # or a log Mapper
from darts.models import LinearRegressionModel

pipeline = Pipeline([BoxCox(lmbda=0)])       # lmbda=0 → log
model = LinearRegressionModel(lags=12, lags_future_covariates=(0, 1))
```

Bias in my backtest went from **+5.2% / −10.9%** (levels) to **−1.4%** (log). Remember the
exp-of-mean bias correction (`× exp(σ²/2)`) if you need unbiased *levels*, or just forecast the
median and say so.

### 3b. Add the calendar as a future covariate — this is free accuracy

CVEs are published on business days. Months have **20 to 23 business days** — a **15% swing**
that no model can currently see. Correlation between detrended monthly counts and business-day
count:

- 2017–2024: **+0.287**
- 2022–2025: **+0.395**

Your "February is a low month" seasonal index (0.889, the lowest of any month) is mostly just
February having fewer days. You are asking twelve models to learn the Gregorian calendar from 117
noisy observations, and the calendar is *known in advance for all time*.

This is the single highest value-per-line change in the review:

```python
from darts.utils.timeseries_generation import datetime_attribute_timeseries

bdays = TimeSeries.from_times_and_values(
    full_index,
    np.array([len(pd.bdate_range(d, d + pd.offsets.MonthEnd(0))) for d in full_index]),
)
month = datetime_attribute_timeseries(full_index, 'month', one_hot=True)
future_cov = bdays.stack(month)

model = LightGBMModel(lags=12, lags_future_covariates=[0])
model.fit(series_log, future_covariates=future_cov)
```

Every regression model in your enabled set (`LinearRegression`, `LightGBM`, `XGBoost`, `CatBoost`,
`RandomForest`) supports `lags_future_covariates` today and all of them currently pass `None`.
Better still: model `count / business_days` directly and multiply back — it removes the effect
rather than asking the model to learn it.

Worth testing as additional future covariates: Patch Tuesday count per month, and days lost to the
late-December holiday window.

### 3c. Training window — your biggest current miss

MASE, h = 1…6, log-trend model, varying training window:

| Training window | 2022–2024 origins | 2025–2026 origins |
|---|---:|---:|
| last 12 months | 1.02 | 1.87 |
| last 18 months | **0.79** | 1.66 |
| last 24 months | 0.83 | 1.54 |
| last 36 months | 0.84 | 1.13 |
| last 48 months | 0.80 | **1.10** |
| **all history (current behaviour)** | **1.11** | **2.13** |

All-history is the worst option in *both* regimes, and catastrophically so in the current one.
The same pattern holds for the levels lag-model: last-72-months scores 1.52 against 1.57 for the
full history.

Options, cheapest first:

1. **Cap the training window** at 48–60 months. One config key.
2. **Exponentially weight observations** by recency (`sample_weight` in LightGBM/XGBoost; darts
   passes it through).
3. **Let the tuner choose the window.** It currently searches `split_ratios` — the wrong knob.
   Training-window length is a real hyperparameter with a real effect; the validation split is an
   evaluation choice that shouldn't be tuned at all (§5).

### 3d. Handle the regime change explicitly

Mean month-over-month log growth:

| Period | MoM | annualised |
|---|---:|---:|
| 2017–2023 | +0.9%/mo | +12%/yr |
| 2024 | +4.4%/mo | +70%/yr |
| 2025 | −0.2%/mo | −2%/yr |
| **2026** | **+14.7%/mo** | **+481%/yr** |

Jan 2026 was 4,302; Aug was 12,260; September is tracking ~15–17k on a business-day-adjusted
run-rate. This is a structural break, not noise, and it's still accelerating. Every model on the
site is currently being graded on how gracefully it fails to see it — which is why all twelve are
badged "Poor."

Practical steps:

- **Damped trend, always.** At h=12 an undamped exponential extrapolation of +14.7%/month gives an
  absurd 2027. A damping parameter φ ∈ [0.8, 0.95] on the log trend is the standard, boring,
  correct answer and it's what keeps a long-horizon forecast defensible. My φ=0.9 variant lost a
  little at short horizons and held up far better at long ones.
- **Detect and surface the break.** A changepoint flag (Prophet already fits changepoints; you can
  also just track a rolling 6-month growth rate against its 3-year distribution) and a banner on
  the site saying "the series entered a new growth regime in 2026; forecasts beyond 6 months are
  wide" would be more honest than a point estimate, and more interesting to your readers.
- **Explain it.** This is a *vulnerability data* site. The 2026 surge has causes — CNA onboarding,
  bulk backfills, AI-assisted disclosure volume. `cna_trend_data.calculate_cna_momentum` already
  computes CNA counts and 12-month growth; that number is loaded, logged, and then never used by
  anything. Active-CNA count is the obvious exogenous driver and you already have it.

### 3e. Publish uncertainty

`ForecastResult.confidence_intervals` exists ([base_forecaster.py:28](../code/core/base_forecaster.py#L28))
and is never populated. `IntervalValidator`
([interval_validation.py](../code/validation/interval_validation.py), 370 lines) is imported by
nothing. The site shows bare point estimates for a 16-month horizon.

Empirical multiplicative intervals from my backtest residuals (log-ratio quantiles):

| horizon | 80% interval | 95% interval | median bias |
|---|---|---|---|
| h=1 | [0.88×, 1.30×] | [0.86×, 1.42×] | 1.00× |
| h=3 | [0.88×, 1.30×] | [0.84×, 1.43×] | 1.02× |
| h=6 | [0.91×, 1.25×] | [0.82×, 1.45×] | 1.02× |
| **h=12** | **[0.94×, 1.46×]** | **[0.85×, 1.87×]** | **1.09×** |

Read the bottom row: an honest 95% interval on a 12-month-ahead forecast spans roughly **−15% to
+87%**. Publishing 52,563 for 2027 as a clean integer, with no band, is the least defensible thing
on the dashboard.

Two ways to get there, both cheap:

- **Conformal / empirical residual quantiles.** Model-agnostic, works with every model you have,
  needs only the rolling-origin backtest from §2. Darts ships
  `ConformalNaiveModel` / `ConformalQRModel` — wrap the chosen forecaster and you get calibrated
  intervals for free.
- **Quantile regression.** `LightGBMModel(likelihood='quantile', quantiles=[0.1, 0.5, 0.9])` gives
  native intervals from a model already in your enabled set.

Then run `IntervalValidator` on them and publish the coverage. "Our 80% interval has covered 78% of
actual months" is the most trust-building sentence a forecasting site can print.

### 3f. Two smaller methodology notes

- **Right-censoring.** `cvelistV5` is backfilled: CVEs published in month *M* keep landing in the
  repo for days or weeks afterward. You correctly exclude the current incomplete month, but the
  *most recent complete* month is also still filling in, and lag models weight it heavily. Keep a
  vintage log (count for month *M* as observed at each daily run — you already run daily), measure
  the revision curve, and either hold out the last 1–2 months or inflate them by the measured
  factor. This is a genuinely novel thing you could publish that nobody else has.
- **The ensemble is untested and includes everything.** `save_results` labels it "weighted ensemble
  forecast (average of all models)" and computes `np.median` over all 12
  ([cve_adapter.py:821](../code/adapters/cve_adapter.py#L821)); `config.json` asks for
  `weighted_mape` over `ensemble_size: 5`, which isn't implemented. In my backtest, combining
  helped relative to the *average* member but lost to the single best:

  | | MASE |
  |---|---:|
  | best single model | 0.978 |
  | mean of all | 1.014 |
  | median of all | 1.019 |
  | mean of top-3 (chosen on prior origins only) | 1.064 |
  | worst member | 1.689 |

  Combination is a good variance-reduction play, but only over a *curated* pool. Drop the models
  that lose to naive, then take a trimmed mean of what's left — and put the ensemble in the
  rankings table so its accuracy is measured like everything else.

---

## 4. The dashboard

### Bugs visible right now on cveforecast.org

1. **The rankings table is misaligned.** The header declares 7 columns
   (`Rank · Model · MAPE · MASE · RMSSE · MAE · Performance`,
   [index.html:227](../web/index.html#L227)) and `populateModelRankings` emits 6 cells
   ([script.js:258](../web/script.js#L258)). MAE renders under "MASE", the Poor/Good badge renders
   under "RMSSE", the expand chevron lands under "MAE", and "Performance" is empty. Either populate
   MASE and RMSSE (you should — §2) or drop the headers.
2. **All 12 models are badged "Poor."** The thresholds (`<10% Excellent`, `<15% Good`,
   `<25% Fair`) were calibrated against a 6% MAPE era. Either rebase them on the naive benchmark
   ("beats naive" / "ties naive" / "loses to naive" is far more meaningful than an absolute MAPE
   cut-off) or acknowledge the regime change in the copy.
3. **The year selector doesn't exist.** `switchYear` looks for `yearBtn2026` / `yearBtn2027`
   ([script.js:134](../web/script.js#L134)); `index.html` contains zero `yearBtn` elements. The
   footer advertises "Forecast Period: October 2026 – December 2027" and there is no way to view
   2027. (Given §1, that's currently merciful.)
4. **The Dec-31 projection marker never renders.**
   `if str(current_year) in yearly_forecast_totals` ([cve_adapter.py:784](../code/adapters/cve_adapter.py#L784))
   tests a *string* against a dict with *integer* keys. Always false. Dead since it was written.
5. **Four of six chart series are hidden by default**, and Chart.js renders hidden legend entries
   struck through — so the legend reads as if four models have been crossed out.

### Content and framing

- **"+134.3% YoY Growth"** is the second-largest number on the page and it comes from the
  constraint formula, not from a model. Once §1 is fixed this should be recomputed — a
  business-day-adjusted extrapolation of the current trend lands nearer **85,000–92,000** for 2026
  (80% band on the remaining months), against the 112,812 currently published. The right answer is
  genuinely uncertain and the honest presentation is a range.
- **Lead with YTD + remainder, not with a model total.** By September, 9/12 of the year is *known*.
  Every method I tested lands within ±5% of the true year-end total when forecasting from a
  September origin, versus ±14% from a March origin. Show the split explicitly:
  "66,665 published + 22,000–30,000 forecast = 89k–97k." That framing is more accurate, more
  legible, and degrades gracefully as the year progresses.
- **The README and Technical Details page are badly stale.** Both claim "LightGBM 6.22% MAPE,
  KalmanFilter 6.26%, TBATS 7.21%." The live site has LightGBM at **35.75%**, ranked 10th of 12.
  That's a 6× discrepancy between your documentation and your dashboard. Generate that table from
  `data.json` at build time rather than hand-maintaining it.
- **Add a forecast-history page.** "Here's what we predicted for August, three, six and nine months
  out, and here's what actually happened" is the most compelling content a forecasting site can
  have, it's what distinguishes you from a chart, and the module to produce it already exists (§5).

### Smaller

- Add `<meta name="description">` and Open Graph tags — this gets shared.
- Publish `data.json` as a documented, versioned API. People will build on it.
- Add a "Methodology & Limitations" page stating the training window, the benchmark, the known
  right-censoring, and the current regime break. Forecasters who publish their limitations get
  taken more seriously, not less.

---

## 5. Dead code — ~2,000 lines of validation that never runs

`run_production_forecast.py` calls `run_all(cve_train_ratio=1.0, cve_validation=False,
cve_diagnostics=False)` ([run_production_forecast.py:53](../code/run_production_forecast.py#L53)).
Consequences:

| Module | Lines | Status in production |
|---|---:|---|
| `validation/time_series_cv.py` | 298 | never called |
| `validation/statistical_tests.py` | 398 | never called |
| `validation/interval_validation.py` | 370 | imported by nothing at all |
| `diagnostics/residual_analysis.py` | 540 | never called |
| `diagnostics/horizon_analysis.py` | 456 | never called |

`code/validation/cv_results.json` is dated 28 June and every entry reads
`{"validated": false, "error": "Unknown error"}` — cross-validation has never successfully
completed. Meanwhile `train_ratio=1.0` makes `val_data` empty, so `train_model` skips metrics
entirely ([base_forecaster.py:160](../code/core/base_forecaster.py#L160)) and every model's
`metrics` dict is `{}` in production.

**This is your best opportunity in the whole review.** The infrastructure to do §2 and §3e properly
is already written and tested. It just isn't plugged in. Run the rolling-origin CV *weekly* (not on
every daily run — it's slow), cache the results to `web/validation.json`, and rank from that
instead of from a single origin.

### The forecast tracker is broken and has been for ~9 months

`web/forecast_history.json` contains **zero snapshots** and was last written in the v0.10 release
commit. Cause: the committed file uses the key `snapshots`, and `ForecastTracker` reads and writes
`forecast_snapshots` ([forecast_tracker.py:118](../code/forecast_tracker.py#L118)). `add_snapshot`
raises `KeyError: 'forecast_snapshots'` — I reproduced it directly — and
`CVEForecaster._save_forecast_snapshot` swallows it in a bare `except Exception` that logs at
WARNING ([cve_adapter.py:634](../code/adapters/cve_adapter.py#L634)).

So the README feature "Forecast Tracking — historical snapshots track prediction evolution and
accuracy over time" has produced nothing, and roughly nine months of daily forecast vintages —
the most valuable dataset this project could accumulate, and the one nobody else has — are gone.

Two-line fix (align the key, or make `_load_history` migrate `snapshots` → `forecast_snapshots`),
then **narrow that `except`** so the next silent failure isn't silent. Same for
`_calculate_forecast_vs_published`, which returns `[], {}` on any exception.

### Tuning

- `config.json` still carries `split_ratio: 0.99` and `optimal_split_ratio: 0.99` for 9 models.
  With 117 months that's **a 1-point validation set** — which is exactly how LightGBM ended up
  recorded at 0.037% MAPE and XGBoost at 0.064%. Those numbers are noise, and they're what the
  monthly tuner compares against when deciding whether a new configuration is an improvement.
- **The split ratio should not be a tuned hyperparameter at all.** Searching over it and keeping
  the best selects the evaluation setup that flatters the model. The tuner has already partly
  recognised this — most grids are pinned to `0.88` with a `# FIXED: Optimal from Issue #2` comment
  — but several still search up to 0.99, and the stale 0.99 results are still live in config.
  Replace the whole mechanism with fixed rolling-origin CV.
- `mase` and `rmsse` are computed by the tuner
  ([comprehensive_tuner.py:1332](../code/tuner/comprehensive_tuner.py#L1332)) and written to
  `config.json` as `0`. Persist the real values.
- Tuning trial counts (19,744 for LightGBM) against a 1–13 point validation set are a very
  efficient way to overfit. Fewer trials against more origins beats more trials against one.

---

## 6. Other engineering notes

- **`freq='M'` with pandas 3.0.3.** `data_loader` resamples with `'ME'` but
  `TimeSeries.from_dataframe` is called with `freq='M'`
  ([cve_adapter.py:95](../code/adapters/cve_adapter.py#L95), [:658](../code/adapters/cve_adapter.py#L658)).
  `'M'` is deprecated and slated for removal. Move to `'ME'`.
- **Horizon off-by-one.** `get_forecast_horizon` returns Oct 2026 → Jan 2028 and
  `forecast_months` computes 16 ([cve_adapter.py:953](../code/adapters/cve_adapter.py#L953)), but
  `predict(16)` starts from the last *trained* month (Aug 2026), so the output runs Sep 2026 →
  Dec 2027. Jan 2028 — the month the docstring says is "needed for the Dec 31 year-end marker" — is
  never produced.
- **Config key mismatch.** `save_results` reads `file_paths['output']`
  ([cve_adapter.py:741](../code/adapters/cve_adapter.py#L741)); `config.json` defines
  `file_paths.output_data`. It works only because the fallback default happens to be the same path.
- **CNA model selection is overfit by construction.** `select_best_model_for_cna` picks the best of
  13 models on a **single 6-month holdout**, independently for ~120 CNAs — about 1,560 comparisons
  on 6 points each. The winner is mostly sampling noise. Either use one global model for all CNAs,
  or select on rolling origins with a complexity penalty, or pool the CNAs into a single global
  model with the CNA as a static covariate (darts supports this natively, and it would be a much
  stronger design).
- **`on: push` triggers the full daily pipeline** on every commit to `main`, including docs-only
  commits. Add `paths-ignore`.
- **Version strings disagree**: `pyproject.toml` 0.11.0, README "0.11 Galway 🇮🇪",
  `docs/ARCHITECTURE.md` "0.11 Phoenix 🔥🐦".

---

## Suggested order of work

**This week — the site is currently wrong**

1. Fix the partial-year comparison in the constraint layer, or disable the layer outright and
   publish raw model output. (§1)
2. Pass `config['forecast_constraints']` instead of the whole config. (§1)
3. Fix the rankings-table column mismatch. (§4.1)
4. Fix the `ForecastTracker` key so vintages start accumulating today — every day this waits is a
   day of data you can't get back. (§5)
5. Update the README and Technical Details accuracy tables, or generate them from `data.json`. (§4)

**Next two weeks — make the numbers defensible**

6. Enable `NaiveSeasonal` / `NaiveDrift` and rank them alongside everything else. (§2)
7. Turn on rolling-origin CV weekly, rank from it, publish MASE as the primary metric. (§2, §5)
8. Add business-day count and month dummies as future covariates. (§3b)
9. Cap the training window at 48–60 months. (§3c)

**The month after — make it a forecasting site rather than a model zoo**

10. Move to log space with a damped trend. (§3a, §3d)
11. Add conformal prediction intervals, validate coverage with the `IntervalValidator` you already
    have, and publish the coverage number. (§3e)
12. Build the forecast-history page off the now-working tracker. (§4)
13. Add CNA count as an exogenous driver — `calculate_cna_momentum` is already computing it. (§3d)
14. Start logging data vintages and measure the revision curve. (§3f)

---

### Reproducing the analysis

The experiments in §2 and §3 were run from `web/cna_data.json` with numpy/pandas/sklearn only.
Rolling-origin backtest with MASE scaling, business-day and seasonal-index analysis, training-window
sweep, combination test, and empirical interval quantiles — all reproducible from the series
reconstruction at the top of this document. Re-run them against the real `cvelistV5` series inside
darts before acting on the specific numbers; the *directions* held across every variant I tried.
