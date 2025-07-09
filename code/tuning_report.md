# CVEForecast Hyperparameter Tuning Results

This report tracks the results of all tuning runs for each model over time. For each model, every run's timestamp, MAPE, MAE, MASE, RMSSE, and hyperparameters are shown.

---

## Tuning Results Over Time

### Prophet
| Timestamp | MAPE | MAE | MASE | RMSSE | Hyperparameters |
|-----------|------|-----|------|-------|----------------|
| 2025-07-07T17:37:41 | 22.71 | 442.90 | 2.88 | 2.99 | yearly_seasonality: true, weekly_seasonality: false, daily_seasonality: false, seasonality_mode: additive, changepoint_prior_scale: 0.1, seasonality_prior_scale: 1.0, n_changepoints: 15, mcmc_samples: 0, interval_width: 0.8 |
| 2025-07-07T17:48:36 | 22.71 | 442.90 | 2.88 | 2.99 | yearly_seasonality: true, weekly_seasonality: false, daily_seasonality: false, seasonality_mode: additive, changepoint_prior_scale: 0.1, seasonality_prior_scale: 1.0, n_changepoints: 15, mcmc_samples: 0, interval_width: 0.8 |

### TBATS
| Timestamp | MAPE | MAE | MASE | RMSSE | Hyperparameters |
|-----------|------|-----|------|-------|----------------|
| 2025-07-07T17:37:41 | 27.52 | 472.66 | 3.07 | 2.88 | season_length: 12 |
| 2025-07-07T17:48:36 | 27.52 | 472.66 | 3.07 | 2.88 | season_length: 12 |

### NaiveDrift
| Timestamp | MAPE | MAE | MASE | RMSSE | Hyperparameters |
|-----------|------|-----|------|-------|----------------|
| 2025-07-07T17:37:41 | 32.25 | 607.39 | 3.94 | 3.49 | (default) |
| 2025-07-07T17:48:36 | 32.25 | 607.39 | 3.94 | 3.49 | (default) |

### Croston
| Timestamp | MAPE | MAE | MASE | RMSSE | Hyperparameters |
|-----------|------|-----|------|-------|----------------|
| 2025-07-07T17:37:41 | 34.16 | 907.81 | 5.89 | 5.35 | (default) |
| 2025-07-07T17:48:36 | 34.16 | 907.81 | 5.89 | 5.35 | (default) |

### NaiveSeasonal
| Timestamp | MAPE | MAE | MASE | RMSSE | Hyperparameters |
|-----------|------|-----|------|-------|----------------|
| 2025-07-07T17:37:41 | 34.17 | 886.73 | 5.76 | 5.28 | K: 12 |
| 2025-07-07T17:48:36 | 34.17 | 886.73 | 5.76 | 5.28 | K: 12 |

---

*Note: This time-series table is based on the first two runs in web/performance_history.json. All runs and models can be included for a complete report.*
