#!/usr/bin/env python3
"""
Generates or updates tuning_report.md from all available tuning results files in hyperparameter_results/.
Scans for all hyperparameter_tuning_results_*.json files and builds a time-series Markdown report for each model.
"""
import os
import glob
import json
from datetime import datetime

RESULTS_DIR = "hyperparameter_results"
REPORT_PATH = "tuning_report.md"

# Find all session files
session_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "hyperparameter_tuning_results_*.json")))

# Collect all results by model
model_history = {}
for path in session_files:
    with open(path, "r") as f:
        session = json.load(f)
    session_id = session.get("session_id", os.path.basename(path))
    start_time = session.get("start_time", "?")
    for model, result in session.get("models_tuned", {}).items():
        if model not in model_history:
            model_history[model] = []
        entry = {
            "session_id": session_id,
            "start_time": start_time,
            "mape": result.get("expected_performance", {}).get("mape"),
            "mae": result.get("expected_performance", {}).get("mae"),
            "mase": result.get("expected_performance", {}).get("mase"),
            "rmsse": result.get("expected_performance", {}).get("rmsse"),
            "hyperparameters": result.get("hyperparameters", {}),
        }
        model_history[model].append(entry)

# Write Markdown report
with open(REPORT_PATH, "w") as f:
    f.write("# CVEForecast Hyperparameter Tuning Results\n\n")
    f.write("This report tracks all tuning runs for each model over time. Each table shows every session's timestamp, MAPE, MAE, MASE, RMSSE, and hyperparameters.\n\n")
    f.write(f"_Generated: {datetime.now().isoformat()}_\n\n")
    for model in sorted(model_history.keys()):
        f.write(f"## {model}\n")
        f.write("| Session | Start Time | MAPE | MAE | MASE | RMSSE | Hyperparameters |\n")
        f.write("|---------|------------|------|-----|------|-------|-----------------|\n")
        for entry in sorted(model_history[model], key=lambda e: e["start_time"]):
            hp_str = ", ".join(f"{k}: {v}" for k, v in entry["hyperparameters"].items()) if entry["hyperparameters"] else "(default)"
            def fmt(val, fmtstr):
                return fmtstr.format(val) if val is not None else "-"
            f.write(f"| {entry['session_id']} | {entry['start_time'][:19]} | {fmt(entry['mape'], '{:.4f}')} | {fmt(entry['mae'], '{:.2f}')} | {fmt(entry['mase'], '{:.2f}')} | {fmt(entry['rmsse'], '{:.2f}')} | {hp_str} |\n")
        f.write("\n")
    f.write("---\n\n")
    f.write("*Note: Report includes all sessions in hyperparameter_results/. Use this script after any tuning run to update.*\n")
