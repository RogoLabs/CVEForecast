"""Generate summary report after hyperparameter tuning."""
import datetime as dt
import json
import os
import pathlib

workspace = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", "."))
status = os.environ.get("JOB_STATUS", "unknown")
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

summary_lines = [
    "# Hyperparameter Tuning Summary",
    "",
    f"**Date:** {now}",
    f"**Status:** {status}",
    ""
]

config_path = workspace / "code" / "config.json"
if config_path.exists():
    data = json.loads(config_path.read_text())
    models = sorted(data.get("models", {}).keys())
    summary_lines.append("## Models in config.json")
    summary_lines.extend([f"- {model}" for model in models])
    summary_lines.append("")

(workspace / "tuning_summary.md").write_text("\n".join(summary_lines))

status_payload = {
    "status": status,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat() + "Z"
}
(workspace / "tuning_status.json").write_text(json.dumps(status_payload, indent=2))
print(f"Report generated: status={status}")
