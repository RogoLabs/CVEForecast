"""Normalize tuner file paths for CI environment."""

import json
import os
import pathlib

cfg_path = pathlib.Path('code/tuner/tuner_config.json')
if not cfg_path.exists():
    print('Tuner config not found, skipping path normalization')
    raise SystemExit(0)

data = json.loads(cfg_path.read_text())
workspace = pathlib.Path(os.environ.get('GITHUB_WORKSPACE', '.'))

updated = False
for key, rel in data.get('file_paths', {}).items():
    abs_path = (workspace / rel).resolve()
    if data['file_paths'][key] != str(abs_path):
        data['file_paths'][key] = str(abs_path)
        updated = True

if updated:
    cfg_path.write_text(json.dumps(data, indent=2))
    print(f'Updated {len(data["file_paths"])} paths')
else:
    print('Paths already normalized')
