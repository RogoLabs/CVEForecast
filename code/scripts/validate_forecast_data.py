"""Validate forecast data JSON files before deployment."""

import json
import sys

REQUIRED_KEYS = ['generated_at', 'model_rankings', 'cumulative_timelines']


def validate():
    try:
        with open('web/data.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print('FAIL: web/data.json not found')
        return False
    except json.JSONDecodeError as e:
        print(f'FAIL: web/data.json is not valid JSON: {e}')
        return False

    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"FAIL: Missing required key '{key}' in data.json")
            return False

    if not data['model_rankings']:
        print('FAIL: model_rankings is empty')
        return False

    print(f'OK: data.json valid ({len(data["model_rankings"])} models)')
    return True


if __name__ == '__main__':
    sys.exit(0 if validate() else 1)
