from __future__ import annotations

import json
from pathlib import Path


def collect_metric_rows(payloads):
    rows = []
    for payload in payloads:
        model_name = payload.get("model_name", "")
        for dataset_name, dataset_payload in (payload.get("datasets") or {}).items():
            if "summary" not in dataset_payload:
                continue
            for method_name, summary in dataset_payload["summary"].items():
                row = {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "method_name": method_name,
                }
                for section in ["faithfulness", "stability", "target_alignment"]:
                    for key, value in (summary.get(section) or {}).items():
                        row[f"{section}.{key}"] = value
                rows.append(row)
    return rows


def load_payloads(paths):
    return [json.loads(Path(path).read_text()) for path in paths]
