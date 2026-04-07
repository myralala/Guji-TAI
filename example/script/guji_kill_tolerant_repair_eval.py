from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.chain_of_thought.dataset_process import name2dataset_module
from evaluation.paper_eval_runner import (
    interpret_subprocess_returncode,
    merge_weighted_metric_summaries,
    summarize_chunk_entries,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--sample_limit", type=int, default=50)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    return ap.parse_args()


def load_payload(out_json: Path, *, model_name: str, sample_limit: int, datasets, methods):
    if out_json.exists():
        payload = json.loads(out_json.read_text())
    else:
        payload = {}
    payload["model_name"] = model_name
    payload["sample_limit"] = sample_limit
    payload["dataset_names"] = list(payload.get("dataset_names") or [])
    for name in datasets:
        if name not in payload["dataset_names"]:
            payload["dataset_names"].append(name)
    payload["method_names"] = list(methods)
    payload["datasets"] = dict(payload.get("datasets") or {})
    return payload


def persist(out_json: Path, payload):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    out_json = Path(args.out_json)
    payload = load_payload(
        out_json,
        model_name=args.model_name,
        sample_limit=args.sample_limit,
        datasets=args.datasets,
        methods=args.methods,
    )

    worker = ROOT / "example" / "script" / "guji_isolated_sample_worker.py"

    for dataset_name in args.datasets:
        dataset = name2dataset_module[dataset_name].get_default_dataset()
        total = min(args.sample_limit, len(dataset))
        entry = dict(payload["datasets"].get(dataset_name) or {})
        entry["methods"] = list(args.methods)
        entry["chunk_size"] = 1
        entry["target_num_samples"] = total
        entry["chunks"] = dict(entry.get("chunks") or {})

        print(f"===== DATASET START {args.model_name} :: {dataset_name} :: total={total} =====", flush=True)
        payload["datasets"][dataset_name] = entry
        persist(out_json, payload)

        for idx in range(total):
            chunk_key = f"{idx}:{idx+1}"
            if chunk_key in entry["chunks"]:
                continue

            fd, tmp_path = tempfile.mkstemp(prefix="guji_sample_", suffix=".json")
            os.close(fd)
            try:
                cmd = [
                    sys.executable,
                    str(worker),
                    "--model_name",
                    args.model_name,
                    "--dataset_name",
                    dataset_name,
                    "--sample_index",
                    str(idx),
                    "--out_json",
                    tmp_path,
                    "--methods",
                    *args.methods,
                ]
                print(f"===== SAMPLE START {args.model_name} :: {dataset_name} :: {chunk_key} =====", flush=True)
                proc = subprocess.run(cmd)
                status, reason = interpret_subprocess_returncode(proc.returncode)
                if status == "ok" and Path(tmp_path).exists():
                    sample_payload = json.loads(Path(tmp_path).read_text())
                    entry["chunks"][chunk_key] = {
                        "num_samples": sample_payload.get("num_samples", 1),
                        "summary": sample_payload["summary"],
                    }
                    print(f"===== SAMPLE DONE {args.model_name} :: {dataset_name} :: {chunk_key} =====", flush=True)
                else:
                    entry["chunks"][chunk_key] = {
                        "num_samples": 1,
                        "skipped": True,
                        "reason": reason,
                        "returncode": proc.returncode,
                    }
                    print(
                        f"===== SAMPLE SKIP {args.model_name} :: {dataset_name} :: {chunk_key} :: status={status} reason={reason} rc={proc.returncode} =====",
                        flush=True,
                    )

                stats = summarize_chunk_entries(entry["chunks"])
                entry["completed_num_samples"] = stats["completed_num_samples"]
                entry["skipped_num_samples"] = stats["skipped_num_samples"]
                if stats["weighted_summaries"]:
                    entry["summary"] = merge_weighted_metric_summaries(stats["weighted_summaries"])
                payload["datasets"][dataset_name] = entry
                persist(out_json, payload)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        print(
            f"===== DATASET DONE {args.model_name} :: {dataset_name} :: completed={entry.get('completed_num_samples', 0)} skipped={entry.get('skipped_num_samples', 0)} =====",
            flush=True,
        )
        payload["datasets"][dataset_name] = entry
        persist(out_json, payload)

    print(f"===== MODEL DONE {args.model_name} =====", flush=True)


if __name__ == "__main__":
    main()
