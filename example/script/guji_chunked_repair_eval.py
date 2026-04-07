from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.chain_of_thought.dataset_process import name2dataset_module
from evaluation.guji_paper_eval import evaluate_dataset_paper
from evaluation.paper_eval_runner import build_chunk_ranges, merge_weighted_metric_summaries


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--sample_limit", type=int, default=50)
    ap.add_argument("--chunk_size", type=int, default=5)
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


def persist_payload(out_json: Path, payload):
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

    for dataset_name in args.datasets:
        dataset = name2dataset_module[dataset_name].get_default_dataset()
        total = min(args.sample_limit, len(dataset))
        chunk_ranges = build_chunk_ranges(total=total, chunk_size=args.chunk_size)

        entry = dict(payload["datasets"].get(dataset_name) or {})
        entry["methods"] = list(args.methods)
        entry["chunk_size"] = args.chunk_size
        entry["target_num_samples"] = total
        entry["chunks"] = dict(entry.get("chunks") or {})

        print(f"===== DATASET START {args.model_name} :: {dataset_name} :: total={total} =====", flush=True)
        persist_payload(out_json, payload)

        for start, end in chunk_ranges:
            chunk_key = f"{start}:{end}"
            if chunk_key in entry["chunks"] and "summary" in entry["chunks"][chunk_key]:
                continue

            chunk_samples = dataset[start:end]
            print(f"===== CHUNK START {args.model_name} :: {dataset_name} :: {chunk_key} =====", flush=True)
            try:
                chunk_summary = evaluate_dataset_paper(
                    dataset=chunk_samples,
                    model_name=args.model_name,
                    dataset_name=dataset_name,
                    sample_limit=len(chunk_samples),
                    selected_methods=args.methods,
                )
                entry["chunks"][chunk_key] = {
                    "num_samples": len(chunk_samples),
                    "summary": chunk_summary,
                }
                weighted = [
                    (chunk_info["num_samples"], chunk_info["summary"])
                    for chunk_info in entry["chunks"].values()
                    if isinstance(chunk_info, dict) and "summary" in chunk_info
                ]
                entry["completed_num_samples"] = sum(weight for weight, _ in weighted)
                entry["summary"] = merge_weighted_metric_summaries(weighted)
                entry.pop("error", None)
                entry.pop("traceback", None)
                print(
                    f"===== CHUNK DONE {args.model_name} :: {dataset_name} :: {chunk_key} :: completed={entry['completed_num_samples']}/{total} =====",
                    flush=True,
                )
            except Exception as exc:
                entry["error"] = repr(exc)
                entry["traceback"] = traceback.format_exc()
                payload["datasets"][dataset_name] = entry
                persist_payload(out_json, payload)
                print(f"===== CHUNK ERROR {args.model_name} :: {dataset_name} :: {chunk_key} :: {exc!r} =====", flush=True)
                raise

            payload["datasets"][dataset_name] = entry
            persist_payload(out_json, payload)

        print(f"===== DATASET DONE {args.model_name} :: {dataset_name} =====", flush=True)
        payload["datasets"][dataset_name] = entry
        persist_payload(out_json, payload)

    print(f"===== MODEL DONE {args.model_name} =====", flush=True)


if __name__ == "__main__":
    main()
