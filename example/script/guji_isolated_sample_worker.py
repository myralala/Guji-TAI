from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.chain_of_thought.dataset_process import name2dataset_module
from evaluation.guji_paper_eval import evaluate_dataset_paper


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--sample_index", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    dataset = name2dataset_module[args.dataset_name].get_default_dataset()
    sample = dataset[args.sample_index : args.sample_index + 1]
    summary = evaluate_dataset_paper(
        dataset=sample,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        sample_limit=1,
        selected_methods=args.methods,
    )
    Path(args.out_json).write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "sample_index": args.sample_index,
                "num_samples": 1,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
