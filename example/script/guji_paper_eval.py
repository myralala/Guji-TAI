import argparse
import json
import sys
from pathlib import Path

base_dir = str(Path(__file__).resolve().parent.parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from data.chain_of_thought.dataset_process import name2dataset_module
from evaluation.guji_paper_eval import evaluate_dataset_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--sample_limit", default=3, type=int)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    if args.dataset_name not in name2dataset_module:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    dataset = name2dataset_module[args.dataset_name].get_default_dataset()
    summary = evaluate_dataset_paper(
        dataset=dataset,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        sample_limit=args.sample_limit,
        selected_methods=args.methods,
    )
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "model_name": args.model_name,
                "sample_limit": args.sample_limit,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
