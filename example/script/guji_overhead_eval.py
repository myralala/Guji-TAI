import argparse
import json
import sys
from pathlib import Path

base_dir = str(Path(__file__).resolve().parent.parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from evaluation.overhead_eval_runner import default_guji_dataset_names, run_overhead_eval
from methods.assist import ALL_SELECTED_METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--sample_limit", default=10, type=int)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    payload = run_overhead_eval(
        model_name=args.model_name,
        out_dir=args.out_dir,
        sample_limit=args.sample_limit,
        dataset_names=args.datasets or default_guji_dataset_names(),
        method_names=args.methods or ALL_SELECTED_METHODS,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
