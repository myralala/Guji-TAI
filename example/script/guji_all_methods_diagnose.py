import argparse
import json
import sys
from pathlib import Path

base_dir = str(Path(__file__).resolve().parent.parent.parent)
if base_dir not in sys.path:
    sys.path.append(base_dir)

from data.chain_of_thought.dataset_process import name2dataset_module
from diagnose.diagnose import diagnosing
from methods import method_load_errors
from methods.assist import get_methods_by_dataset_and_model_name
from util.explanation_fusion import fuse_explanations


def run(dataset_name: str, model_name: str, sample_index: int, selected_methods=None, task_aware: bool = False, target_mode: str = "auto"):
    if dataset_name not in name2dataset_module:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_module = name2dataset_module[dataset_name]
    dataset = dataset_module.get_default_dataset()
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_name} is empty.")

    sample_index = max(0, min(sample_index, len(dataset) - 1))
    sample = dataset[sample_index]

    methods = selected_methods or get_methods_by_dataset_and_model_name(
        dataset_name=dataset_name,
        model_name=model_name,
    )
    if not methods:
        raise RuntimeError(
            f"No runnable methods. method_load_errors={method_load_errors}"
        )

    results = {}
    for method_name in methods:
        results[method_name] = diagnosing(
            sample=sample,
            model_name_or_path=model_name,
            method=method_name,
        )
    fused_result = fuse_explanations(list(results.values())) if task_aware else None
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "sample_index": sample_index,
        "sample_prompt": sample.get("prompt", ""),
        "method_names": methods,
        "task_aware": task_aware,
        "target_mode": target_mode,
        "results": results,
        "fused_result": fused_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--sample_index", default=0, type=int)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--task_aware", action="store_true")
    parser.add_argument("--target_mode", default="auto", type=str)
    args = parser.parse_args()

    payload = run(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        sample_index=args.sample_index,
        selected_methods=args.methods,
        task_aware=args.task_aware,
        target_mode=args.target_mode,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
