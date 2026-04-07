from __future__ import annotations

from statistics import mean

from evaluation.paper_metrics import evaluate_explanation_sample
from methods.assist import get_methods_by_dataset_and_model_name


def summarize_dataset_paper_metrics(metric_list):
    if not metric_list:
        return {}

    summary = {}
    for section in ["faithfulness", "stability", "target_alignment"]:
        keys = metric_list[0].get(section, {}).keys()
        summary[section] = {
            key: round(mean(item[section][key] for item in metric_list), 6)
            for key in keys
        }
    return summary


def evaluate_dataset_paper(dataset, model_name, dataset_name, sample_limit=3, selected_methods=None):
    methods = selected_methods or get_methods_by_dataset_and_model_name(dataset_name, model_name)
    aggregated = {method: [] for method in methods}

    for sample in dataset[:sample_limit]:
        for method in methods:
            aggregated[method].append(
                evaluate_explanation_sample(
                    sample=sample,
                    model_name=model_name,
                    method_name=method,
                )
            )

    return {
        method: summarize_dataset_paper_metrics(metric_list)
        for method, metric_list in aggregated.items()
    }
