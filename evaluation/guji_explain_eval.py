from __future__ import annotations

from statistics import mean

from diagnose.diagnose import diagnosing
from evaluation.faithfulness_metrics import (
    comprehensiveness_score,
    perturbation_drop,
    sufficiency_score,
)
from evaluation.stability_metrics import explanation_stability
from evaluation.task_utility_metrics import compression_retention
from methods.assist import get_methods_by_dataset_and_model_name


def summarize_method_metrics(result):
    evidence_spans = result.get("evidence_spans", []) or []
    span_texts = [span.get("text") or span.get("span") for span in evidence_spans if span.get("text") or span.get("span")]
    top_score = max([float(span.get("score", 0.0)) for span in evidence_spans], default=0.0)

    return {
        "faithfulness": {
            "comprehensiveness": comprehensiveness_score(top_score, top_score / 2 if top_score else 0.0),
            "sufficiency": sufficiency_score(max(top_score, 1e-6), top_score),
            "perturbation_drop": perturbation_drop(top_score, top_score / 2 if top_score else 0.0),
        },
        "stability": {
            "self_overlap": explanation_stability(span_texts, span_texts),
        },
        "task_utility": {
            "compression_retention": compression_retention(max(top_score, 1e-6), top_score),
        },
    }


def evaluate_dataset(dataset, model_name, dataset_name, sample_limit=5, selected_methods=None):
    methods = selected_methods or get_methods_by_dataset_and_model_name(dataset_name, model_name)
    aggregated = {method: [] for method in methods}

    for sample in dataset[:sample_limit]:
        for method in methods:
            result = diagnosing(sample=sample, model_name_or_path=model_name, method=method)
            aggregated[method].append(summarize_method_metrics(result))

    summary = {}
    for method, metric_list in aggregated.items():
        if not metric_list:
            summary[method] = {}
            continue
        summary[method] = {
            "faithfulness": {
                metric: round(mean(item["faithfulness"][metric] for item in metric_list), 6)
                for metric in metric_list[0]["faithfulness"]
            },
            "stability": {
                metric: round(mean(item["stability"][metric] for item in metric_list), 6)
                for metric in metric_list[0]["stability"]
            },
            "task_utility": {
                metric: round(mean(item["task_utility"][metric] for item in metric_list), 6)
                for metric in metric_list[0]["task_utility"]
            },
        }
    return summary
