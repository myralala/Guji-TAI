from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_ORDER = [
    "Attribution",
    "Attention Weights",
    "FiNE",
    "Logit Lens",
]

METHOD_LABELS = {
    "Attribution": "Attribution",
    "Attention Weights": "Attention Weights",
    "FiNE": "FiNE",
    "Logit Lens": "Logit Lens",
}

DATASET_LABELS = {
    "自动标点_chain_of_thought": "自动标点",
    "缺字补全_chain_of_thought": "缺字补全",
    "传统文化问答_0_shot": "传统文化问答",
    "关系抽取_0_shot": "关系抽取",
}


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _method_value(summary: dict, field: str) -> float:
    return float(summary[field]["mean"])


def build_markdown(payload: dict) -> str:
    overall = payload["overall_method_summary"]
    datasets = payload["datasets"]

    lines = [
        "# Overhead Paper Tables",
        "",
        f"- Model: `{payload['model_name']}`",
        f"- Samples per dataset: `{payload['sample_limit']}`",
        "",
        "## Table 1. Overall interpretability overhead",
        "",
        "| Method | Explain Time (s/sample) | Time Overhead (x) | Peak GPU (GB) | Memory Overhead (x) |",
        "|---|---:|---:|---:|---:|",
    ]

    for method_name in METHOD_ORDER:
        summary = overall[method_name]
        lines.append(
            "| {method} | {time} | {ratio} | {gpu} | {mem_ratio} |".format(
                method=METHOD_LABELS[method_name],
                time=_format_float(_method_value(summary, "explain_time_seconds")),
                ratio=_format_float(_method_value(summary, "time_overhead_ratio"), digits=2),
                gpu=_format_float(_method_value(summary, "explain_peak_gpu_gb")),
                mem_ratio=_format_float(_method_value(summary, "memory_overhead_ratio"), digits=2),
            )
        )

    lines.extend(
        [
            "",
            "## Table 2. Task-level time overhead ratio (x)",
            "",
            "| Task | Attribution | Attention Weights | FiNE | Logit Lens |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for dataset_name in DATASET_LABELS:
        info = datasets[dataset_name]["method_summaries"]
        lines.append(
            "| {task} | {a} | {aw} | {fine} | {ll} |".format(
                task=DATASET_LABELS[dataset_name],
                a=_format_float(_method_value(info["Attribution"], "time_overhead_ratio"), digits=2),
                aw=_format_float(_method_value(info["Attention Weights"], "time_overhead_ratio"), digits=2),
                fine=_format_float(_method_value(info["FiNE"], "time_overhead_ratio"), digits=2),
                ll=_format_float(_method_value(info["Logit Lens"], "time_overhead_ratio"), digits=2),
            )
        )

    lines.extend(
        [
            "",
            "## Table 3. Task-level peak GPU memory (GB)",
            "",
            "| Task | Attribution | Attention Weights | FiNE | Logit Lens |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for dataset_name in DATASET_LABELS:
        info = datasets[dataset_name]["method_summaries"]
        lines.append(
            "| {task} | {a} | {aw} | {fine} | {ll} |".format(
                task=DATASET_LABELS[dataset_name],
                a=_format_float(_method_value(info["Attribution"], "explain_peak_gpu_gb")),
                aw=_format_float(_method_value(info["Attention Weights"], "explain_peak_gpu_gb")),
                fine=_format_float(_method_value(info["FiNE"], "explain_peak_gpu_gb")),
                ll=_format_float(_method_value(info["Logit Lens"], "explain_peak_gpu_gb")),
            )
        )

    lines.extend(
        [
            "",
            "Note: all numbers are means over the reduced setting (`4 tasks x 4 methods x 10 samples`).",
            "",
        ]
    )
    return "\n".join(lines)


def build_latex(payload: dict) -> str:
    overall = payload["overall_method_summary"]
    datasets = payload["datasets"]

    overall_rows = []
    for method_name in METHOD_ORDER:
        summary = overall[method_name]
        overall_rows.append(
            "{method} & {time} & {ratio} & {gpu} & {mem_ratio} \\\\".format(
                method=METHOD_LABELS[method_name],
                time=_format_float(_method_value(summary, "explain_time_seconds")),
                ratio=_format_float(_method_value(summary, "time_overhead_ratio"), digits=2),
                gpu=_format_float(_method_value(summary, "explain_peak_gpu_gb")),
                mem_ratio=_format_float(_method_value(summary, "memory_overhead_ratio"), digits=2),
            )
        )

    ratio_rows = []
    gpu_rows = []
    for dataset_name in DATASET_LABELS:
        info = datasets[dataset_name]["method_summaries"]
        task = DATASET_LABELS[dataset_name]
        ratio_rows.append(
            "{task} & {a} & {aw} & {fine} & {ll} \\\\".format(
                task=task,
                a=_format_float(_method_value(info["Attribution"], "time_overhead_ratio"), digits=2),
                aw=_format_float(_method_value(info["Attention Weights"], "time_overhead_ratio"), digits=2),
                fine=_format_float(_method_value(info["FiNE"], "time_overhead_ratio"), digits=2),
                ll=_format_float(_method_value(info["Logit Lens"], "time_overhead_ratio"), digits=2),
            )
        )
        gpu_rows.append(
            "{task} & {a} & {aw} & {fine} & {ll} \\\\".format(
                task=task,
                a=_format_float(_method_value(info["Attribution"], "explain_peak_gpu_gb")),
                aw=_format_float(_method_value(info["Attention Weights"], "explain_peak_gpu_gb")),
                fine=_format_float(_method_value(info["FiNE"], "explain_peak_gpu_gb")),
                ll=_format_float(_method_value(info["Logit Lens"], "explain_peak_gpu_gb")),
            )
        )

    return f"""% Auto-generated from reduced overhead evaluation results
% Model: {payload['model_name']}
% Sample limit per dataset: {payload['sample_limit']}

\\begin{{table}}[t]
\\centering
\\caption{{Overall interpretability overhead on Qwen3-8B.}}
\\label{{tab:overhead-overall}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Method & Explain Time (s/sample) & Time Overhead ($\\times$) & Peak GPU (GB) & Memory Overhead ($\\times$) \\\\
\\midrule
{chr(10).join(overall_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\caption{{Task-level time overhead ratio ($\\times$).}}
\\label{{tab:overhead-ratio}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrr}}
\\toprule
Task & Attribution & Attention Weights & FiNE & Logit Lens \\\\
\\midrule
{chr(10).join(ratio_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}

\\begin{{table}}[t]
\\centering
\\caption{{Task-level peak GPU memory (GB).}}
\\label{{tab:overhead-gpu}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrr}}
\\toprule
Task & Attribution & Attention Weights & FiNE & Logit Lens \\\\
\\midrule
{chr(10).join(gpu_rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()

    input_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    markdown = build_markdown(payload)
    latex = build_latex(payload)

    (out_dir / "paper_overhead_tables.md").write_text(markdown, encoding="utf-8")
    (out_dir / "paper_overhead_tables.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
