import threading

import numpy as np
import torch
from tqdm import tqdm

from . import CausalTraceHyperParams
from .causal_tracing import (
    calculate_hidden_flow,
    collect_embedding_gaussian,
    collect_embedding_std,
    collect_embedding_tdist,
    plot_trace_heatmap,
)
from util.fileutil import get_temp_file_with_prefix_suffix
from util.guji_labels import causal_trace_plot_meta
from util.runtime import get_cached_model_tok_runtime
from util.sample_adapter import infer_triple_subject


lock_kn = threading.Lock()


def _resolve_trace_target(sample):
    explanation_target = sample.get("explanation_target", {}) or {}
    target_type = explanation_target.get("target_type", "")

    if target_type in {"missing_position", "decision_position", "answer_unit"}:
        focus_text = explanation_target.get("focus_text")
        if focus_text:
            return str(focus_text)

    if target_type == "target_triple":
        triple = explanation_target.get("target_triple") or sample.get("target_triple")
        if isinstance(triple, dict):
            joined = " ".join(str(x) for x in [triple.get("subject"), triple.get("predicate"), triple.get("object")] if x)
            if joined:
                return joined
        if triple:
            return str(triple)

    output_segment = explanation_target.get("output_segment")
    if output_segment:
        return str(output_segment)

    subject = sample.get("triple_subject") or infer_triple_subject(sample)
    if subject:
        return str(subject)
    return str(sample.get("prompt", ""))[:8]


def _extract_key_layers_and_pairs(score_matrix, input_tokens, top_k=3):
    score_array = np.array(score_matrix, dtype=float)
    if score_array.size == 0:
        return [], []

    flat = np.argpartition(score_array.flatten(), -min(top_k, score_array.size))[-min(top_k, score_array.size):]
    rows, cols = np.unravel_index(flat, score_array.shape)
    ranked = sorted(
        zip(rows.tolist(), cols.tolist()),
        key=lambda item: score_array[item[0], item[1]],
        reverse=True,
    )
    pairs = [
        {
            "layer": int(col),
            "token": str(input_tokens[row]),
            "score": round(float(score_array[row, col]), 6),
        }
        for row, col in ranked
    ]
    key_layers = sorted({pair["layer"] for pair in pairs})
    return key_layers, pairs


def diagnose(sample, model_name_or_path, hparams=None):
    result = {
        "origin_data": {
            "model_output": None,
            "prob": None,
            "tokens": None,
            "subject range": None,
            "Restoring state score": None,
            "Restoring MLP score": None,
            "Restoring Attn score": None,
        },
        "image": [],
    }

    with lock_kn:
        hparams = (
            CausalTraceHyperParams.from_model_name_or_path(model_name_or_path)
            if hparams is None
            else hparams
        )
        mt = get_cached_model_tok_runtime(
            model_name_or_path=model_name_or_path,
            hparams_model_path=hparams.model_path,
        )

        subject = _resolve_trace_target(sample)

        uniform_noise = False
        noise_level = hparams.noise_level
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(mt, [subject])
        elif noise_level == "m":
            noise_level = collect_embedding_gaussian(mt)
        elif noise_level.startswith("t"):
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

        for kind in tqdm([None, "attn", "mlp"], desc="Causal tracing..."):
            rt = calculate_hidden_flow(
                mt,
                sample["prompt"],
                subject,
                kind=kind,
                noise=noise_level,
                uniform_noise=uniform_noise,
                replace=hparams.replace,
                window=hparams.window,
            )

            numpy_result = {
                k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v
                for k, v in rt.items()
            }
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind

            tmp_png_file = get_temp_file_with_prefix_suffix(suffix=".png")
            plot_trace_heatmap(plot_result, savepdf=tmp_png_file, modelname=mt.model_type)
            plot_meta = causal_trace_plot_meta(kind=kind, modelname=mt.model_type, window=hparams.window)
            name = plot_meta["image_name"]

            score = plot_result["scores"].tolist()
            tem_score = np.array(score)
            max_indices = np.argpartition(tem_score.flatten(), -3)[-3:]
            max_indices = np.unravel_index(max_indices, tem_score.shape)
            max_indices = list(zip(*max_indices))
            token_layer_pairs = [
                f'Layer_{ind[1]}-Token_{rt["input_tokens"][ind[0]]}' for ind in max_indices
            ]
            img_des = (
                plot_meta["image_des"]
                if kind == "mlp"
                else ""
            )
            img_res = (
                "For each component, we are computing the top 3 causal tracing scores corresponding to the token-layer pair: \n"
                + ", ".join(token_layer_pairs)
                + "."
            )

            result["image"].append(
                {
                    "image_name": name,
                    "image_path": tmp_png_file,
                    "image_des": img_des,
                    "image_res": img_res,
                }
            )
            result["origin_data"]["model_output"] = rt["answer"]
            result["origin_data"]["prob"] = rt["high_score"].item()
            result["output"] = rt["answer"]
            result["origin_data"]["tokens"] = rt["input_tokens"]
            result["origin_data"]["subject range"] = rt["subject_range"]
            result["origin_data"][name + " score"] = score
            key_layers, critical_pairs = _extract_key_layers_and_pairs(score, rt["input_tokens"])
            result["key_layers"] = sorted(set(result.get("key_layers", []) + key_layers))
            result["evidence_spans"] = result.get("evidence_spans", []) + critical_pairs
            result["causal_effect_score"] = max(
                [pair["score"] for pair in critical_pairs],
                default=0.0,
            )
            result["origin_data"]["trace_target"] = subject

        result["result_des"] = ""
        return result
