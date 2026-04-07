from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List


def _as_text(span: Dict) -> str:
    return str(span.get("text") or span.get("span") or "").strip()


def _as_score(span: Dict):
    score = span.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _merge_spans(results: Iterable[Dict]) -> List[Dict]:
    merged = {}
    order = []

    for result in results:
        for span in result.get("evidence_spans", []) or []:
            text = _as_text(span)
            if not text:
                continue
            score = _as_score(span)
            if text not in merged:
                merged[text] = {"text": text}
                if score is not None:
                    merged[text]["score"] = score
                order.append(text)
                continue
            existing_score = merged[text].get("score")
            if score is not None and (existing_score is None or score > existing_score):
                merged[text]["score"] = score

    return [merged[text] for text in order]


def _merge_sorted_unique(results: Iterable[Dict], key: str) -> List:
    values = set()
    for result in results:
        for value in result.get(key, []) or []:
            values.add(value)
    return sorted(values)


def _average_pairwise_overlap(sets: List[set]) -> float:
    if len(sets) < 2:
        return 1.0 if sets else 0.0

    scores = []
    for left, right in combinations(sets, 2):
        union = left | right
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(left & right) / len(union))
    return sum(scores) / len(scores)


def fuse_explanations(results: List[Dict]) -> Dict:
    results = results or []
    evidence_spans = _merge_spans(results)
    key_layers = _merge_sorted_unique(results, "key_layers")
    key_heads = _merge_sorted_unique(results, "key_heads")
    key_neurons = _merge_sorted_unique(results, "key_neurons")

    span_sets = []
    layer_sets = []
    for result in results:
        span_sets.append({_as_text(span) for span in (result.get("evidence_spans", []) or []) if _as_text(span)})
        layer_sets.append(set(result.get("key_layers", []) or []))

    consistency = {
        "num_results": len(results),
        "span_overlap": _average_pairwise_overlap(span_sets),
        "layer_overlap": _average_pairwise_overlap(layer_sets),
    }

    return {
        "evidence_spans": evidence_spans,
        "key_layers": key_layers,
        "key_heads": key_heads,
        "key_neurons": key_neurons,
        "consistency": consistency,
    }
