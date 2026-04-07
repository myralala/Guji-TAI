"""Shared helpers for cleaning tokens and producing ranked text spans."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_PUNCT_SPLIT_RE = re.compile(r"[，。！？；：、,.!?;:\n\r\t]+")
_SPACE_TOKENS = ("▁", "Ġ")
_SPECIAL_TOKENS = {
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
}
_SPAN_CONTENT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


@dataclass(frozen=True)
class _TokenMeta:
    text: str
    is_bpe: bool
    needs_space: bool
    is_separator: bool


def _normalize_token(raw_token: object) -> _TokenMeta | None:
    if raw_token is None:
        return None

    token = str(raw_token).strip()
    if not token or token in _SPECIAL_TOKENS:
        return None

    is_bpe = token.startswith("##")
    if is_bpe:
        token = token[2:]

    needs_space = False
    while token and token[0] in _SPACE_TOKENS:
        needs_space = True
        token = token[1:]

    if not token:
        return None

    if _PUNCT_SPLIT_RE.fullmatch(token):
        return _TokenMeta(text=token, is_bpe=False, needs_space=False, is_separator=True)

    if not _SPAN_CONTENT_RE.search(token):
        return None

    return _TokenMeta(
        text=token,
        is_bpe=is_bpe,
        needs_space=needs_space and not is_bpe,
        is_separator=False,
    )


def _is_cjk_char(value: str) -> bool:
    if not value:
        return False
    char = value[0]
    return "\u4e00" <= char <= "\u9fff"


def _requires_space(prev_text: str, token_text: str, needs_space: bool) -> bool:
    if needs_space:
        return True
    if not prev_text:
        return False
    if _is_cjk_char(prev_text[-1]) or _is_cjk_char(token_text):
        return False
    return True


def _to_score(raw_score: object) -> float | None:
    if raw_score is None:
        return None
    if hasattr(raw_score, "item") and callable(raw_score.item):
        try:
            return float(raw_score.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return None


def build_ranked_spans(
    tokens: Sequence[str],
    scores: Sequence[object],
    top_k: int | None = 5,
) -> list[dict[str, float | str]]:
    """Merge token-level evidence into ranked contiguous spans.

    Args:
        tokens: Token strings reported by an explanation method.
        scores: Numeric salience aligned with ``tokens``.
        top_k: How many spans to return. ``None`` returns all spans.
    """
    if not tokens or not scores:
        return []

    spans: list[dict[str, float | str]] = []
    current_text = ""
    current_score = 0.0

    def _flush() -> None:
        nonlocal current_text, current_score
        if current_text:
            spans.append({"text": current_text, "score": round(current_score, 6)})
        current_text = ""
        current_score = 0.0

    for token, raw_score in zip(tokens, scores):
        score = _to_score(raw_score)
        if score is None:
            continue

        meta = _normalize_token(token)
        if meta is None:
            continue

        if meta.is_separator:
            _flush()
            continue

        if meta.is_bpe:
            if not current_text:
                current_text = meta.text
            else:
                current_text += meta.text
            current_score += score
            continue

        if current_text and _requires_space(current_text, meta.text, meta.needs_space):
            current_text += " "

        current_text += meta.text
        current_score += score

    _flush()

    sorted_spans = sorted(spans, key=lambda span: span["score"], reverse=True)

    if top_k is None:
        return sorted_spans
    limit = int(top_k)
    if limit <= 0:
        return []
    return sorted_spans[:limit]


__all__ = ["build_ranked_spans"]
