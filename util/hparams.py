import json
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import List

@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_model_name_or_path(cls, model_name_or_path):
        hparams_dir = Path(inspect.getfile(cls)).parent / "hparams"
        hparams_json_path = resolve_hparams_json_path(hparams_dir, model_name_or_path)
        return cls.from_json(hparams_json_path)


def _safe_stem(x: str) -> str:
    return x.strip().replace("\\", "/").split("/")[-1]


def _normalize_key(x: str) -> str:
    return "".join(ch for ch in x.lower() if ch.isalnum())


def get_hparams_name_candidates(model_name_or_path: str) -> List[str]:
    raw = str(model_name_or_path or "").strip()
    if not raw:
        return []

    candidates = []

    # direct forms
    candidates.append(raw)
    candidates.append(raw.replace("/", "_"))
    candidates.append(raw.replace("/", "_").replace("-", "_"))

    # last segment forms
    stem = _safe_stem(raw)
    if stem:
        candidates.append(stem)
        candidates.append(stem.replace("-", "_"))
        # common suffix normalization: internlm3-8b-instruct -> internlm3_8b
        if stem.endswith("-instruct"):
            stem_no_suffix = stem[: -len("-instruct")]
            candidates.append(stem_no_suffix)
            candidates.append(stem_no_suffix.replace("-", "_"))

    # de-dup and skip empty strings
    ordered = []
    seen = set()
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def resolve_hparams_json_path(hparams_dir: Path, model_name_or_path: str) -> Path:
    for name in get_hparams_name_candidates(model_name_or_path):
        candidate = hparams_dir / f"{name}.json"
        if candidate.exists():
            return candidate

    # Fallback: compare normalized names (handles mixed legacy naming styles)
    wanted = {_normalize_key(x) for x in get_hparams_name_candidates(model_name_or_path)}
    for p in hparams_dir.glob("*.json"):
        if _normalize_key(p.stem) in wanted:
            return p

    tried = ", ".join(f"{x}.json" for x in get_hparams_name_candidates(model_name_or_path))
    raise FileNotFoundError(f"No hparams json found under {hparams_dir}. Tried: {tried}")
