from pathlib import Path
from typing import Optional, Tuple

from util.model_tokenizer import get_cached_model_tok


def _clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_like_local_path(path_text: str) -> bool:
    if not path_text:
        return False
    if path_text.startswith((".", "/", "\\")):
        return True
    # windows drive style, e.g. C:\...
    if len(path_text) >= 2 and path_text[1] == ":":
        return True
    return False


def resolve_runtime_model_name_and_path(
    model_name_or_path: str,
    hparams_model_path: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    runtime_name = _clean(model_name_or_path) or _clean(hparams_model_path)
    configured_path = _clean(hparams_model_path)

    if not configured_path:
        return runtime_name, None

    # If hparams uses an absolute/relative local path that does not exist, ignore it.
    if _looks_like_local_path(configured_path) and not Path(configured_path).exists():
        return runtime_name, None

    return runtime_name, configured_path


def get_cached_model_tok_runtime(model_name_or_path: str, hparams_model_path: Optional[str] = None):
    runtime_name, runtime_path = resolve_runtime_model_name_and_path(
        model_name_or_path=model_name_or_path,
        hparams_model_path=hparams_model_path,
    )
    return get_cached_model_tok(model_name=runtime_name, model_path=runtime_path)

