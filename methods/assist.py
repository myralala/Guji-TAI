from pathlib import Path

from data.chain_of_thought.dataset_process import name2dataset_module
from methods import method_name2sub_module
from util.hparams import resolve_hparams_json_path

# Must match registered method display names in methods/__init__.py.
ALL_SELECTED_METHODS = [
    "Attention Weights",
    "Attribution",
    "CausalTracing",
    "FiNE",
    "KN",
    "Logit Lens",
]

DATASET_TYPE_2_METHODS = {
    # normalized Chinese labels
    "问答类": ALL_SELECTED_METHODS,
    "生成类": ALL_SELECTED_METHODS,
    "结构化理解类": ALL_SELECTED_METHODS,
    # normalized English labels
    "qa": ALL_SELECTED_METHODS,
    "generation": ALL_SELECTED_METHODS,
    "structured_understanding": ALL_SELECTED_METHODS,
    # current mojibake fallback labels in existing files
    "闂瓟绫?": ALL_SELECTED_METHODS,
    "鐢熸垚绫?": ALL_SELECTED_METHODS,
    "缁撴瀯鍖栫悊瑙ｇ被": ALL_SELECTED_METHODS,
}

AUTO_DERIVABLE_KEYS = {
    "prompts",
    "triple_subject",
}

DATASET_NAME_HINT_2_METHODS = {
    "自动标点": ["Attention Weights", "Attribution", "CausalTracing", "Logit Lens"],
    "标点": ["Attention Weights", "Attribution", "CausalTracing", "Logit Lens"],
    "缺字": ["Attribution", "CausalTracing", "FiNE", "KN", "Logit Lens"],
    "补全": ["Attribution", "CausalTracing", "FiNE", "KN", "Logit Lens"],
    "翻译": ["Attribution", "Attention Weights", "Logit Lens"],
    "释义": ["Attribution", "Attention Weights", "Logit Lens"],
    "关系": ["Attribution", "Attention Weights", "CausalTracing", "Logit Lens"],
    "问答": ["Attribution", "CausalTracing", "FiNE", "KN", "Logit Lens"],
    "推理": ["Attribution", "CausalTracing", "FiNE", "KN", "Logit Lens"],
}


def _is_template_compatible(support_template_keys, requires_input_keys):
    support = set(support_template_keys)
    required = set(requires_input_keys)
    unresolved = required - support
    return unresolved.issubset(AUTO_DERIVABLE_KEYS)


def get_task_aware_methods_for_dataset_name(dataset_name: str):
    for hint, methods in DATASET_NAME_HINT_2_METHODS.items():
        if hint in dataset_name:
            return methods
    return list(ALL_SELECTED_METHODS)


def get_methods_by_dataset(dataset_name: str):
    """Field-based method filtering."""
    method_names = []
    if dataset_name not in name2dataset_module:
        return method_names

    dataset_module = name2dataset_module[dataset_name]
    support_template_keys = dataset_module.support_template_keys

    for method_name, method_module in method_name2sub_module.items():
        requires_input_keys = method_module.requires_input_keys
        if _is_template_compatible(support_template_keys, requires_input_keys):
            method_names.append(method_name)
    return method_names


def get_methods_by_dataset_type(dataset_name: str):
    """Dataset-type priority routing with field compatibility guard."""
    if dataset_name not in name2dataset_module:
        return []

    dataset_module = name2dataset_module[dataset_name]
    dataset_info = getattr(dataset_module, "dataset_info", {})
    dataset_type = dataset_info.get("dataset_type", "")

    preferred_methods = get_task_aware_methods_for_dataset_name(dataset_name)
    if preferred_methods == ALL_SELECTED_METHODS:
        preferred_methods = DATASET_TYPE_2_METHODS.get(dataset_type, ALL_SELECTED_METHODS)
    support_template_keys = dataset_module.support_template_keys
    valid_methods = []

    for method_name in preferred_methods:
        if method_name not in method_name2sub_module:
            continue
        method_module = method_name2sub_module[method_name]
        requires_input_keys = method_module.requires_input_keys
        if _is_template_compatible(support_template_keys, requires_input_keys):
            valid_methods.append(method_name)

    return valid_methods


def get_methods_by_model_name(model_name: str):
    """Keep methods that have hparams for the current model."""
    method_names = []
    for method_name, method_module in method_name2sub_module.items():
        hparams_dir = Path(method_module.__file__).parent / "hparams"
        try:
            resolve_hparams_json_path(hparams_dir, model_name)
            method_names.append(method_name)
        except FileNotFoundError:
            continue
    return method_names


def get_methods_by_dataset_and_model_name(dataset_name: str, model_name: str):
    """Apply dataset filtering then model hparams filtering."""
    methods1 = get_methods_by_dataset_type(dataset_name=dataset_name)

    if not methods1:
        methods1 = get_methods_by_dataset(dataset_name=dataset_name)

    methods2 = get_methods_by_model_name(model_name=model_name)

    # keep stable ordering from dataset side
    return [m for m in methods1 if m in methods2]


def sort_methods_by_cost_time(method_names: list):
    return sorted(
        method_names,
        key=lambda name: method_name2sub_module[name].cost_seconds_per_query,
    )

