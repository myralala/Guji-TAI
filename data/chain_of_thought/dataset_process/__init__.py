from pathlib import Path
import importlib.util
import sys

dataset_list = []
name2dataset_module = dict()
dataset_load_errors = {}

CURRENT_DIR = Path(__file__).absolute().parent

# 关键：把 dataset_process 目录加入 sys.path
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

SKIP_MODULES = {
    "__init__",
    "guji_common",
    "dataset_base",
}

for module_file_path in CURRENT_DIR.glob("*.py"):
    module_name = module_file_path.stem

    if module_name in SKIP_MODULES:
        continue

    try:
        spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "dataset_info"):
            continue

        name2dataset_module[module.dataset_info["name"]] = module

        if "user_input" not in module_file_path.name:
            dataset_list.append(module.dataset_info)

    except Exception as e:
        dataset_load_errors[module_name] = str(e)

if __name__ == "__main__":
    print(name2dataset_module, "\n", dataset_list)
