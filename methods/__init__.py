import importlib
import pkgutil

import methods


method_name2diagnose_fun = {}
method_name2sub_module = {}
method_load_errors = {}

ALLOW_METHOD_PACKAGES = {
    "attention_weights",
    "attribution_integrated_grads",
    "causal_trace",
    "FiNE",
    "kn",
    "logit_lens",
}


for _, modname, ispkg in pkgutil.iter_modules(methods.__path__):
    if not ispkg or modname not in ALLOW_METHOD_PACKAGES:
        continue

    try:
        submodule = importlib.import_module(f"{methods.__name__}.{modname}")
        method_name2diagnose_fun[submodule.name] = submodule.diagnose
        method_name2sub_module[submodule.name] = submodule
    except Exception as e:
        method_load_errors[modname] = str(e)


support_methods = list(method_name2diagnose_fun.keys())

