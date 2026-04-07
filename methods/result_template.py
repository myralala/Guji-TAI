result = {   
    "origin_data": "any", # Raw data for each interpretation method, used for subsequent interaction. The specific meaning should be explained in the README of each interpretation method.
    "image": [{"image_name": "xxx", "image_path": "xxx"}, {"image_name": "xxx", "image_path": "xxx"}], # Name and path of each image
    "table": [{"table_name": "xxx1", "table_list": [{"a1": 1, "b1": 2}, {"a2": 3, "b2": 4}]}, 
                {"table_name": "xxx2", "table_list": [{"a2": 1, "b2": 2}, {"a2": 3, "b2": 4}]}] # Name and content of each table, where the content is organized as List[Dict]. Each Dict represents a row with its corresponding values. 
}


def normalize_method_result(raw_result, sample, method_name):
    raw_result = raw_result or {}
    sample = sample or {}

    normalized = dict(raw_result)

    normalized["task_type"] = sample.get("task_type", "") or ""
    normalized["sub_task"] = sample.get("sub_task", "") or ""
    normalized["target_unit"] = sample.get("explanation_target", {}) or {}
    normalized["origin_data"] = raw_result.get("origin_data", {}) or {}
    normalized["evidence_spans"] = raw_result.get("evidence_spans", []) or []
    normalized["key_layers"] = raw_result.get("key_layers", []) or []
    normalized["key_heads"] = raw_result.get("key_heads", []) or []
    normalized["key_neurons"] = raw_result.get("key_neurons", []) or []
    normalized["faithfulness"] = raw_result.get("faithfulness", {}) or {}
    normalized["consistency"] = raw_result.get("consistency", {}) or {}
    normalized["image"] = raw_result.get("image", []) or []
    normalized["table"] = raw_result.get("table", []) or []
    normalized["method_name"] = method_name

    return normalized
