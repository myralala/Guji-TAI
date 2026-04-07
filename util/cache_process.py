import os
import sys
from pathlib import Path
import threading
import json

lock_cache = threading.Lock()
cache_path = Path(__file__).parent/"tmp"/"cache.json"
cache_data = json.loads(cache_path.read_text()) if cache_path.exists() else dict()


def _has_missing_image_path(value):
    if isinstance(value, dict):
        image_path = value.get("image_path")
        if image_path and not Path(image_path).exists():
            return True
        return any(_has_missing_image_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_missing_image_path(item) for item in value)
    return False


def is_cache_entry_usable(data):
    if data is None:
        return False
    return not _has_missing_image_path(data)


def read_cache(cache_path=cache_path, cache_data=cache_data, key="key"):
    return cache_data.get(key, None), key in cache_data
    
def write_cache(key, data, cache_path=cache_path, cache_data=cache_data):
    with lock_cache:
        cache_data[key] = data
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=1))


def delete_cache(key, cache_path=cache_path, cache_data=cache_data):
    with lock_cache:
        if key not in cache_data:
            return
        cache_data.pop(key, None)
        cache_path.write_text(json.dumps(cache_data, ensure_ascii=False, indent=1))

def generate_cache_key(sample, method_name, model_name):
    key = method_name+model_name+sample["prompt"]
    return key
