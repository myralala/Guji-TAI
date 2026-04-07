def compression_retention(original_metric, compressed_metric):
    original = float(original_metric)
    if original == 0:
        return 0.0
    return round(max(min(float(compressed_metric) / original, 1.0), 0.0), 6)
