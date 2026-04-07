def explanation_stability(reference_items, perturbed_items):
    left = set(reference_items or [])
    right = set(perturbed_items or [])
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)
