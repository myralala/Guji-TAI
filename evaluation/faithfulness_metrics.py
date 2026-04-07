def comprehensiveness_score(original_score, ablated_score):
    return round(max(float(original_score) - float(ablated_score), 0.0), 6)


def sufficiency_score(original_score, kept_score):
    original = float(original_score)
    if original == 0:
        return 0.0
    return round(max(min(float(kept_score) / original, 1.0), 0.0), 6)


def perturbation_drop(original_score, perturbed_score):
    return round(max(float(original_score) - float(perturbed_score), 0.0), 6)
