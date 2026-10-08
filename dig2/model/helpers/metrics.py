def to_scalar_metric(value):
    """Return scalar metric value from plain float or (mean, std) tuple/list."""
    if isinstance(value, (tuple, list)) and len(value) > 0:
        return float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_main_accuracy(metrics_dict):
    """Find the most likely primary accuracy key from a metrics dictionary."""
    if not isinstance(metrics_dict, dict):
        return None, None

    preferred_suffixes = [
        "/tree_acc",
        "/planar_acc",
        "/sbm_acc",
        "/comm20_acc",
        "/acc",
        "/accuracy",
    ]
    keys = list(metrics_dict.keys())

    for suffix in preferred_suffixes:
        for key in keys:
            key_l = key.lower()
            if key_l.endswith(suffix):
                value = to_scalar_metric(metrics_dict[key])
                if value is not None:
                    return key, value

    for key in keys:
        key_l = key.lower()
        if "acc" in key_l and "frac" not in key_l:
            value = to_scalar_metric(metrics_dict[key])
            if value is not None:
                return key, value

    return None, None
