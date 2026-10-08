import numpy as np


def _augment_accuracy_summaries(self, metrics_dict):
    """Add macro/min/max summaries across available domain accuracy metrics."""
    if not isinstance(metrics_dict, dict):
        return

    acc_items = []
    for key, value in metrics_dict.items():
        key_l = str(key).lower()
        if not key_l.endswith("_acc"):
            continue
        if key_l.startswith("summary/"):
            continue
        scalar = self._to_scalar_metric(value)
        if scalar is None:
            continue
        acc_items.append((str(key), float(scalar)))

    if not acc_items:
        return

    vals = [v for _, v in acc_items]
    metrics_dict["summary/acc_domain_macro"] = float(np.mean(vals))
    metrics_dict["summary/acc_domain_min"] = float(np.min(vals))
    metrics_dict["summary/acc_domain_max"] = float(np.max(vals))
    metrics_dict["summary/acc_domain_count"] = int(len(vals))

    for key, value in acc_items:
        safe_key = key.replace("/", "_")
        metrics_dict[f"summary/acc_domain_breakout/{safe_key}"] = float(value)

