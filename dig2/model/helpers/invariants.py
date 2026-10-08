def _assert_model_input_dim_invariants(self, where: str = "runtime"):
    """Fail fast if dataset_infos input dims disagree with instantiated model layers."""
    if not hasattr(self, "model"):
        return

    expected = {
        "X": int(self.input_dims.get("X", -1)),
        "E": int(self.input_dims.get("E", -1)),
        "y": int(self.input_dims.get("y", -1)),
    }

    model_in = {}
    # Only check available module paths; do not swallow mismatch errors.
    if hasattr(self.model, "mlp_in_X") and len(self.model.mlp_in_X) > 0:
        model_in["X"] = int(self.model.mlp_in_X[0].in_features)
    if hasattr(self.model, "mlp_in_E") and len(self.model.mlp_in_E) > 0:
        model_in["E"] = int(self.model.mlp_in_E[0].in_features)
    if hasattr(self.model, "mlp_in_y") and len(self.model.mlp_in_y) > 0:
        # The transformer appends its timestep embedding to y internally.
        model_in["y"] = int(self.model.mlp_in_y[0].in_features) - int(self.model.time_dim)

    mismatches = {}
    for key, v in model_in.items():
        if expected.get(key, v) != v:
            mismatches[key] = {
                "dataset_infos": expected.get(key),
                "model_in_features": v,
            }

    if mismatches:
        raise RuntimeError(
            "Model/input-dim invariant failed: "
            f"where={where}, mismatches={mismatches}, "
            f"dataset={getattr(self.cfg.dataset, 'name', None)}, "
            f"model_type={getattr(self.cfg.model, 'type', None)}, "
            f"conditional={getattr(self.cfg.general, 'conditional', None)}, "
            f"conditional_domain={getattr(self.cfg.general, 'conditional_domain', None)}"
        )

