import torch.nn.functional as F


def resolve_pr_readout_layers(readout_layers, readout_stride, n_layers: int):
    """Resolve the effective progressive-readout depth list from raw config values (explicit
    list > stride > terminal-only), terminal depth always included. Single source of truth for
    this resolution -- used both by `ProgressiveReadoutCombineMixin._pr_parse_config` (the
    live, authoritative parse) and by `GraphDiscreteInterpolativeModel.__init__`'s `cond_slice_dim`
    sizing, which needs the same
    list's *length* before the host `GraphTransformer` is constructed
    -- keeping both call sites in exact agreement by construction rather than duplicated logic."""
    if readout_layers:
        layers = [int(x) for x in readout_layers]
    elif int(readout_stride or 0) > 0:
        layers = list(range(int(readout_stride), n_layers + 1, int(readout_stride)))
    else:
        layers = [n_layers]
    layers = sorted({l for l in layers if 1 <= l <= n_layers})
    if not layers or layers[-1] != n_layers:
        layers = sorted(set(layers) | {n_layers})
    return layers


def _yaml_bareword(value, default: str) -> str:
    """Config scalar meant as a bareword string. YAML 1.1 parses an unquoted `off` as False,
    which here means the literal word "off"."""
    if isinstance(value, bool):
        return "off" if value is False else str(value)
    return str(value)


def _progressive_readout_config(self):
    cfg = getattr(getattr(self.cfg, "general", None), "progressive_readout", None)
    readout_layers = getattr(cfg, "readout_layers", None) if cfg is not None else None
    return {
        "enabled": bool(getattr(cfg, "enabled", False)),
        "readout_layers": [int(x) for x in readout_layers] if readout_layers else None,
        "readout_stride": int(getattr(cfg, "readout_stride", 0) or 0),
        "head_mode": str(getattr(cfg, "head_mode", "shared")).strip().lower(),
        "gate_mode": _yaml_bareword(getattr(cfg, "gate_mode", "off"), "off").strip().lower(),
        "combine_space": str(getattr(cfg, "combine_space", "logit")).strip().lower(),
        "gate_hidden_dim": int(getattr(cfg, "gate_hidden_dim", 64) or 64),
        "gate_bias_init": str(getattr(cfg, "gate_bias_init", "uniform")).strip().lower(),
        "combine_y": bool(getattr(cfg, "combine_y", True)),
        "deep_supervision_weight": float(getattr(cfg, "deep_supervision_weight", 0.0) or 0.0),
        "deep_supervision_include_terminal": bool(
            getattr(cfg, "deep_supervision_include_terminal", False)
        ),
        "log_gate_weights": bool(getattr(cfg, "log_gate_weights", True)),
    }


def _compute_progressive_readout_deep_supervision_loss(self, layer_predictions, true_X, true_E, node_mask, domain_ids=None, graph_weight=None):
    """Per-depth CE against the same targets as the main loss, for non-terminal configured
    depths (terminal excluded by default -- it's already supervised by self.train_loss, so
    including it here would double-count its gradient contribution).

    Uses plain F.cross_entropy directly rather than the stateful TrainLossDiscrete metric,
    to avoid polluting its epoch-accumulated node/edge accuracy stats with extra-depth data.

    ``domain_ids`` ([B] long, optional): when given, each depth's logits are hard-masked to
    the sample's domain vocabulary V_d before the CE (same `_apply_domain_vocab_mask` as the
    main loss), so intermediate readout heads get zero gradient on out-of-domain classes.

    ``graph_weight`` ([B] float, optional): per-graph loss weights (the per-domain weights used
    by anchored fine-tuning). When given, each node/edge term carries its graph's weight in a
    weighted mean, mirroring the main loss's `(ce*w).sum()/w.sum()`.
    """
    cfg = self.progressive_readout_cfg
    weight = float(cfg.get("deep_supervision_weight", 0.0))
    zero = true_X.new_tensor(0.0)
    if weight <= 0 or not layer_predictions:
        return {"total": zero, "per_depth": {}}

    include_terminal = bool(cfg.get("deep_supervision_include_terminal", False))
    terminal_depth = max(layer_predictions.keys())
    target_X = true_X.argmax(dim=-1)
    target_E = true_E.argmax(dim=-1)
    nm = node_mask.squeeze(-1) if node_mask.dim() == 3 else node_mask
    nm = nm.bool()
    pair_mask = nm.unsqueeze(1) & nm.unsqueeze(2)

    per_depth = {}
    total = zero
    n_terms = 0
    for depth, ph in layer_predictions.items():
        if depth == terminal_depth and not include_terminal:
            continue
        logits_X, logits_E = ph.X, ph.E
        if domain_ids is not None:
            logits_X, logits_E = self._apply_domain_vocab_mask(logits_X, logits_E, domain_ids)
        px = logits_X[nm]
        pe = logits_E[pair_mask]
        ce = zero
        if graph_weight is None:
            if px.numel() > 0:
                ce = ce + F.cross_entropy(px, target_X[nm])
            if pe.numel() > 0:
                ce = ce + F.cross_entropy(pe, target_E[pair_mask])
        else:
            _gw = graph_weight.to(px.dtype)
            B_, n_ = nm.shape
            if px.numel() > 0:
                wx = _gw.view(-1, 1).expand(B_, n_)[nm]
                ce = ce + (F.cross_entropy(px, target_X[nm], reduction="none") * wx).sum() \
                    / wx.sum().clamp_min(1e-12)
            if pe.numel() > 0:
                we = _gw.view(-1, 1, 1).expand(B_, n_, n_)[pair_mask]
                ce = ce + (F.cross_entropy(pe, target_E[pair_mask], reduction="none") * we).sum() \
                    / we.sum().clamp_min(1e-12)
        per_depth[depth] = ce.detach()
        total = total + ce
        n_terms += 1

    if n_terms > 0:
        total = weight * (total / n_terms)
    return {"total": total, "per_depth": per_depth}
