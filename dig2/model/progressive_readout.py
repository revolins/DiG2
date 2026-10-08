"""Progressive multi-depth readout for GraphTransformer.

The backbone exposes a denoising prediction at several transformer depths instead of only the
terminal one, and combines them with a per-sample gate over ``self.pr_readout_layers``.

The host class calls, from its own ``__init__`` (after its ``mlp_out_X/E/y`` exist):
    1. ``self._pr_parse_config(progressive_readout_cfg, n_layers)``
    2. builds ``self.pr_extra_heads`` for ``head_mode="per_depth"``
    3. ``self._pr_init_gate()``
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from dig2 import utils
from dig2.model.helpers.readout import resolve_pr_readout_layers


class ProgressiveReadoutCombineMixin:
    """Progressive-readout config parsing, gating and per-depth combination."""

    def _pr_parse_config(self, progressive_readout_cfg: Optional[Dict], n_layers: int):
        """Parses `general.progressive_readout.*` into `self.pr_*` attributes and
        `self.pr_readout_layers` (terminal depth always included)."""
        _pr = dict(progressive_readout_cfg or {})
        self.progressive_readout_enabled = bool(_pr.get("enabled", False))
        self.pr_head_mode = str(_pr.get("head_mode", "shared")).strip().lower()
        self.pr_gate_mode = str(_pr.get("gate_mode", "off")).strip().lower()
        if self.pr_gate_mode not in ("off", "mean", "domain_gated"):
            raise ValueError(f"Unknown progressive_readout.gate_mode: {self.pr_gate_mode!r}")
        combine_space = str(_pr.get("combine_space", "logit")).strip().lower()
        if combine_space != "logit":
            raise ValueError(f"progressive_readout.combine_space must be 'logit', got {combine_space!r}")
        self.pr_combine_y = bool(_pr.get("combine_y", True))
        self.pr_gate_hidden_dim = int(_pr.get("gate_hidden_dim", 64) or 64)
        self.pr_gate_bias_init = str(_pr.get("gate_bias_init", "uniform")).strip().lower()
        self.pr_readout_layers = resolve_pr_readout_layers(
            _pr.get("readout_layers", None), _pr.get("readout_stride", 0), n_layers
        )

    def _pr_init_gate(self):
        """Builds the learned gate MLP (`self.pr_gate`) for gate_mode='domain_gated'. Call
        after `_pr_parse_config` and after `self.cond_slice_dim` is set."""
        self.pr_gate = None
        if not self.progressive_readout_enabled:
            return

        if self.pr_gate_mode == "domain_gated":
            n_depths = len(self.pr_readout_layers)
            self.pr_gate = nn.Sequential(
                nn.Linear(self.cond_slice_dim, self.pr_gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.pr_gate_hidden_dim, n_depths),
            )
            # Zero-init final layer: softmax(0) == uniform 1/K for every sample at step 0,
            # i.e. domain_gated STARTS as literal gate_mode="mean" and is free to deviate
            # per-domain as training proceeds.
            nn.init.zeros_(self.pr_gate[-1].weight)
            nn.init.zeros_(self.pr_gate[-1].bias)
            if self.pr_gate_bias_init == "terminal_heavy":
                with torch.no_grad():
                    bias = torch.full((n_depths,), -4.0)
                    bias[-1] = 4.0
                    self.pr_gate[-1].bias.copy_(bias)

    def _pr_combine(
        self, depth_preds: Dict[int, "utils.PlaceHolder"], cond_slice, domain_idx,
        bs: int, node_mask,
    ):
        """Combines per-depth predictions into one PlaceHolder: a per-sample convex
        combination of the per-depth logits, with weights from the gate (`domain_gated`),
        uniform (`mean`), or terminal-only (`off`)."""
        depths = self.pr_readout_layers
        terminal = depth_preds[self.n_layers]

        if self.pr_gate_mode == "off":
            # Terminal-only: numerically identical to the non-progressive path.
            n = len(depths)
            w = torch.zeros(bs, n, device=terminal.X.device, dtype=terminal.X.dtype)
            w[:, -1] = 1.0
            return terminal, w

        if self.pr_gate_mode == "mean":
            n = len(depths)
            w = torch.full((bs, n), 1.0 / n, device=terminal.X.device, dtype=terminal.X.dtype)
        else:
            gate_in = cond_slice if cond_slice is not None else terminal.y.new_zeros(bs, self.cond_slice_dim)
            w = self.pr_gate(gate_in.float()).softmax(dim=-1)

        combined_X = combined_E = combined_y = 0.0
        for k, depth in enumerate(depths):
            ph = depth_preds[depth]
            wk = w[:, k]
            combined_X = combined_X + wk.view(bs, 1, 1) * ph.X
            combined_E = combined_E + wk.view(bs, 1, 1, 1) * ph.E
            if self.pr_combine_y:
                combined_y = combined_y + wk.view(bs, 1) * ph.y
        if not self.pr_combine_y:
            combined_y = terminal.y

        # Each depth's PlaceHolder is already masked; a convex per-sample combination (weights
        # sum to 1) of already-masked tensors is still correctly masked -- no re-mask needed.
        return utils.PlaceHolder(X=combined_X, E=combined_E, y=combined_y), w
