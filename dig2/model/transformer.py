import math
import warnings

import torch
import torch.utils.checkpoint as _torch_ckpt
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from typing import Dict, Optional

from dig2 import utils
from dig2.interpolant import interpolant_utils
from dig2.model.layers import Xtoy, Etoy, masked_softmax
from dig2.model.progressive_readout import ProgressiveReadoutCombineMixin


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        cond_slice_dim: int = 0,
        cond_attn_cfg: Optional[Dict] = None,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(
            dx,
            de,
            dy,
            n_head,
            cond_slice_dim=cond_slice_dim,
            cond_attn_cfg=cond_attn_cfg,
            **kw,
        )

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y,
        node_mask: Tensor,
        cond_slice: Optional[Tensor] = None,
    ):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """
        newX, newE, new_y = self.self_attn(
            X,
            E,
            y,
            node_mask=node_mask,
            cond_slice=cond_slice,
        )

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, cond_slice_dim: int = 0, cond_attn_cfg: Optional[Dict] = None, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dropout_attn = Dropout(0.1)
        self.dropout_X = Dropout(0.1)
        self.dropout_E = Dropout(0.1)

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head
        self.cond_slice_dim = int(max(0, cond_slice_dim))
        _cfg = cond_attn_cfg or {}
        self.enable_qkv_residual = bool(_cfg.get("enable_qkv_residual", False)) and self.cond_slice_dim > 0
        self.enable_head_gate = bool(_cfg.get("enable_head_gate", False)) and self.cond_slice_dim > 0
        self.enable_xey_film = bool(_cfg.get("enable_xey_film", False)) and self.cond_slice_dim > 0

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        if self.enable_qkv_residual:
            self.cond_q = Linear(self.cond_slice_dim, dx)
            self.cond_k = Linear(self.cond_slice_dim, dx)
            self.cond_v = Linear(self.cond_slice_dim, dx)
            qkv_gain = float(_cfg.get("qkv_gain_init", 0.05))
            self.cond_q_gain = nn.Parameter(torch.tensor(qkv_gain, dtype=torch.float32))
            self.cond_k_gain = nn.Parameter(torch.tensor(qkv_gain, dtype=torch.float32))
            self.cond_v_gain = nn.Parameter(torch.tensor(qkv_gain, dtype=torch.float32))

        if self.enable_head_gate:
            self.cond_head_gate = Linear(self.cond_slice_dim, n_head)
            gate_gain = float(_cfg.get("gate_gain_init", 0.05))
            self.cond_gate_gain = nn.Parameter(torch.tensor(gate_gain, dtype=torch.float32))

        if self.enable_xey_film:
            self.cond_x_film = Linear(self.cond_slice_dim, 2 * dx)
            self.cond_e_film = Linear(self.cond_slice_dim, 2 * de)
            self.cond_y_film = Linear(self.cond_slice_dim, 2 * dy)
            film_gain = float(_cfg.get("xey_film_gain_init", 0.05))
            self.cond_film_gain = nn.Parameter(torch.tensor(film_gain, dtype=torch.float32))

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask, cond_slice: Optional[Tensor] = None):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X)
        K = self.k(X)
        V_raw = self.v(X)

        cond_gate = None
        if (
            cond_slice is not None
            and cond_slice.dim() == 2
            and cond_slice.size(0) == X.size(0)
            and cond_slice.size(-1) >= self.cond_slice_dim
            and self.cond_slice_dim > 0
        ):
            cond_gate = cond_slice[:, : self.cond_slice_dim].float()
            if self.enable_qkv_residual:
                Q = Q + torch.tanh(self.cond_q_gain) * self.cond_q(cond_gate).unsqueeze(1)
                K = K + torch.tanh(self.cond_k_gain) * self.cond_k(cond_gate).unsqueeze(1)
                V_raw = V_raw + torch.tanh(self.cond_v_gain) * self.cond_v(cond_gate).unsqueeze(1)

        Q = Q * x_mask  # (bs, n, dx)
        K = K * x_mask  # (bs, n, dx)
        interpolant_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        if (
            self.enable_head_gate
            and cond_slice is not None
            and cond_slice.dim() == 2
            and cond_slice.size(0) == X.size(0)
            and cond_slice.size(-1) >= self.cond_slice_dim
        ):
            cond_gate = cond_slice[:, : self.cond_slice_dim].float()
            head_gate = torch.tanh(self.cond_head_gate(cond_gate)).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
            Y = Y * (1.0 + torch.tanh(self.cond_gate_gain) * head_gate)

        interpolant_utils.assert_correctly_masked(
            Y, (e_mask1 * e_mask2).unsqueeze(-1)
        )

        # Incorporate edge features to the self attention scores.
        E1 = self.e_mul(E)                      # bs, n, n, dx
        E1.mul_(e_mask1).mul_(e_mask2)
        # In-place masking is autograd-safe here: the masks are grad-free views of node_mask.
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_add(E)                      # bs, n, n, dx
        E2.mul_(e_mask1).mul_(e_mask2)
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE)                      # bs, n, n, de
        newE.mul_(e_mask1).mul_(e_mask2)
        if self.enable_xey_film and cond_gate is not None:
            eg, eb = torch.chunk(self.cond_e_film(cond_gate), chunks=2, dim=-1)
            gain = torch.tanh(self.cond_film_gain)
            eg = torch.tanh(eg).unsqueeze(1).unsqueeze(1)
            eb = torch.tanh(eb).unsqueeze(1).unsqueeze(1)
            newE = newE * (1.0 + gain * eg) + gain * eb
            newE.mul_(e_mask1).mul_(e_mask2)
        interpolant_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = V_raw.mul_(x_mask)  # bs, n, dx  (V_raw is dead after this line)
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX)
        newX.mul_(x_mask)
        if self.enable_xey_film and cond_gate is not None:
            xg, xb = torch.chunk(self.cond_x_film(cond_gate), chunks=2, dim=-1)
            gain = torch.tanh(self.cond_film_gain)
            xg = torch.tanh(xg).unsqueeze(1)
            xb = torch.tanh(xb).unsqueeze(1)
            newX = newX * (1.0 + gain * xg) + gain * xb
            newX.mul_(x_mask)
        interpolant_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy
        if self.enable_xey_film and cond_gate is not None:
            yg, yb = torch.chunk(self.cond_y_film(cond_gate), chunks=2, dim=-1)
            gain = torch.tanh(self.cond_film_gain)
            new_y = new_y * (1.0 + gain * torch.tanh(yg)) + gain * torch.tanh(yb)

        return newX, newE, new_y


def _resolve_dim_ffy(hidden_dims, default: int = 2048) -> int:
    """Width of the y-channel FFN: `hidden_dims["dim_ffy"]`, or `default` (with a warning)
    when missing or non-positive."""
    try:
        val = int(hidden_dims.get("dim_ffy", 0) or 0)
    except (TypeError, ValueError):
        val = 0
    if val <= 0:
        warnings.warn(
            "model.hidden_dims.dim_ffy is missing or non-positive; falling back to "
            f"{default}. Set dim_ffy explicitly to size the y-channel FFN on purpose.",
            category=UserWarning, stacklevel=2,
        )
        return int(default)
    return val


class GraphTransformer(nn.Module, ProgressiveReadoutCombineMixin):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: nn.ReLU(),
        act_fn_out: nn.ReLU(),
        domain_time_modulation: bool = False,
        domain_cond_dim: int = 0,
        cond_slice_dim: int = 0,
        cond_attn_cfg: Optional[Dict] = None,
        domain_time_hidden_dim: int = 64,
        progressive_readout_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        # Activation checkpointing for the transformer stack. Set from the LightningModule
        # (cfg.model.grad_checkpointing).
        self.grad_checkpointing = False
        self.dim_ffy_effective = _resolve_dim_ffy(hidden_dims)
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]
        # Sinusoidal timestep embedding width, appended to y before mlp_in_y.
        self.time_dim = time_dim = 64
        self.domain_time_modulation = bool(domain_time_modulation)
        self.domain_cond_dim = int(max(0, domain_cond_dim))
        self.cond_slice_dim = int(max(0, cond_slice_dim))
        self.cond_attn_cfg = dict(cond_attn_cfg or {})
        self.domain_time_film = None
        if self.domain_time_modulation and self.domain_cond_dim > 0 and time_dim > 0:
            hidden_dim = int(max(16, domain_time_hidden_dim))
            self.domain_time_film = nn.Sequential(
                nn.Linear(self.domain_cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * time_dim),
            )
        else:
            self.domain_time_modulation = False

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"] + time_dim, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    cond_slice_dim=self.cond_slice_dim,
                    cond_attn_cfg=self.cond_attn_cfg,
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dim_ffy=self.dim_ffy_effective,
                )
                for i in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

        # Progressive multi-depth readout: a denoising prediction at several transformer depths,
        # combined per sample by a gate (ProgressiveReadoutCombineMixin).
        self._pr_parse_config(progressive_readout_cfg, n_layers)

        self.pr_extra_heads = None
        if self.progressive_readout_enabled and self.pr_head_mode == "per_depth":
            self.pr_extra_heads = nn.ModuleDict()
            for depth in self.pr_readout_layers:
                if depth == n_layers:
                    continue  # terminal always uses mlp_out_X/E/y
                hX = nn.Sequential(
                    nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
                    act_fn_out,
                    nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
                )
                hE = nn.Sequential(
                    nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
                    act_fn_out,
                    nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
                )
                # Warm-start: copy the (freshly-initialized) terminal head's weights so every
                # per-depth head starts identical to mlp_out_X/E/y at init, then diverges under
                # gradient as it specializes to that depth's hidden-state statistics.
                hX.load_state_dict(self.mlp_out_X.state_dict())
                hE.load_state_dict(self.mlp_out_E.state_dict())
                self.pr_extra_heads[f"X_{depth}"] = hX
                self.pr_extra_heads[f"E_{depth}"] = hE
                if self.pr_combine_y:
                    hy = nn.Sequential(
                        nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
                        act_fn_out,
                        nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
                    )
                    hy.load_state_dict(self.mlp_out_y.state_dict())
                    self.pr_extra_heads[f"y_{depth}"] = hy

        self._pr_init_gate()

    def forward(
        self, X, E, y, node_mask,
        cond_slice: Optional[Tensor] = None,
        domain_idx: Optional[Tensor] = None,
        return_layer_predictions: bool = False,
    ):
        bs, n = X.shape[0], X.shape[1]

        expected_x_in = int(self.mlp_in_X[0].in_features)
        if int(X.size(-1)) != expected_x_in:
            raise RuntimeError(
                "GraphTransformer input mismatch before mlp_in_X: "
                f"got X.size(-1)={int(X.size(-1))}, expected={expected_x_in}, "
                f"X.shape={tuple(X.shape)}, E.shape={tuple(E.shape)}, y.shape={tuple(y.shape)}"
            )

        diag_mask = torch.eye(n, device=E.device)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        if cond_slice is None and self.cond_slice_dim > 0 and y.size(-1) >= self.cond_slice_dim:
            cond_slice = y[:, : self.cond_slice_dim]
        time_emb = timestep_embedding(y[:, -1].unsqueeze(-1), 64)
        if (
            self.domain_time_modulation
            and self.domain_time_film is not None
            and y.size(-1) >= self.domain_cond_dim
        ):
            y_domain = y[:, : self.domain_cond_dim].float()
            film = self.domain_time_film(y_domain)
            gamma, beta = torch.chunk(film, chunks=2, dim=-1)
            gamma = torch.tanh(gamma)
            beta = torch.tanh(beta)
            # Small FiLM gain keeps modulation stable while still domain-aware.
            time_emb = time_emb * (1.0 + 0.25 * gamma) + 0.25 * beta
        y = torch.hstack([y, time_emb])

        after_in = utils.PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)
        ).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        if not self.progressive_readout_enabled:
            for layer in self.tf_layers:
                X, E, y = self._run_tf_layer(layer, X, E, y, node_mask, cond_slice)
            X = self.mlp_out_X(X) + X_to_out
            E = (self.mlp_out_E(E) + E_to_out) * diag_mask
            y = self.mlp_out_y(y) + y_to_out
            E = 1 / 2 * (E + torch.transpose(E, 1, 2))
            return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)

        return self._progressive_forward(
            X, E, y, node_mask, cond_slice, domain_idx, diag_mask,
            X_to_out, E_to_out, y_to_out,
            return_layer_predictions=return_layer_predictions,
        )

    def _run_tf_layer(self, layer, X, E, y, node_mask, cond_slice):
        """Run one transformer layer, optionally under activation checkpointing.

        Checkpointing keeps only the layer inputs and recomputes the interior during backward
        (~30% more compute for a large cut in the (B, N, N, dx) activation memory). torch's
        `preserve_rng_state=True` default replays the same RNG draws on recompute, so dropout
        is unchanged. Only active while training with grad enabled.
        """
        if not (self.grad_checkpointing and self.training and torch.is_grad_enabled()):
            return layer(X, E, y, node_mask, cond_slice=cond_slice)

        def _fn(_X, _E, _y):
            # node_mask / cond_slice are captured, not checkpoint inputs: they never require grad.
            return layer(_X, _E, _y, node_mask, cond_slice=cond_slice)

        return _torch_ckpt.checkpoint(_fn, X, E, y, use_reentrant=False)

    def _progressive_forward(
        self, X, E, y, node_mask, cond_slice, domain_idx, diag_mask,
        X_to_out, E_to_out, y_to_out, return_layer_predictions: bool,
    ):
        """Runs the transformer stack once, taking a masked (X, E, y) prediction at every
        depth in ``self.pr_readout_layers`` (terminal depth always included), then combines
        them via ``self._pr_combine``."""
        bs = X.shape[0]
        readout_set = set(self.pr_readout_layers)
        depth_preds = {}  # depth(int) -> masked utils.PlaceHolder
        for li, layer in enumerate(self.tf_layers):
            depth = li + 1
            X, E, y = self._run_tf_layer(layer, X, E, y, node_mask, cond_slice)
            if depth not in readout_set:
                continue
            if depth == self.n_layers or self.pr_head_mode == "shared":
                Xh, Eh, yh = self.mlp_out_X(X), self.mlp_out_E(E), self.mlp_out_y(y)
            else:
                Xh = self.pr_extra_heads[f"X_{depth}"](X)
                Eh = self.pr_extra_heads[f"E_{depth}"](E)
                yh = self.pr_extra_heads[f"y_{depth}"](y) if self.pr_combine_y else self.mlp_out_y(y)
            Xh = Xh + X_to_out
            Eh = (Eh + E_to_out) * diag_mask
            yh = yh + y_to_out
            Eh = 0.5 * (Eh + Eh.transpose(1, 2))
            depth_preds[depth] = utils.PlaceHolder(X=Xh, E=Eh, y=yh).mask(node_mask)

        combined, weights = self._pr_combine(
            depth_preds, cond_slice, domain_idx, bs, node_mask,
        )
        if return_layer_predictions:
            return combined, depth_preds, weights
        return combined



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding
