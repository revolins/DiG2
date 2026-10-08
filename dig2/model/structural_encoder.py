from typing import List, Tuple, Union

import torch
import torch.nn as nn

from dig2.model.extra_features import RRWPFeatures


def _make_mlp(in_dim: int, mid_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    """Build a depth-layer MLP: in_dim → [mid_dim]×(depth-1) → out_dim with SiLU activations.

    All intermediate layers use mid_dim; depth controls layer count only.
    depth=1 → single Linear (no activation); depth=2 → Linear→SiLU→Linear.
    """
    assert depth >= 1, f"MLP depth must be >= 1, got {depth}"
    if depth == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: list = [nn.Linear(in_dim, mid_dim), nn.SiLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(mid_dim, mid_dim), nn.SiLU()]
    layers.append(nn.Linear(mid_dim, out_dim))
    return nn.Sequential(*layers)


class DomainStructuralEncoder(nn.Module):
    """Message-passing encoder for graph-level structural embeddings.

    Processes dense (B, N, *) graph tensors and produces a single graph-level
    vector per sample. Messages are a function of both neighbor node features
    AND the projected edge-type vector (full E feature, concatenated with RRWP
    edge features), with post-LN residual updates for stability. RRWP node features
    are concatenated to the node input, and log(1 + graph size) to the pooled vector.

    At training time the encoder is called on noisy X_t / E_t (differentiable)
    and on clean X / E inside torch.no_grad() for EMA prototype updates.

    Args:
        node_dim:   input node feature dimension
        edge_dim:   input edge feature dimension
        hidden_dim: int → uniform hidden width; list of length n_layers+1 → per-layer
                    boundary widths [H₀, H₁, …, H_{n_layers}] where H_l is the
                    output dim of layer l-1 and input dim of layer l.
        output_dim: graph-level embedding dimension
        n_layers:   number of message-passing rounds
        mp_depth:   int → uniform MLP depth; list of length n_layers → per-layer depth.
                    Depth counts linear layers (depth=2 = Linear→SiLU→Linear, default).
        rrwp_steps: number of RRWP steps (k), > 0.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: Union[int, List[int]],
        output_dim: int,
        n_layers: int = 3,
        mp_depth: Union[int, List[int]] = 2,
        rrwp_steps: int = 0,
    ):
        super().__init__()

        if rrwp_steps <= 0:
            raise ValueError("rrwp_steps must be > 0")
        self.rrwp_steps = rrwp_steps
        self._rrwp = RRWPFeatures(k=rrwp_steps, normalize=True)

        if isinstance(hidden_dim, int):
            hdims = [hidden_dim] * (n_layers + 1)
        else:
            hdims = list(hidden_dim)
            if len(hdims) != n_layers + 1:
                raise ValueError(
                    f"hidden_dim list length must equal n_layers+1={n_layers+1}, "
                    f"got {len(hdims)}"
                )

        if isinstance(mp_depth, int):
            depths = [mp_depth] * n_layers
        else:
            depths = list(mp_depth)
            if len(depths) != n_layers:
                raise ValueError(
                    f"mp_depth list length must equal n_layers={n_layers}, got {len(depths)}"
                )

        self.node_in = nn.Linear(node_dim + rrwp_steps, hdims[0])

        # Edge features are re-projected from the raw input E (+ RRWP) at every layer.
        self.edge_projs = nn.ModuleList([
            nn.Linear(edge_dim + rrwp_steps, hdims[l]) for l in range(n_layers)
        ])
        self.msg_fns = nn.ModuleList([
            _make_mlp(2 * hdims[l], hdims[l], hdims[l], depths[l])
            for l in range(n_layers)
        ])
        self.update_fns = nn.ModuleList([
            _make_mlp(2 * hdims[l], hdims[l], hdims[l + 1], depths[l])
            for l in range(n_layers)
        ])
        self.skip_projs = nn.ModuleList([
            nn.Identity() if hdims[l] == hdims[l + 1]
            else nn.Linear(hdims[l], hdims[l + 1])
            for l in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hdims[l + 1]) for l in range(n_layers)])

        self.graph_out = nn.Sequential(
            nn.Linear(hdims[n_layers] + 1, hdims[n_layers]),
            nn.SiLU(),
            nn.Linear(hdims[n_layers], output_dim),
        )

        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_rrwp(
        self, E: torch.Tensor, node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RRWP node/edge features from the edge-presence weights of E.

        Computed under no_grad and detached: RRWPFeatures' internal `1/row_sum`
        degree-normalization produces a 0*inf -> NaN gradient for any exact-zero
        row (which padding/invalid-node rows always are once masked below). RRWP
        is a fixed structural input feature; gradient still flows into
        node_in/edge_projs from how they process it.
        """
        with torch.no_grad():
            if E.size(-1) > 1:
                w = E[..., 1:].sum(-1)          # (B, N, N)
            else:
                w = 1.0 - E[..., 0]              # (B, N, N)
            valid = node_mask.float()
            w = w * valid.unsqueeze(1) * valid.unsqueeze(2)   # zero out invalid-node entries
            rrwp_edge = self._rrwp(w.detach(), k=self.rrwp_steps)       # (B, N, N, k)
            n = rrwp_edge.shape[1]
            diag_index = torch.arange(n, device=rrwp_edge.device)
            rrwp_node = rrwp_edge[:, diag_index, diag_index, :]  # (B, N, k)
        return rrwp_edge, rrwp_node

    def forward(self, X: torch.Tensor, E: torch.Tensor, node_mask: torch.Tensor):
        """
        Args:
            X:         (B, N, node_dim)   — node features (noisy or clean)
            E:         (B, N, N, edge_dim) — edge features
            node_mask: (B, N) bool or float — valid node indicator
        Returns:
            (B, output_dim) graph-level structural embedding.
        """
        vmask = node_mask.float().unsqueeze(-1)   # (B, N, 1) — broadcasts to any H

        rrwp_edge, rrwp_node = self._compute_rrwp(E, node_mask)

        X_in = torch.cat([X, rrwp_node], dim=-1)
        h = self.node_in(X_in) * vmask               # (B, N, H₀)

        # Aggregation weight: edge presence = 1 - p(no-edge class)
        if E.size(-1) > 1:
            edge_w = E[..., 1:].sum(-1, keepdim=True)   # (B, N, N, 1)
        else:
            edge_w = 1.0 - E[..., :1]

        edge_valid = (
            node_mask.float().unsqueeze(-1) *
            node_mask.float().unsqueeze(-2)
        ).unsqueeze(-1)                                   # (B, N, N, 1)
        agg_w = edge_w * edge_valid                       # (B, N, N, 1) — combined weight

        n = h.shape[1]
        E_in = torch.cat([E, rrwp_edge], dim=-1)

        for l, (mf, uf, sp, nm) in enumerate(
            zip(self.msg_fns, self.update_fns, self.skip_projs, self.norms)
        ):
            e_feat = self.edge_projs[l](E_in)  # (B, N, N, H_l)

            msg_in = torch.cat(
                [h.unsqueeze(1).expand(-1, n, -1, -1), e_feat], dim=-1
            )                                                           # (B, N, N, 2H_l)
            per_edge_msg = mf(msg_in)                                  # (B, N, N, H_l)

            neighbor_weight = agg_w.sum(dim=2)                                  # (B, N, 1)
            agg = (per_edge_msg * agg_w).sum(dim=2) / neighbor_weight.clamp(min=1.0)  # (B, N, H_l)

            h = nm(sp(h) + uf(torch.cat([h, agg], dim=-1))) * vmask  # (B, N, H_{l+1})

        valid_count = vmask.sum(dim=1).clamp(min=1.0)   # (B, 1)
        graph_emb = (h * vmask).sum(dim=1) / valid_count  # (B, H_{n_layers})
        graph_emb = torch.cat([graph_emb, torch.log1p(valid_count)], dim=-1)
        return self.graph_out(graph_emb)  # (B, output_dim)
