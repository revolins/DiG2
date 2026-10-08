"""Row-subsetted CE and teacher-KL for the edge-pathway shared-weight fine-tune (Routes B/C).

Both functions select a SUBSET OF GRAPHS (rows) of a dense batch and reduce over that subset's valid
nodes / valid pairs only. That is what the anchor and the Route-C protected gradient need and what
neither existing loss offers:

  * `TrainLossDiscrete` (metrics/train_metrics.py) is a stateful torchmetrics Metric -- calling it a
    second time per step pollutes its epoch accumulators -- and it flattens by filtering on nonzero
    one-hot targets, which is a per-class weighted-CE shape, not a per-row subset.
  * `domain_helpers.domain_prior_kl` has the right masked-mean-over-the-dense-grid idiom but no row
    subsetting (it averages over the whole batch).

So these follow the deep-supervision idiom (readout_helpers.py:136-172): build the
node mask / pair mask, index with it, reduce with a clamped denominator.

X and E are combined as `loss_X + lambda_E * loss_E`, matching TrainLossDiscrete's
`loss_X + lambda_train[0] * loss_E` (train_metrics.py:340) with lambda_train=[5,0], so these terms sit
on the same scale as the main loss and a weight of ~1 means "comparable to the task loss".

EMPTY-SUBSET SAFETY: with no selected row (or no valid pair among them) each returns an exact zero that
is still connected to the graph (`logits.sum() * 0`), so `.backward()` works and the logged key exists
on every step. Under DDP every rank must emit the same keys every step (the invariant documented at
graph_discrete_interpolative_model.py:2465-2470), and a Python float would break that.
"""

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def _masks(node_mask: torch.Tensor, row_mask: torch.Tensor):
    """(node_mask[B,n], row_mask[B] bool) -> (node sel [B,n] bool, pair sel [B,n,n] bool)."""
    nm = node_mask.bool()
    rm = row_mask.view(-1, 1)
    node_sel = nm & rm
    pair_sel = (nm.unsqueeze(1) & nm.unsqueeze(2)) & rm.unsqueeze(-1)
    # Drop self-pairs: the model never predicts a self-edge and the dense diagonal is structurally
    # absent, so including it would dilute the mean with a constant.
    eye = torch.eye(nm.size(1), dtype=torch.bool, device=nm.device).unsqueeze(0)
    pair_sel = pair_sel & ~eye
    return node_sel, pair_sel


def _zero_like(*tensors: torch.Tensor) -> torch.Tensor:
    """Graph-connected exact zero (so backward works on an empty subset)."""
    out = None
    for t in tensors:
        z = t.sum() * 0.0
        out = z if out is None else out + z
    return out


def masked_row_ce(pred_X: torch.Tensor, pred_E: torch.Tensor,
                  true_X: torch.Tensor, true_E: torch.Tensor,
                  node_mask: torch.Tensor, row_mask: torch.Tensor,
                  lambda_E: float = 5.0) -> Tuple[torch.Tensor, int, int]:
    """Plain CE over the selected rows. Returns (loss, n_nodes_used, n_pairs_used).

    `true_X`/`true_E` are one-hot dense targets (as training_step holds them); converted with argmax,
    which is exact for one-hot and is what the deep-supervision helper does.
    """
    node_sel, pair_sel = _masks(node_mask, row_mask)
    n_nodes = int(node_sel.sum())
    n_pairs = int(pair_sel.sum())
    if n_nodes == 0 and n_pairs == 0:
        return _zero_like(pred_X, pred_E), 0, 0
    loss = _zero_like(pred_X, pred_E)
    if n_nodes > 0:
        tx = true_X[node_sel]
        loss = loss + F.cross_entropy(pred_X[node_sel], tx.argmax(dim=-1), reduction="mean")
    if n_pairs > 0:
        te = true_E[pair_sel]
        loss = loss + float(lambda_E) * F.cross_entropy(
            pred_E[pair_sel], te.argmax(dim=-1), reduction="mean")
    return loss, n_nodes, n_pairs


def _kl_cells(logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) per selected cell, NaN-safe under vocabulary masking.

    Both inputs are the V_d-MASKED logits the main loss uses (`train_domain_vocab_mask: true` sets
    out-of-domain classes to -inf). For those classes p_t == 0 and (logp_t - logp_s) is
    (-inf) - (-inf) = NaN, so the product must be zeroed explicitly. `F.kl_div(..., log_target=True)`
    produces the same NaN and is therefore NOT used. Masked logits are used deliberately: the
    unmasked variant would place anchor gradient on classes the domain does not have.
    """
    logp_s = F.log_softmax(logits_s, dim=-1)
    logp_t = F.log_softmax(logits_t, dim=-1)
    p_t = logp_t.exp()
    term = p_t * (logp_t - logp_s)
    return term.masked_fill(p_t == 0, 0.0).sum(dim=-1)


def masked_row_kl(logits_s_X: torch.Tensor, logits_s_E: torch.Tensor,
                  logits_t_X: torch.Tensor, logits_t_E: torch.Tensor,
                  node_mask: torch.Tensor, row_mask: torch.Tensor,
                  lambda_E: float = 5.0,
                  channels: Sequence[str] = ("X", "E")) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(teacher || student) over the selected rows. Returns (combined, kl_X, kl_E).

    Reverse KL (teacher as the target distribution) is the standard distillation direction: it is
    mode-covering with respect to the teacher, i.e. it penalizes the student for dropping mass the
    frozen base puts somewhere.
    """
    node_sel, pair_sel = _masks(node_mask, row_mask)
    zero = _zero_like(logits_s_X, logits_s_E)
    kl_x, kl_e = zero, zero
    if "X" in channels and int(node_sel.sum()) > 0:
        kl_x = _kl_cells(logits_s_X[node_sel], logits_t_X[node_sel]).mean()
    if "E" in channels and int(pair_sel.sum()) > 0:
        kl_e = _kl_cells(logits_s_E[pair_sel], logits_t_E[pair_sel]).mean()
    return kl_x + float(lambda_E) * kl_e, kl_x, kl_e


def masked_row_argmax_disagreement(logits_s_E: torch.Tensor, logits_t_E: torch.Tensor,
                                   node_mask: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    """Fraction of the selected rows' valid pairs where the edge argmax differs teacher-vs-student.

    The cheapest available proxy for "will generation actually change": KL can move while every
    decision stays put, and the sampler consumes decisions.
    """
    _, pair_sel = _masks(node_mask, row_mask)
    n = int(pair_sel.sum())
    if n == 0:
        return logits_s_E.sum() * 0.0
    a = logits_s_E[pair_sel].argmax(dim=-1)
    b = logits_t_E[pair_sel].argmax(dim=-1)
    return (a != b).float().mean()


def row_mask_for_domains(domain_idx: torch.Tensor, wanted_idx: Iterable[int]) -> torch.Tensor:
    """[B] bool mask for 'this graph belongs to one of `wanted_idx`'.

    MUST be fed `data.domain_idx` (stamped per sample in datasets/unified_dataset.py:1079-1080),
    which always carries the TRUE domain -- never `data.y`, which CFG dropout mutates in place
    (graph_discrete_interpolative_model.py:2479-2510), and never `_domain_indices_from_y`, which maps nulled
    rows to the CFG null index. A CFG-nulled protected row is still a protected row: the null branch
    shares these edge weights and generation uses it at every guidance step.
    """
    did = domain_idx.view(-1)
    out = torch.zeros_like(did, dtype=torch.bool)
    for d in wanted_idx:
        out |= did == int(d)
    return out
