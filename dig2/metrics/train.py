
import torch
import torch.nn as nn
import torch.nn.functional as F

from dig2.metrics.abstract import (CrossEntropyMetric, KLDMetric,)
from dig2.ddp_utils import all_reduce_sum_




class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train, kld=False, node_loss_enabled: bool = True,
                 per_graph_ce_enabled: bool = False):
        super().__init__()
        self.lambda_train = lambda_train
        self.node_loss_enabled = bool(node_loss_enabled)
        # Opt-in per-graph CE (default off, zero cost when off). Every CE here is otherwise
        # a batch-pooled scalar: the `reduction="none"` vectors are flat over the batch and
        # are reduced on the next line, so no per-graph or per-domain value survives. A
        # per-DOMAIN theta head needs a per-domain gradient, hence this.
        #
        # Stores per-graph CE NUMERATOR and DENOMINATOR rather than a mean, so that
        # sum(num)/sum(den) reproduces the pooled scalar exactly, and so a caller can form
        # a correct per-domain mean by summing both over that domain's graphs. Taking a
        # mean here would make per-domain aggregation a mean-of-means, which is wrong
        # whenever graphs differ in node/edge count -- which they always do.
        self.per_graph_ce_enabled = bool(per_graph_ce_enabled)
        self.last_per_graph_ce = None
        if not kld:
            self.node_loss = CrossEntropyMetric() if self.node_loss_enabled else None
            self.edge_loss = CrossEntropyMetric()
        else:
            self.node_loss = KLDMetric() if self.node_loss_enabled else None
            self.edge_loss = KLDMetric()
        self.y_loss = CrossEntropyMetric()
        # Set (only when log=True) each forward() call so the caller -- a LightningModule,
        # which this plain nn.Module can't self.log() from directly -- can retrieve and
        # self.log_dict() it, so metrics.csv carries the primary CE/NLL trajectory.
        self.last_to_log = None
        # Set by forward() under DDP: the per-term local/global denominators and the
        # resulting scale factors. None on a single GPU. Purely diagnostic.
        self.last_denoms = None

    @staticmethod
    def _scatter_num_den(bs, gidx, ce_none, weight):
        """-> (ce_sum[bs], w_sum[bs]) grouped by graph.

        `weight=None` means unweighted, i.e. w=1, so w_sum is the per-graph count and
        ce_sum/w_sum is the plain per-graph mean CE. Kept as two tensors rather than a
        ratio so callers can aggregate across graphs (or across a domain's graphs)
        correctly: sum(ce_sum)/sum(w_sum), never a mean of means.
        """
        dev, dt = ce_none.device, ce_none.dtype
        w = torch.ones_like(ce_none) if weight is None else weight.to(dt)
        ce_sum = torch.zeros(bs, device=dev, dtype=dt).index_add_(0, gidx, ce_none * w)
        w_sum = torch.zeros(bs, device=dev, dtype=dt).index_add_(0, gidx, w)
        return ce_sum, w_sum

    def _accumulate_per_graph_ce(
        self, bs, gidx_x, flat_pred_X, flat_true_X, flat_node_weight,
        gidx_e, flat_pred_E, flat_true_E, flat_edge_weight,
    ):
        """Per-graph CE for X and E, detached (diagnostic/aggregation use, not a loss path).

        Recomputes `reduction="none"` CE rather than reusing the caller's, because the
        unweighted branches above never build one -- they go straight through the
        torchmetrics accumulator, which reduces with `reduction="sum"` into a single
        scalar. The extra cross_entropy is why this is opt-in.
        """
        out = {}
        with torch.no_grad():
            if flat_pred_X is not None and flat_pred_X.numel() > 0 and gidx_x is not None:
                ce_x = F.cross_entropy(
                    flat_pred_X, torch.argmax(flat_true_X, dim=-1), reduction="none"
                )
                out["x_ce_sum"], out["x_w_sum"] = self._scatter_num_den(
                    bs, gidx_x, ce_x, flat_node_weight
                )
            else:
                out["x_ce_sum"] = out["x_w_sum"] = torch.zeros(bs, device=flat_pred_E.device)

            if flat_pred_E.numel() > 0:
                ce_e = F.cross_entropy(
                    flat_pred_E, torch.argmax(flat_true_E, dim=-1), reduction="none"
                )
                out["e_ce_sum"], out["e_w_sum"] = self._scatter_num_den(
                    bs, gidx_e, ce_e, flat_edge_weight
                )
            else:
                out["e_ce_sum"] = out["e_w_sum"] = torch.zeros(bs, device=flat_pred_E.device)
        return out

    def forward(
        self,
        masked_pred_X,
        masked_pred_E,
        pred_y,
        true_X,
        true_E,
        true_y,
        log: bool,
        edge_weight=None,
        node_weight=None,
        ddp_world_size: int = 1,
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean.
        edge_weight : optional (bs, n, n) per-pair weights for the edge CE (Layer 2
            per-domain present-edge balancing, and/or per-domain loss-weight
            rebalancing). When given, loss_E is a weighted mean over non-padded
            pairs; the edge metric is still updated unweighted so logged E_CE
            stays comparable across configs.
        node_weight : optional (bs, n) per-node weights for the node CE (per-domain
            loss-weight rebalancing). When given, loss_X is a weighted mean over
            non-padded nodes; the node metric is still updated unweighted so logged
            X_CE stays comparable across configs."""
        # Per-graph grouping indices, captured BEFORE the flattening reshapes below.
        # Every masked flat tensor here comes from a (bs, n, ...) -> (bs*n*..., ...) view
        # followed by a boolean row mask, so each surviving row's graph is recoverable as
        # arange(bs).repeat_interleave(stride)[mask]. That is the exact inverse of how
        # `node_weight`/`edge_weight` (built per graph) are mapped INTO this flat space a
        # few lines down, so the two are guaranteed consistent.
        _pg = self.per_graph_ce_enabled
        if _pg:
            _bs, _n = masked_pred_E.size(0), masked_pred_E.size(1)
            _dev = masked_pred_E.device
            _gidx_e = torch.arange(_bs, device=_dev).repeat_interleave(_n * _n)
            _gidx_x = torch.arange(_bs, device=_dev).repeat_interleave(_n)

        if edge_weight is not None:
            edge_weight = torch.reshape(edge_weight, (-1,))  # (bs * n * n,)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)
        # print("dimension of masked_pred_E", masked_pred_E.shape)

        # Remove masked rows
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_E = true_E[mask_E, :]  # torch.Size([857058, 2])
        flat_pred_E = masked_pred_E[mask_E, :] # torch.Size([857058, 2])
        flat_edge_weight = edge_weight[mask_E] if edge_weight is not None else None

        zero = torch.zeros([], device=masked_pred_E.device)
        if self.node_loss_enabled:
            true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
            masked_pred_X = torch.reshape(
                masked_pred_X, (-1, masked_pred_X.size(-1))
            )  # (bs * n, dx)
            mask_X = (true_X != 0.0).any(dim=-1)
            flat_true_X = true_X[mask_X, :]
            flat_pred_X = masked_pred_X[mask_X, :]
            flat_node_weight = (
                torch.reshape(node_weight, (-1,))[mask_X]
                if node_weight is not None
                else None
            )
            if (
                flat_node_weight is not None
                and true_X.numel() > 0
                and flat_pred_X.numel() > 0
            ):
                # Per-domain loss-weight rebalancing (weighted mean CE), mirroring
                # the edge_weight branch below.
                ce_none_X = F.cross_entropy(
                    flat_pred_X, torch.argmax(flat_true_X, dim=-1), reduction="none"
                )
                loss_X = (ce_none_X * flat_node_weight).sum() / flat_node_weight.sum().clamp(min=1e-8)
                _den_X = flat_node_weight.sum()
                # keep the unweighted metric updated so logged X_CE remains comparable
                self.node_loss(flat_pred_X, flat_true_X)
            else:
                loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else zero
                _den_X = (
                    torch.as_tensor(float(flat_pred_X.shape[0]), device=zero.device)
                    if true_X.numel() > 0
                    else zero
                )
        else:
            loss_X = zero
            _den_X = zero

        # print("flat_pred_E shape", flat_pred_E.shape, "flat_true_E", flat_true_E.shape)
        # improve the weights of positive samples. 20 not good -> 
        # 
        # num_pos = flat_true_E[:, 1].sum()
        # num_neg = flat_true_E.size(0) - num_pos
        # # print("positive vs negative", num_pos/num_neg) 0.09
        # w_pos = (num_neg / (num_pos + 1e-8))
        # weights = torch.tensor([1, w_pos], dtype=torch.float32).to(flat_pred_E.device)
        if flat_edge_weight is not None and true_E.numel() > 0 and flat_pred_E.numel() > 0:
            # Layer 2: per-domain present-edge balanced edge loss (weighted mean CE).
            ce_none = F.cross_entropy(
                flat_pred_E, torch.argmax(flat_true_E, dim=-1), reduction="none"
            )
            loss_E = (ce_none * flat_edge_weight).sum() / flat_edge_weight.sum().clamp(min=1e-8)
            _den_E = flat_edge_weight.sum()
            # keep the unweighted metric updated so logged E_CE remains comparable
            self.edge_loss(flat_pred_E, flat_true_E)
        else:
            loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else zero
            _den_E = (
                torch.as_tensor(float(flat_pred_E.shape[0]), device=zero.device)
                if true_E.numel() > 0
                else zero
            )

        # --- opt-in per-graph CE (numerator / denominator, never a mean) ----------------
        if _pg:
            self.last_per_graph_ce = self._accumulate_per_graph_ce(
                bs=_bs,
                gidx_x=(_gidx_x[mask_X] if (self.node_loss_enabled and true_X.numel() > 0) else None),
                flat_pred_X=(flat_pred_X if self.node_loss_enabled else None),
                flat_true_X=(flat_true_X if self.node_loss_enabled else None),
                flat_node_weight=(flat_node_weight if self.node_loss_enabled else None),
                gidx_e=_gidx_e[mask_E],
                flat_pred_E=flat_pred_E,
                flat_true_E=flat_true_E,
                flat_edge_weight=flat_edge_weight,
            )
        else:
            self.last_per_graph_ce = None

        # print(loss_E) # KL divergence loss
        loss_y = self.y_loss(pred_y, true_y) if pred_y.numel() > 0 else zero
        _den_y = (
            torch.as_tensor(float(pred_y.size(0)), device=zero.device)
            if pred_y.numel() > 0
            else zero
        )
        # if mu is not None and logvar is not None:
        #     kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # else:
        #     kld_loss = 0.0
        # print("loss_E", loss_E, "kld_loss", kld_loss, "mu", mu, "logvar", logvar)
        # ------------------------------------------------------------- DDP loss scaling
        # Each of loss_X / loss_E / loss_y is a MEAN over its own denominator (nodes,
        # edge slots, graphs), so a single scalar factor cannot fix all three -- they are
        # scaled independently.
        #
        # A 1-GPU run computes  S/N  over the whole global batch. Under DDP each rank
        # holds a disjoint shard and computes  S_r/N_r , and DDP averages the ranks'
        # gradients, i.e. optimizes  (1/W) * sum_r L_r . Setting
        #     L_r = W * (S_r / N_global)   ==   (S_r / N_r) * (N_r / N_global) * W
        # makes  sum_r L_r == W * (S_global / N_global) , so the DDP average reproduces
        # the exact global-batch mean. That is what makes an N-GPU step numerically the
        # same optimization step as a 1-GPU step over the union batch.
        #
        # The denominators are captured from whichever branch actually ran above, rather
        # than recomputed here, so this cannot drift from the real reduction. (The
        # weighted/unweighted branch choice is batch-content dependent via
        # `(per_graph_x == 1.0).all()`, but that short-circuit only fires when every
        # weight IS 1.0, in which case sum(w) == count -- so the two branches agree
        # numerically and mixing them across ranks is consistent.)
        #
        # At ddp_world_size == 1 none of this runs: no collective, no extra tensor, no
        # multiply. That is the single-GPU byte-for-byte bar.
        _global_present = None
        if ddp_world_size > 1:
            _dens = torch.stack([
                _den_X.reshape(()).to(zero.dtype),
                _den_E.reshape(()).to(zero.dtype),
                _den_y.reshape(()).to(zero.dtype),
            ]).detach()
            _global = all_reduce_sum_(_dens.clone())
            _scale = (_dens / _global.clamp(min=1e-12)) * float(ddp_world_size)
            loss_X = loss_X * _scale[0]
            loss_E = loss_E * _scale[1]
            loss_y = loss_y * _scale[2]
            self.last_denoms = {
                "local": [float(v) for v in _dens],
                "global": [float(v) for v in _global],
                "scale": [float(v) for v in _scale],
            }
            _global_present = _global > 0
        else:
            self.last_denoms = None

        # `.compute()` on these metrics is a COLLECTIVE (dist_reduce_fx="sum"). The
        # guards below are rank-local predicates, so a rank that skipped one while others
        # entered it would hang the group. Under DDP the gate is made global. In practice
        # every rank holds >= 1 graph (enforced by min_graphs_per_batch >= world_size plus
        # the empty-shard raise), so these are already unanimous -- this is belt-and-braces
        # against a whole class of hang for three extra floats in a reduce we already do.
        _gate_X = self.node_loss_enabled and true_X.numel() > 0
        _gate_E = true_E.numel() > 0
        _gate_y = true_y.numel() > 0
        if _global_present is not None:
            _gate_X = self.node_loss_enabled and bool(_global_present[0])
            _gate_E = bool(_global_present[1])
            _gate_y = bool(_global_present[2])

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": (
                    self.node_loss.compute()
                    if _gate_X
                    else -1
                ),
                "train_loss/E_CE": (
                    self.edge_loss.compute() if _gate_E else -1
                ),
                "train_loss/y_CE": self.y_loss.compute() if _gate_y else -1,
            }
            # if wandb.run:
            #     wandb.log(to_log, commit=True)
            self.last_to_log = to_log
        else:
            self.last_to_log = None
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y


    def reset(self):
        metrics_to_reset = [self.edge_loss, self.y_loss]
        if self.node_loss_enabled and self.node_loss is not None:
            metrics_to_reset.insert(0, self.node_loss)
        for metric in metrics_to_reset:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute()
            if (self.node_loss_enabled and self.node_loss is not None and self.node_loss.total_samples > 0)
            else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = (
            self.y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
        }
        # if wandb.run:
        #     wandb.log(to_log, commit=False)

        return to_log
