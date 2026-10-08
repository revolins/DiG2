import time
import os
import contextlib
from collections.abc import Mapping

import numpy as np
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hashlib
from dig2.ddp_utils import (ddp_topology, all_reduce_min_, all_reduce_sum_, barrier as ddp_barrier, disable_metric_sync,)
from dig2.model.helpers.metrics import to_scalar_metric, extract_main_accuracy
from dig2.model.helpers import invariants as invariant_helpers
from dig2.model.helpers import domain as domain_helpers
from dig2.model.helpers import eval_summary as eval_summary_helpers
from dig2.model.helpers import readout as readout_helpers

from dig2.procedures.anchor_ft import AnchorFTMixin
from dig2.model.transformer import GraphTransformer

from dig2.metrics.train import TrainLossDiscrete
from dig2 import utils
from dig2.interpolant.noise_distribution import NoiseDistribution
from dig2.interpolant.time_distorter import TimeDistorter
from dig2.interpolant.utils import p_xt_g_x1
from dig2.interpolant import interpolant_utils
from omegaconf import DictConfig, OmegaConf
from dig2.debug import hotpath_asserts_enabled


class GraphDiscreteInterpolativeModel(AnchorFTMixin, pl.LightningModule):
    def _to_dense_maybe_tokenized(self, data):
        """Build dense graph tensors from either legacy one-hot fields or token fields.

        Legacy path: delegates to utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        Token path: reconstructs one-hot transiently using global vocab carried by datamodule.
        """
        if not (hasattr(data, "x_token") and hasattr(data, "edge_token")):
            return utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        datamodule = getattr(getattr(self, "trainer", None), "datamodule", None)
        global_vocab = getattr(datamodule, "global_vocab", None)
        if not isinstance(global_vocab, Mapping):
            # fallback for safety if token fields exist but vocab unavailable
            return utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        Sx = int(global_vocab.get("Sx", 0))
        Se = int(global_vocab.get("Se", 0))
        if Sx <= 0 or Se <= 0:
            return utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        x_tok = data.x_token.long()
        e_tok = data.edge_token.long()
        x = F.one_hot(x_tok, num_classes=Sx).float()
        edge_attr = F.one_hot(e_tok, num_classes=Se).float()
        return utils.to_dense(x, data.edge_index, edge_attr, data.batch)

    # === Core lifecycle and runtime ===
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        visualization_tools_by_domain=None,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg

        # DDP topology, resolved ONCE. Every DDP branch in this class keys off
        # `self._ddp_world_size > 1`; at 1 those branches are not taken, so the
        # single-GPU path runs the pre-DDP code verbatim (no collective, no extra
        # tensor, no changed arithmetic, no extra RNG draw).
        #
        # Read here rather than from self.trainer/torch.distributed because the model is
        # built before the Trainer exists. Note Lightning's `self.global_rank` is also
        # unavailable until then, which is why this is a separate attribute.
        self._ddp_rank, self._ddp_world_size = ddp_topology(cfg)
        self._ddp_on = self._ddp_world_size > 1

        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32

        # Helper method to get the correct base path for outputs
        self.get_output_base_path = lambda: getattr(cfg, 'hier_dir', cfg.general.name)
        self.conditional = bool(
            getattr(cfg.general, "conditional", False)
            or getattr(cfg.general, "conditional_domain", False)
        )
        self._domain_y_dim = None
        self._rrwp_y_dim = None
        self._cond_slice_layout_locked = False
        self.test_labels = test_labels

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dict(dataset_infos.input_dims)  # mutable copy — may be expanded below
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos

        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0].item())

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.visualization_tools_by_domain = {
            str(k).strip().lower(): v
            for k, v in (visualization_tools_by_domain or {}).items()
        }
        self.extra_features = extra_features
        self.domain_features = domain_features
        # Runtime caches used by domain-aware helpers.
        # Initialize these early because some config builders call
        # _get_domain_to_idx() during __init__.
        self._cached_domain_to_idx = None
        self._domain_spec_locked = False
        self._domain_spec_source = "unresolved"
        self._locked_y_width = None
        self._cached_domain_hist = {}
        self._cached_domain_one_hot_rows = {}

        # Optional learned embedding over one-hot domain labels (ablateable).
        self.use_domain_embedding = bool(getattr(cfg.general, "domain_embedding", False))
        self.domain_emb_dim = int(getattr(cfg.general, "domain_emb_dim", 0) or 0)
        self.cfg_null_mode = str(getattr(cfg.general, "cfg_null_mode", "zeros")).lower()
        if self.cfg_null_mode not in {"zeros", "uniform", "learned"}:
            self.cfg_null_mode = "zeros"
        self.domain_emb_table = None
        self.domain_null_idx = None
        self._domain_emb_warned = False

        if self.use_domain_embedding:
            n_domains = max(len(self._get_domain_to_idx()), 1)
            emb_dim = self.domain_emb_dim if self.domain_emb_dim > 0 else n_domains
            self.domain_emb_dim = int(emb_dim)
            _hidden = self.domain_emb_dim * 4
            self._domain_emb_n_inputs = n_domains
            n_embeddings = n_domains
            if self.cfg_null_mode == "learned":
                self.domain_null_idx = n_domains
                n_embeddings += 1
            self.domain_emb_table = nn.Embedding(n_embeddings, self.domain_emb_dim)
            torch.nn.init.orthogonal_(self.domain_emb_table.weight)
            # Residual MLP: output = x + down(act(up(x))).
            # Starts as identity (zero-init on down-projection bias), learns deviations.
            # This preserves the orthogonal init of domain_emb_table during training
            # without requiring an explicit regularization term.
            self._domain_emb_mlp_up   = nn.Linear(self.domain_emb_dim, _hidden)
            self._domain_emb_mlp_act  = nn.SiLU()
            self._domain_emb_mlp_down = nn.Linear(_hidden, self.domain_emb_dim)
            torch.nn.init.orthogonal_(self._domain_emb_mlp_up.weight)
            torch.nn.init.zeros_(self._domain_emb_mlp_up.bias)
            torch.nn.init.orthogonal_(self._domain_emb_mlp_down.weight)
            torch.nn.init.zeros_(self._domain_emb_mlp_down.bias)

        # Optional structural encoder: GNN that processes graph state and produces
        # a per-sample embedding used alongside the domain label for conditioning.
        # Combines an EMA prototype (from clean graphs) with a current-state encoding
        # (from noisy X_t), giving the model explicit structural statistics per domain.
        _struct_cfg = getattr(cfg.general, "structural_encoder", None)
        self.structural_encoder_enabled = bool(getattr(_struct_cfg, "enabled", False))
        self.structural_emb_dim = int(getattr(_struct_cfg, "output_dim", 32) or 32)
        self.proto_ema_decay = float(getattr(_struct_cfg, "proto_ema_decay", 0.99))
        # Loss-free transform applied to each per-domain mean before it enters the prototype
        # EMA (center_l2): subtract a running mean of the observed per-domain means, then
        # L2-normalize. No gradient path and no learnable parameters.
        self.proto_center_momentum = float(getattr(_struct_cfg, "proto_center_momentum", 0.99))
        self.structural_encoder = None
        self.struct_fusion = None
        if self.structural_encoder_enabled:
            from dig2.model.structural_encoder import DomainStructuralEncoder

            def _cfg_to_int_or_list(val, default):
                if val is None:
                    return default
                try:
                    return int(val)
                except (TypeError, ValueError):
                    return [int(v) for v in val]

            _se_hidden   = _cfg_to_int_or_list(getattr(_struct_cfg, "hidden_dim", None), 64)
            _se_layers   = int(getattr(_struct_cfg, "n_layers", 3) or 3)
            _se_mp_depth = _cfg_to_int_or_list(getattr(_struct_cfg, "mp_depth", None), 2)
            _se_rrwp_steps = int(
                getattr(_struct_cfg, "rrwp_steps", None)
                or getattr(cfg.model, "rrwp_steps", 8)
                or 8
            )
            self.structural_encoder = DomainStructuralEncoder(
                node_dim=dataset_infos.output_dims["X"],
                edge_dim=dataset_infos.output_dims["E"],
                hidden_dim=_se_hidden,
                output_dim=self.structural_emb_dim,
                n_layers=_se_layers,
                mp_depth=_se_mp_depth,
                rrwp_steps=_se_rrwp_steps,
            )

            self.struct_fusion = nn.Sequential(
                nn.Linear(2 * self.structural_emb_dim, self.structural_emb_dim),
                nn.SiLU(),
            )
            n_dom_proto = max(len(self._get_domain_to_idx()), 1)
            self.register_buffer(
                "domain_prototypes",
                torch.zeros(n_dom_proto, self.structural_emb_dim),
            )
            # Registered ONLY under a centering proto_transform, mirroring
            # structural_null_prototype above: an unconditional buffer would add
            # state_dict keys and break strict loads of every existing checkpoint.
            # proto_transform='none'/'l2' therefore stay checkpoint-compatible.
            self.register_buffer(
                "proto_running_mean", torch.zeros(self.structural_emb_dim)
            )
            self.register_buffer(
                "proto_center_initialized", torch.zeros((), dtype=torch.bool)
            )
            # input_dims['y'] must include the structural embedding for every model
            # type (compute_extra_data's contract check assumes this unconditionally,
            # not just for 'gt') — applied once here rather than inside the gt-only
            # branch below.
            self.input_dims["y"] += self.structural_emb_dim

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)

        # Precompute per-domain valid class masks from training marginals.
        # Used by train_domain_vocab_mask to zero gradients for out-of-vocab classes.
        _dld = getattr(self.noise_dist, "domain_limit_dist", {})
        if len(_dld) > 0:
            _sample_ph = next(iter(_dld.values()))
            _Sx = _sample_ph.X.shape[0]
            _Se = _sample_ph.E.shape[0]
            _x_valid = torch.zeros(len(_dld), _Sx, dtype=torch.bool)
            _e_valid = torch.zeros(len(_dld), _Se, dtype=torch.bool)
            for _d, _ph in _dld.items():
                _x_valid[int(_d)] = _ph.X > 0
                _e_valid[int(_d)] = _ph.E > 0
            self.register_buffer("_train_domain_x_valid", _x_valid)
            self.register_buffer("_train_domain_e_valid", _e_valid)
        else:
            self._train_domain_x_valid = None
            self._train_domain_e_valid = None

        self.prior_mode = str(getattr(cfg.model, "prior_mode", "default")).strip().lower()
        if self.prior_mode not in {"default", "domain_specific"}:
            self.prior_mode = "default"
        if self.prior_mode == "domain_specific":
            if len(getattr(self.noise_dist, "domain_limit_dist", {})) <= 0:
                raise ValueError(
                    "prior_mode=domain_specific requested, but no valid domain priors were built."
                )
            prior_warnings = getattr(self.noise_dist, "domain_prior_warnings", []) or []
            if prior_warnings:
                raise ValueError(f"Domain prior warnings: {prior_warnings}")
        self.limit_dist = self.noise_dist.get_limit_dist(prior_mode=self.prior_mode)

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.progressive_readout_cfg = self._progressive_readout_config()
        self.progressive_readout_enabled = bool(self.progressive_readout_cfg["enabled"])
        self.progressive_readout_deep_sup_weight = float(
            self.progressive_readout_cfg["deep_supervision_weight"]
        )
        if (not self.progressive_readout_enabled) and self.progressive_readout_deep_sup_weight > 0:
            raise ValueError(
                "progressive_readout.deep_supervision_weight > 0 requires "
                "progressive_readout.enabled=True."
            )

        # How loudly to report the two silent passthrough branches in
        # `_apply_domain_vocab_mask` (domain_helpers.py). Resolved once here so the free
        # function can read a plain attribute via getattr -- it is called from ~20 sites
        # including hot loops, some with stub `self` objects that have no `.cfg`.
        _vm_cfg = getattr(cfg.general, "vocab_mask", None)
        self._vocab_mask_passthrough_warn = str(
            getattr(_vm_cfg, "passthrough_warn", "off") or "off"
        ).strip().lower()
        if self._vocab_mask_passthrough_warn not in {"off", "once", "always"}:
            self._vocab_mask_passthrough_warn = "off"
        self._vocab_mask_passthrough_warned = set()

        gt_domain_cond_dim = 0
        if bool(getattr(cfg.general, "conditional_domain", False)):
            if self.use_domain_embedding and self.domain_emb_dim > 0:
                gt_domain_cond_dim = self.domain_emb_dim
            else:
                gt_domain_cond_dim = int(max(len(self._get_domain_to_idx()), 0))
        # Structural encoder expands the domain conditioning slice so per-layer
        # FiLM sees both the learned domain embedding and a structural signal.
        # (input_dims['y'] is already expanded above, unconditionally.)
        if self.structural_encoder_enabled:
            gt_domain_cond_dim += self.structural_emb_dim
        cond_cfg_raw = getattr(cfg.model, "cond_attn", None)
        if isinstance(cond_cfg_raw, DictConfig):
            cond_cfg_raw = OmegaConf.to_container(cond_cfg_raw, resolve=True) or {}
        elif not isinstance(cond_cfg_raw, Mapping):
            cond_cfg_raw = {}
        cond_attn_cfg = dict(cond_cfg_raw)
        cond_slice_dim = int(gt_domain_cond_dim)
        if (
            self.progressive_readout_enabled
            and self.progressive_readout_cfg["gate_mode"] in ("domain_gated", "domain_argmax")
            and cond_slice_dim <= 0
        ):
            raise ValueError(
                f"progressive_readout.gate_mode={self.progressive_readout_cfg['gate_mode']} "
                "requires a non-empty cond_slice (cond_slice_dim > 0); enable "
                "general.conditional_domain or use gate_mode=mean/off instead."
            )
        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            domain_time_modulation=bool(
                self.use_domain_embedding
                and bool(getattr(cfg.general, "conditional_domain", False))
            ),
            domain_cond_dim=gt_domain_cond_dim,
            cond_slice_dim=cond_slice_dim,
            cond_attn_cfg=cond_attn_cfg,
            domain_time_hidden_dim=int(getattr(cfg.general, "domain_time_hidden_dim", 64) or 64),
            progressive_readout_cfg=self.progressive_readout_cfg,
        )
        # Hard invariant: dataset-computed input dims must match actual model input layers.
        self._assert_model_input_dim_invariants(where="init")

        # Only "cfg" is captured (not the full auto-captured arg list): dataset_infos,
        # visualization_tools, extra_features, domain_features, etc. are custom runtime
        # objects, not yaml-serializable, and a real logger (e.g. CSVLogger) will try to
        # dump self.hparams to hparams.yaml at the start of every run.
        self.save_hyperparameters("cfg")
        # logging
        self.start_epoch_time = None
        self.val_start_time = None
        self.test_start_time = None
        self.sampling_start_time = None
        self.train_epoch_times = []
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0


        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
        )



        # Cache of per-domain limit distributions stacked into (n_domains, S)
        # tensors for the vectorized domain-specific prior in apply_noise.
        self._domain_limit_stack_cache = None
        self._freeze_prototypes = False

        # train.anchor_ft (anchored fine-tuning): fine-tune the SHARED edge pathway toward the
        # target domains while a KL anchor holds the protected domains' predictions in place.
        self._anchor_ft_enabled = False
        self._anchor_ft_trainable_names = set()
        self._anchor_ft_params = []
        self._anchor_ft_param_names = []
        self._anchor_ft_teacher = None
        _et_cfg = getattr(cfg.train, "anchor_ft", None)
        if _et_cfg is not None and bool(getattr(_et_cfg, "enabled", False)):
            self._anchor_ft_setup(_et_cfg, cfg)
    # === Wrappers (delegating helpers) ===

    def _get_domain_limit_stack(self, width: int):
        """Stack per-domain limit distributions into (width, Sx)/(width, Se) tensors
        for the vectorized domain-specific prior. Returns None (→ caller falls back
        to the per-domain loop) unless every domain id in [0, width) has a
        registered domain prior, so prior_strict error semantics are preserved."""
        cached = self._domain_limit_stack_cache
        if cached is not None and cached["width"] == int(width):
            return cached
        domain_limits = getattr(self.noise_dist, "domain_limit_dist", {})
        if not all(int(d) in domain_limits for d in range(int(width))):
            raise ValueError(
                f"domain-specific prior missing for some domain id in [0, {int(width)}): "
                f"registered={sorted(int(d) for d in domain_limits)}"
            )
        limits = [
            self._get_limit_dist_checked(d, context="apply_noise")
            for d in range(int(width))
        ]
        cached = {
            "width": int(width),
            "X": torch.stack([l.X.float() for l in limits]).to(self.device),
            "E": torch.stack([l.E.float() for l in limits]).to(self.device),
        }
        self._domain_limit_stack_cache = cached
        return cached

    def log(self, name, value, *args, **kwargs):
        """Default `sync_dist=True` under DDP for every metric logged from this module.

        Every self.log here is a rank-LOCAL scalar; without syncing, metrics.csv shows
        rank 0's shard only, which under stride sharding is a 1/N subsample. Overriding
        here rather than editing ~60 call sites also means a self.log added later is
        correct by default.

        Callers that pass `sync_dist` or `rank_zero_only` explicitly are left alone --
        that is how the variable-key per-domain diagnostics opt out.

        Inert at world size 1: the branch is not taken and this forwards byte-identically.

        CAVEAT: Lightning's sync reduce is "mean", so an epoch aggregate becomes a
        mean-of-means across ranks -- exact only when ranks carry equal weight. Fine for
        diagnostics; the LOSS is all-reduced exactly and separately (see train_metrics).
        """
        if (
            self._ddp_on
            and "sync_dist" not in kwargs
            and not kwargs.get("rank_zero_only", False)
        ):
            kwargs["sync_dist"] = True
        return super().log(name, value, *args, **kwargs)

    def log_dict(self, dictionary, *args, **kwargs):
        """Same defaulting as `log` -- see its docstring."""
        if (
            self._ddp_on
            and "sync_dist" not in kwargs
            and not kwargs.get("rank_zero_only", False)
        ):
            kwargs["sync_dist"] = True
        return super().log_dict(dictionary, *args, **kwargs)

    def training_step(self, data, i):
        _batch_idx = int(i)
        if self._ddp_world_size == 1:
            if data.edge_index.numel() == 0:
                self.print("Found a batch with no edges. Skipping.")
                return
        else:
            # A bare per-rank `return` skips backward() on THIS rank only and deadlocks
            # every other rank inside the gradient all-reduce. The skip decision therefore
            # has to be unanimous, which needs its own tiny collective here: the whole
            # forward is skipped, so it cannot be folded into the denominator reduce
            # inside TrainLossDiscrete. One float, ~10us.
            #
            # See also the invariant already documented in on_before_optimizer_step:
            # "backward() is always called unconditionally on every rank (never add a
            # pre-backward, per-rank-only skip elsewhere)".
            _ok = torch.tensor(
                [1.0 if data.edge_index.numel() > 0 else 0.0], device=self.device
            )
            all_reduce_min_(_ok)
            if float(_ok.item()) == 0.0:
                self.print(
                    "Found a batch with no edges on at least one rank. Skipping on ALL ranks."
                )
                return

        y_cond_for_weight = getattr(data, "y", None)
        if self.conditional:
            cfg_dropout_prob = self._cfg_dropout_prob()
            # One Bernoulli per GRAPH. `_build_uncond_y` is row-wise, so building it for
            # the whole batch and selecting rows is equivalent to building it per row.
            _y = data.y
            _u = self._build_uncond_y(_y)
            _keep = torch.rand(_y.size(0), device=_y.device) < cfg_dropout_prob
            data.y = torch.where(_keep.view(-1, *([1] * (_y.dim() - 1))), _u, _y)
            self.log(
                "train/cfg_dropout_applied",
                float(_keep.float().mean()),
                on_step=False, on_epoch=True, prog_bar=False, logger=True,
            )

        dense_data, node_mask = self._to_dense_maybe_tokenized(data)

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask, prior_y=y_cond_for_weight)
        extra_data = self.compute_extra_data(noisy_data)
        layer_predictions = None
        gate_weights = None
        _want_progressive_readout_aux = (
            self.progressive_readout_enabled and self.progressive_readout_deep_sup_weight > 0
        )
        # train.anchor_ft: frozen-base prediction on the SAME noisy inputs, taken BEFORE the student
        # forward and inside fork_rng so the student still draws identical dropout masks.
        _et_step = _batch_idx
        _et_teacher_pred = None
        if self._anchor_ft_want_teacher(_et_step):
            _et_teacher_pred = self._anchor_ft_teacher_forward(noisy_data, extra_data, node_mask)

        if _want_progressive_readout_aux:
            pred, layer_predictions, gate_weights = self.forward(
                noisy_data,
                extra_data,
                node_mask,
                return_layer_predictions=True,
            )
        else:
            pred = self.forward(noisy_data, extra_data, node_mask)
        if (
            getattr(self.cfg.general, "train_domain_vocab_mask", False)
            and hasattr(data, "domain_idx")
        ):
            # Hard-mask out-of-domain (V_d = semantic) class logits before CE. Support is the
            # semantic declared set (falls back to marginal buffers in non-semantic mode).
            _pred_X_loss, _pred_E_loss = self._apply_domain_vocab_mask(
                pred.X, pred.E, data.domain_idx
            )
        else:
            _pred_X_loss, _pred_E_loss = pred.X, pred.E

        _node_weight, _edge_weight = self._domain_loss_weight_tensors(y_cond_for_weight, node_mask)
        loss = self.train_loss(
            masked_pred_X=_pred_X_loss,
            masked_pred_E=_pred_E_loss,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
            edge_weight=_edge_weight,
            node_weight=_node_weight,
            ddp_world_size=self._ddp_world_size,
        )
        if self.train_loss.last_to_log is not None:
            # Restores the primary CE trajectory to metrics.csv -- previously computed
            # here every log_every_steps and silently discarded (train_metrics.py's
            # TrainLossDiscrete.forward built this dict for a since-removed wandb.log
            # call). on_epoch=True matches every other train/* metric's convention in
            # this file, averaging the per-log-step snapshots across the epoch.
            self.log_dict(
                self.train_loss.last_to_log,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        if layer_predictions is not None:
            _pr_domain_ids = (
                data.domain_idx
                if (
                    getattr(self.cfg.general, "train_domain_vocab_mask", False)
                    and hasattr(data, "domain_idx")
                )
                else None
            )
            # Under anchored fine-tuning the protected domains carry domain_loss_weight 0. The deep-
            # supervision loss must get the same per-domain weight as the main CE, otherwise the
            # protected rows would keep training the shared trunk through this term.
            _ds_gw = None
            if self._anchor_ft_enabled:
                _ds_gw = self._domain_loss_multiplier(y_cond_for_weight, channel="e")
            pr_loss = self._compute_progressive_readout_deep_supervision_loss(
                layer_predictions=layer_predictions,
                true_X=X,
                true_E=E,
                node_mask=node_mask,
                domain_ids=_pr_domain_ids,
                graph_weight=_ds_gw,
            )
            loss = loss + pr_loss["total"]
            self.log(
                "train/progressive_readout_deep_sup_total",
                pr_loss["total"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            for depth, v in pr_loss["per_depth"].items():
                self.log(
                    f"train/progressive_readout_ce_l{depth}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )
            if gate_weights is not None and self.progressive_readout_cfg.get("log_gate_weights", True):
                gw = gate_weights.detach()
                mean_w = gw.mean(dim=0)
                for i, depth in enumerate(self.model.pr_readout_layers):
                    self.log(
                        f"train/progressive_readout_gate_l{depth}",
                        mean_w[i],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                    )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            # was `i % ...`, which the readout-loop rebinding made permanently False
            log=_batch_idx % self.log_every_steps == 0,
        )

        # EMA prototype update: encode the clean (true) graphs from this batch and
        # blend into the per-domain prototype bank. Uses y_cond_for_weight (pre-CFG-dropout
        # domain labels) so the update is always conditioned on the true domain.
        # Skipped under anchored fine-tuning (freeze_prototypes): this block is also the writer of
        # the SHARED proto_running_mean (via _apply_proto_transform below), so leaving it on would
        # move every domain's prototype centering.
        if (
            self.structural_encoder_enabled
            and self.structural_encoder is not None
            and y_cond_for_weight is not None
            and not getattr(self, "_freeze_prototypes", False)
        ):
            with torch.no_grad():
                clean_emb = self.structural_encoder(
                    X.float(), E.float(), node_mask
                )  # (B, structural_emb_dim)
                domain_idx = y_cond_for_weight.argmax(dim=-1)
                ema = self.proto_ema_decay
                # Vectorized segment mean over domains (replaces a Python loop
                # with per-domain .unique() host syncs): prototypes for domains
                # absent from this batch are left untouched.
                n_dom = int(self.domain_prototypes.size(0))
                counts = torch.zeros(n_dom, device=clean_emb.device, dtype=clean_emb.dtype)
                counts.index_add_(0, domain_idx, torch.ones_like(domain_idx, dtype=clean_emb.dtype))
                sums = torch.zeros(
                    n_dom, clean_emb.size(-1), device=clean_emb.device, dtype=clean_emb.dtype
                )
                sums.index_add_(0, domain_idx, clean_emb)
                if self._ddp_on:
                    # ONE fused collective, n_dom*(D+1) floats. See module note: this MUST
                    # happen before the mean and before _apply_proto_transform.
                    _nd, _D = int(counts.numel()), int(sums.size(-1))
                    _packed = torch.cat([sums.reshape(-1), counts.reshape(-1)])
                    all_reduce_sum_(_packed)
                    sums = _packed[: _nd * _D].view(_nd, _D)
                    counts = _packed[_nd * _D :].view(_nd)
                mean_emb = sums / counts.clamp(min=1.0).unsqueeze(-1)
                present_rows = counts > 0
                mean_emb = self._apply_proto_transform(mean_emb, present_rows)
                updated = ema * self.domain_prototypes + (1.0 - ema) * mean_emb
                present = present_rows.unsqueeze(-1)
                self.domain_prototypes.copy_(
                    torch.where(present, updated, self.domain_prototypes)
                )

        if self._anchor_ft_enabled:
            loss = self._anchor_ft_terms(
                loss=loss, data=data, node_mask=node_mask, X=X, E=E,
                student_X=_pred_X_loss, student_E=_pred_E_loss,
                teacher_pred=_et_teacher_pred, step_idx=_et_step,
            )

        return {"loss": loss}

    def configure_optimizers(self):
        _params = self.parameters()
        _wd = self.cfg.train.weight_decay
        if self._anchor_ft_enabled:
            # Only the parameters being trained reach the optimizer, so nothing else can move even
            # through weight decay. Asserted, not assumed: the trainable set must equal the set
            # recorded at __init__. Gradient clipping then applies to the edge set alone.
            _named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
            _names = {n for n, _ in _named}
            _expect = self._anchor_ft_trainable_names
            if not _named:
                raise RuntimeError("[anchor-FT] no trainable parameters")
            if _names != _expect:
                raise RuntimeError(
                    f"[anchor-FT] trainable set changed since __init__: "
                    f"+{sorted(_names - _expect)[:4]} -{sorted(_expect - _names)[:4]}")
            _params = [p for _, p in _named]
            _wd = 0.0
        opt = torch.optim.AdamW(
            _params,
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=_wd,
        )
        warmup_steps = int(getattr(self.cfg.train, "warmup_steps", 0) or 0)
        if warmup_steps <= 0:
            return opt
        # Linear LR warmup: 0 -> 1x lr over `warmup_steps` optimizer steps, then held at 1x.
        def lr_lambda(step):
            return min(1.0, float(step + 1) / float(warmup_steps))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    def on_before_optimizer_step(self, optimizer):
        """Safety net: zero any non-finite gradient before the optimizer applies it.

        Gradient-NORM clipping (train.clip_grad, applied later in Lightning's default hook
        ordering via configure_gradient_clipping) can only rescale an already-finite gradient --
        once a parameter's grad is NaN/Inf, the norm computed for clipping is itself NaN, so
        clipping cannot repair or prevent the corruption, and optimizer.step() applies a NaN
        update that permanently poisons that parameter (and its AdamW moment buffers) from this
        step forward. This hook intercepts before that happens: it fires after backward()/DDP's
        gradient allreduce complete but before clipping, so it sees the real synced gradient.

        DDP-safe with no extra collective needed: allreduce is already synchronous inside
        backward(), so every rank holds the bit-identical post-allreduce `.grad` by the time this
        runs, and `torch.isfinite(...).all()` evaluated independently per rank on already-
        identical inputs yields the same skip/zero decision everywhere -- as long as backward()
        is always called unconditionally on every rank (never add a pre-backward, per-rank-only
        skip elsewhere). Zeroing (not skipping the whole step) leaves every other, unaffected
        parameter free to take its legitimate update this batch.
        """
        if self._anchor_ft_enabled:
            with torch.no_grad():
                _sq = sum(float((p.grad * p.grad).sum()) for p in self._anchor_ft_params
                          if p.grad is not None)
            self.log("train/edge_grad_norm", _sq ** 0.5, on_step=True, on_epoch=True,
                     prog_bar=False, logger=True, rank_zero_only=False)

        bad = []
        for name, p in self.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad.zero_()
                bad.append(name)
        if bad:
            self._nan_grad_skip_count = getattr(self, "_nan_grad_skip_count", 0) + 1
            if self._nan_grad_skip_count <= 20 or self._nan_grad_skip_count % 100 == 0:
                self.print(
                    f"[nan_grad_guard] step={self.global_step} epoch={self.current_epoch} "
                    f"zeroed non-finite grad on {len(bad)} param(s) (first: {bad[0]}); "
                    f"skip_count={self._nan_grad_skip_count}"
                )
            if not self._ddp_on:
                self.log(
                    "train/nan_grad_skip", 1.0,
                    on_step=True, on_epoch=True, prog_bar=False, logger=True,
                    rank_zero_only=False,
                )
        if self._ddp_on:
            # Unconditional: every rank must emit this key on every step or the reduction
            # collective mismatches. Value carries the same signal (1.0 == fired).
            self.log(
                "train/nan_grad_skip", 1.0 if bad else 0.0,
                on_step=True, on_epoch=True, prog_bar=False, logger=True,
                rank_zero_only=False,
            )

    @torch.no_grad()
    def _apply_proto_transform(self, mean_emb, present_mask):
        """Loss-free center_l2 transform of the per-domain mean, applied BEFORE the prototype EMA.

        Subtracts a running mean of the observed per-domain means, then L2-normalizes.
        `present_mask` (bool, [n_domains]) marks domains actually observed. It is load-bearing:
        `mean_emb` rows for absent domains are exactly zero (`sums / counts.clamp(min=1)`), so
        including them would drag the running mean toward the origin and silently corrupt the
        centering for every other domain.
        """
        out = mean_emb
        if bool(present_mask.any()):
            batch_mean = out[present_mask].mean(dim=0).detach()
            if bool(self.proto_center_initialized):
                m = self.proto_center_momentum
                self.proto_running_mean.mul_(m).add_(
                    batch_mean.to(self.proto_running_mean.dtype), alpha=(1.0 - m)
                )
            else:
                # Seed from the first observed batch rather than from zeros, so the very
                # first EMA writes are not centered against an all-zero mean.
                self.proto_running_mean.copy_(batch_mean.to(self.proto_running_mean.dtype))
                self.proto_center_initialized.fill_(True)
        out = out - self.proto_running_mean.to(out.dtype).unsqueeze(0)
        return F.normalize(out, dim=-1)

    def on_fit_start(self) -> None:
        # Activation checkpointing on the transformer stack (cfg.model.grad_checkpointing):
        # trades ~30% compute for a large cut in the (B, N, N, dx) activation memory.
        # Bit-exact: torch replays RNG state on recompute.
        _gc = bool(getattr(getattr(self.cfg, "model", None), "grad_checkpointing", False))
        if _gc:
            if hasattr(self.model, "grad_checkpointing"):
                self.model.grad_checkpointing = True
                self.print("[model] activation checkpointing ENABLED on the transformer stack")
            else:
                self.print(
                    f"[model] WARNING: model.grad_checkpointing=true but "
                    f"{type(self.model).__name__} does not support it -- ignored"
                )
        if self._ddp_on:
            # `sampling_metrics` is scored on rank 0 alone (see on_validation_epoch_end).
            # SamplingMolecularMetrics holds four torchmetrics.Metric submodules whose
            # .compute() is a COLLECTIVE by default, so a single-rank call would hang the
            # others. Disabling sync is also the semantically right choice: rank 0 already
            # holds the complete sample set, so a cross-rank reduce would double-count.
            # (SpectreSamplingMetrics, used for the structural domains, is a plain
            # nn.Module and is unaffected -- this covers both.)
            _n = disable_metric_sync(self.sampling_metrics)
            self.print(
                f"[ddp] rank {self.global_rank}/{self._ddp_world_size}: "
                f"disabled cross-rank sync on {_n} sampling metric(s); "
                f"validation generation is rank-0 only"
            )
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )

        self._lock_domain_contract_once(context="on_fit_start")

        # train.anchor_ft: snapshot the frozen teacher HERE, not in __init__ -- general.eval_resume
        # loads the checkpoint after the model is constructed, so an __init__ snapshot would capture
        # the randomly initialised weights. After _lock_domain_contract_once so the domain->index map
        # used by the row masks is final.
        if self._anchor_ft_enabled:
            self._anchor_ft_domain_indices("target")       # raises now, not mid-training
            self._anchor_ft_domain_indices("protected")
            if self._anchor_ft_teacher is None:
                self._anchor_ft_capture_teacher()
    def on_train_epoch_start(self) -> None:
        #self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        if self.start_epoch_time is None: self.start_epoch_time = 0.0
        training_epoch_time = time.time() - self.start_epoch_time
        self.train_epoch_times.append(training_epoch_time)
        to_log = self.train_loss.log_epoch_metrics()
        # Once-per-epoch primary CE summary to metrics.csv.
        self.log_dict(to_log, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if self.start_epoch_time is None: self.start_epoch_time = time.time()
        self.print(
            f"Train Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {training_epoch_time:.1f}s "
        )

        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.val_start_time = time.time()
        self.sampling_metrics.reset()
        if not self._domain_spec_locked:
            self._lock_domain_contract_once(context="on_validation_epoch_start")

    def validation_step(self, data, i):
        return

    def _ddp_shard_domain_budgets(self, budgets):
        """Split a {domain: count} generation budget across ranks, summing to the total.

        Earlier ranks absorb the remainder, so the GLOBAL total is preserved exactly --
        which matters because downstream metric denominators assume it. Splitting
        per-domain (rather than by total) also keeps each rank's domain mix proportional,
        so no domain is generated entirely on one rank.
        """
        from dig2.ddp_utils import shard_count
        if not self._ddp_on or not budgets:
            return budgets
        r, ws = int(self.global_rank), int(self._ddp_world_size)
        return {d: shard_count(int(n), r, ws) for d, n in budgets.items()}

    def _ddp_gather_samples(self, samples, labels):
        """all_gather generated graphs + labels; return the rank-ordered union.

        Payload is a list of {"graph": [atom_types, edge_types], "dataset_name", ...} with
        CPU int64 tensors, so it is picklable as-is (~10-20 MB total at a 504-graph
        budget). `samples` and `labels` are gathered TOGETHER so the positional join that
        _compute_output_isolation_diag / _compute_embed_separation_diag rely on survives.

        Order becomes rank-major rather than domain-major; every consumer is
        permutation-invariant (set statistics and per-graph metrics alike).
        """
        from dig2.ddp_utils import all_gather_objects
        if not self._ddp_on:
            return samples, labels
        parts = all_gather_objects((samples, labels))
        out_s, out_l = [], []
        for part_s, part_l in parts:
            out_s.extend(part_s)
            out_l.extend(part_l)
        return out_s, out_l

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        _sample_now = self.val_counter % self.cfg.general.sample_every_val == 0
        if self._ddp_on:
            # Generation is distributed over ranks; scoring and the results file are rank 0 only.
            _generate_now = _sample_now
            _sample_now = _sample_now and self.global_rank == 0
        else:
            _generate_now = _sample_now
        if _generate_now:
            print("Starting to sample")
            self.sampling_start_time = time.time()
            # general.save_samples_val: write the per-graph generated samples on the validation path.
            # Single-GPU only: under DDP each rank would write just its own shard.
            _save_val = bool(getattr(self.cfg.general, "save_samples_val", False))
            if _save_val and self._ddp_on:
                raise RuntimeError(
                    "general.save_samples_val=true is single-GPU only: under DDP each rank writes "
                    "its own generated_samples_rank{N}.pkl shard while the val txt reports the "
                    "gathered union, so per-graph analysis would see only 1/world_size of the "
                    "samples. Re-run with a single device."
                )
            with contextlib.ExitStack() as _rng_stack:
                if self._ddp_on:
                    # Every rank's global RNG was seeded identically by seed_everything, so
                    # without a rank offset all ranks generate the SAME graphs and the
                    # gather returns N copies. Fork so the training stream is restored
                    # afterwards, and fork ONLY this rank's device.
                    _rng_stack.enter_context(torch.random.fork_rng(devices=[self.device]))
                    _s = (int(self.cfg.train.seed)
                          + 1_000_003 * int(self.global_rank)
                          + 7919 * int(self.current_epoch)) % (2**31 - 1)
                    torch.manual_seed(_s)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(_s)
                if self._use_mixed_per_domain_sampling(is_test=False):
                    domain_to_idx = self._get_domain_to_idx()
                    if len(domain_to_idx) <= 0:
                        raise RuntimeError("domain_to_idx empty before validation per-domain sampling")
                    samples, labels = self.sample_per_domain(
                        is_test=False,
                        per_domain_samples=self._ddp_shard_domain_budgets(
                            self._get_per_domain_budgets(is_test=False)
                        ),
                        domain_to_idx=domain_to_idx,
                        save_samples=_save_val,
                        # only rank 0 writes chain PNGs -- cfg.hier_dir is shared
                        save_visualization=(not self._ddp_on) or self.global_rank == 0,
                    )
                else:
                    samples, labels = self.sample(
                        is_test=False, save_samples=_save_val,
                        save_visualization=(not self._ddp_on) or self.global_rank == 0,
                    )
            samples, labels = self._ddp_gather_samples(samples, labels)
            if self._ddp_on and torch.cuda.is_available():
                # Return the generation arena to the allocator before training resumes.
                _before = torch.cuda.memory_reserved() / 2**30
                torch.cuda.empty_cache()
                self.print(
                    f"[ddp] released generation arena: reserved "
                    f"{_before:.2f} -> {torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                )
        if _sample_now:
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=False)
            if not to_log:
                to_log = {
                    "warning/empty_metrics": "sampling_metrics_returned_empty_dict"
                }
            sampling_time = time.time() - self.sampling_start_time
            to_log["sampling_time"] = f"{sampling_time:.2f}"
            acc_key, acc_value = self._extract_main_accuracy(to_log)
            if acc_value is not None:
                to_log["summary/acc_main"] = acc_value
                to_log["summary/acc_main_key"] = acc_key
            self._augment_accuracy_summaries(to_log)
            # Store results
            filename = os.path.join(self.cfg.hier_dir, f"val_epoch{self.current_epoch}_res.txt")
            # Add training time statistics
            if self.train_epoch_times:
                to_log["train_time_max"] = f"{max(self.train_epoch_times):.2f}"
                to_log["train_time_min"] = f"{min(self.train_epoch_times):.2f}"
                to_log["train_time_avg"] = f"{sum(self.train_epoch_times)/len(self.train_epoch_times):.2f}"
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        val_epoch_time = time.time() - self.val_start_time
        if _sample_now:
            # Re-open file to add epoch time
            filename = os.path.join(self.cfg.hier_dir, f"val_epoch{self.current_epoch}_res.txt")
            with open(filename, "a") as file:
                file.write(f"validation_epoch_time: {val_epoch_time:.2f}\n")

        # Ranks that skipped the (slow) sampling block above must wait for rank 0 here,
        # or they race ahead into the next training epoch and the ranks drift apart in
        # the shared draw stream that stride sharding depends on.
        if self._ddp_on:
            ddp_barrier()

        self.print("Finished validation.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_start_time = time.time()
        self.sampling_metrics.reset()
        if not self._domain_spec_locked:
            self._lock_domain_contract_once(context="on_test_epoch_start")
    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:
        print("Starting to sample")
        self.sampling_start_time = time.time()
        if self._use_mixed_per_domain_sampling(is_test=True):
            domain_to_idx = self._get_domain_to_idx()
            if len(domain_to_idx) <= 0:
                raise RuntimeError("domain_to_idx empty before test per-domain sampling")
            samples, labels = self.sample_per_domain(
                is_test=True,
                per_domain_samples=self._get_per_domain_budgets(is_test=True),
                domain_to_idx=domain_to_idx,
                save_samples=self.cfg.general.save_samples,
                save_visualization=True,
            )
        else:
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=True,
            )

        to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)
        if not to_log:
            to_log = {"warning/empty_metrics": "sampling_metrics_returned_empty_dict"}
        sampling_time = time.time() - self.sampling_start_time
        to_log["sampling_time"] = f"{sampling_time:.2f}"
        acc_key, acc_value = self._extract_main_accuracy(to_log)
        if acc_value is not None:
            to_log["summary/acc_main"] = acc_value
            to_log["summary/acc_main_key"] = acc_key
        self._augment_accuracy_summaries(to_log)

        test_epoch_time = time.time() - self.test_start_time
        filename = os.path.join(self.cfg.hier_dir, f"test_epoch{self.current_epoch}_res.txt")
        with open(filename, "w") as file:
            for key, value in to_log.items():
                file.write(f"{key}: {value}\n")
            file.write(f"test_epoch_time: {test_epoch_time:.2f}\n")
        self.print("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):
        # Otherwise, generate new samples
        if is_test:
            samples_to_generate = self.cfg.general.final_model_samples_to_generate
            samples_left_to_generate = samples_to_generate
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        # sample.rng_safe_per_domain / rng_safe_per_batch / trajectory_seed_offset: paired
        # (common-random-numbers) sampling, keyed on the dataset name instead of a domain name.
        _sm = getattr(self.cfg, "sample", None)
        _rng_safe = bool(getattr(_sm, "rng_safe_per_domain", False))
        _rng_safe_batch = _rng_safe and bool(getattr(_sm, "rng_safe_per_batch", False))
        _rng_key = str(getattr(getattr(self.cfg, "dataset", None), "name", "single")).strip().lower()
        _batch_idx = 0
        if _rng_safe:
            self.print(
                f"[rng-safe-single] paired sampling active for '{_rng_key}' "
                f"(per_batch={_rng_safe_batch}, traj_offset="
                f"{getattr(_sm, 'trajectory_seed_offset', None)})"
            )
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                end="",
                flush=True,
            )
            _gen = _gen_cpu = None
            if _rng_safe:
                _bi = _batch_idx if _rng_safe_batch else None
                _gen = self._domain_rng_generator(_rng_key, device=self.device, batch_idx=_bi)
                _gen_cpu = self._domain_rng_generator(_rng_key, device="cpu", batch_idx=_bi)
                # sample_batch's trajectory_seed_offset block reads these to build its seed.
                self._z0_current_domain_name = _rng_key
                self._z0_current_batch_idx = _batch_idx

            _sample_max_nodes = int(
                getattr(getattr(self.cfg, "sample", None), "max_nodes_per_batch", 0) or 0
            )
            if _sample_max_nodes > 0:
                # The node-count draw (DistributionNodes.sample_n) accepts no generator, so under
                # rng_safe it runs in a forked global stream seeded per batch.
                if _rng_safe:
                    with torch.random.fork_rng(devices=[self.device]):
                        _n_seed = self._domain_rng_generator(
                            f"{_rng_key}#nodes", device="cpu", batch_idx=_batch_idx
                        ).initial_seed()
                        torch.manual_seed(int(_n_seed))
                        bs, packed_num_nodes = self._pack_single_domain_sample_batch(
                            _sample_max_nodes, samples_left_to_generate
                        )
                else:
                    bs, packed_num_nodes = self._pack_single_domain_sample_batch(
                        _sample_max_nodes, samples_left_to_generate
                    )
                to_generate = bs
            else:
                bs = 2 * self.cfg.train.batch_size
                packed_num_nodes = None
                to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=packed_num_nodes,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
                generator=_gen,
                generator_cpu=_gen_cpu,
            )
            _batch_idx += 1
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def _use_mixed_per_domain_sampling(self, is_test: bool) -> bool:
        mixed_cfg = getattr(getattr(self.cfg, "evaluation", None), "mixed", None)
        if mixed_cfg is None:
            return False
        if not bool(getattr(mixed_cfg, "enabled", False)):
            return False
        budgets = self._get_per_domain_budgets(is_test=is_test)
        return isinstance(budgets, Mapping) and len(budgets) > 0

    def _get_per_domain_budgets(self, is_test: bool):
        mixed_cfg = getattr(getattr(self.cfg, "evaluation", None), "mixed", None)
        if mixed_cfg is None:
            return {}
        def _to_plain_dict(value):
            if value is None:
                return {}
            if isinstance(value, DictConfig):
                return OmegaConf.to_container(value, resolve=True) or {}
            if isinstance(value, Mapping):
                return dict(value)
            return {}
        if is_test:
            return _to_plain_dict(
                getattr(mixed_cfg, "per_domain_samples_test", None)
                or getattr(mixed_cfg, "per_domain_samples", {})
            )
        return _to_plain_dict(getattr(mixed_cfg, "per_domain_samples", {}))

    def _resolve_domain_to_idx_authoritative(self):
        """Resolve domain->index map, preferring trainer datamodule when attached."""
        trainer_obj = getattr(self, "_trainer", None)
        datamodule = getattr(trainer_obj, "datamodule", None) if trainer_obj is not None else None
        if datamodule is not None and hasattr(datamodule, "domain_to_idx"):
            dm_map = dict(getattr(datamodule, "domain_to_idx") or {})
            if len(dm_map) > 0:
                return dm_map, "datamodule"

        raw_names = getattr(getattr(self.cfg, "dataset", None), "name", "")
        if isinstance(raw_names, str):
            domain_names = [
                name.strip().lower()
                for name in raw_names.split(",")
                if name.strip()
            ]
        elif isinstance(raw_names, Mapping):
            domain_names = [str(k).strip().lower() for k in raw_names.keys() if str(k).strip()]
        else:
            try:
                domain_names = [str(v).strip().lower() for v in raw_names if str(v).strip()]
            except Exception:
                domain_names = []

        return ({name: idx for idx, name in enumerate(domain_names)}, "cfg_fallback")

    def _peek_domain_y_width(self):
        """Best-effort lookup for y one-hot width from runtime data."""
        trainer_obj = getattr(self, "_trainer", None)
        datamodule = getattr(trainer_obj, "datamodule", None) if trainer_obj is not None else None
        if datamodule is None:
            return None

        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is not None:
            try:
                if len(train_dataset) > 0:
                    sample0 = train_dataset[0]
                    y0 = getattr(sample0, "y", None)
                    if y0 is not None and hasattr(y0, "size") and y0.numel() > 0:
                        return int(y0.size(-1))
            except Exception:
                pass

        try:
            batch = next(iter(datamodule.train_dataloader()))
            y = getattr(batch, "y", None)
            if y is not None and hasattr(y, "size") and y.numel() > 0:
                return int(y.size(-1))
        except Exception:
            pass
        return None

    def _lock_domain_contract_once(self, context: str = ""):
        """Validate and freeze domain conditioning contract once at startup."""
        if not bool(getattr(self.cfg.general, "conditional_domain", False)):
            return
        if self._domain_spec_locked:
            return

        domain_to_idx, source = self._resolve_domain_to_idx_authoritative()
        if len(domain_to_idx) == 0:
            raise RuntimeError(f"[{context}] Could not resolve domain_to_idx in conditional_domain mode")

        idx_sorted = sorted(int(v) for v in domain_to_idx.values())
        expected_idx = list(range(len(domain_to_idx)))
        if idx_sorted != expected_idx:
            raise RuntimeError(
                f"[{context}] Non-contiguous domain indices resolved: {domain_to_idx}. "
                f"Expected contiguous indices {expected_idx}."
            )

        y_width = self._peek_domain_y_width()
        if y_width is not None and int(y_width) != int(len(domain_to_idx)):
            raise RuntimeError(
                f"[{context}] Domain contract mismatch: len(domain_to_idx)={len(domain_to_idx)} "
                f"!= observed y width={y_width}."
            )

        if isinstance(getattr(self, "input_dims", None), Mapping) and ("y" in self.input_dims):
            input_y = int(self.input_dims["y"])
            if y_width is not None:
                min_required = int(y_width) + 1  # +time
                if input_y < min_required:
                    raise RuntimeError(
                        f"[{context}] Invalid input y contract: input_dims['y']={input_y} < "
                        f"y_width+time={min_required}."
                    )

        self._cached_domain_to_idx = dict(domain_to_idx)
        self._domain_spec_source = str(source)
        self._locked_y_width = None if y_width is None else int(y_width)
        self._domain_spec_locked = True

        self.print(
            "[domain_contract] locked "
            f"source={self._domain_spec_source} "
            f"n_domains={len(domain_to_idx)} "
            f"y_width={self._locked_y_width}"
        )

    def _get_domain_to_idx(self):
        if not hasattr(self, "_cached_domain_to_idx"):
            self._cached_domain_to_idx = None
        if getattr(self, "_domain_spec_locked", False) and self._cached_domain_to_idx is not None:
            return dict(self._cached_domain_to_idx)

        domain_to_idx, source = self._resolve_domain_to_idx_authoritative()
        if len(domain_to_idx) > 0:
            self._cached_domain_to_idx = dict(domain_to_idx)
            self._domain_spec_source = str(source)
            return dict(self._cached_domain_to_idx)

        if self._cached_domain_to_idx is not None:
            return dict(self._cached_domain_to_idx)
        return {}

    def _domain_one_hot_rows(self, num_domains: int, *, dtype=torch.float32):
        key = (int(num_domains), str(self.device), str(dtype))
        cached = self._cached_domain_one_hot_rows.get(key)
        if cached is None:
            cached = torch.eye(int(num_domains), device=self.device, dtype=dtype)
            self._cached_domain_one_hot_rows[key] = cached
        return cached

    def _sample_domain_num_nodes(self, domain_name: str, batch_size: int, generator=None):
        """Sample node counts from domain-specific support when available.

        `generator` (optional torch.Generator, same device as hist_t i.e. self.device):
        None (default) draws from the default global RNG stream, byte-identical to
        pre-existing behavior. This call happens BEFORE sample_batch's own RNG consumption
        (it decides graph size, sample_batch decides graph content), so it needed its own
        thread of the domain generator to close the rng_safe_per_domain fix -- otherwise
        node counts alone would still leak cross-domain RNG-position-dependence even with
        sample_batch's draws fixed.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "domain_specs"):
            return None
        spec = (getattr(datamodule, "domain_specs", {}) or {}).get(domain_name)
        if not isinstance(spec, dict):
            return None
        hist = spec.get("n_nodes_hist", None)
        if hist is None:
            return None

        cache_key = str(domain_name)
        cached = self._cached_domain_hist.get(cache_key)
        if cached is None:
            hist_t = torch.tensor(hist, dtype=torch.float, device=self.device)
            hist_t = hist_t / hist_t.sum().clamp(min=1e-8)
            min_n = int(spec.get("min_n_nodes", 1))
            max_n = int(spec.get("max_n_nodes", int(hist_t.numel() - 1)))
            cached = (hist_t, min_n, max_n)
            self._cached_domain_hist[cache_key] = cached
        hist_t, min_n, max_n = cached
        sampled = torch.multinomial(hist_t, num_samples=batch_size, replacement=True, generator=generator)
        sampled = sampled.clamp(min=min_n, max=max_n)
        return sampled.long()

    def _domain_expected_n_nodes(self, domain_name: str) -> int:
        """Mean node count for a domain from the cached histogram, or node_dist.max_n fallback."""
        cached = self._cached_domain_hist.get(str(domain_name))
        if cached is None:
            self._sample_domain_num_nodes(domain_name, 1)
            cached = self._cached_domain_hist.get(str(domain_name))
        if cached is not None:
            hist_t, _min_n, _max_n = cached
            mean = float((hist_t * torch.arange(len(hist_t), device=hist_t.device)).sum())
            return max(1, int(mean + 0.5))
        return int(getattr(self.node_dist, "max_n", 64) or 64)

    def _pack_single_domain_sample_batch(self, node_budget: int, samples_left: int):
        """Directly interpret a sampling batch's graph count from real sampled node
        counts, for the single-domain sample() path -- mirrors the node-budget
        greedy packing NodeBudgetBatchSampler (abstract_dataset.py) already does
        for training batches, instead of dividing node_budget by an aggregate
        stat like node_dist.max_n (the coarser estimate used previously, and
        still used by _domain_expected_n_nodes's multi-domain fallback).

        Draws node counts one at a time from self.node_dist until adding the
        next graph would exceed node_budget, or samples_left is reached. Always
        includes at least one graph, even if it alone exceeds the budget (a
        generated graph can't be "skipped" the way an oversized real graph can
        be at training time -- see abstract_dataset.py's own tolerance note).

        Returns (bs, num_nodes): num_nodes is a LongTensor of the exact sampled
        per-graph node counts, threaded straight into sample_batch(num_nodes=...)
        so the counts drawn here are reused rather than redrawn.
        """
        counts = []
        total = 0
        while len(counts) < samples_left:
            n = int(self.node_dist.sample_n(1, self.device).item())
            if counts and total + n > node_budget:
                break
            counts.append(n)
            total += n
        num_nodes = torch.tensor(counts, device=self.device, dtype=torch.long)
        return len(counts), num_nodes

    def _sample_batch_size_for_domain(self, domain_name: str, left: int) -> int:
        """Compute per-batch graph count from node budget, or fall back to 2*train.batch_size."""
        sample_max_nodes = int(
            getattr(getattr(self.cfg, "sample", None), "max_nodes_per_batch", 0) or 0
        )
        if sample_max_nodes > 0:
            mean_n = self._domain_expected_n_nodes(domain_name)
            return max(1, min(left, sample_max_nodes // mean_n))
        return min(left, 2 * self.cfg.train.batch_size)

    @torch.no_grad()
    def _trajectory_seed(self, domain_name, batch_idx, offset):
        """Deterministic seed for the REVERSE-TRAJECTORY draws only, decoupled from z_T.

        `sample.trajectory_seed_offset` re-runs the SAME z_T under different trajectory draws,
        which separates per-graph trajectory noise from real differences between sampling
        settings. md5 (not str hash(), which is PYTHONHASHSEED-randomized) keeps the seed stable
        across processes.
        """
        base_seed = int(getattr(getattr(self.cfg, "train", None), "seed", 42))
        key = f"{str(domain_name).strip().lower()}#b{int(batch_idx or 0)}#traj{int(offset)}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return (base_seed + int(digest[:8], 16)) % (2**31 - 1)

    def _domain_rng_generator(self, domain_name: str, device=None, batch_idx=None):
        """Build a torch.Generator seeded deterministically from (cfg.train.seed, domain_name),
        independent of any other domain's presence, sample count, or position in
        per_domain_samples. Only called when `sample.rng_safe_per_domain=true` -- see that
        flag's docstring in sample_per_domain for the reproducibility issue this fixes.

        `device` defaults to self.device. torch.Generator objects are device-locked (a
        generator built for one device raises if handed a tensor on another), and this
        model's noise-prior tensors (`limit_dist.X/E`, consumed by
        sample_discrete_feature_noise) live on CPU while step-transition probabilities
        (consumed by sample_discrete_features) live on self.device -- so callers need one
        generator per device, both seeded identically for the same domain (see
        sample_per_domain, which builds both).

        Uses hashlib.md5 (not Python's built-in hash()) since str hash() is randomized
        per-process by default (PYTHONHASHSEED) and would defeat the whole point of a
        deterministic, cross-process-stable seed.
        """
        base_seed = int(getattr(getattr(self.cfg, "train", None), "seed", 42))
        domain_key = str(domain_name).strip().lower()
        if batch_idx is not None:
            # Per-batch reseeding (sample.rng_safe_per_batch). Without it, a domain's
            # batches share one advancing stream, and because the reverse loop consumes a
            # SAMPLE_STEPS-DEPENDENT number of draws, batch 2 onward gets a different z_T
            # at different step counts -- which silently destroys common-random-numbers
            # pairing for every domain whose quota exceeds one batch. Measured: pairing
            # held for only the first batch (sbm 19/40, qm9 113/512, zinc 89/512).
            domain_key = f"{domain_key}#b{int(batch_idx)}"
        digest = hashlib.md5(domain_key.encode("utf-8")).hexdigest()
        domain_offset = int(digest[:8], 16)  # stable 32-bit value from the hash
        seed = (base_seed + domain_offset) % (2**31 - 1)
        generator = torch.Generator(device=device if device is not None else self.device)
        generator.manual_seed(seed)
        return generator

    def sample_per_domain(
        self,
        *,
        is_test,
        per_domain_samples,
        domain_to_idx,
        save_samples,
        save_visualization,
    ):
        """Generate exact per-domain sample quotas using forced one-hot conditioning.

        Returns a metadata-rich sample list, where each item is a dict:
          {
            "graph": [atom_types, edge_types],
            "dataset_name": str,
            "dataset_id": int,
          }

        `sample.rng_safe_per_domain` (and `rng_safe_per_batch`) give each domain (and batch) its
        own generator seeded from (train.seed, domain[, batch]), so a domain's samples never depend
        on any other domain's quota. Off by default (one shared global stream); the evaluation
        protocol turns both on.
        """
        all_samples = []
        all_labels = []
        domain_samples_for_viz = {}
        global_graph_id = 0
        num_domains = len(domain_to_idx)
        _rng_safe_per_domain = bool(getattr(self.cfg.sample, "rng_safe_per_domain", False))
        _rng_safe_per_batch = bool(getattr(self.cfg.sample, "rng_safe_per_batch", False)) and _rng_safe_per_domain

        for domain_name, target_n in per_domain_samples.items():
            domain_name = str(domain_name).strip().lower()
            if domain_name not in domain_to_idx:
                continue
            target_n = int(target_n)
            if target_n <= 0:
                continue

            domain_id = domain_to_idx[domain_name]
            left = target_n
            # One generator pair per domain (device-locked, see _domain_rng_generator),
            # constructed once and reused across all of that domain's batches below (so
            # batch-to-batch draws within a domain still advance deterministically from its
            # own seed, same as the shared-stream default did -- only cross-domain
            # independence changes). Both seeded identically for this domain.
            domain_generator = self._domain_rng_generator(domain_name, device=self.device) if _rng_safe_per_domain else None
            domain_generator_cpu = self._domain_rng_generator(domain_name, device=torch.device("cpu")) if _rng_safe_per_domain else None
            samples_left_to_save = (
                self.cfg.general.final_model_samples_to_save
                if is_test
                else self.cfg.general.samples_to_save
            )
            chains_left_to_save = (
                self.cfg.general.final_model_chains_to_save
                if is_test
                else self.cfg.general.chains_to_save
            )

            _batch_idx = 0
            while left > 0:
                bs = self._sample_batch_size_for_domain(domain_name, left)
                if _rng_safe_per_batch:
                    # Fresh, batch-indexed stream so this batch's z_T does not depend on
                    # how many draws earlier batches' trajectories consumed.
                    domain_generator = self._domain_rng_generator(
                        domain_name, device=self.device, batch_idx=_batch_idx)
                    domain_generator_cpu = self._domain_rng_generator(
                        domain_name, device=torch.device("cpu"), batch_idx=_batch_idx)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                self._z0_current_domain_name = domain_name
                self._z0_current_domain_id = int(domain_id)
                self._z0_current_batch_idx = _batch_idx
                resolved_sample_steps = self._resolve_sample_steps(domain_name)
                num_chain_steps = min(self.number_chain_steps, resolved_sample_steps)

                use_domain_conditioning = getattr(self.cfg.general, "conditional_domain", False)
                if use_domain_conditioning:
                    one_hot_rows = self._domain_one_hot_rows(num_domains, dtype=torch.float32)
                    forced_condition = one_hot_rows[int(domain_id)].unsqueeze(0).expand(bs, -1)
                else:
                    # Unconditional model: null conditioning — zero vector, no domain signal.
                    # Samples are still labeled with dataset_name for MixedMetricDispatcher routing.
                    # Use input_dims['y']-1 (subtracts the time slot) to match data.y.size(1)
                    # so compute_extra_data's contract check does not go negative.
                    y_dim = int(self.input_dims["y"]) - 1
                    forced_condition = torch.zeros(bs, y_dim, device=self.device)
                domain_num_nodes = self._sample_domain_num_nodes(
                    domain_name, bs, generator=domain_generator)

                cur_samples, cur_labels = self.sample_batch(
                    global_graph_id,
                    bs,
                    num_nodes=domain_num_nodes,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=num_chain_steps,
                    save_visualization=False,
                    forced_condition=forced_condition,
                    force_domain_mask=use_domain_conditioning,
                    sample_steps=resolved_sample_steps,
                    generator=domain_generator,
                    generator_cpu=domain_generator_cpu,
                )

                domain_samples_for_viz.setdefault(domain_name, []).extend(cur_samples)

                for i, graph in enumerate(cur_samples):
                    all_samples.append(
                        {
                            "graph": graph,
                            "dataset_name": domain_name,
                            "dataset_id": int(domain_id),
                        }
                    )
                    all_labels.append(
                        cur_labels[i] if i < len(cur_labels) else forced_condition[i].detach().cpu()
                    )

                global_graph_id += bs
                left -= bs
                _batch_idx += 1
                samples_left_to_save -= to_save
                chains_left_to_save -= chains_save

        if save_visualization and self.visualization_tools is not None:
            self._visualize_domain_samples(domain_samples_for_viz, is_test=is_test)

        if save_samples and len(all_samples) > 0:
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(all_samples, f)

        return all_samples, all_labels

    def _is_molecular_domain(self, domain_name: str) -> bool:
        domain_name = str(domain_name).strip().lower()
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "domain_type_by_name"):
            domain_type = (getattr(datamodule, "domain_type_by_name", {}) or {}).get(domain_name)
            if isinstance(domain_type, str):
                return domain_type.strip().lower() == "molecular"

        sampling_metrics = getattr(self, "sampling_metrics", None)
        mol_eval_map = getattr(sampling_metrics, "molecular_evaluator_by_domain", {})
        return domain_name in mol_eval_map

    def _get_domain_visualizer(self, domain_name: str):
        domain_name = str(domain_name).strip().lower()

        if domain_name in self.visualization_tools_by_domain:
            return self.visualization_tools_by_domain[domain_name]

        if self._is_molecular_domain(domain_name):
            raise ValueError(
                "Missing molecular visualization tool for domain "
                f"'{domain_name}'. Available domain visualizers: "
                f"{sorted(self.visualization_tools_by_domain.keys())}"
            )

        return self.visualization_tools

    def _validate_molecular_samples_against_decoder(self, domain_name: str, samples, visualizer):
        dataset_infos = getattr(visualizer, "dataset_infos", None)
        atom_decoder = getattr(dataset_infos, "atom_decoder", None)
        if atom_decoder is None:
            raise ValueError(
                f"Domain '{domain_name}' visualizer has no atom_decoder in dataset_infos."
            )

        decoder_len = len(atom_decoder)
        max_atom_id = -1
        for sample in samples:
            atom_types = sample[0]
            if torch.is_tensor(atom_types):
                ids = atom_types.detach().view(-1).long()
                ids = ids[ids >= 0]
                if ids.numel() > 0:
                    max_atom_id = max(max_atom_id, int(ids.max().item()))

        if max_atom_id >= decoder_len:
            raise ValueError(
                "Molecular visualization decoder mismatch for domain "
                f"'{domain_name}': observed atom id {max_atom_id} but decoder size is {decoder_len}."
            )

    def _visualize_domain_samples(self, domain_samples_for_viz, *, is_test: bool):
        if not isinstance(domain_samples_for_viz, Mapping):
            return

        save_cap = (
            int(getattr(self.cfg.general, "final_model_samples_to_save", 0))
            if is_test
            else int(getattr(self.cfg.general, "samples_to_save", 0))
        )

        for domain_name, samples in domain_samples_for_viz.items():
            if not samples:
                continue
            visualizer = self._get_domain_visualizer(domain_name)
            if visualizer is None:
                continue

            # Apply global→local remap for semantic_global_v1 so visualization uses local indices.
            _sm = getattr(self, "sampling_metrics", None)
            _me = (getattr(_sm, "molecular_evaluator_by_domain", {}) or {}).get(domain_name)
            _ar = getattr(_me, "global_to_local_atom_map", None)
            _er = getattr(_me, "global_to_local_edge_map", None)
            if _ar is not None or _er is not None:
                _ar_cpu = _ar.cpu() if _ar is not None else None
                _er_cpu = _er.cpu() if _er is not None else None
                samples = [
                    [_ar_cpu[s[0].long()] if _ar_cpu is not None else s[0],
                     _er_cpu[s[1].long()] if _er_cpu is not None else s[1]]
                    for s in samples
                ]

            if self._is_molecular_domain(domain_name):
                self._validate_molecular_samples_against_decoder(
                    domain_name=domain_name,
                    samples=samples,
                    visualizer=visualizer,
                )

            num_to_save = min(len(samples), save_cap if save_cap > 0 else len(samples))
            base_path = self.cfg.hier_dir
            result_path = os.path.join(
                base_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}/domain_{domain_name}",
            )
            os.makedirs(result_path, exist_ok=True)
            visualizer.visualize(result_path, samples, num_to_save)

    def evaluate_samples(self, samples, labels, is_test):
        print("Computing sampling metrics...")
        return self.sampling_metrics.forward(
            samples,
            ref_metrics=self.dataset_info.ref_metrics,
            name=self.cfg.general.name,
            current_epoch=self.current_epoch,
            val_counter=-1,
            test=is_test,
            local_rank=self.local_rank,
            labels=labels if self.conditional else None,
        )
    def apply_noise(self, X, E, y, node_mask, t=None, prior_y=None):
        """Sample noise and apply it to the data.

        ``prior_y`` (optional): conditioning tensor used ONLY to select the per-domain
        noise prior. Pass the pre-CFG-dropout y so CFG-dropped rows are still noised with
        each sample's true-domain prior; the nulled ``y`` stays the conditioning signal in
        ``noisy_data["y_t"]``.
        """

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        y_prior = y
        if prior_y is not None:
            y_prior = prior_y

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        use_domain_prior = (
            self.prior_mode == "domain_specific"
            and y_prior is not None
            and y_prior.dim() == 2
            and y_prior.size(-1) > 0
            and len(getattr(self.noise_dist, "domain_limit_dist", {})) > 0
        )
        if use_domain_prior:
            domain_ids = torch.argmax(y_prior, dim=-1)
            limit_stack = self._get_domain_limit_stack(width=int(y_prior.size(-1)))
            # Vectorized path: same interpolation as p_xt_g_x1, but with a
            # per-sample limit vector gathered by domain id. One kernel,
            # no per-domain Python loop, no .item() host syncs.
            t_time = t_float.squeeze(-1)[:, None, None]
            lim_X = limit_stack["X"][domain_ids]  # (B, Sx)
            lim_E = limit_stack["E"][domain_ids]  # (B, Se)
            X1_onehot = F.one_hot(X_1_label, num_classes=int(limit_stack["X"].size(-1))).float()
            E1_onehot = F.one_hot(E_1_label, num_classes=int(limit_stack["E"].size(-1))).float()
            prob_X_t = t_time * X1_onehot + (1 - t_time) * lim_X[:, None, :]
            prob_E_t = (
                t_time[:, None] * E1_onehot
                + (1 - t_time[:, None]) * lim_E[:, None, None, :]
            )
            if hotpath_asserts_enabled():
                assert ((prob_X_t.sum(-1) - 1).abs() < 1e-4).all() and (
                    (prob_E_t.sum(-1) - 1).abs() < 1e-4
                ).all()
            prob_X_t = prob_X_t.clamp(min=0.0, max=1.0)
            prob_E_t = prob_E_t.clamp(min=0.0, max=1.0)
        else:
            prob_X_t, prob_E_t = p_xt_g_x1(
                X1=X_1_label,
                E1=E_1_label,
                t=t_float,
                limit_dist=self.limit_dist,
            )

        # step 4 - sample noised data
        sampled_t = interpolant_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(
        self,
        noisy_data,
        extra_data,
        node_mask,
        return_layer_predictions: bool = False,
    ):
        """Denoiser forward.

        ``return_layer_predictions``: also return the per-depth progressive-readout predictions
        and gate weights (used by the deep-supervision loss).
        """
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        self._ensure_cond_slice_layout(noisy_data["y_t"])
        if not bool(getattr(self.cfg.general, "conditional_domain", False)):
            # Domain-blind: zero-fill the y_t slot so mlp_in_y receives input_dims['y']
            # dimensions but no domain signal leaks through.
            y_cond = torch.zeros_like(noisy_data["y_t"])
            y_domain = noisy_data["y_t"][:, :0]  # 0-dim; _domain_indices_from_y returns None
        else:
            y_domain, _y_rrwp = self._split_cond_targets(noisy_data["y_t"])
            y_cond = self._inject_domain_embedding(y_domain)
        # Structural conditioning: computed once, injected into both y and cond_slice.
        _structural_emb = None
        if self.structural_encoder_enabled and self.structural_encoder is not None:
            _structural_emb = self._build_structural_cond(
                X_t=noisy_data["X_t"],
                E_t=noisy_data["E_t"],
                y_t=noisy_data["y_t"],
                node_mask=node_mask,
            )
            y_cond = torch.cat([y_cond, _structural_emb], dim=-1)
        y = torch.hstack((y_cond, extra_data.y)).float()

        self._assert_model_input_dim_invariants(where="forward_pre")

        expected_x_in = int(self.input_dims["X"])
        if int(X.size(-1)) != expected_x_in:
            raise RuntimeError(
                f"Input X feature mismatch in forward: got {int(X.size(-1))}, "
                f"expected {expected_x_in}. conditional_domain={getattr(self.cfg.general, 'conditional_domain', None)}"
            )

        cond_slice = self._assemble_cond_slice(noisy_data["y_t"], structural_emb=_structural_emb)
        domain_idx = self._domain_indices_from_y(y_domain)
        gt_kwargs = dict(
            cond_slice=cond_slice,
            domain_idx=domain_idx,
        )
        if return_layer_predictions:
            return self.model(X, E, y, node_mask, return_layer_predictions=True, **gt_kwargs)
        return self.model(X, E, y, node_mask, **gt_kwargs)

    def _ensure_cond_slice_layout(self, y_cond: torch.Tensor):
        if y_cond is None or y_cond.dim() != 2:
            return
        if self._cond_slice_layout_locked:
            return
        n_total = int(y_cond.size(-1))
        domain_dim = int(max(len(self._get_domain_to_idx()), 0)) if bool(getattr(self.cfg.general, "conditional_domain", False)) else 0
        rrwp_dim = int(max(n_total - domain_dim, 0))
        self._domain_y_dim = int(domain_dim)
        self._rrwp_y_dim = int(rrwp_dim)
        self._cond_slice_layout_locked = True

    def _split_cond_targets(self, y_cond: torch.Tensor):
        if y_cond is None:
            return y_cond, None
        self._ensure_cond_slice_layout(y_cond)
        domain_dim = int(self._domain_y_dim or 0)
        rrwp_dim = int(self._rrwp_y_dim or 0)
        y_domain = y_cond[:, :domain_dim] if domain_dim > 0 else y_cond[:, :0]
        y_rrwp = y_cond[:, domain_dim : domain_dim + rrwp_dim] if rrwp_dim > 0 else y_cond[:, :0]
        return y_domain, y_rrwp

    def _assemble_cond_slice(self, y_cond: torch.Tensor,
                             structural_emb: torch.Tensor = None):
        if y_cond is None or y_cond.dim() != 2:
            return None
        y_domain, y_rrwp = self._split_cond_targets(y_cond)
        y_domain = self._inject_domain_embedding(y_domain)
        base = y_domain.float()
        if structural_emb is not None:
            return torch.cat([base, structural_emb], dim=-1)
        return base

    def _build_structural_cond(
        self,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        y_t: torch.Tensor,
        node_mask: torch.Tensor,
    ):
        """Structural conditioning = fuse(EMA prototype[domain], encode(X_t, E_t)).

        The current-state encoding (from noisy X_t) is differentiable and trains the encoder
        end-to-end. The prototype embedding (from the EMA bank of clean graphs) is detached and
        serves as a non-gradient anchor for domain identity.
        """
        current_emb = self.structural_encoder(
            X_t.float(), E_t.float(), node_mask,
        )  # (B, structural_emb_dim)
        proto_emb = self._structural_proto_for_y(y_t)
        return self.struct_fusion(torch.cat([proto_emb, current_emb], dim=-1))

    def _proto_y_slice(self, y):
        """The domain slice of a conditioning tensor, which the prototype argmax runs over.

        With `conditional_domain=false` the domain slice is 0-wide; fall back to the full tensor.
        """
        y_dom, _ = self._split_cond_targets(y)
        if y_dom is None or int(y_dom.size(-1)) == 0:
            return y
        return y_dom

    def _structural_proto_for_y(self, y_for_proto):
        """Per-sample EMA structural prototype for a conditioning tensor; zeros for CFG-null rows."""
        n_proto = int(self.domain_prototypes.size(0))
        y_sel = self._proto_y_slice(y_for_proto)
        domain_idx = y_sel.argmax(dim=-1).clamp(0, n_proto - 1)      # (B,)
        null_rows = y_sel.float().abs().sum(dim=-1) <= 1e-8          # (B,)
        proto_emb = self.domain_prototypes[domain_idx].detach()
        null_vec = torch.zeros_like(proto_emb)
        return torch.where(null_rows.view(-1, 1), null_vec, proto_emb)

    def _inject_domain_embedding(self, y_cond: torch.Tensor) -> torch.Tensor:
        """Optionally enrich one-hot domain conditioning with a learned embedding."""
        if not bool(getattr(self, "use_domain_embedding", False)):
            return y_cond
        if y_cond is None:
            return y_cond

        if not bool(getattr(self.cfg.general, "conditional_domain", False)):
            if not self._domain_emb_warned:
                self.print(
                    "[domain_embedding] Enabled while general.conditional_domain=false; "
                    "skipping domain embedding."
                )
                self._domain_emb_warned = True
            return y_cond

        n_domains = int(y_cond.size(-1))
        if self.domain_emb_table is None:
            return y_cond
        _emb_n_in = int(getattr(self, "_domain_emb_n_inputs", n_domains))
        if _emb_n_in != n_domains:
            raise RuntimeError(
                "domain_embedding input mismatch: "
                f"y_cond has {n_domains} channels, but domain_emb_table expects {_emb_n_in}"
            )

        def _residual_mlp(x):
            return x + self._domain_emb_mlp_down(
                self._domain_emb_mlp_act(self._domain_emb_mlp_up(x))
            )

        domain_idx = self._domain_indices_from_y(y_cond)
        return _residual_mlp(self.domain_emb_table(domain_idx))

    def _domain_indices_from_y(self, y_cond: torch.Tensor):
        """Resolve raw domain one-hot rows to embedding indices, including learned CFG null."""
        if y_cond is None or y_cond.dim() != 2 or y_cond.size(-1) == 0:
            return None
        domain_idx = y_cond.argmax(dim=-1)
        null_idx = getattr(self, "domain_null_idx", None)
        if null_idx is None:
            return domain_idx
        null_rows = y_cond.float().abs().sum(dim=-1) <= 1e-8
        # `torch.any(...)` in a Python `if` forces a GPU->CPU sync; measured at 3 syncs per
        # step in BOTH the training and sampling loops. `torch.where` is unconditional and
        # bit-identical: identical values, no host round-trip, no `.clone()`.
        return torch.where(
            null_rows, torch.full_like(domain_idx, int(null_idx)), domain_idx
        )

    def _direct_domain_emb_expansion(self, n_raw_y: int) -> int:
        """Channels added to y by the learned domain embedding in forward() (0 when off).

        forward() calls _inject_domain_embedding AFTER compute_extra_data, expanding y_t from
        n_raw_y -> domain_emb_dim. Callers that reason about channel counts before that expansion
        must subtract this value from input_dims["y"] to get the pre-expansion required size.
        """
        gen = self.cfg.general
        if bool(getattr(gen, "domain_embedding", False)):
            emb_dim = int(getattr(gen, "domain_emb_dim", 0) or 0)
            return max(0, emb_dim - n_raw_y)
        return 0

    def _build_uncond_y(self, y_cond: torch.Tensor) -> torch.Tensor:
        """Build unconditional conditioning tensor for classifier-free guidance.

        - conditional_domain: use all-zeros one-hot (null-domain token)
        - legacy conditioning: keep -1 sentinel behavior
        """
        if bool(getattr(self.cfg.general, "conditional_domain", False)):
            y_dom, y_rrwp = self._split_cond_targets(y_cond)
            dom = torch.zeros_like(y_dom, device=y_cond.device)
            if y_rrwp is not None and y_rrwp.size(-1) > 0:
                rrwp = torch.zeros_like(y_rrwp, device=y_cond.device)
                return torch.cat((dom, rrwp), dim=-1)
            return dom
        return torch.ones_like(y_cond, device=y_cond.device) * -1

    def _cfg_dropout_prob(self) -> float:
        explicit = getattr(self.cfg.general, "cfg_dropout_prob", None)
        if explicit is not None:
            return float(explicit)
        # Slightly lower default for mixed-domain runs to reduce over-regularization.
        domain_names = [
            name.strip().lower()
            for name in str(getattr(self.cfg.dataset, "name", "")).split(",")
            if name.strip()
        ]
        if bool(getattr(self.cfg.general, "conditional_domain", False)) and len(domain_names) > 1:
            return 0.05
        return 0.1

    def _domain_loss_weight_tensors(self, y_cond, node_mask):
        """Broadcast per-graph domain_loss_weight to node/edge weight tensors consumed by
        TrainLossDiscrete's weighted-mean-CE path.

        The weight is applied per graph before reduction, so each graph's node/edge terms carry
        its own domain's weight regardless of batch composition. X and E are resolved
        independently (`_domain_loss_multiplier(channel=...)`) since they scale differently under
        node_budget batching (O(n) vs O(n^2) per graph).

        Returns (node_weight, edge_weight): each None when that channel's per-graph weight is
        exactly 1.0 for every graph, so trivial weights are a strict no-op.
        """
        if y_cond is None or y_cond.numel() == 0 or node_mask is None:
            return None, None
        per_graph_x = self._domain_loss_multiplier(y_cond, channel="x")  # [B]
        per_graph_e = self._domain_loss_multiplier(y_cond, channel="e")  # [B]
        n = node_mask.size(1)

        node_weight = None
        if not bool((per_graph_x == 1.0).all()):
            node_weight = per_graph_x.view(-1, 1).expand(-1, n).to(node_mask.device)
        edge_weight = None
        if not bool((per_graph_e == 1.0).all()):
            edge_weight = per_graph_e.view(-1, 1, 1).expand(-1, n, n)
        return node_weight, edge_weight

    def _domain_loss_multiplier(self, y_cond: torch.Tensor, channel: str = "x"):
        """Return per-sample loss multipliers from cfg.train.domain_loss_weight map.

        Each domain's entry may be a plain number (applied uniformly to every
        channel -- the original flat schema) or a mapping with per-channel keys,
        e.g. `{x: 0.64, e: 0.40}` (resolved via `channel`, defaulting to 1.0 / a
        no-op if that channel's key is absent). `channel` is "x" (node loss) or
        "e" (edge loss); X and E scale differently (O(n) vs O(n^2) per graph)
        under node_budget batching, so they generally need different weights.
        """
        if y_cond is None or y_cond.numel() == 0:
            return torch.ones(1, device=self.device)

        weights_cfg = getattr(getattr(self.cfg, "train", None), "domain_loss_weight", None)
        if weights_cfg is None:
            return torch.ones(y_cond.size(0), device=y_cond.device)

        if isinstance(weights_cfg, DictConfig):
            weights_map = OmegaConf.to_container(weights_cfg, resolve=True) or {}
        elif isinstance(weights_cfg, Mapping):
            weights_map = dict(weights_cfg)
        else:
            return torch.ones(y_cond.size(0), device=y_cond.device)

        y_domain, _ = self._split_cond_targets(y_cond)
        domain_to_idx = self._domain_name_to_index()
        if not domain_to_idx:
            return torch.ones(y_cond.size(0), device=y_cond.device)

        # per_domain depends only on (channel, weights_map, domain_to_idx, vocab size) --
        # all static for the whole run, not on the current batch -- so cache it instead
        # of rebuilding it via a Python loop on every call (called up to 4x/step: X/E
        # weight tensors + X/E logging).
        _pd_cache = getattr(self, "_domain_loss_multiplier_cache", None)
        if _pd_cache is None:
            _pd_cache = {}
            self._domain_loss_multiplier_cache = _pd_cache
        cache_key = (channel, int(y_domain.size(-1)), str(y_cond.device))
        per_domain = _pd_cache.get(cache_key)
        if per_domain is None:
            per_domain = torch.ones(y_domain.size(-1), device=y_cond.device)
            for domain_name, domain_idx in domain_to_idx.items():
                if int(domain_idx) >= y_domain.size(-1):
                    continue
                if domain_name in weights_map:
                    entry = weights_map[domain_name]
                    if isinstance(entry, Mapping):
                        per_domain[int(domain_idx)] = float(entry.get(channel, 1.0))
                    else:
                        per_domain[int(domain_idx)] = float(entry)
            _pd_cache[cache_key] = per_domain

        if y_domain.size(-1) == 0:
            return torch.ones(y_cond.size(0), device=y_cond.device)
        return (y_domain.float() * per_domain.unsqueeze(0)).sum(dim=-1).clamp(min=1e-8)

    def _domain_guidance_weight(self, y_cond: torch.Tensor):
        """Per-sample CFG guidance weights: general.guidance_weight_by_domain[domain] when
        set, otherwise the flat general.guidance_weight."""
        base_weight = float(getattr(self.cfg.general, "guidance_weight", 1.0))
        if y_cond is None or y_cond.numel() == 0:
            return torch.full((1,), base_weight, device=self.device)
        y_domain, _ = self._split_cond_targets(y_cond)

        by_domain_cfg = getattr(self.cfg.general, "guidance_weight_by_domain", None)
        if by_domain_cfg is None:
            return torch.full((y_cond.size(0),), base_weight, device=y_cond.device)

        if isinstance(by_domain_cfg, DictConfig):
            by_domain = OmegaConf.to_container(by_domain_cfg, resolve=True) or {}
        elif isinstance(by_domain_cfg, Mapping):
            by_domain = dict(by_domain_cfg)
        else:
            by_domain = {}
        by_domain = {k: v for k, v in by_domain.items() if v is not None}

        if not bool(getattr(self.cfg.general, "conditional_domain", False)):
            return torch.full((y_cond.size(0),), base_weight, device=y_cond.device)

        domain_to_idx = self._domain_name_to_index()
        per_domain = torch.full((y_domain.size(-1),), base_weight, device=y_cond.device)
        for domain_name, domain_idx in domain_to_idx.items():
            if int(domain_idx) >= y_domain.size(-1):
                continue
            if domain_name in by_domain:
                per_domain[int(domain_idx)] = float(by_domain[domain_name])
        if y_domain.size(-1) == 0:
            return torch.full((y_cond.size(0),), base_weight, device=y_cond.device)
        return (y_domain.float() * per_domain.unsqueeze(0)).sum(dim=-1)

    def _resolve_sample_steps(self, domain_name: str) -> int:
        """Reverse-process step count for one domain: cfg.sample.sample_steps_by_domain[domain]
        when set, otherwise the flat cfg.sample.sample_steps."""
        base = int(self.cfg.sample.sample_steps)
        by_domain_cfg = getattr(self.cfg.sample, "sample_steps_by_domain", None)
        by_domain = {}
        if isinstance(by_domain_cfg, DictConfig):
            by_domain = OmegaConf.to_container(by_domain_cfg, resolve=True) or {}
        elif isinstance(by_domain_cfg, Mapping):
            by_domain = dict(by_domain_cfg)
        key = str(domain_name).strip().lower()
        if key in by_domain and by_domain[key] is not None:
            return int(by_domain[key])

        return base

    def _get_limit_dist_checked(self, domain_idx=None, context: str = ""):
        cur, source = self.noise_dist.resolve_limit_dist(
            domain_idx=domain_idx,
            prior_mode=self.prior_mode,
        )
        if self.prior_mode == "domain_specific" and domain_idx is not None:
            did = int(domain_idx)
            if did not in getattr(self.noise_dist, "domain_limit_dist", {}):
                raise ValueError(f"Missing domain prior for domain_idx={did} ({context}).")
        return cur

    def _get_constant_sampling_masks_cached(self, node_mask):
        """(inverse_edge_mask, diag_mask) for `node_mask`, cached against the last-seen
        `node_mask` object -- both depend only on node_mask/n, which are built once
        before sample_batch()'s denoising loop and never reassigned, so this turns
        `sample_steps` rebuilds into one computation per sample_batch() call.

        The cache holds `node_mask` itself and hits only on identity. Keying on
        `id(node_mask)` alone would be unsafe: CPython recycles freed ids, so a later
        same-shape batch could inherit another batch's padding mask."""
        cached = getattr(self, "_sampling_masks_cache", None)
        if cached is not None and cached[0] is node_mask:
            return cached[1]
        n = node_mask.shape[1]
        bs = node_mask.shape[0]
        inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag_mask = torch.eye(n, device=node_mask.device).unsqueeze(0).expand(bs, -1, -1)
        masks = (inverse_edge_mask, diag_mask)
        self._sampling_masks_cache = (node_mask, masks)
        return masks

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = True,
        forced_condition=None,
        force_domain_mask: bool = False,
        sample_steps: int = None,
        generator=None,
        generator_cpu=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :param sample_steps: int (optional) number of reverse-diffusion steps;
            defaults to cfg.sample.sample_steps if None
        :param generator_cpu: optional CPU-device torch.Generator, used only for the initial
            noise draw (sample_discrete_feature_noise consumes the CPU-resident limit_dist
            tensors). See `generator`'s docstring above and sample_per_domain's
            `sample.rng_safe_per_domain`.
        :param generator: optional torch.Generator. None (default) draws from the default
            global RNG stream, byte-identical to pre-existing behavior. Set by
            sample_per_domain when `sample.rng_safe_per_domain=true` -- see that flag's
            docstring for the reproducibility issue this fixes.
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        # Per-batch cache must never carry across sample_batch() calls (see
        # _get_constant_sampling_masks_cached for the leak an id-keyed carry-over caused).
        self._sampling_masks_cache = None
        _sample_steps = (
            int(sample_steps) if sample_steps is not None else int(self.cfg.sample.sample_steps)
        )
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        zT_domain_idx = None
        if forced_condition is not None and forced_condition.numel() > 0:
            try:
                zT_domain_idx = int(torch.argmax(forced_condition[0]).item())
            except Exception:
                zT_domain_idx = None
        elif (
            self.prior_mode == "domain_specific"
            and self.conditional
            and getattr(self.cfg.general, "conditional_domain", False)
        ):
            n_domains = max(len(self._get_domain_to_idx()), 1)
            zT_domain_idx = int(torch.randint(low=0, high=n_domains, size=(1,), device=self.device).item())
        zT_limit_dist = self._get_limit_dist_checked(
            domain_idx=zT_domain_idx,
            context="sample_batch_zT",
        )
        # torch.Generator is device-locked to the tensor it draws from; limit_dist.X/E is
        # CPU-resident in this codebase's current setup, but pick whichever of the two
        # available generators actually matches the target tensor's device (rather than
        # assuming CPU) so this stays correct if that ever changes. Neither matching falls
        # back to the default global stream (None) rather than crashing.
        _zT_target_device = zT_limit_dist.X.device.type
        if generator_cpu is not None and generator_cpu.device.type == _zT_target_device:
            _zT_generator = generator_cpu
        elif generator is not None and generator.device.type == _zT_target_device:
            _zT_generator = generator
        else:
            _zT_generator = None
        z_T = interpolant_utils.sample_discrete_feature_noise(
            limit_dist=zT_limit_dist,
            node_mask=node_mask,
            generator=_zT_generator,
        )
        if self.conditional:
            if forced_condition is not None:
                z_T.y = forced_condition
            elif getattr(self.cfg.general, "conditional_domain", False):
                domain_names = [
                    name.strip().lower()
                    for name in str(self.cfg.dataset.name).split(",")
                    if name.strip()
                ]
                n_domains = len(domain_names)
                condition = torch.zeros(batch_size, n_domains, device=self.device)
                row_idx = torch.arange(batch_size, device=self.device)
                if zT_domain_idx is not None:
                    assigned = torch.full_like(row_idx, int(zT_domain_idx) % max(n_domains, 1))
                else:
                    assigned = (row_idx + batch_id) % max(n_domains, 1)
                condition[row_idx, assigned] = 1.0
                z_T.y = condition
            else:
                raise NotImplementedError(
                    "conditional sampling without domain conditioning needs a forced_condition"
                )
        X, E, y = z_T.X, z_T.E, z_T.y

        # Trajectory-seed decoupling (sample.trajectory_seed_offset). Deliberately placed
        # AFTER the z_T draw above and BEFORE the reverse loop below, so z_T is unchanged
        # while every subsequent trajectory draw comes from a fresh stream. Left as None
        # (the default) nothing is reseeded and behavior is byte-identical.
        _traj_off = getattr(self.cfg.sample, "trajectory_seed_offset", None)
        if _traj_off is not None and generator is not None:
            _traj_seed = self._trajectory_seed(
                getattr(self, "_z0_current_domain_name", None),
                getattr(self, "_z0_current_batch_idx", 0),
                _traj_off,
            )
            generator.manual_seed(int(_traj_seed))
            if generator_cpu is not None:
                generator_cpu.manual_seed(int(_traj_seed))

        mask_condition = forced_condition if forced_condition is not None else y

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Pre-compute undistorted time sequences and write indices outside the loop
        _T = _sample_steps
        _t_base = torch.arange(_T, dtype=y.dtype, device=y.device).unsqueeze(1) / _T  # (T,1)
        _s_base = _t_base + (1.0 / _T)
        write_index_seq = [
            min(int(np.ceil(np.round(float(_s_base[i]) / chain_time_unit, 6))), number_chain_steps)
            for i in range(_T)
        ]

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        _last_step_idx = _sample_steps - 1
        for t_int in tqdm(range(0, _sample_steps)):
            # this state — use pre-allocated base tensors (views, no alloc)
            t_array = _t_base[t_int].expand(batch_size, 1)
            t_norm = t_array
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = _s_base[t_int].expand(batch_size, 1)
            s_norm = s_array

            # using round for precision (pre-computed)
            write_index = write_index_seq[t_int]

            # Distort time
            t_norm = self.time_distorter.sample_ft(t_norm, self.cfg.sample.time_distortion)
            s_norm = self.time_distorter.sample_ft(s_norm, self.cfg.sample.time_distortion)

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm, s_norm, X, E, y, node_mask,
                force_domain_mask=force_domain_mask,
                domain_mask_condition=mask_condition,
                last_step=(t_int == _last_step_idx),
                generator=generator,
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs. Move each tensor to CPU once, then slice on CPU:
        # per-graph .cpu() calls each force a GPU->CPU sync (B syncs per batch).
        X_cpu = X.cpu()
        E_cpu = E.cpu()
        y_cpu = y.cpu() if torch.is_tensor(y) else y
        n_nodes_cpu = n_nodes.cpu() if torch.is_tensor(n_nodes) else n_nodes
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = int(n_nodes_cpu[i])
            atom_types = X_cpu[i, :n].clone()
            edge_types = E_cpu[i, :n, :n].clone()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y_cpu[i].clone())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            #current_path = os.getcwd()
            current_path = self.cfg.hier_dir
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times[:, i].numpy(),
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            #current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def _bridge_posterior_probs(self, pi_X, pi_E, X_t, E_t, t, s):
        """Exact finite-step reverse posterior for the linear-interpolation path.

        Closed-form pinned bridge of the interpolation forward p_t(x_t|x1)=t*x1+(1-t)*prior.
        Per dimension, with current state i=argmax(x_t) and predicted clean distribution
        pi=p(x1|x_t):

            p(x_s = j | x_t = i) = a * pi_j + (1 - a) * delta_{j,i},
            a = (s - t) / (1 - t).

        With probability a the dimension "resolves" to the predicted clean distribution;
        otherwise it keeps its current value. At s=1 (a=1) it reduces to prob=pi. X_t / E_t
        are one-hot, so they are exactly the delta_{j,i} terms.
        """
        t_col = t.reshape(-1).to(pi_X.dtype)   # (B,)
        s_col = s.reshape(-1).to(pi_X.dtype)   # (B,)
        a = ((s_col - t_col) / (1.0 - t_col).clamp(min=1e-6)).clamp(0.0, 1.0)
        a_x = a.view(-1, 1, 1)
        a_e = a.view(-1, 1, 1, 1)
        prob_X = a_x * pi_X + (1.0 - a_x) * X_t.to(pi_X.dtype)
        prob_E = a_e * pi_E + (1.0 - a_e) * E_t.to(pi_E.dtype)
        return prob_X, prob_E

    def _posterior_guided_pi(
        self,
        logits_X_cond,
        logits_E_cond,
        logits_X_uncond,
        logits_E_uncond,
        pred_X,
        pred_E,
        y_t,
        node_mask,
        posterior_cfg,
    ):
        """Predicted-x1 distribution pi for the bridge posterior, with logit-level
        classifier-free guidance:
            guided = uncond + w * (cond - uncond);  pi = softmax(guided),
        where w is the per-sample `_domain_guidance_weight`. When CFG is off (or the model is
        unconditional) pi is softmax(cond logits) == pred_X / pred_E, reused directly.
        """
        do_cfg = bool(
            self.conditional and posterior_cfg and logits_X_uncond is not None
        )
        if not do_cfg:
            return pred_X, pred_E
        gw = self._domain_guidance_weight(y_t).to(logits_X_cond.device)
        gx = gw.view(-1, 1, 1)
        ge = gw.view(-1, 1, 1, 1)
        pi_X = F.softmax(logits_X_uncond + gx * (logits_X_cond - logits_X_uncond), dim=-1)
        pi_E = F.softmax(logits_E_uncond + ge * (logits_E_cond - logits_E_uncond), dim=-1)
        return pi_X, pi_E
    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        force_domain_mask: bool = False,
        domain_mask_condition=None,
        last_step=None,
        generator=None,
    ):
        """Samples from zs ~ p(zs | zt) with the posterior-bridge reverse sampler.

        The prediction logits are hard-masked to each domain's vocabulary V_d, classifier-free
        guidance is applied at the logit level (`_posterior_guided_pi`), and the step
        distribution is the closed-form interpolation-path posterior (`_bridge_posterior_probs`).
        On the last step the graph is drawn from the conditional prediction.

        `generator` (optional torch.Generator): None draws from the default global RNG stream.
        See sample_per_domain's `sample.rng_safe_per_domain` docstring.
        """
        bs, n, dx = X_t.shape
        _posterior_cfg = bool(getattr(self.cfg.sample, "posterior_cfg", True))

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        if self.conditional:
            # Fuse cond + uncond into one doubled-batch forward pass:
            # graph state (X_t, E_t) is identical for both; only y_t differs.
            uncond_y = self._build_uncond_y(y_t)
            noisy_data_2x = {
                "X_t":       torch.cat([X_t, X_t], dim=0),
                "E_t":       torch.cat([E_t, E_t], dim=0),
                "y_t":       torch.cat([y_t, uncond_y], dim=0),
                "t":         torch.cat([t, t], dim=0),
                "node_mask": torch.cat([node_mask, node_mask], dim=0),
            }
            node_mask_2x = torch.cat([node_mask, node_mask], dim=0)
            # The extra features depend only on X_t/E_t/node_mask/t, which are identical in
            # both halves, so compute them once on the un-doubled batch and duplicate.
            _extra_data_1x = self.compute_extra_data(noisy_data)
            extra_data_2x = utils.PlaceHolder(
                X=torch.cat([_extra_data_1x.X, _extra_data_1x.X], dim=0),
                E=torch.cat([_extra_data_1x.E, _extra_data_1x.E], dim=0),
                y=torch.cat([_extra_data_1x.y, _extra_data_1x.y], dim=0),
            )
            pred_2x = self.forward(noisy_data_2x, extra_data_2x, node_mask_2x)
            pred_X = F.softmax(pred_2x.X[:bs], dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred_2x.E[:bs], dim=-1)  # bs, n, n, d0
            # Pre-softmax logits for the logit-level CFG.
            logits_X_cond, logits_E_cond = pred_2x.X[:bs], pred_2x.E[:bs]
            logits_X_uncond, logits_E_uncond = pred_2x.X[bs:], pred_2x.E[bs:]
        else:
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
            logits_X_cond, logits_E_cond = pred.X, pred.E
            logits_X_uncond = logits_E_uncond = None

        # Hard-mask the prediction logits to the domain vocabulary V_d, mirroring the training loss
        # mask, so the categorical draw can never emit an invalid class. Both CFG halves are masked
        # to the SAME domain's V_d, which keeps the cond/uncond difference purely semantic.
        if y_t is not None and y_t.numel() > 0 and y_t.dim() == 2:
            _vm_dom = y_t.argmax(-1)
            logits_X_cond, logits_E_cond = self._apply_domain_vocab_mask(
                logits_X_cond, logits_E_cond, _vm_dom
            )
            pred_X = F.softmax(logits_X_cond, dim=-1)
            pred_E = F.softmax(logits_E_cond, dim=-1)
            if logits_X_uncond is not None:
                logits_X_uncond, logits_E_uncond = self._apply_domain_vocab_mask(
                    logits_X_uncond, logits_E_uncond, _vm_dom
                )

        pi_X, pi_E = self._posterior_guided_pi(
            logits_X_cond, logits_E_cond, logits_X_uncond, logits_E_uncond,
            pred_X, pred_E, y_t, node_mask, _posterior_cfg,
        )
        prob_X, prob_E = self._bridge_posterior_probs(pi_X, pi_E, X_t, E_t, t, s)

        # Host-side last-step flag from the sampling loop; the tensor comparison
        # fallback (s[0] == 1.0) forces a GPU sync every step.
        _is_last_step = bool(last_step) if last_step is not None else bool(s[0] == 1.0)
        if _is_last_step:
            prob_X, prob_E = pred_X, pred_E

        _inv_edge_mask, _diag_mask = self._get_constant_sampling_masks_cached(node_mask)
        sampled_s = interpolant_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask, generator=generator,
            inverse_edge_mask=_inv_edge_mask, diag_mask=_diag_mask,
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(self.limit_dist.X)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(self.limit_dist.E)).float()

        # Symmetry check syncs the host every denoising step; gated off by default.
        if hotpath_asserts_enabled():
            assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""
        extra_features = self.extra_features(noisy_data)
        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        # Build a deterministic feature pool independent of conditional flags.
        # Then resolve final extra dims from the model contract:
        # input_dims[channel] = noisy_channel_dim + extra_channel_dim.
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)
        pool_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        pool_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        pool_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        target_extra_x = int(self.input_dims["X"]) - int(noisy_data["X_t"].size(-1))
        target_extra_e = int(self.input_dims["E"]) - int(noisy_data["E_t"].size(-1))
        # y gets +t appended below, so reserve one slot for time.
        # For direct domain embedding, _inject_domain_embedding() expands y_t from n_raw → emb_dim
        # inside forward() — that expansion hasn't happened yet here, so subtract it.
        _dom_exp = self._direct_domain_emb_expansion(int(noisy_data["y_t"].size(-1)))
        _struct_exp = self.structural_emb_dim if self.structural_encoder_enabled else 0
        target_extra_y_no_time = int(self.input_dims["y"]) - int(noisy_data["y_t"].size(-1)) - 1 - _dom_exp - _struct_exp

        if target_extra_x < 0 or target_extra_e < 0 or target_extra_y_no_time < 0:
            raise RuntimeError(
                "Invalid extra-feature contract from input_dims: "
                f"target_extra_x={target_extra_x}, target_extra_e={target_extra_e}, "
                f"target_extra_y_no_time={target_extra_y_no_time}, "
                f"input_dims={dict(self.input_dims)}, "
                f"noisy_dims={{'X': {int(noisy_data['X_t'].size(-1))}, 'E': {int(noisy_data['E_t'].size(-1))}, 'y': {int(noisy_data['y_t'].size(-1))}}}"
            )

        if pool_X.size(-1) < target_extra_x or pool_E.size(-1) < target_extra_e or pool_y.size(-1) < target_extra_y_no_time:
            raise RuntimeError(
                "Feature providers do not expose enough channels for configured input contract: "
                f"available=(X:{int(pool_X.size(-1))}, E:{int(pool_E.size(-1))}, y:{int(pool_y.size(-1))}), "
                f"required=(X:{target_extra_x}, E:{target_extra_e}, y:{target_extra_y_no_time}), "
                f"conditional={getattr(self.cfg.general, 'conditional', None)}, "
                f"conditional_domain={getattr(self.cfg.general, 'conditional_domain', None)}"
            )

        extra_X = pool_X[..., :target_extra_x]
        extra_E = pool_E[..., :target_extra_e]
        extra_y = pool_y[..., :target_extra_y_no_time]

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        # Final safety: ensure concatenation contract exactly matches model input dims.
        if int(noisy_data["X_t"].size(-1) + extra_X.size(-1)) != int(self.input_dims["X"]):
            raise RuntimeError("X contract mismatch after extra feature assembly")
        if int(noisy_data["E_t"].size(-1) + extra_E.size(-1)) != int(self.input_dims["E"]):
            raise RuntimeError("E contract mismatch after extra feature assembly")
        if int(noisy_data["y_t"].size(-1) + _dom_exp + _struct_exp + extra_y.size(-1)) != int(self.input_dims["y"]):
            raise RuntimeError("y contract mismatch after extra feature assembly")

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # === Wrapper delegation layer ===
    def _domain_name_to_index(self):
        return self._get_domain_to_idx()

    def _to_scalar_metric(self, value):
        return to_scalar_metric(value)

    def _extract_main_accuracy(self, metrics_dict):
        return extract_main_accuracy(metrics_dict)

    def _assert_model_input_dim_invariants(self, *args, **kwargs):
        return invariant_helpers._assert_model_input_dim_invariants(self, *args, **kwargs)

    def _progressive_readout_config(self, *args, **kwargs):
        return readout_helpers._progressive_readout_config(self, *args, **kwargs)

    def _compute_progressive_readout_deep_supervision_loss(self, *args, **kwargs):
        return readout_helpers._compute_progressive_readout_deep_supervision_loss(
            self, *args, **kwargs
        )

    def _apply_domain_vocab_mask(self, *args, **kwargs):
        return domain_helpers._apply_domain_vocab_mask(self, *args, **kwargs)

    def _augment_accuracy_summaries(self, *args, **kwargs):
        return eval_summary_helpers._augment_accuracy_summaries(self, *args, **kwargs)
