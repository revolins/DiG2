"""Anchored fine-tuning (``train.anchor_ft``).

Fine-tunes the SHARED edge pathway of a trained model toward the target domains while a
KL(frozen teacher || student) anchor on the protected domains' predictions pins their output
distribution to the starting model. Only the parameters matched by the selected scope are
trainable; the frozen teacher is a plain dict of tensors (not a submodule), so the state_dict
of a fine-tuned model is identical in structure to the base model's.
"""
import torch

# train.anchor_ft trainable scope: the EDGE pathway + the EDGE readout heads.
# Deliberately code constants, not config: a typo'd regex in a yaml would silently WIDEN the set of
# weights a "targeted" fine-tune is allowed to move, and the resulting run would look fine.
# sbm's failure mode is block structure and planar's is planarity -- both edge-placement decisions --
# so attention q/k/v/x_out (which build the edge update), the e_mul/e_add/e_out edge mixers, the edge
# FFN and the edge heads are in scope, while the X/y heads stay frozen (protecting atom/bond-type
# marginals by construction). With readout_layers [8, 10, 12] the progressive-readout edge heads are
# E_8/E_10; depth 12 is terminal and reads through mlp_out_E.
ANCHOR_FT_PATTERNS = (
    r"^model\.tf_layers\.\d+\.self_attn\.(q|k|v|x_out|e_mul|e_add|e_out)\.(weight|bias)$",
    r"^model\.tf_layers\.\d+\.(linE1|linE2)\.(weight|bias)$",
    r"^model\.mlp_out_E\.\d+\.(weight|bias)$",
    r"^model\.pr_extra_heads\.E_\d+\.\d+\.(weight|bias)$",
)

# Wider scopes, selected by NAME via train.anchor_ft.scope (an unknown name raises). Still code
# constants for the reason above. Every scope keeps the X/y output heads, the input MLPs, the
# structural encoder and the domain-conditioning tables frozen.
#  edge_node: + the per-layer node FFN. sbm's residual failure after the edge tune is block-size
#             imbalance with the right block COUNT -- a node-partition decision, not edge placement.
#  layers:    + every transformer-layer weight (attention incl. cond_*/y paths, all FFNs, norms).
ANCHOR_FT_SCOPES = {
    "edge": ANCHOR_FT_PATTERNS,
    "edge_node": ANCHOR_FT_PATTERNS + (
        r"^model\.tf_layers\.\d+\.(linX1|linX2)\.(weight|bias)$",
    ),
    "layers": (
        r"^model\.tf_layers\.\d+\..+$",
        r"^model\.mlp_out_E\.\d+\.(weight|bias)$",
        r"^model\.pr_extra_heads\.E_\d+\.\d+\.(weight|bias)$",
    ),
}


class AnchorFTMixin:
    """Mixed into the LightningModule; adds no parameters."""

    def _anchor_ft_setup(self, et_cfg, cfg) -> None:
        """Freeze everything except ANCHOR_FT_PATTERNS; assert every condition the study depends on."""
        import re as _re
        _scope = str(getattr(et_cfg, "scope", "edge") or "edge")
        if _scope not in ANCHOR_FT_SCOPES:
            raise ValueError(f"train.anchor_ft.scope={_scope!r} not in {sorted(ANCHOR_FT_SCOPES)}")
        self._anchor_ft_scope = _scope
        _pats = [_re.compile(x) for x in ANCHOR_FT_SCOPES[_scope]]
        for _p in self.parameters():
            _p.requires_grad_(False)
        for n, p in self.named_parameters():
            if any(rx.match(n) for rx in _pats):
                p.requires_grad_(True)
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        self._anchor_ft_param_names = [n for n, _ in named]
        self._anchor_ft_params = [p for _, p in named]
        self._anchor_ft_trainable_names = set(self._anchor_ft_param_names)
        n_tensors = len(named)
        n_params = sum(p.numel() for _, p in named)
        exp_t = getattr(et_cfg, "expect_trainable_tensors", None)
        exp_p = getattr(et_cfg, "expect_trainable_params", None)
        if exp_t is not None and int(exp_t) != n_tensors:
            raise RuntimeError(f"[anchor-FT] expected {int(exp_t)} trainable tensors, got {n_tensors}")
        if exp_p is not None and int(exp_p) != n_params:
            raise RuntimeError(f"[anchor-FT] expected {int(exp_p)} trainable params, got {n_params}")
        if n_tensors == 0:
            raise RuntimeError("[anchor-FT] no parameter matched ANCHOR_FT_PATTERNS")

        # Freezing the prototype EMA also freezes the SHARED centering statistic
        # (proto_running_mean): its only writer, _apply_proto_transform, is called from the EMA write
        # path, so a single PROTECTED-domain row in a batch would otherwise move the centering for
        # every domain -- drift with no gradient involved at all.
        self._freeze_prototypes = bool(getattr(et_cfg, "freeze_prototypes", True))
        if not self._freeze_prototypes:
            print("[anchor-FT] WARNING freeze_prototypes=false: domain_prototypes and the SHARED "
                  "proto_running_mean will drift for every domain", flush=True)

        self._anchor_ft_targets = [str(d).strip().lower()
                                   for d in (getattr(et_cfg, "target_domains", None) or [])]
        self._anchor_ft_protected = [str(d).strip().lower()
                                     for d in (getattr(et_cfg, "protected_domains", None) or [])]
        if not self._anchor_ft_targets or not self._anchor_ft_protected:
            raise ValueError("train.anchor_ft needs non-empty target_domains and protected_domains")
        _overlap = set(self._anchor_ft_targets) & set(self._anchor_ft_protected)
        if _overlap:
            raise ValueError(f"train.anchor_ft target/protected overlap: {sorted(_overlap)}")
        self._anchor_ft_lambda_E = float(getattr(et_cfg, "lambda_E", 5.0))
        self._anchor_ft_anchor_w = float(getattr(et_cfg, "anchor_kl_weight", 0.0) or 0.0)
        self._anchor_ft_channels = tuple(getattr(et_cfg, "anchor_channels", None) or ("X", "E"))
        self._anchor_ft_monitor_every = int(getattr(et_cfg, "monitor_every", 25) or 25)
        # Resume support. Lightning 2.5 restores WEIGHTS before on_fit_start but global_step only
        # after it, so a plain general.resume passes the step-0 guard and would snapshot the RESUMED
        # student as the "frozen" teacher. A resumed run must name the base checkpoint explicitly.
        self._anchor_ft_teacher_ckpt = getattr(et_cfg, "teacher_ckpt", None) or None
        if getattr(cfg.general, "resume", None) and not self._anchor_ft_teacher_ckpt:
            raise ValueError("train.anchor_ft with general.resume needs train.anchor_ft.teacher_ckpt "
                             "(the frozen base); otherwise the resumed student becomes the teacher")
        self._anchor_ft_enabled = True
        print(f"[anchor-FT] targets={self._anchor_ft_targets} protected={self._anchor_ft_protected} "
              f"scope={self._anchor_ft_scope} trainable tensors={n_tensors} params={n_params} "
              f"anchor_kl_weight={self._anchor_ft_anchor_w} "
              f"freeze_prototypes={self._freeze_prototypes}", flush=True)

    def _anchor_ft_domain_indices(self, which: str):
        d2i = self._get_domain_to_idx()
        names = self._anchor_ft_targets if which == "target" else self._anchor_ft_protected
        missing = [d for d in names if d not in d2i]
        if missing:
            raise KeyError(f"train.anchor_ft {which}_domains {missing} not in model domains "
                           f"{sorted(d2i)}")
        return [int(d2i[d]) for d in names]

    def _anchor_ft_row_mask(self, data, which: str):
        """[B] bool mask over graphs. Requires data.domain_idx (the TRUE domain); data.y is mutated
        in place by CFG dropout and would mislabel every nulled row."""
        from dig2.procedures.anchor_ft_losses import row_mask_for_domains
        if not hasattr(data, "domain_idx"):
            raise RuntimeError("train.anchor_ft requires data.domain_idx on every batch")
        return row_mask_for_domains(data.domain_idx, self._anchor_ft_domain_indices(which))

    def _anchor_ft_capture_teacher(self) -> None:
        """Snapshot the trainable tensors as the frozen teacher.

        Taken at FIT START, not in __init__: general.eval_resume loads the checkpoint into the model
        after construction (main.py), so an __init__ snapshot would capture the randomly initialised
        weights. Because only these tensors can ever move, they fully determine the frozen base.

        Stored as a plain dict attribute (~16 MB fp32), deliberately NOT a submodule and NOT buffers:
        a submodule would enter self.parameters() (and therefore AdamW, since configure_optimizers
        snapshots it) and state_dict() (doubling the checkpoint and breaking the strict resume load).
        """
        sd = dict(self.named_parameters())
        if self._anchor_ft_teacher_ckpt:
            # Explicit base: read the trainable tensors from that checkpoint, never from self.
            _ck = torch.load(str(self._anchor_ft_teacher_ckpt), map_location="cpu", weights_only=False)
            _base = _ck.get("state_dict", _ck)
            _missing = [n for n in self._anchor_ft_param_names if n not in _base]
            if _missing:
                raise RuntimeError(f"[anchor-FT] teacher_ckpt lacks {len(_missing)} trainable "
                                   f"tensors, e.g. {_missing[:3]}")
            self._anchor_ft_teacher = {
                n: _base[n].to(device=sd[n].device, dtype=sd[n].dtype).clone()
                for n in self._anchor_ft_param_names}
            print(f"[anchor-FT] teacher loaded from {self._anchor_ft_teacher_ckpt}", flush=True)
        else:
            if int(self.global_step) != 0:
                raise RuntimeError(f"[anchor-FT] teacher snapshot must happen at step 0, "
                                   f"got global_step={int(self.global_step)}")
            self._anchor_ft_teacher = {n: sd[n].detach().clone() for n in self._anchor_ft_param_names}
        print(f"[anchor-FT] teacher snapshot: {len(self._anchor_ft_teacher)} tensors, "
              f"{sum(t.numel() for t in self._anchor_ft_teacher.values()):,} params", flush=True)

    def _anchor_ft_want_teacher(self, step_idx: int) -> bool:
        if not self._anchor_ft_enabled or self._anchor_ft_teacher is None:
            return False
        if self._anchor_ft_anchor_w > 0.0:
            return True          # the anchor needs it every step
        every = max(1, self._anchor_ft_monitor_every)
        return (int(step_idx) % every) == 0     # monitor-only arms pay for it occasionally

    def _anchor_ft_teacher_forward(self, noisy_data, extra_data, node_mask):
        """Frozen-base prediction on the SAME inputs, with no RNG side effects.

        `functional_call` substitutes the teacher tensors for one call and mutates nothing -- swapping
        `p.data` instead would rebind storage that autograd's saved tensors alias, silently corrupting
        the student's backward.

        Called BEFORE the student forward and wrapped in fork_rng: the extra forward consumes dropout
        draws, and without restoring the stream the student would see different masks, so teacher and
        student would differ at step 0 even with identical weights.
        """
        from torch.func import functional_call
        _devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=_devs, enabled=True), torch.no_grad():
            return functional_call(self, self._anchor_ft_teacher,
                                   (noisy_data, extra_data, node_mask))

    def _anchor_ft_terms(self, *, loss, data, node_mask, X, E, student_X, student_E,
                         teacher_pred, step_idx):
        """B2's anchor term and the loss-space drift monitor.

        Every log key is emitted on EVERY step regardless of batch composition: domains are drawn
        i.i.d. per graph, so a batch can legitimately contain no protected (or no target) row, and a
        conditionally-emitted key mismatches the metric reduction (the same invariant the empty-edge
        skip documents).
        """
        from dig2.procedures.anchor_ft_losses import (masked_row_argmax_disagreement, masked_row_ce, masked_row_kl,)
        prot = self._anchor_ft_row_mask(data, "protected")
        tgt = self._anchor_ft_row_mask(data, "target")
        lam = self._anchor_ft_lambda_E

        # No target row in this batch -> drop the task loss entirely.
        # MEASURED SUBTLETY this guards: domain_loss_weight suppresses the protected domains through a
        # weighted MEAN, (ce*w).sum()/w.sum(), which is SCALE-INVARIANT. In a mixed batch the protected
        # rows' 1e-8 weights are negligible beside the targets' 1.0 and they drop out as intended, but
        # in an all-protected batch the uniform tiny weights cancel and the protected domains would
        # train at FULL strength -- the exact opposite of the intent. Domains are drawn i.i.d. per
        # graph, so at 60/40 target/protected and ~8 graphs/batch this is ~7e-4 of batches (~17 of
        # 25k steps): rare, real, and silent. The anchor below is still applied, so the step is not
        # wasted; it becomes a pure anchoring step.
        _no_tgt = not bool(tgt.any())
        if _no_tgt:
            loss = loss * 0.0
        self.log("train/anchor_ft_no_target_batch", 1.0 if _no_tgt else 0.0, on_step=True,
                 on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False)
        t_X = t_E = None
        if teacher_pred is not None:
            # Same V_d mask the student's loss logits got, so the KL never places gradient on classes
            # the domain does not have.
            if (getattr(self.cfg.general, "train_domain_vocab_mask", False)
                    and hasattr(data, "domain_idx")):
                t_X, t_E = self._apply_domain_vocab_mask(
                    teacher_pred.X, teacher_pred.E, data.domain_idx)
            else:
                t_X, t_E = teacher_pred.X, teacher_pred.E

        # ---- B2: anchor the protected rows to the frozen base
        anchor_kl = student_X.sum() * 0.0
        if self._anchor_ft_anchor_w > 0.0 and t_X is not None:
            anchor_kl, _kx, _ke = masked_row_kl(
                student_X, student_E, t_X, t_E, node_mask, prot,
                lambda_E=lam, channels=self._anchor_ft_channels)
            loss = loss + self._anchor_ft_anchor_w * anchor_kl
        self.log("train/anchor_ft_anchor_kl", anchor_kl.detach(), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, rank_zero_only=False)
        # ---- drift monitor (no sampling; in-run validation metrics use the buggy sbm instrument)
        if t_X is not None and (int(step_idx) % max(1, self._anchor_ft_monitor_every)) == 0:
            with torch.no_grad():
                _s_ce, _, _ = masked_row_ce(student_X, student_E, X, E, node_mask, prot, lambda_E=lam)
                _t_ce, _, _ = masked_row_ce(t_X, t_E, X, E, node_mask, prot, lambda_E=lam)
                _kl, _kx, _ke = masked_row_kl(student_X, student_E, t_X, t_E, node_mask, prot,
                                              lambda_E=lam, channels=("X", "E"))
                _dis = masked_row_argmax_disagreement(student_E, t_E, node_mask, prot)
                _tg_s, _, _ = masked_row_ce(student_X, student_E, X, E, node_mask, tgt, lambda_E=lam)
                _tg_t, _, _ = masked_row_ce(t_X, t_E, X, E, node_mask, tgt, lambda_E=lam)
                _logs = {
                    "drift/prot_ce_student": _s_ce, "drift/prot_ce_teacher": _t_ce,
                    "drift/prot_ce_delta": _s_ce - _t_ce, "drift/prot_kl_X": _kx,
                    "drift/prot_kl_E": _ke, "drift/prot_disagree_E": _dis,
                    "drift/tgt_ce_student": _tg_s, "drift/tgt_ce_delta": _tg_s - _tg_t,
                }
                for _d in self._anchor_ft_protected:
                    _m = self._anchor_ft_row_mask_one(data, _d)
                    _a, _, _ = masked_row_ce(student_X, student_E, X, E, node_mask, _m, lambda_E=lam)
                    _b, _, _ = masked_row_ce(t_X, t_E, X, E, node_mask, _m, lambda_E=lam)
                    _logs[f"drift/{_d}/ce_delta"] = _a - _b
                # on_step too: main.py owns Trainer(log_every_n_steps=50), so epoch-only logging of a
                # 250-step epoch yields ONE point and no drift trajectory to read a tripwire off.
                for _k, _v in _logs.items():
                    self.log(_k, _v.detach() if torch.is_tensor(_v) else _v, on_step=True,
                             on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False)
        return loss

    def _anchor_ft_row_mask_one(self, data, domain_name: str):
        from dig2.procedures.anchor_ft_losses import row_mask_for_domains
        d2i = self._get_domain_to_idx()
        return row_mask_for_domains(data.domain_idx, [int(d2i[str(domain_name).strip().lower()])])
