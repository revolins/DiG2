import warnings

import torch


def _domain_semantic_allowed_sets(self):
    """Return (atom_allowed, edge_allowed) dicts for semantic_global_v1 mode.

    Primary path: uses valid_x/e_mask from trainer.datamodule.global_vocab.
    Fallback path: uses static _SEMANTIC_NODE_MAPS_BASE + SEMANTIC_EDGE_MAPS when masks
    are unavailable (stale cache, no trainer, standalone inference).
    Returns (None, None) when not in semantic_global_v1 mode.
    Returned dicts map domain_idx -> list of valid GLOBAL class indices.
    """
    try:
        # LightningModule.trainer is a @property that raises RuntimeError (not
        # AttributeError) when the module isn't attached to a Trainer -- plain
        # getattr(self, "trainer", None) does NOT catch that (getattr only swallows
        # AttributeError), so a standalone/no-Trainer caller (e.g.
        # train_joint_mae_pretrain.py, which deliberately never calls trainer.fit())
        # would otherwise crash here instead of reaching the documented static
        # fallback below.
        dm = getattr(self.trainer, "datamodule", None)
    except RuntimeError:
        dm = None
    gv = getattr(dm, "global_vocab", None)

    _is_semantic = isinstance(gv, dict) and gv.get("feature_schema") == "semantic_global_v1"

    if _is_semantic:
        valid_x = gv.get("valid_x_mask")
        valid_e = gv.get("valid_e_mask")
        if valid_x is not None and valid_e is not None:
            atom_allowed, edge_allowed = {}, {}
            for name, didx in self._domain_name_to_index().items():
                d = int(didx)
                if d < valid_x.size(0):
                    ax = torch.where(valid_x[d])[0].tolist()
                    if ax:
                        atom_allowed[d] = ax
                if d < valid_e.size(0):
                    ae = torch.where(valid_e[d])[0].tolist()
                    if ae:
                        edge_allowed[d] = ae
            return atom_allowed, edge_allowed
        # masks absent (stale cache) — fall through to static fallback
    else:
        # Check config directly (handles stale cache where global_vocab is None)
        ds_cfg = getattr(getattr(self, "cfg", None), "dataset", None)
        u6_cfg = getattr(ds_cfg, "unified", None)
        if getattr(u6_cfg, "feature_schema", None) != "semantic_global_v1":
            return None, None
        # semantic mode confirmed via config — fall through to static fallback

    # Static fallback: compute from constants, always correct, cache-independent.
    # Mirrors the logic in _build_global_vocab_semantic() without needing the cache.
    try:
        from dig2.data.unified.dataset import _SEMANTIC_NODE_MAPS_BASE, SEMANTIC_EDGE_MAPS
    except ImportError:
        return None, None

    ds_cfg = getattr(getattr(self, "cfg", None), "dataset", None)
    _remove_h = bool(getattr(ds_cfg, "remove_h", False)) if ds_cfg is not None else False
    _node_maps = {**_SEMANTIC_NODE_MAPS_BASE, "qm9": [1, 2, 3, 4, 5] if not _remove_h else [2, 3, 4, 5]}

    atom_allowed, edge_allowed = {}, {}
    for name, didx in self._domain_name_to_index().items():
        d = int(didx)
        key = str(name).strip().lower()
        nm = _node_maps.get(key)
        em = SEMANTIC_EDGE_MAPS.get(key)
        if nm:
            atom_allowed[d] = list(nm)
        if em:
            edge_allowed[d] = list(em)

    return (atom_allowed, edge_allowed) if (atom_allowed or edge_allowed) else (None, None)


def _domain_vocab_valid_masks(self, Sx: int, Se: int, device=None):
    """Boolean per-domain valid-class masks ``V_d`` = the SEMANTIC declared set.

    Returns ``(V_x, V_e)`` with shapes ``[n_domains, Sx]`` / ``[n_domains, Se]`` where a
    ``True`` marks a class that is valid for that domain (Zinc allows I but not H, QM9 allows
    C/N/O/F only, structural = generic). Built once from ``_domain_semantic_allowed_sets`` and
    cached on ``self._Vd_mask_cache``. Returns ``(None, None)`` when not in semantic_global_v1
    mode (masking then no-ops, preserving legacy behavior).

    On first build it runs the ``observed ⊆ V_d`` guard: every class that actually occurs in a
    domain's data (``_train_domain_*_valid``, the marginal support) must be inside ``V_d``. A
    stale/incorrect semantic map (or a data artifact) trips it immediately.
    """
    cache = getattr(self, "_Vd_mask_cache", None)
    if cache is not None:
        Vx, Ve = cache
        if Vx is None:
            return None, None
        if int(Vx.size(-1)) == int(Sx) and int(Ve.size(-1)) == int(Se):
            if device is not None:
                # `_Vd_mask_cache` holds CPU tensors, so the old code paid a BLOCKING
                # pageable-host->device copy on every cache HIT. Measured at 8.0 syncs per
                # training step (5 call sites: the main loss + one per readout depth) and 4.0
                # per reverse sampling step -- the largest single training site after the
                # assert and masked_softmax fixes.
                # Memoize the device-resident copies in a SEPARATE dict so `_Vd_mask_cache`
                # keeps its exact contract, including the `(None, None)` non-semantic sentinel
                # checked above. Callers never mutate these (`_apply_domain_vocab_mask` does
                # `~Vx[dom]`, which allocates), so sharing one instance is safe.
                _key = (str(device), int(Sx), int(Se))
                _dc = getattr(self, "_Vd_mask_device_cache", None)
                if _dc is None:
                    _dc = {}
                    try:
                        self._Vd_mask_device_cache = _dc
                    except Exception:      # stub `self` objects in the smoke scripts
                        _dc = None
                if _dc is not None:
                    _hit = _dc.get(_key)
                    if _hit is not None:
                        return _hit
                Vx, Ve = Vx.to(device), Ve.to(device)
                if _dc is not None:
                    _dc[_key] = (Vx, Ve)
            return Vx, Ve
        # output dims changed unexpectedly — rebuild

    atom_allowed, edge_allowed = _domain_semantic_allowed_sets(self)
    if not atom_allowed and not edge_allowed:
        self._Vd_mask_cache = (None, None)
        try:
            self._Vd_mask_device_cache = {}
        except Exception:
            pass
        return None, None

    idx_map = self._domain_name_to_index()
    max_idx = max(int(v) for v in idx_map.values()) if idx_map else -1
    for d in list((atom_allowed or {}).keys()) + list((edge_allowed or {}).keys()):
        max_idx = max(max_idx, int(d))
    n_dom = max_idx + 1

    Vx = torch.zeros((n_dom, int(Sx)), dtype=torch.bool)
    Ve = torch.zeros((n_dom, int(Se)), dtype=torch.bool)
    for d, allowed in (atom_allowed or {}).items():
        idx = [c for c in allowed if 0 <= int(c) < int(Sx)]
        if idx:
            Vx[int(d), idx] = True
    for d, allowed in (edge_allowed or {}).items():
        idx = [c for c in allowed if 0 <= int(c) < int(Se)]
        if idx:
            Ve[int(d), idx] = True

    # Guard: any domain we will actually use must have a non-empty V_d row.
    for name, didx in idx_map.items():
        d = int(didx)
        if d < n_dom and not bool(Vx[d].any()) and not bool(Ve[d].any()):
            raise ValueError(
                f"V_d is empty for domain '{name}' (idx {d}); missing semantic map entry. "
                "Add it to _SEMANTIC_NODE_MAPS_BASE / SEMANTIC_EDGE_MAPS."
            )

    _assert_observed_subset_Vd(self, Vx, Ve)

    self._Vd_mask_cache = (Vx, Ve)
    # Drop stale device copies: a rebuild means the masks changed.
    try:
        self._Vd_mask_device_cache = {}
    except Exception:
        pass
    if device is not None:
        Vx, Ve = Vx.to(device), Ve.to(device)
    return Vx, Ve


def _assert_observed_subset_Vd(self, Vx, Ve):
    """Assert observed ⊆ V_d: every class occurring in a domain's data is in its semantic map.

    ``_train_domain_*_valid`` are the marginal (data-occurrence) supports built at init. A True
    there that is False in ``V_d`` means the data carries a class the semantic map forbids — a
    stale map or a data artifact. Skipped when the marginal buffers are unavailable.
    """
    obs_x = getattr(self, "_train_domain_x_valid", None)
    obs_e = getattr(self, "_train_domain_e_valid", None)
    for obs, V, chan in ((obs_x, Vx, "node"), (obs_e, Ve, "edge")):
        if obs is None or V is None:
            continue
        n = min(int(obs.size(0)), int(V.size(0)))
        c = min(int(obs.size(-1)), int(V.size(-1)))
        if n == 0 or c == 0:
            continue
        leaked = obs[:n, :c] & ~V[:n, :c].to(obs.device)
        if bool(leaked.any()):
            bad = torch.nonzero(leaked, as_tuple=False).tolist()
            raise ValueError(
                f"observed ⊈ V_d ({chan}): data contains class(es) outside the semantic map "
                f"at (domain_idx, class) {bad[:8]}{'...' if len(bad) > 8 else ''}. "
                "Fix the domain's semantic map or the data preprocessing."
            )


def _warn_vocab_mask_passthrough(self, kind: str, detail: str):
    """Announce the two silent passthrough returns in `_apply_domain_vocab_mask`.

    Those two `return logits_X, logits_E` branches disable domain vocabulary masking for the
    call, with no exception and no log line -- contradicting the docstring three lines above
    them, which claims the fallback "never silently disables masking". A schema/vocab-width
    drift would therefore switch masking off everywhere, and the only symptom would be a slow
    rise in out-of-domain predictions that reads as ordinary model weakness.

    `kind="absent"` is often LEGITIMATE (non-semantic vocab, train_domain_vocab_mask=false),
    so it is informational. `kind="shape_mismatch"` is a real config/vocab bug and is always
    worth a UserWarning: any nonzero count during a training epoch means invalid classes are
    reachable.

    Controlled by `general.vocab_mask.passthrough_warn` (off | once | always).
    """
    mode = str(getattr(self, "_vocab_mask_passthrough_warn", "off") or "off").lower()
    if mode == "off":
        return
    if mode == "once":
        seen = getattr(self, "_vocab_mask_passthrough_warned", None)
        if seen is None:
            seen = set()
            try:
                self._vocab_mask_passthrough_warned = seen
            except Exception:
                return
        if kind in seen:
            return
        seen.add(kind)
    msg = (
        f"[SILENT-FALLBACK] domain vocabulary masking was DISABLED for this call "
        f"({kind}): {detail}. Out-of-domain classes are reachable in the returned logits. "
        f"This is not auto-fixed here -- set general.vocab_mask.passthrough_warn='off' to "
        f"silence, or fix the vocab table so the mask applies."
    )
    if kind == "shape_mismatch":
        warnings.warn(msg, category=UserWarning, stacklevel=3)
    print(msg, flush=True)


def _apply_domain_vocab_mask(self, logits_X, logits_E, domain_ids, neg: float = -1e9):
    """Hard-mask out-of-`V_d` class logits per sample (value replacement via ``masked_fill``,
    so invalid classes get exactly zero gradient and zero sampling probability).

    ``logits_X``: ``[B, N, Sx]``, ``logits_E``: ``[B, N, N, Se]``, ``domain_ids``: ``[B]`` long.
    Returns NEW tensors (inputs untouched). Falls back to the per-domain marginal support when
    the semantic ``V_d`` is unavailable.
    """
    Vx, Ve = _domain_vocab_valid_masks(self, Sx=logits_X.size(-1), Se=logits_E.size(-1), device=logits_X.device)
    if Vx is None or Ve is None:
        # Non-semantic vocab: fall back to the marginal support (legacy behavior) so this is a
        # strict superset of the old mask and never silently disables masking.
        Vx = getattr(self, "_train_domain_x_valid", None)
        Ve = getattr(self, "_train_domain_e_valid", None)
        if Vx is None or Ve is None:
            _warn_vocab_mask_passthrough(
                self, "absent",
                "no semantic V_d and no marginal-support fallback table "
                "(_train_domain_x_valid/_train_domain_e_valid) is available",
            )
            return logits_X, logits_E
        if int(Vx.size(-1)) != int(logits_X.size(-1)) or int(Ve.size(-1)) != int(logits_E.size(-1)):
            _warn_vocab_mask_passthrough(
                self, "shape_mismatch",
                f"marginal-support table width does not match the output head "
                f"(X: mask={int(Vx.size(-1))} vs logits={int(logits_X.size(-1))}; "
                f"E: mask={int(Ve.size(-1))} vs logits={int(logits_E.size(-1))})",
            )
            return logits_X, logits_E
        Vx = Vx.to(logits_X.device)
        Ve = Ve.to(logits_E.device)
    dom = domain_ids.reshape(-1).long().to(Vx.device)
    inv_x = ~Vx[dom]  # [B, Sx]
    inv_e = ~Ve[dom]  # [B, Se]

    logits_X = logits_X.masked_fill(inv_x.unsqueeze(1), neg)                 # broadcast over N
    logits_E = logits_E.masked_fill(inv_e.unsqueeze(1).unsqueeze(1), neg)    # broadcast over N, N
    return logits_X, logits_E

