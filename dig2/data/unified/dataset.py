from dig2 import paths
import gc
import json
import os
import pathlib
import time
from collections.abc import Iterable
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from dig2.data.qm9 import QM9Dataset, RemoveYTransform
from dig2.data.spectre import SpectreGraphDataset
from dig2.data import zinc as zinc_dataset


# ---------------------------------------------------------------------------
# Semantic global feature vocabulary (schema: semantic_global_v1)
# ---------------------------------------------------------------------------

# Global node token → chemical/structural meaning
SEMANTIC_NODE_VOCAB = {
    "generic": 0,  # structural node with no chemical identity (synthetic domains)
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "P": 6,
    "S": 7,
    "Cl": 8,
    "Br": 9,
    "I": 10,
}
SEMANTIC_NODE_VOCAB_SIZE = 11  # Sx

# Global edge token → bond/connectivity meaning
# Index 0 must remain no_edge to stay compatible with encode_no_edge()
SEMANTIC_EDGE_VOCAB = {
    "no_edge": 0,   # absent edge — set by encode_no_edge in dense conversion
    "generic": 1,   # structural edge, no chemical bond type (synthetic domains)
    "SINGLE": 2,
    "DOUBLE": 3,
    "TRIPLE": 4,
}
SEMANTIC_EDGE_VOCAB_SIZE = 5  # Se
# To add aromatic support: extend to Se=6 with AROMATIC:5, bump schema_version, and extend qm9 edge map to [0,2,3,4,5].

# Per-domain node maps: local argmax index → global semantic index.
# Align with each dataset's atom_decoder order.
# QM9 is remove_h-dependent and is built at runtime in _build_global_vocab_semantic().
_SEMANTIC_NODE_MAPS_BASE: Dict[str, List[int]] = {
    "sbm":       [0],                          # single type: generic
    "comm20":    [0],
    "planar":    [0],
    "tree":      [0],
    "zinc":      [2, 3, 4, 5, 6, 7, 8, 9, 10],  # C, N, O, F, P, S, Cl, Br, I
}

# Per-domain edge maps: local argmax index → global semantic index.
# Molecular: {no_bond:0, SINGLE:1, DOUBLE:2, TRIPLE:3} → {no_edge:0, SINGLE:2, DOUBLE:3, TRIPLE:4}
# Synthetic: {no_edge:0, edge:1} → {no_edge:0, generic:1}
SEMANTIC_EDGE_MAPS: Dict[str, List[int]] = {
    "sbm":       [0, 1],
    "comm20":    [0, 1],
    "planar":    [0, 1],
    "tree":      [0, 1],
    "zinc":      [0, 2, 3, 4],
    "qm9":       [0, 2, 3, 4],
}

# ---------------------------------------------------------------------------


class _DomainSubset(Dataset):
    def __init__(self, parent: Dataset, indices: List[int]):
        self.parent = parent
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


class UnifiedDataset(Dataset):
    """RAM-backed unified dataset for SBM, Comm20, Zinc, QM9, Planar, Tree.

    Canonical split source: JSONL manifests under
      {root}/manifests/{train|val|test}.jsonl

    If a manifest is missing, it is auto-generated once from the legacy per-domain
    datasets, then reused for subsequent runs.
    """

    CANONICAL_DOMAINS = ["sbm", "comm20", "zinc", "qm9", "planar", "tree"]
    SPECTRE_DOMAINS = {"sbm", "comm20", "planar", "tree"}

    @staticmethod
    def _domain_condition_tag(conditional_domain: bool) -> str:
        return "dcond_t" if bool(conditional_domain) else "dcond_f"

    @classmethod
    def _parse_domain_tokens(cls, raw_value) -> List[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            parts = raw_value.split(",")
        elif isinstance(raw_value, Iterable):
            parts = list(raw_value)
        else:
            parts = [raw_value]

        out: List[str] = []
        seen = set()
        for p in parts:
            name = str(p).strip().lower()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    @classmethod
    def _resolve_active_domains(cls, cfg) -> List[str]:
        ds_cfg = getattr(cfg, "dataset", None)
        unified_cfg = getattr(ds_cfg, "unified", None)

        requested = cls._parse_domain_tokens(getattr(ds_cfg, "name", None))
        if len(requested) == 0:
            requested = list(cls.CANONICAL_DOMAINS)

        unknown = [d for d in requested if d not in cls.CANONICAL_DOMAINS]
        if unknown:
            raise ValueError(
                f"Unsupported unified domain(s): {unknown}. "
                f"Supported domains: {cls.CANONICAL_DOMAINS}"
            )

        order_cfg = cls._parse_domain_tokens(
            getattr(unified_cfg, "domain_order", None) if unified_cfg is not None else None
        )
        if len(order_cfg) > 0:
            order_unknown = [d for d in order_cfg if d not in cls.CANONICAL_DOMAINS]
            if order_unknown:
                raise ValueError(
                    "dataset.unified.domain_order includes unknown domains: "
                    f"{order_unknown}"
                )
            requested_set = set(requested)
            ordered = [d for d in order_cfg if d in requested_set]
            missing = [d for d in requested if d not in set(ordered)]
            if missing:
                raise ValueError(
                    "dataset.unified.domain_order must include all selected "
                    f"dataset.name domains. Missing: {missing}"
                )
            return ordered

        # Default to canonical-stable ordering filtered by requested set.
        requested_set = set(requested)
        return [d for d in cls.CANONICAL_DOMAINS if d in requested_set]

    @classmethod
    def _domain_tag(cls, domain_names: List[str]) -> str:
        if not domain_names:
            return "none"
        return "-".join(domain_names)

    @classmethod
    def _resolve_unified_cache_cfg(cls, cfg):
        ds_cfg = getattr(cfg, "dataset", None)
        unified_cfg = getattr(ds_cfg, "unified", None)
        cache_mode = str(getattr(unified_cfg, "cache_mode", "ram")).strip().lower()
        if cache_mode != "ram":
            raise ValueError(f"Unsupported unified cache_mode '{cache_mode}'. Use 'ram'.")
        feature_schema = str(getattr(unified_cfg, "feature_schema", "padded_onehot_v1")).strip().lower()
        if feature_schema != "semantic_global_v1":
            raise ValueError(
                f"Unsupported unified feature_schema '{feature_schema}'. Use 'semantic_global_v1'."
            )
        keep_legacy_onehot_fields = bool(getattr(unified_cfg, "keep_legacy_onehot_fields", True))
        return {
            "cache_mode": cache_mode,
            "schema_version": int(getattr(unified_cfg, "schema_version", 1)),
            "remove_h": bool(getattr(ds_cfg, "remove_h", False)),
            "aromatic": bool(getattr(ds_cfg, "aromatic", False)),
            "verbose": bool(getattr(unified_cfg, "verbose", True)),
            "log_memory": bool(getattr(unified_cfg, "log_memory", False)),
            "feature_schema": feature_schema,
            "keep_legacy_onehot_fields": keep_legacy_onehot_fields,
        }

    @staticmethod
    def _try_rss_gb():
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux ru_maxrss is KB
            return float(rss_kb) / (1024.0 * 1024.0)
        except Exception:
            return None

    @classmethod
    def _log(cls, enabled: bool, msg: str):
        if enabled:
            print(f"[unified] {msg}", flush=True)

    @classmethod
    def _log_memory(cls, enabled: bool, prefix: str):
        if not enabled:
            return
        rss_gb = cls._try_rss_gb()
        if rss_gb is not None:
            print(f"[unified][mem] {prefix}: rss_max_gb={rss_gb:.3f}", flush=True)

    @staticmethod
    def _root_overrides(unified_cfg) -> dict:
        """dataset.unified.domain_root_overrides -> {domain: realpath}. Empty by default.

        Points one domain's raw source at another directory (e.g. a freshly generated
        train pool for the adapter fine-tunes) without touching the other domains.
        """
        raw = getattr(unified_cfg, "domain_root_overrides", None) or {}
        try:
            items = dict(raw).items()
        except Exception:
            items = []
        base = paths.repo_root()
        return {str(k).strip().lower(): os.path.realpath(os.path.join(base, str(v))) for k, v in items if v}

    @classmethod
    def _root_override_tag(cls, unified_cfg) -> str:
        """Cache-dir suffix for a root override; EMPTY when none, so every existing cache keeps its
        directory. Without a distinct directory an override run would read (or clobber) the default
        cache and silently train on the original source."""
        ov = cls._root_overrides(unified_cfg)
        if not ov:
            return ""
        import hashlib as _hl
        key = ";".join(f"{k}={ov[k]}" for k in sorted(ov))
        return "_ro" + _hl.md5(key.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _fixed_signature_fields(cfg) -> Dict:
        """Signature keys that every existing cache carries, at the only values this loader builds.

        They must stay in the signature with exactly these values or existing caches stop
        validating. rrwp_codebook_steps follows model.rrwp_steps.
        """
        model_cfg = getattr(cfg, "model", None)
        return {
            "precompute_rrwp": False,
            "rrwp_steps": 0,
            "rrwp_codebook_enabled": False,
            "rrwp_codebook_k": 1,
            "rrwp_codebook_steps": max(1, int(getattr(model_cfg, "rrwp_steps", 10) or 10)),
            "rrwp_codebook_pool": "mean",
            "rrwp_codebook_seed": 42,
        }

    @classmethod
    def _expected_signature(cls, cfg, split: str):
        split = str(split).strip().lower()
        cache_cfg = cls._resolve_unified_cache_cfg(cfg)
        active_domains = cls._resolve_active_domains(cfg)
        unified_cfg = getattr(getattr(cfg, "dataset", None), "unified", None)
        _ov = cls._root_overrides(unified_cfg)
        sig = {
            "schema_version": int(cache_cfg["schema_version"]),
            "split": split,
            "domain_order": list(active_domains),
            "remove_h": bool(cache_cfg["remove_h"]),
            "aromatic": bool(cache_cfg["aromatic"]),
            "feature_schema": str(cache_cfg.get("feature_schema", "padded_onehot_v1")),
            "keep_legacy_onehot_fields": bool(cache_cfg.get("keep_legacy_onehot_fields", True)),
            "domain_condition_tag": cls._domain_condition_tag(
                bool(getattr(getattr(cfg, "general", None), "conditional_domain", False))
            ),
            **cls._fixed_signature_fields(cfg),
        }
        # Only when set: adding the key unconditionally would change the signature of every
        # existing cache and force full rebuilds for runs that never use an override. Must match
        # _signature() below (the one written into the cache).
        if _ov:
            sig["domain_root_overrides"] = dict(sorted(_ov.items()))
        return sig

    @classmethod
    def is_cache_ready(cls, cfg, split: str, root: str) -> bool:
        split = str(split).strip().lower()
        cache_cfg = cls._resolve_unified_cache_cfg(cfg)
        cache_mode = cache_cfg["cache_mode"]
        expected_sig = cls._expected_signature(cfg, split)
        dcond_tag = cls._domain_condition_tag(
            bool(getattr(getattr(cfg, "general", None), "conditional_domain", False))
        )
        _sv = int(cache_cfg.get("schema_version", 1))
        active_domains = cls._resolve_active_domains(cfg)
        domain_tag = cls._domain_tag(active_domains)
        _ro = cls._root_override_tag(getattr(getattr(cfg, "dataset", None), "unified", None))
        cache_dir = os.path.join(root, f"cache__{dcond_tag}_v{_sv}{_ro}__{domain_tag}")
        cls._log(cache_cfg.get("verbose", True), f"cache_check split={split} mode={cache_mode}")

        cache_path = os.path.join(cache_dir, f"{split}_ram.pt")
        meta_path = os.path.join(cache_dir, f"{split}_ram.meta.pt")
        if not os.path.exists(cache_path):
            return False
        # Fast path: tiny sidecar avoids loading the full cache just for sig check.
        if os.path.exists(meta_path):
            try:
                payload = torch.load(meta_path, weights_only=False, map_location="cpu")
                return isinstance(payload, dict) and payload.get("meta", {}) == expected_sig
            except Exception:
                pass
        try:
            payload = torch.load(cache_path, weights_only=False, map_location="cpu")
        except Exception:
            return False
        return isinstance(payload, dict) and payload.get("meta", {}) == expected_sig

    @classmethod
    def build_cache_if_needed(cls, cfg, split: str, root: str, conditional_domain: bool = False):
        if cls.is_cache_ready(cfg, split, root):
            return None
        return cls(cfg, split, root, conditional_domain)

    def __init__(self, cfg, split: str, root: str, conditional_domain: bool = False):
        init_t0 = time.perf_counter()
        self.cfg = cfg
        self.split = str(split).strip().lower()
        self.root = root
        self.conditional_domain = bool(conditional_domain)
        self.domain_condition_tag = self._domain_condition_tag(self.conditional_domain)

        self.dataset_names = self._resolve_active_domains(cfg)
        self.domain_to_idx = {name: i for i, name in enumerate(self.dataset_names)}
        self.num_domains = len(self.dataset_names)
        self.domain_type_by_name = {
            "sbm": "synthetic",
            "comm20": "synthetic",
            "planar": "synthetic",
            "tree": "synthetic",
            "zinc": "molecular",
            "qm9": "molecular",
        }

        _domain_tag = self._domain_tag(self.dataset_names)
        _unified_cfg = getattr(getattr(self.cfg, "dataset", None), "unified", None)
        # The manifest fixes each domain's ROW COUNT. Without the root-override tag here, an
        # override run reused the base manifest (128 sbm rows) and silently read only the first
        # 128 graphs of an 8,192-graph pool. Empty tag when no override -> base path unchanged.
        self.manifest_dir = os.path.join(
            self.root, "manifests", f"domains__{_domain_tag}{self._root_override_tag(_unified_cfg)}")
        _sv = int(getattr(_unified_cfg, "schema_version", 1))
        _ro = self._root_override_tag(_unified_cfg)
        self.cache_dir = os.path.join(
            self.root,
            f"cache__{self.domain_condition_tag}_v{_sv}{_ro}__{_domain_tag}",
        )
        os.makedirs(self.manifest_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.manifest_path = os.path.join(self.manifest_dir, f"{self.split}.jsonl")
        self._ensure_manifest()
        self.manifest_rows = self._read_manifest(self.manifest_path)
        self._validate_manifest_rows(self.manifest_rows)

        unified_cfg = getattr(getattr(self.cfg, "dataset", None), "unified", None)
        cache_cfg = self._resolve_unified_cache_cfg(self.cfg)
        self.cache_mode = cache_cfg["cache_mode"]
        self.feature_schema = cache_cfg["feature_schema"]
        self.keep_legacy_onehot_fields = bool(cache_cfg.get("keep_legacy_onehot_fields", True))
        self.verbose = bool(cache_cfg.get("verbose", True))
        self.log_memory = bool(cache_cfg.get("log_memory", False))
        self.sample_watchdog_s = float(getattr(unified_cfg, "sample_watchdog_s", 2.0))

        self.global_vocab = None

        self._log(self.verbose, f"init split={self.split} cache_mode=ram")
        self.data_list, self.n_nodes_by_global_idx = self._load_or_build_ram_cache()
        self.total_length = len(self.data_list)
        self.global_vocab_meta = self._extract_global_vocab_meta(self.data_list)
        self.domain_id_by_global_idx = torch.tensor(
            [int(d.domain_idx.item()) for d in self.data_list],
            dtype=torch.long,
        )
        (
            self.node_token_counts,
            self.edge_token_counts,
            self.per_domain_node_token_counts,
            self.per_domain_edge_token_counts,
        ) = self._compute_full_token_counts(self.data_list, self.global_vocab_meta)

        self.domain_indices = self._build_domain_indices(self.domain_id_by_global_idx)
        self._log_memory(self.log_memory, f"after_init split={self.split}")
        self._log(self.verbose, f"init_done split={self.split} len={len(self)} elapsed_s={time.perf_counter()-init_t0:.2f}")
        self._log(
            self.verbose,
            f"cache_variant split={self.split} tag={self.domain_condition_tag} cache_dir={self.cache_dir}",
        )

        if self.global_vocab_meta is not None:
            valid_x = torch.tensor(self.global_vocab_meta["valid_x_mask"], dtype=torch.bool)
            valid_e = torch.tensor(self.global_vocab_meta["valid_e_mask"], dtype=torch.bool)
            self.global_vocab = {
                "Sx": int(self.global_vocab_meta["Sx"]),
                "Se": int(self.global_vocab_meta["Se"]),
                "node_offset": dict(self.global_vocab_meta["node_offset"]),
                "edge_offset": dict(self.global_vocab_meta["edge_offset"]),
                "node_dims": dict(self.global_vocab_meta["node_dims"]),
                "edge_dims": dict(self.global_vocab_meta["edge_dims"]),
                "valid_x_mask": valid_x,
                "valid_e_mask": valid_e,
                "feature_schema": str(self.feature_schema),
            }

    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, f"{self.split}_ram.pt")

    def _signature(self) -> Dict:
        ds_cfg = getattr(self.cfg, "dataset", None)
        unified_cfg = getattr(ds_cfg, "unified", None)
        _ov = self._root_overrides(unified_cfg)
        sig = {
            "schema_version": int(getattr(unified_cfg, "schema_version", 1)),
            "split": self.split,
            "domain_order": list(self.dataset_names),
            "remove_h": bool(getattr(ds_cfg, "remove_h", False)),
            "aromatic": bool(getattr(ds_cfg, "aromatic", False)),
            "feature_schema": str(self.feature_schema),
            "keep_legacy_onehot_fields": bool(self.keep_legacy_onehot_fields),
            "domain_condition_tag": str(self.domain_condition_tag),
            **self._fixed_signature_fields(self.cfg),
        }
        if _ov:  # must match _expected_signature()
            sig["domain_root_overrides"] = dict(sorted(_ov.items()))
        return sig

    def _build_global_vocab_semantic(self, remove_h: bool):
        """Build a fixed, semantically grounded global vocabulary (semantic_global_v1).

        Node tokens are grounded in chemistry (generic=0, H=1, C=2, …).
        Edge tokens are grounded in bond chemistry (no_edge=0, generic=1, SINGLE=2, …).
        Token 0 for edges must stay as no_edge to stay compatible with encode_no_edge().
        """
        Sx = SEMANTIC_NODE_VOCAB_SIZE
        Se = SEMANTIC_EDGE_VOCAB_SIZE

        # QM9 atom order depends on whether hydrogens are retained
        qm9_node_map = (
            [1, 2, 3, 4, 5] if not remove_h  # H, C, N, O, F
            else [2, 3, 4, 5]                 # C, N, O, F
        )
        node_maps = {**_SEMANTIC_NODE_MAPS_BASE, "qm9": qm9_node_map}

        node_map_tensors: Dict[str, torch.Tensor] = {}
        edge_map_tensors: Dict[str, torch.Tensor] = {}
        valid_x_mask = torch.zeros((self.num_domains, Sx), dtype=torch.bool)
        valid_e_mask = torch.zeros((self.num_domains, Se), dtype=torch.bool)

        for domain_name in self.dataset_names:
            if domain_name not in node_maps:
                raise ValueError(
                    f"semantic_global_v1: no node map defined for domain '{domain_name}'. "
                    "Add it to _SEMANTIC_NODE_MAPS_BASE or handle it in _build_global_vocab_semantic()."
                )
            if domain_name not in SEMANTIC_EDGE_MAPS:
                raise ValueError(
                    f"semantic_global_v1: no edge map defined for domain '{domain_name}'. "
                    "Add it to SEMANTIC_EDGE_MAPS."
                )
            d = int(self.domain_to_idx[domain_name])
            nm = torch.tensor(node_maps[domain_name], dtype=torch.long)
            em = torch.tensor(SEMANTIC_EDGE_MAPS[domain_name], dtype=torch.long)
            node_map_tensors[domain_name] = nm
            edge_map_tensors[domain_name] = em
            valid_x_mask[d, nm] = True
            valid_e_mask[d, em] = True

        node_dims = {dn: len(node_maps[dn]) for dn in self.dataset_names}
        edge_dims = {dn: len(SEMANTIC_EDGE_MAPS[dn]) for dn in self.dataset_names}

        return {
            "Sx": Sx,
            "Se": Se,
            # node_offset / edge_offset kept for structural compatibility; unused in semantic mode
            "node_offset": {dn: 0 for dn in self.dataset_names},
            "edge_offset": {dn: 0 for dn in self.dataset_names},
            "node_dims": node_dims,
            "edge_dims": edge_dims,
            "node_map_tensors": node_map_tensors,
            "edge_map_tensors": edge_map_tensors,
            "valid_x_mask": valid_x_mask,
            "valid_e_mask": valid_e_mask,
            "feature_schema": "semantic_global_v1",
        }

    def _global_vocab_to_meta(self, global_vocab):
        if global_vocab is None:
            return None
        return {
            "Sx": int(global_vocab["Sx"]),
            "Se": int(global_vocab["Se"]),
            "node_offset": dict(global_vocab["node_offset"]),
            "edge_offset": dict(global_vocab["edge_offset"]),
            "node_dims": dict(global_vocab["node_dims"]),
            "edge_dims": dict(global_vocab["edge_dims"]),
            "valid_x_mask": global_vocab["valid_x_mask"].to(torch.bool).cpu().tolist(),
            "valid_e_mask": global_vocab["valid_e_mask"].to(torch.bool).cpu().tolist(),
            "feature_schema": str(global_vocab.get("feature_schema", self.feature_schema)),
        }

    def _extract_global_vocab_meta(self, data_list):
        meta = None
        if isinstance(data_list, list) and len(data_list) > 0:
            maybe = getattr(data_list[0], "global_vocab", None)
            if isinstance(maybe, dict):
                meta = maybe
        if meta is None:
            return None
        # Gracefully handle old caches that predate valid_x/e_mask fields.
        if meta.get("valid_x_mask") is None or meta.get("valid_e_mask") is None:
            return None
        return {
            "Sx": int(meta["Sx"]),
            "Se": int(meta["Se"]),
            "node_offset": dict(meta["node_offset"]),
            "edge_offset": dict(meta["edge_offset"]),
            "node_dims": dict(meta["node_dims"]),
            "edge_dims": dict(meta["edge_dims"]),
            "valid_x_mask": list(meta["valid_x_mask"]),
            "valid_e_mask": list(meta["valid_e_mask"]),
            "feature_schema": str(meta.get("feature_schema", self.feature_schema)),
        }

    def _ensure_manifest(self):
        if os.path.exists(self.manifest_path):
            return

        datasets = self._instantiate_domain_datasets()
        rows = []
        for domain_name in self.dataset_names:
            domain_idx = self.domain_to_idx[domain_name]
            ds = datasets[domain_name]
            for sample_idx in range(len(ds)):
                rows.append(
                    {
                        "sample_id": f"{domain_name}_{sample_idx:08d}",
                        "split": self.split,
                        "domain_name": domain_name,
                        "domain_idx": int(domain_idx),
                        "sample_idx": int(sample_idx),
                    }
                )

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        del datasets
        gc.collect()

    def _read_manifest(self, path: str) -> List[Dict]:
        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _validate_manifest_rows(self, rows: List[Dict]):
        for row in rows:
            domain_name = str(row["domain_name"]).strip().lower()
            domain_idx = int(row["domain_idx"])
            if domain_name not in self.domain_to_idx:
                raise ValueError(f"Unknown manifest domain '{domain_name}' in {self.manifest_path}")
            expected_idx = self.domain_to_idx[domain_name]
            if domain_idx != expected_idx:
                raise ValueError(
                    f"Manifest domain_idx mismatch for {domain_name}: {domain_idx} != {expected_idx}"
                )
            split = str(row.get("split", self.split)).strip().lower()
            if split != self.split:
                raise ValueError(
                    f"Manifest split mismatch: row split '{split}' != dataset split '{self.split}'"
                )

    def _instantiate_domain_datasets(self):
        base_path = paths.repo_root()
        base_root = os.path.join(base_path, str(self.cfg.dataset.datadir))
        datasets = {}
        _ov = self._root_overrides(getattr(getattr(self.cfg, "dataset", None), "unified", None))
        for domain_name in self.dataset_names:
            domain_root = _ov.get(domain_name, os.path.join(base_root, domain_name))
            if domain_name in _ov:
                print(f"[unified] domain_root_overrides: {domain_name} <- {domain_root}", flush=True)
            if domain_name in self.SPECTRE_DOMAINS:
                datasets[domain_name] = SpectreGraphDataset(
                    dataset_name=domain_name,
                    split=self.split,
                    root=domain_root,
                )
            elif domain_name == "qm9":
                datasets[domain_name] = QM9Dataset(
                    stage=self.split,
                    root=domain_root,
                    remove_h=self.cfg.dataset.remove_h,
                    aromatic=self.cfg.dataset.aromatic,
                    target_prop=getattr(self.cfg.general, "target", None),
                    transform=RemoveYTransform(),
                )
            elif domain_name == "zinc":
                datasets[domain_name] = zinc_dataset.ZINCDataset(
                    split=self.split,
                    root=domain_root,
                    remove_h=self.cfg.dataset.remove_h,
                    aromatic=self.cfg.dataset.aromatic,
                    transform=zinc_dataset.RemoveYTransform(),
                )
            else:
                raise ValueError(f"Unsupported unified domain '{domain_name}'")
        return datasets

    def _load_n_nodes_sidecar(self, expected_sig, n_nodes_path, data_list):
        """Load n_nodes from sidecar if valid; otherwise compute from data_list and write sidecar."""
        if os.path.exists(n_nodes_path):
            try:
                sc = torch.load(n_nodes_path, weights_only=False, map_location="cpu")
                if isinstance(sc, dict) and sc.get("meta", {}) == expected_sig:
                    return list(sc["n_nodes"])
            except Exception:
                pass
        n_nodes = [int(d.x.size(0)) for d in data_list]
        try:
            torch.save({"meta": expected_sig, "n_nodes": n_nodes}, n_nodes_path)
        except Exception:
            pass
        return n_nodes

    def _load_or_build_ram_cache(self):
        cache_path = self._cache_path()
        meta_path = os.path.join(self.cache_dir, f"{self.split}_ram.meta.pt")
        n_nodes_path = os.path.join(self.cache_dir, f"{self.split}_ram.n_nodes.pt")
        expected_sig = self._signature()
        ds_cfg = getattr(self.cfg, "dataset", None)
        unified_cfg = getattr(ds_cfg, "unified", None)
        force_rebuild = bool(getattr(unified_cfg, "force_rebuild", False))

        if (not force_rebuild) and os.path.exists(cache_path):
            # Fast path: sidecar sig check avoids loading the full cache payload.
            if os.path.exists(meta_path):
                try:
                    meta_payload = torch.load(meta_path, weights_only=False, map_location="cpu")
                    if isinstance(meta_payload, dict) and meta_payload.get("meta", {}) == expected_sig:
                        payload = torch.load(cache_path, weights_only=False, map_location="cpu")
                        if isinstance(payload, dict):
                            data_list = payload.get("data_list", None)
                            if isinstance(data_list, list):
                                self._log(self.verbose, f"ram_cache_hit split={self.split} path={cache_path} n={len(data_list)}")
                                n_nodes = self._load_n_nodes_sidecar(expected_sig, n_nodes_path, data_list)
                                return data_list, n_nodes
                except Exception:
                    pass
            # Slow path: full file sig check (backward compat, no sidecar yet).
            try:
                payload = torch.load(cache_path, weights_only=False, map_location="cpu")
                if isinstance(payload, dict) and payload.get("meta", {}) == expected_sig:
                    data_list = payload.get("data_list", None)
                    if isinstance(data_list, list):
                        self._log(self.verbose, f"ram_cache_hit split={self.split} path={cache_path} n={len(data_list)}")
                        n_nodes = self._load_n_nodes_sidecar(expected_sig, n_nodes_path, data_list)
                        return data_list, n_nodes
            except Exception:
                pass

        self._log(self.verbose, f"ram_cache_build split={self.split} path={cache_path}")

        data_list = self._build_preprocessed_data_list()
        n_nodes = [int(d.x.size(0)) for d in data_list]

        torch.save({"meta": expected_sig, "data_list": data_list}, cache_path)
        torch.save({"meta": expected_sig}, meta_path)
        torch.save({"meta": expected_sig, "n_nodes": n_nodes}, n_nodes_path)
        self._log(self.verbose, f"ram_cache_saved split={self.split} n={len(data_list)}")
        return data_list, n_nodes

    def _build_preprocessed_data_list(self):
        domain_datasets = self._instantiate_domain_datasets()
        global_vocab = self._build_global_vocab_semantic(
            bool(getattr(getattr(self.cfg, "dataset", None), "remove_h", False))
        )

        data_list = []
        for row in self.manifest_rows:
            domain_name = str(row["domain_name"]).strip().lower()
            domain_idx = int(row["domain_idx"])
            sample_idx = int(row["sample_idx"])
            raw = domain_datasets[domain_name][sample_idx]
            data = self._normalize_one_sample(
                data=raw,
                domain_name=domain_name,
                domain_idx=domain_idx,
                global_vocab=global_vocab,
            )
            data_list.append(data)
        del domain_datasets
        gc.collect()
        global_vocab_meta = self._global_vocab_to_meta(global_vocab)
        for sample in data_list:
            sample.global_vocab = global_vocab_meta
        return data_list

    def _normalize_one_sample(self, data, domain_name: str, domain_idx: int, global_vocab):
        t0 = time.perf_counter()
        nm = global_vocab["node_map_tensors"][domain_name]
        em = global_vocab["edge_map_tensors"][domain_name]
        x_local = torch.argmax(data.x, dim=-1).long()
        e_local = torch.argmax(data.edge_attr, dim=-1).long()
        # Vectorized lookup: local domain index → global semantic index
        data.x_token = nm[x_local]
        data.edge_token = em[e_local]
        if not self.keep_legacy_onehot_fields:
            data.x = torch.zeros((int(x_local.numel()), 0), dtype=torch.float)
            data.edge_attr = torch.zeros((int(e_local.numel()), 0), dtype=torch.float)
        norm_dt = time.perf_counter() - t0
        if self.verbose and norm_dt > self.sample_watchdog_s:
            self._log(
                True,
                f"slow_normalize split={self.split} domain={domain_name} domain_idx={domain_idx} normalize_s={norm_dt:.3f}",
            )

        onehot = torch.zeros((1, self.num_domains), dtype=torch.float)
        onehot[0, domain_idx] = 1.0
        data.domain_idx = torch.tensor([domain_idx], dtype=torch.long)
        data.dataset_name = domain_name
        data.domain_onehot = onehot
        data.y = onehot if self.conditional_domain else torch.zeros_like(onehot)
        data.n_nodes = torch.tensor([int(data.x.size(0))], dtype=torch.long)
        return data

    def _compute_full_token_counts(self, data_list, global_vocab_meta):
        """Exact (not sampled) per-domain node/edge class marginals from in-memory data.

        Used for cache_mode="ram", where the whole split is already materialized, so there is
        no need to estimate these counts from a bounded probe (see
        project_masking_cfg_rate_explosion memory: a small random sample can miss genuinely
        rare-but-real classes, e.g. zinc's Phosphorus/Iodine, which then get hard-masked to
        -1e9 by train_domain_vocab_mask and cause catastrophic CE spikes when they later
        appear in a real training batch). Accumulates directly in the global Sx/Se id space
        (no domain-local truncation).
        """
        gv = global_vocab_meta if isinstance(global_vocab_meta, dict) else {}
        Sx = int(gv.get("Sx", 0))
        Se = int(gv.get("Se", 0))
        edge_off = dict(gv.get("edge_offset", {}))
        node_token_counts = torch.zeros(Sx, dtype=torch.float) if Sx > 0 else None
        edge_token_counts = torch.zeros(Se, dtype=torch.float) if Se > 0 else None
        per_domain_node_counts: dict = {}
        per_domain_edge_counts: dict = {}

        for d in data_list:
            if not (hasattr(d, "x_token") and hasattr(d, "edge_token")):
                continue
            domain_idx = int(d.domain_idx.item())
            if node_token_counts is not None and d.x_token is not None:
                x_tok = d.x_token.view(-1).long()
                valid_x = (x_tok >= 0) & (x_tok < Sx)
                if torch.any(valid_x):
                    bc = torch.bincount(x_tok[valid_x], minlength=Sx).float()
                    node_token_counts += bc
                    d_ntok = per_domain_node_counts.get(domain_idx)
                    if d_ntok is None:
                        per_domain_node_counts[domain_idx] = bc.clone()
                    else:
                        d_ntok += bc
            if edge_token_counts is not None and d.edge_token is not None:
                # No-edge pairs aren't materialized as sparse edges, so their count must be
                # added back in manually at that domain's own "no_edge" global id (se0, always
                # 0 under semantic_global_v1).
                se0 = int(edge_off.get(str(getattr(d, "dataset_name", "")), 0))
                n_nodes_g = int(d.x.size(0))
                n_edges_g = int(d.edge_token.numel())
                n_non = max(0, n_nodes_g * (n_nodes_g - 1) - n_edges_g)
                edge_tok = d.edge_token.view(-1).long()
                valid_e = (edge_tok >= 0) & (edge_tok < Se)
                bc = torch.bincount(edge_tok[valid_e], minlength=Se).float() if torch.any(valid_e) else torch.zeros(Se, dtype=torch.float)
                if n_non > 0 and 0 <= se0 < Se:
                    bc[se0] += float(n_non)
                edge_token_counts += bc
                d_etok = per_domain_edge_counts.get(domain_idx)
                if d_etok is None:
                    per_domain_edge_counts[domain_idx] = bc.clone()
                else:
                    d_etok += bc

        return node_token_counts, edge_token_counts, per_domain_node_counts, per_domain_edge_counts

    def _build_domain_indices(self, domain_id_tensor: torch.Tensor):
        domain_indices = {name: [] for name in self.dataset_names}
        for global_idx, domain_idx in enumerate(domain_id_tensor.tolist()):
            domain_name = self.dataset_names[int(domain_idx)]
            domain_indices[domain_name].append(global_idx)
        return domain_indices

    def __len__(self):
        return int(self.total_length)

    def _ensure_required_fields(self, sample, global_idx: int):
        """Backfill fields expected by downstream collate/metrics for cache compatibility."""
        if not hasattr(sample, "idx"):
            sample.idx = torch.tensor([int(global_idx)], dtype=torch.long)
        if not hasattr(sample, "n_nodes"):
            if hasattr(sample, "x") and torch.is_tensor(sample.x):
                n_nodes = int(sample.x.size(0))
            elif hasattr(sample, "x_token") and torch.is_tensor(sample.x_token):
                n_nodes = int(sample.x_token.size(0))
            else:
                n_nodes = 0
            sample.n_nodes = torch.tensor([n_nodes], dtype=torch.long)
        return sample

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")
        sample = self.data_list[idx]
        return self._ensure_required_fields(sample, idx)

    def get_domain_dataset(self, domain_name: str):
        domain_key = str(domain_name).strip().lower()
        if domain_key not in self.domain_indices:
            raise KeyError(f"Unknown domain '{domain_key}'")
        return _DomainSubset(self, self.domain_indices[domain_key])
