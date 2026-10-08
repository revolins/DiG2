from dig2 import paths
import os
import pathlib
import gc
import time

from dig2.data.abstract_dataset import AbstractDataModule
from dig2.data.unified.dataset import UnifiedDataset


class UnifiedDataModule(AbstractDataModule):
    """Single datamodule for six-domain unified RAM-backed loading."""

    def __init__(self, cfg):
        t0 = time.perf_counter()
        self.cfg = cfg
        self._all_domain_type_by_name = {
            "sbm": "synthetic",
            "comm20": "synthetic",
            "planar": "synthetic",
            "tree": "synthetic",
            "zinc": "molecular",
            "qm9": "molecular",
        }
        self.dataset_names = []
        self.domain_to_idx = {}
        self.domain_type_by_name = {}

        base_path = paths.repo_root()
        root_subdir = str(getattr(getattr(cfg.dataset, "unified", None), "root_subdir", "unified"))
        unified_root = os.path.join(base_path, cfg.dataset.datadir, root_subdir)
        conditional_domain = bool(getattr(cfg.general, "conditional_domain", False))
        unified_cfg = getattr(cfg.dataset, "unified", None)
        auto_build_missing_splits = bool(getattr(unified_cfg, "auto_build_missing_splits", True))
        verbose = bool(getattr(unified_cfg, "verbose", True))
        log_memory = bool(getattr(unified_cfg, "log_memory", False))

        def _log(msg: str):
            if verbose:
                print(f"[unified-dm] {msg}", flush=True)

        def _log_mem(prefix: str):
            if not log_memory:
                return
            try:
                import resource
                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_gb = float(rss_kb) / (1024.0 * 1024.0)
                print(f"[unified-dm][mem] {prefix}: rss_max_gb={rss_gb:.3f}", flush=True)
            except Exception:
                pass

        splits = ["train", "val", "test"]
        _log(f"checking cache readiness for splits={splits}")
        missing_splits = [
            split for split in splits
            if not UnifiedDataset.is_cache_ready(cfg, split, unified_root)
        ]
        _log(f"missing_splits={missing_splits}")

        if len(missing_splits) > 0:
            if not auto_build_missing_splits:
                raise RuntimeError(
                    "Missing unified caches for splits: "
                    f"{missing_splits}. Set dataset.unified.auto_build_missing_splits=true "
                    "or prebuild these splits first."
                )

            for split in missing_splits:
                split_t0 = time.perf_counter()
                _log_mem(f"before_build split={split}")
                _log(f"building missing split={split}")
                tmp = UnifiedDataset.build_cache_if_needed(
                    cfg=cfg,
                    split=split,
                    root=unified_root,
                    conditional_domain=conditional_domain,
                )
                if tmp is not None:
                    del tmp
                gc.collect()
                _log_mem(f"after_build split={split}")
                _log(f"finished split={split} elapsed_s={time.perf_counter()-split_t0:.2f}")

        _log("instantiating runtime split datasets")
        datasets = {
            "train": UnifiedDataset(cfg, "train", unified_root, conditional_domain),
            "val": UnifiedDataset(cfg, "val", unified_root, conditional_domain),
            "test": UnifiedDataset(cfg, "test", unified_root, conditional_domain),
        }
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

        self.dataset_names = list(
            getattr(self.train_dataset, "dataset_names", UnifiedDataset.CANONICAL_DOMAINS)
        )
        self.domain_to_idx = dict(
            getattr(
                self.train_dataset,
                "domain_to_idx",
                {name: i for i, name in enumerate(self.dataset_names)},
            )
        )
        self.domain_type_by_name = {
            name: self._all_domain_type_by_name.get(name, "unknown")
            for name in self.dataset_names
        }
        self.domain_specs = self._build_domain_specs(self.train_dataset)

        train_vocab = getattr(self.train_dataset, "global_vocab", None)
        if isinstance(train_vocab, dict):
            self.global_vocab = {
                "Sx": int(train_vocab.get("Sx", 0)),
                "Se": int(train_vocab.get("Se", 0)),
                "node_offset": dict(train_vocab.get("node_offset", {})),
                "edge_offset": dict(train_vocab.get("edge_offset", {})),
                "node_dims": dict(train_vocab.get("node_dims", {})),
                "edge_dims": dict(train_vocab.get("edge_dims", {})),
                "valid_x_mask": train_vocab.get("valid_x_mask"),
                "valid_e_mask": train_vocab.get("valid_e_mask"),
                "feature_schema": str(train_vocab.get("feature_schema", "padded_onehot_v1")),
            }
        else:
            self.global_vocab = None

        _log_mem("after_datamodule_init")
        _log(f"datamodule_ready elapsed_s={time.perf_counter()-t0:.2f}")

    def _build_domain_specs(self, train_dataset) -> dict:
        from collections import Counter

        specs = {}
        domain_indices = getattr(train_dataset, "domain_indices", {})
        dataset_names = getattr(train_dataset, "dataset_names", [])
        n_nodes_by_global_idx = getattr(train_dataset, "n_nodes_by_global_idx", None)

        if n_nodes_by_global_idx is None:
            raise RuntimeError(
                "[unified-dm] n_nodes_by_global_idx is missing from the dataset index. "
                "The cache is stale — set dataset.unified.force_rebuild=true and rerun to rebuild it."
            )

        for name in dataset_names:
            indices = domain_indices.get(name, [])
            if not indices:
                specs[name] = {"min_n_nodes": 1, "max_n_nodes": 1, "n_nodes_hist": [0.0, 1.0]}
                continue

            n_nodes_list = [n_nodes_by_global_idx[i] for i in indices]

            min_n = int(min(n_nodes_list))
            max_n = int(max(n_nodes_list))
            counts = Counter(n_nodes_list)
            hist = [0.0] * (max_n + 1)
            for n, c in counts.items():
                if 0 <= n <= max_n:
                    hist[n] = float(c)
            total = sum(hist)
            if total > 0:
                hist = [v / total for v in hist]
            specs[name] = {"min_n_nodes": min_n, "max_n_nodes": max_n, "n_nodes_hist": hist}
            print(
                f"[unified-dm] domain_specs[{name}]: min={min_n}, max={max_n}, n={len(n_nodes_list)}",
                flush=True,
            )

        return specs

    def __getitem__(self, item):
        return self.inner[item]
