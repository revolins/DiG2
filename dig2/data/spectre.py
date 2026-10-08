import pickle as pkl
import time

from networkx import to_numpy_array

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from dig2.data.abstract_dataset import AbstractDatasetInfos


class SpectreGraphDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.num_graphs = len(self.data.n_nodes)

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        return [self.split + ".pt"]

    def download(self):
        """
        Download raw graph files. Taken from PyG QM9 class
        """
        if self.dataset_name == "sbm":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/sbm.pkl"
        elif self.dataset_name == "planar":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/planar.pkl"
        elif self.dataset_name == "tree":
            raw_url = "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/tree.pkl"
        elif self.dataset_name == "comm20":
            raw_url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt"
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name in ["tree", "sbm", "planar"]:
            with open(file_path, "rb") as f:
                dataset = pkl.load(f)
            train_data = dataset["train"]
            val_data = dataset["val"]
            test_data = dataset["test"]

            train_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in train_data
            ]
            val_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in val_data
            ]
            test_data = [
                torch.Tensor(to_numpy_array(graph)).fill_diagonal_(0)
                for graph in test_data
            ]
        else:
            (
                adjs,
                eigvals,
                eigvecs,
                n_nodes,
                max_eigval,
                min_eigval,
                same_sample,
                n_max,
            ) = torch.load(file_path, weights_only=False)

            g_cpu = torch.Generator().manual_seed(1234)
            self.num_graphs = 200
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len
            indices = torch.randperm(self.num_graphs, generator=g_cpu)
            print(
                f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}"
            )
            train_indices = indices[:train_len]
            val_indices = indices[train_len : train_len + val_len]
            test_indices = indices[train_len + val_len :]

            train_data = []
            val_data = []
            test_data = []

            for i, adj in enumerate(adjs):
                if i in train_indices:
                    train_data.append(adj)
                if i in val_indices:
                    val_data.append(adj)
                if i in test_indices:
                    test_data.append(adj)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {"train": 0, "val": 1, "test": 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]], weights_only=False)

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def _try_load_precomputed_stats(datamodule, domain_to_idx):
    """Return pre-computed histogram tensors from the unified train dataset, or None if absent."""
    from collections import Counter

    train_ds = getattr(datamodule, "train_dataset", None)
    if train_ds is None:
        return None

    n_nodes_by_global_idx = getattr(train_ds, "n_nodes_by_global_idx", None)
    node_token_counts = getattr(train_ds, "node_token_counts", None)
    edge_token_counts = getattr(train_ds, "edge_token_counts", None)
    per_domain_node_tc = getattr(train_ds, "per_domain_node_token_counts", None)
    per_domain_edge_tc = getattr(train_ds, "per_domain_edge_token_counts", None)

    if any(x is None for x in [n_nodes_by_global_idx, node_token_counts, edge_token_counts]):
        return None

    counts = Counter(n_nodes_by_global_idx)
    max_n = max(counts.keys()) if counts else 1
    n_nodes_hist = torch.zeros(max_n + 1, dtype=torch.float)
    for n, c in counts.items():
        if n >= 0:
            n_nodes_hist[n] = float(c)
    n_nodes = n_nodes_hist / n_nodes_hist.sum().clamp(min=1e-8)

    ntc = torch.tensor(node_token_counts, dtype=torch.float)
    etc = torch.tensor(edge_token_counts, dtype=torch.float)
    node_types = ntc / ntc.sum().clamp(min=1e-8)
    edge_types = etc / etc.sum().clamp(min=1e-8)

    domain_node_types = None
    domain_edge_types = None
    if per_domain_node_tc and per_domain_edge_tc and domain_to_idx:
        d_dim = max(int(v) for v in domain_to_idx.values()) + 1
        x_dim = len(node_token_counts)
        e_dim = len(edge_token_counts)
        dom_x = torch.zeros((d_dim, x_dim), dtype=torch.float)
        dom_e = torch.zeros((d_dim, e_dim), dtype=torch.float)
        for did, ntok in per_domain_node_tc.items():
            did = int(did)
            lim = min(x_dim, len(ntok))
            if 0 <= did < d_dim and lim > 0:
                dom_x[did, :lim] = torch.tensor(ntok[:lim], dtype=torch.float)
        for did, etok in per_domain_edge_tc.items():
            did = int(did)
            lim = min(e_dim, len(etok))
            if 0 <= did < d_dim and lim > 0:
                dom_e[did, :lim] = torch.tensor(etok[:lim], dtype=torch.float)
        dom_x = dom_x / dom_x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        dom_e = dom_e / dom_e.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        domain_node_types = dom_x
        domain_edge_types = dom_e

    return {
        "n_nodes": n_nodes,
        "node_types": node_types,
        "edge_types": edge_types,
        "domain_node_types": domain_node_types,
        "domain_edge_types": domain_edge_types,
    }


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        t0 = time.perf_counter()

        def _log(msg: str):
            print(f"[data-info] {msg}", flush=True)

        def _log_mem(stage: str):
            try:
                import resource
                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_gb = float(rss_kb) / (1024.0 * 1024.0)
                print(f"[data-info][mem] stage={stage} rss_max_gb={rss_gb:.3f}", flush=True)
            except Exception:
                pass

        _log("stage=init_begin")
        # self.datamodule = datamodule
        if hasattr(datamodule, "dataset_names") and len(datamodule.dataset_names) > 1:
            self.dataset_name = ",".join(datamodule.dataset_names)
        elif hasattr(datamodule.inner, "dataset_name"):
            self.dataset_name = datamodule.inner.dataset_name
        else:
            self.dataset_name = str(dataset_config.name)

        domain_to_idx = getattr(datamodule, "domain_to_idx", None)
        self.domain_to_idx = dict(domain_to_idx) if isinstance(domain_to_idx, dict) else {}

        # Exact per-domain statistics precomputed by the unified train dataset. A bounded random
        # probe is deliberately not used: it can miss genuinely rare-but-real classes and hand
        # train_domain_vocab_mask a corrupted validity mask (see
        # project_masking_cfg_rate_explosion memory).
        _log("stage=unified_fastpath_begin")
        _log_mem("unified_fastpath_begin")
        _precomp = _try_load_precomputed_stats(datamodule, self.domain_to_idx)
        if _precomp is None:
            raise RuntimeError(
                "SpectreDatasetInfos needs the unified train dataset's precomputed node-count and "
                "token statistics (n_nodes_by_global_idx, node/edge_token_counts)."
            )
        _log("stage=unified_fastpath_precomputed_stats skipping_probe=true")
        self.n_nodes = _precomp["n_nodes"]
        self.node_types = _precomp["node_types"]
        self.edge_types = _precomp["edge_types"]
        self.domain_node_types = _precomp.get("domain_node_types")
        self.domain_edge_types = _precomp.get("domain_edge_types")
        _log(
            f"stage=unified_fastpath_done max_n_nodes={int(self.n_nodes.numel()-1)} "
            f"x_dim={int(self.node_types.numel())} e_dim={int(self.edge_types.numel())}"
        )
        _log_mem("unified_fastpath_done")

        _log("stage=complete_infos_begin")
        super().complete_infos(self.n_nodes, self.node_types)
        _log(f"stage=init_done total_elapsed_s={time.perf_counter()-t0:.2f}")
        _log_mem("init_done")
