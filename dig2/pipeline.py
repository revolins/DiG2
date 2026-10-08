"""Builds everything a DiG2 model needs besides its config: the datamodule, dataset infos,
sampling metrics, visualizers and extra-feature providers.

Two data paths:
  * `dataset.loader_backend: unified` -- the multi-domain unified loader (planar, sbm, tree,
    comm20, qm9, zinc), scored by MixedMetricDispatcher.
  * `dataset.name: guacamol | moses` -- a standalone single-domain molecular dataset.
"""
import os

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dig2 import paths
from dig2.metrics.abstract import TrainAbstractMetricsDiscrete
from dig2.model.extra_features import DummyExtraFeatures, ExtraFeatures


# ----------------------------------------------------------------------------- unified path
def _build_lightweight_molecular_infos(domain_name: str, cfg: DictConfig, split_dataset, datamodule):
    """Dataset infos for one molecular domain of the unified loader, sized to the domain-native
    vocabulary (the molecular evaluators work in domain-local atom/bond indices)."""
    from dig2.data import qm9, zinc

    if domain_name == "qm9":
        base_infos = qm9.QM9infos(datamodule=None, cfg=cfg)
    elif domain_name == "zinc":
        base_infos = zinc.ZINCinfos(datamodule=None, cfg=cfg)
    else:
        raise ValueError(f"Unsupported molecular domain for unified infos: {domain_name}")
    if len(split_dataset) == 0:
        raise ValueError(f"Empty split dataset for domain '{domain_name}'")

    sample = split_dataset[0]
    sample_x_dim = int(sample.x.size(-1)) if torch.is_tensor(getattr(sample, "x", None)) and sample.x.dim() >= 2 else 0
    sample_e_dim = (
        int(sample.edge_attr.size(-1))
        if torch.is_tensor(getattr(sample, "edge_attr", None)) and sample.edge_attr.dim() >= 2 else 0
    )
    has_tokens = hasattr(sample, "x_token") and hasattr(sample, "edge_token")

    x_dim = int(len(base_infos.atom_decoder))
    edge_types = getattr(base_infos, "edge_types", None)
    e_dim = int(edge_types.numel()) if edge_types is not None and hasattr(edge_types, "numel") else sample_e_dim
    base_infos.output_dims = {"X": x_dim, "E": e_dim, "y": 0}
    base_infos.input_dims = {"X": x_dim, "E": e_dim, "y": 1}

    # The semantic schema may store empty legacy one-hot fields and carry token ids instead.
    if (sample_x_dim == 0 or sample_e_dim == 0) and has_tokens:
        gv = getattr(datamodule, "global_vocab", None)
        if isinstance(gv, dict):
            sample_x_dim = int((gv.get("node_dims", {}) or {}).get(domain_name, sample_x_dim))
            sample_e_dim = int((gv.get("edge_dims", {}) or {}).get(domain_name, sample_e_dim))

    if sample_x_dim > 0 and sample_e_dim > 0:
        if x_dim > sample_x_dim:
            raise ValueError(
                f"Domain-native node dimension {x_dim} exceeds the unified feature width "
                f"{sample_x_dim} for domain '{domain_name}'."
            )
        if e_dim > sample_e_dim:
            # Native edge_types may include an aromatic slot with probability 0.0 that is absent
            # from the semantic schema. Clamp to the schema width.
            e_dim = sample_e_dim
            if edge_types is not None and edge_types.numel() > e_dim:
                base_infos.edge_types = edge_types[:e_dim]
            base_infos.output_dims = {"X": x_dim, "E": e_dim, "y": 0}
            base_infos.input_dims = {"X": x_dim, "E": e_dim, "y": 1}
    elif not has_tokens:
        raise ValueError(f"Could not infer molecular feature widths for domain '{domain_name}'.")
    return base_infos


def _decode_domain_smiles(split_dataset, atom_map, edge_map, atom_decoder):
    """SMILES of every molecule in one unified split, in dataset order.

    Unified samples carry global semantic token ids (`x_token`, `edge_token`) instead of
    one-hot features, so they are mapped to the domain-local atom / bond indices first and
    then decoded exactly as the standalone QM9 / ZINC loaders do: molecules that fail to
    sanitize are dropped."""
    from dig2.metrics.rdkit_functions import build_molecule_with_partial_charges, mol2smiles

    smiles = []
    for k in range(len(split_dataset)):
        sample = split_dataset[k]
        atoms = atom_map[sample.x_token.view(-1).long()]
        n = int(atoms.numel())
        bonds = torch.zeros(n, n, dtype=torch.long)
        ei = sample.edge_index.long()
        if ei.numel():
            bonds[ei[0], ei[1]] = edge_map[sample.edge_token.view(-1).long()]
        smile = mol2smiles(build_molecule_with_partial_charges(atoms, bonds, atom_decoder))
        if smile is not None:
            smiles.append(smile)
    return smiles


def _load_or_compute_domain_smiles(cfg: DictConfig, domain: str, split_datasets, dataset_infos,
                                   atom_map, edge_map):
    """Train/val/test SMILES for one molecular domain: cached .npy files under the data
    directory when present, otherwise decoded from the unified split datasets and cached."""
    suffix = "no_h" if bool(cfg.dataset.remove_h) else "h"
    data_dir = os.path.join(paths.repo_root(), str(cfg.dataset.datadir))
    smiles = {}
    for split in ["train", "val", "test"]:
        filename = f"{split}_smiles_{suffix}.npy"
        # QM9's files sit at the data root, ZINC's under data/zinc/ (the layout of the original
        # loaders). Never fall back across domains: ZINC must not pick up QM9's root-level file.
        path = os.path.join(data_dir, "zinc", filename) if domain == "zinc" else os.path.join(data_dir, filename)
        if os.path.exists(path):
            smiles[split] = np.load(path, allow_pickle=True).tolist()
            continue
        print(f"[{domain}] decoding {split} SMILES (cached at {path})")
        smiles[split] = _decode_domain_smiles(split_datasets[split], atom_map, edge_map,
                                              dataset_infos.atom_decoder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(smiles[split]))
    return smiles


def _build_molecular_evaluators(cfg: DictConfig, datamodule):
    """Per-domain molecular evaluators and visualizers for the unified loader. Generated graphs
    use the global semantic vocabulary; each evaluator remaps them to its domain-local indices."""
    from dig2.data.unified.dataset import SEMANTIC_EDGE_MAPS, SEMANTIC_EDGE_VOCAB_SIZE, SEMANTIC_NODE_VOCAB_SIZE
    from dig2.metrics.molecular import SamplingMolecularMetrics
    from dig2.metrics.visualization import MolecularVisualization

    evaluators, visualizers = {}, {}
    splits = {"train": datamodule.train_dataset, "val": datamodule.val_dataset, "test": datamodule.test_dataset}
    gv = getattr(datamodule, "global_vocab", None)
    Se = int(gv.get("Se", SEMANTIC_EDGE_VOCAB_SIZE)) if isinstance(gv, dict) else SEMANTIC_EDGE_VOCAB_SIZE
    Sx = int(gv.get("Sx", SEMANTIC_NODE_VOCAB_SIZE)) if isinstance(gv, dict) else SEMANTIC_NODE_VOCAB_SIZE

    for domain in datamodule.dataset_names:
        domain = str(domain).strip().lower()
        if datamodule.domain_type_by_name.get(domain) != "molecular":
            continue
        domain_splits = {s: ds.get_domain_dataset(domain) for s, ds in splits.items()}
        domain_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        domain_cfg.dataset.name = domain
        infos = _build_lightweight_molecular_infos(domain, domain_cfg, domain_splits["train"], datamodule)
        # Global semantic edge index -> local bond index (global 'generic' maps to no-bond).
        edge_map = torch.zeros(Se, dtype=torch.long)
        for local_idx, global_idx in enumerate(SEMANTIC_EDGE_MAPS[domain]):
            edge_map[global_idx] = local_idx
        # Global semantic atom index -> local decoder index, from the domain's valid-node mask.
        valid_x = gv.get("valid_x_mask") if isinstance(gv, dict) else None
        atom_map = torch.zeros(Sx, dtype=torch.long)
        valid_globals = torch.where(valid_x[int(datamodule.domain_to_idx[domain])])[0]
        for local_idx, global_idx in enumerate(valid_globals.tolist()):
            atom_map[global_idx] = local_idx
        smiles = _load_or_compute_domain_smiles(domain_cfg, domain, domain_splits, infos, atom_map, edge_map)

        evaluators[domain] = SamplingMolecularMetrics(
            infos, smiles, domain_cfg,
            add_virtual_states=False,
            global_to_local_edge_map=edge_map,
            global_to_local_atom_map=atom_map,
        )
        visualizers[domain] = MolecularVisualization(remove_h=domain_cfg.dataset.remove_h, dataset_infos=infos)
    return evaluators, visualizers


def build_unified_pipeline(cfg, compute_reference: bool = True):
    from dig2.data.spectre import SpectreDatasetInfos
    from dig2.data.unified.datamodule import UnifiedDataModule
    from dig2.metrics.mixed_dispatcher import MixedMetricDispatcher
    from dig2.metrics.spectre_utils import CombinedSpectreSamplingMetrics
    from dig2.metrics.visualization import NonMolecularVisualization

    datamodule = UnifiedDataModule(cfg)
    synthetic_metrics = CombinedSpectreSamplingMetrics(
        datamodule=datamodule, compute_emd=False, domain_names=datamodule.dataset_names,
    )
    molecular_evaluators, visualizers_by_domain = _build_molecular_evaluators(cfg, datamodule)
    unified_metrics = getattr(cfg.dataset, "unified_metrics", None)
    sampling_metrics = MixedMetricDispatcher(
        domain_type_by_name=datamodule.domain_type_by_name,
        synthetic_evaluator=synthetic_metrics,
        molecular_evaluator_by_domain=molecular_evaluators,
        use_unified_metrics=bool(getattr(unified_metrics, "enabled", False)),
        shared_prefilter=bool(getattr(unified_metrics, "shared_prefilter", True)),
        shared_cache=bool(getattr(unified_metrics, "shared_cache", True)),
    )
    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
    extra_features = ExtraFeatures(cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos)
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features,
    )
    if compute_reference:
        dataset_infos.compute_reference_metrics(datamodule=datamodule, sampling_metrics=sampling_metrics)

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": TrainAbstractMetricsDiscrete(),
        "sampling_metrics": sampling_metrics,
        "visualization_tools": NonMolecularVisualization(dataset_name=cfg.dataset.name),
        "visualization_tools_by_domain": visualizers_by_domain,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }
    return datamodule, model_kwargs


# ----------------------------------------------------------------------------- single-domain path
def build_single_molecular_pipeline(cfg, compute_reference: bool = True):
    """Standalone GuacaMol / MOSES training and evaluation (DeFoG/DiGress-style molecular metrics)."""
    from dig2.metrics.molecular import SamplingMolecularMetrics
    from dig2.metrics.molecular_discrete import TrainMolecularMetricsDiscrete
    from dig2.metrics.visualization import MolecularVisualization
    from dig2.model.extra_features_molecular import ExtraMolecularFeatures

    name = str(cfg.dataset.name)
    if name == "guacamol":
        from dig2.data import guacamol as module
        datamodule = module.GuacamolDataModule(cfg)
        dataset_infos = module.Guacamolinfos(datamodule, cfg)
    elif name == "moses":
        from dig2.data import moses as module
        datamodule = module.MosesDataModule(cfg)
        dataset_infos = module.MOSESinfos(datamodule, cfg)
    else:
        raise ValueError(f"Single-domain molecular dataset must be guacamol or moses, got {name!r}")
    dataset_smiles = module.get_smiles(raw_dir=datamodule.train_dataset.raw_dir, filter_dataset=cfg.dataset.filter)

    extra_features = ExtraFeatures(cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features,
    )
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, dataset_smiles, cfg)
    if compute_reference:
        dataset_infos.compute_reference_metrics(datamodule=datamodule, sampling_metrics=sampling_metrics)
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": TrainMolecularMetricsDiscrete(dataset_infos),
        "sampling_metrics": sampling_metrics,
        "visualization_tools": MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos),
        "visualization_tools_by_domain": None,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }
    return datamodule, model_kwargs


def build_pipeline(cfg, compute_reference: bool = True):
    """(datamodule, model_kwargs) for any supported dataset configuration."""
    if str(getattr(cfg.dataset, "loader_backend", "") or "").strip().lower() == "unified":
        return build_unified_pipeline(cfg, compute_reference=compute_reference)
    return build_single_molecular_pipeline(cfg, compute_reference=compute_reference)


def build_model(cfg, compute_reference: bool = True):
    from dig2.model.graph_discrete_interpolative_model import GraphDiscreteInterpolativeModel

    datamodule, model_kwargs = build_pipeline(cfg, compute_reference=compute_reference)
    return GraphDiscreteInterpolativeModel(cfg=cfg, **model_kwargs), datamodule
