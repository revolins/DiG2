import os

from torchmetrics import MeanAbsoluteError

### packages for visualization
from dig2.metrics.rdkit_functions import compute_molecular_metrics
import torch
from torchmetrics import Metric
try:
    import wandb
except ImportError:
    wandb = None
import torch.nn as nn
import pandas as pd

from dig2.metrics.abstract import compute_ratios
from dig2.metrics.rdkit_functions import _domain_suffixed
from dig2.metrics.moses_style import compute_moses_metrics



class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, dataset_smiles, cfg, add_virtual_states=False,
                 global_to_local_edge_map=None, global_to_local_atom_map=None):
        super().__init__()
        di = dataset_infos
        self.add_virtual_states = bool(add_virtual_states)
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)

        # When a global→local atom remap is provided the histogram must be sized for the
        # domain-local atom vocabulary, not for the global vocab (Sx=11).
        has_atom_remap = global_to_local_atom_map is not None
        atom_decoder = getattr(di, "atom_decoder", None)
        if has_atom_remap and atom_decoder is not None:
            local_x_dim = int(len(atom_decoder))
        else:
            local_x_dim = di.output_dims["X"]
        num_node_types = local_x_dim + 1 if self.add_virtual_states else local_x_dim

        self.generated_node_dist = GeneratedNodesDistribution(num_node_types)

        # When a global→local edge remap is provided the histogram must be sized for the
        # domain-local vocabulary (e.g. 4 for QM9/ZINC), not for the global vocab (Se=5).
        has_edge_remap = global_to_local_edge_map is not None
        edge_types_ref = getattr(di, "edge_types", None)
        if has_edge_remap and edge_types_ref is not None and hasattr(edge_types_ref, "numel"):
            local_e_dim = int(edge_types_ref.numel())
        else:
            local_e_dim = di.output_dims["E"]
        num_edge_types = local_e_dim + 1 if self.add_virtual_states else local_e_dim

        self.generated_edge_dist = GeneratedEdgesDistribution(num_edge_types)
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)
        self.cfg = cfg

        if global_to_local_edge_map is not None:
            self.register_buffer(
                "global_to_local_edge_map",
                torch.as_tensor(global_to_local_edge_map, dtype=torch.long),
            )
        else:
            self.global_to_local_edge_map = None

        if global_to_local_atom_map is not None:
            self.register_buffer(
                "global_to_local_atom_map",
                torch.as_tensor(global_to_local_atom_map, dtype=torch.long),
            )
        else:
            self.global_to_local_atom_map = None

        # Skip the node-dimension consistency check when a global→local atom remap is active
        # because output_dims["X"] reflects the global vocab size (Sx), not the local one.
        if not has_atom_remap and atom_decoder is not None:
            expected_node_bins = int(len(atom_decoder)) + (1 if self.add_virtual_states else 0)
            if int(num_node_types) != expected_node_bins:
                raise ValueError(
                    "SamplingMolecularMetrics node-dimension mismatch: "
                    f"output_dims['X'](+virtual)={int(num_node_types)} but "
                    f"len(atom_decoder)(+virtual)={expected_node_bins}."
                )

        # Skip the edge-dimension consistency check when a global→local edge remap is active
        # because output_dims["E"] reflects the global vocab size (Se), not the local one.
        if not has_edge_remap and edge_types_ref is not None and hasattr(edge_types_ref, "numel"):
            expected_edge_bins = int(edge_types_ref.numel()) + (1 if self.add_virtual_states else 0)
            if int(num_edge_types) != expected_edge_bins:
                raise ValueError(
                    "SamplingMolecularMetrics edge-dimension mismatch: "
                    f"output_dims['E'](+virtual)={int(num_edge_types)} but "
                    f"len(edge_types)(+virtual)={expected_edge_bins}."
                )

        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer("n_target_dist", n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        if self.add_virtual_states:
            node_target_dist = torch.cat(
                [node_target_dist, torch.zeros(1, device=node_target_dist.device, dtype=node_target_dist.dtype)]
            )
        self.register_buffer("node_target_dist", node_target_dist)

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        if self.add_virtual_states:
            edge_target_dist = torch.cat(
                [edge_target_dist, torch.zeros(1, device=edge_target_dist.device, dtype=edge_target_dist.dtype)]
            )
        self.register_buffer("edge_target_dist", edge_target_dist)

        valency_target_dist = di.valency_distribution.type_as(
            self.generated_valency_dist.edgepernode_dist
        )
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer("valency_target_dist", valency_target_dist)
        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = dataset_smiles["train"] if dataset_smiles != None else dataset_smiles # Necessary for regressor guidance
        self.val_smiles = dataset_smiles["val"] if dataset_smiles != None else dataset_smiles
        self.test_smiles = dataset_smiles["test"] if dataset_smiles != None else dataset_smiles
        self.dataset_info = di

    def _remap_molecules(self, molecules):
        """Remap global atom/edge vocab indices to domain-local indices."""
        has_atom = self.global_to_local_atom_map is not None
        has_edge = self.global_to_local_edge_map is not None
        if not has_atom and not has_edge:
            return molecules
        remapped = []
        for (atoms, edges) in molecules:
            new_atoms = self.global_to_local_atom_map[atoms.long()] if has_atom else atoms
            new_edges = self.global_to_local_edge_map[edges.long()] if has_edge else edges
            remapped.append((new_atoms, new_edges))
        return remapped

    def forward(
        self,
        molecules: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
        prevalidated=False,
        context=None,
    ):
        molecules = self._remap_molecules(molecules)
        stability, rdkit_metrics, all_smiles, to_log = compute_molecular_metrics(
            molecules, self.train_smiles, self.dataset_info, labels, self.cfg, test
        )

        if test and local_rank == 0:
            with open(_domain_suffixed("final_smiles.txt", self.dataset_info), "w") as fp:
                for smiles in all_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print("All smiles saved")

            print(all_smiles)
            df = pd.DataFrame(all_smiles, columns=["SMILES"])
            df.to_csv(_domain_suffixed("final_smiles.csv", self.dataset_info), index=False)
            print("All SMILES saved to CSV")

        # `compute_fcd` (legacy, per-domain config flag) gates the single backward-compatible
        # `to_log["fcd"]` key consumed by compute_ratios() below. `compute_moses_metrics` (new)
        # gates the full MOSES-style Filters/FCD/SNN/Scaf bundle. Both draw from a single
        # compute_moses_metrics() call (only at test time -- it's expensive: a ChemNet forward
        # pass plus pairwise fingerprint similarity over the whole reference set) so we never
        # pay for FCD twice.
        run_fcd = bool(getattr(self.dataset_info, "compute_fcd", False))
        run_moses_metrics = bool(getattr(self.dataset_info, "compute_moses_metrics", False))

        if test and (run_fcd or run_moses_metrics):
            reference_smiles_by_split = {"train": self.train_smiles, "test": self.test_smiles}
            # MOSESinfos' "val" split is actually MOSES's scaffold-holdout test_scaffolds.csv
            # (see MOSESDataset.download()/get_smiles() in datasets/moses_dataset.py -- the
            # dataset's own val/test file naming is swapped relative to upstream MOSES), so it
            # doubles as the TestSF reference the Scaf metric expects.
            if str(getattr(self.dataset_info, "name", "")).strip().lower() == "moses":
                reference_smiles_by_split["test_scaffolds"] = self.val_smiles

            moses_metrics = compute_moses_metrics(all_smiles, reference_smiles_by_split)

            if run_fcd:
                to_log["fcd"] = moses_metrics["FCD"]
            if run_moses_metrics:
                for key in ("Filters", "FCD", "SNN", "Scaf"):
                    to_log[key] = moses_metrics[key]
        else:
            if run_fcd:
                print("FCD computation requested but only runs at test time. Skipping.")
            to_log["fcd"] = -1
        # The legacy key is only filled when compute_fcd is on. With compute_fcd off and
        # compute_moses_metrics on, the MOSES bundle (FCD/SNN/Filters/Scaf) runs but "fcd"
        # was never set, and the print below raised KeyError -- discarding the MOSES values
        # and every metric computed after this point. -1 is the key's existing "not
        # computed" value; the real FCD stays under "FCD".
        to_log.setdefault("fcd", -1)

        print("fcd", to_log["fcd"])

        print("Starting custom metrics")
        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

        for i, atom_type in enumerate(self.dataset_info.atom_encoder.keys()):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
            # ["No bond", "Single", "Double", "Triple"]
        ):
            if j < len(generated_edge_dist):
                generated_probability = generated_edge_dist[j]
                target_probability = self.edge_target_dist[j]
                to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                    generated_probability - target_probability
                ).item()

        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                generated_probability - target_probability
            ).item()

        n_mae = self.n_dist_mae.compute()
        node_mae = self.node_dist_mae.compute()
        edge_mae = self.edge_dist_mae.compute()
        valency_mae = self.valency_dist_mae.compute()

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=["fcd"] if self.dataset_info.compute_fcd else [],
        )
        to_log.update(ratios)

        if wandb is not None and wandb.run:
            wandb.log(to_log, commit=False)
            wandb.run.summary["Gen n distribution"] = generated_n_dist
            wandb.run.summary["Gen node distribution"] = generated_node_dist
            wandb.run.summary["Gen edge distribution"] = generated_edge_dist
            wandb.run.summary["Gen valency distribution"] = generated_valency_dist

            wandb.log(
                {
                    "basic_metrics/n_mae": n_mae,
                    "basic_metrics/node_mae": node_mae,
                    "basic_metrics/edge_mae": edge_mae,
                    "basic_metrics/valency_mae": valency_mae,
                },
                commit=False,
            )

        if local_rank == 0:
            print("Custom metrics computed.")
        if local_rank == 0:
            valid_unique_molecules = rdkit_metrics[1]
            text_filename = f"graphs/{name}/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt"
            if not os.path.exists(os.path.dirname(text_filename)):
                os.makedirs(os.path.dirname(text_filename))
            textfile = open(
                text_filename,
                "w",
            )
            textfile.writelines(valid_unique_molecules)
            textfile.close()
            print("Stability metrics:", stability, "--", rdkit_metrics[0])

        return to_log

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)



class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__(dist_sync_on_step=False, sync_on_compute=False)
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            edge_types[edge_types == 5] = 0.0  # zero out virtual states
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        kwargs.setdefault("dist_sync_on_step", False)
        kwargs.setdefault("sync_on_compute", False)
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)



# Bonds MSE







