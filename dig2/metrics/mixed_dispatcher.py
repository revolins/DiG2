from collections.abc import Mapping

import torch

from dig2.metrics.unified_context import UnifiedMetricContext


class MixedMetricDispatcher(torch.nn.Module):
    """Dispatch mixed-domain evaluation to synthetic or molecular evaluators.

    Expected generated sample format for mixed-mode evaluation:
      {"graph": [X, E], "dataset_name": str, "dataset_id": int}
    """

    def __init__(
        self,
        *,
        domain_type_by_name,
        synthetic_evaluator,
        molecular_evaluator_by_domain,
        use_unified_metrics=False,
        shared_prefilter=True,
        shared_cache=True,
    ):
        super().__init__()
        self.domain_type_by_name = {
            str(k).strip().lower(): str(v).strip().lower()
            for k, v in (domain_type_by_name or {}).items()
        }
        self.synthetic_evaluator = synthetic_evaluator
        self.molecular_evaluator_by_domain = molecular_evaluator_by_domain or {}
        self.use_unified_metrics = bool(use_unified_metrics)
        self.shared_prefilter = bool(shared_prefilter)
        self.shared_cache = bool(shared_cache)

    def reset(self):
        if hasattr(self.synthetic_evaluator, "reset"):
            self.synthetic_evaluator.reset()
        for evaluator in self.molecular_evaluator_by_domain.values():
            if hasattr(evaluator, "reset"):
                evaluator.reset()

    def _safe_num_nodes(self, mol):
        try:
            return int(mol[0].shape[0])
        except Exception:
            return None

    def _molecular_graph_decoder_validity(self, mol, evaluator):
        """Check whether atom ids in `mol` are valid for evaluator's atom decoder.

        Returns tuple: (is_valid: bool, reason: str)
        """
        dataset_info = getattr(evaluator, "dataset_info", None)
        atom_decoder = getattr(dataset_info, "atom_decoder", None)
        if atom_decoder is None:
            return False, "missing_atom_decoder"

        try:
            atom_types = mol[0]
        except Exception:
            return False, "malformed_graph"

        if torch.is_tensor(atom_types):
            atom_ids = atom_types.detach().view(-1).long()
        else:
            atom_ids = torch.as_tensor(atom_types).view(-1).long()

        if atom_ids.numel() == 0:
            return False, "empty_atoms"
        if torch.any(atom_ids < 0):
            return False, "negative_atom_index"

        # Apply global→local remap when the evaluator carries one (e.g. Zinc/QM9 in unified
        # semantic vocab).  Without this, Br (global 9) and I (global 10) are incorrectly
        # dropped as out-of-bounds against Zinc's local decoder_size=9 before the correct
        # remap inside SamplingMolecularMetrics.forward() ever runs.
        g2l = getattr(evaluator, "global_to_local_atom_map", None)
        if g2l is not None:
            g2l_cpu = g2l.cpu()
            if torch.any(atom_ids >= g2l_cpu.numel()):
                return False, "atom_index_oob"
            atom_ids = g2l_cpu[atom_ids]
            if torch.any(atom_ids < 0):
                return False, "atom_not_in_domain"

        decoder_size = int(len(atom_decoder))
        if decoder_size <= 0:
            return False, "empty_atom_decoder"
        if torch.any(atom_ids >= decoder_size):
            return False, "atom_index_oob"

        return True, "ok"

    def _partition_generated_graphs(self, generated_graphs):
        synthetic_graphs = []
        molecular_graphs_by_domain = {}

        for item in generated_graphs:
            domain_name = str(item.get("dataset_name", "")).strip().lower()
            domain_type = self.domain_type_by_name.get(domain_name, "synthetic")
            if domain_type == "molecular":
                molecular_graphs_by_domain.setdefault(domain_name, []).append(item["graph"])
            else:
                synthetic_graphs.append(item)

        return synthetic_graphs, molecular_graphs_by_domain

    def _build_eval_records(self, generated_graphs):
        records = []
        for idx, item in enumerate(generated_graphs):
            domain_name = str(item.get("dataset_name", "")).strip().lower()
            domain_type = self.domain_type_by_name.get(domain_name, "synthetic")
            graph = item.get("graph", None)
            n_nodes = self._safe_num_nodes(graph) if graph is not None else None
            records.append(
                {
                    "record_id": idx,
                    "graph": graph,
                    "dataset_name": domain_name,
                    "domain_type": domain_type,
                    "n_nodes": n_nodes,
                    "drop_reason": None,
                }
            )
        return records

    def _shared_prefilter(self, records):
        kept = []
        dropped_by_domain = {}
        for rec in records:
            if rec.get("graph", None) is None:
                rec["drop_reason"] = "missing_graph"
            elif rec.get("n_nodes", None) is None:
                rec["drop_reason"] = "malformed_graph"
            elif int(rec["n_nodes"]) <= 0:
                rec["drop_reason"] = "empty_graph"

            if rec.get("drop_reason", None) is None:
                kept.append(rec)
                continue

            domain = rec.get("dataset_name", "unknown")
            reasons = dropped_by_domain.setdefault(domain, {})
            reason = rec["drop_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        return kept, dropped_by_domain

    def forward(
        self,
        generated_graphs,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
    ):
        if len(generated_graphs) == 0:
            raise AssertionError("Returned Empty Generated Graphs, likely failure to map metadata?")

        # Plain [X, E] graphs are used in reference-style paths. Delegate to the
        # synthetic evaluator to keep synthetic metric machinery available.
        if not isinstance(generated_graphs[0], Mapping):
            return self.synthetic_evaluator.forward(
                generated_graphs,
                ref_metrics=ref_metrics,
                name=name,
                current_epoch=current_epoch,
                val_counter=val_counter,
                local_rank=local_rank,
                test=test,
                labels=labels,
            )

        if self.use_unified_metrics:
            context = UnifiedMetricContext()
            records = self._build_eval_records(generated_graphs)
            if self.shared_cache:
                context.record_by_id = {int(r["record_id"]): r for r in records}
            dropped_prefilter = {}
            if self.shared_prefilter:
                records, dropped_prefilter = self._shared_prefilter(records)

            synthetic_graphs = []
            molecular_graphs_by_domain = {}
            for rec in records:
                domain_name = rec["dataset_name"]
                item = {"graph": rec["graph"], "dataset_name": domain_name}
                if rec["domain_type"] == "molecular":
                    molecular_graphs_by_domain.setdefault(domain_name, []).append(rec["graph"])
                else:
                    synthetic_graphs.append(item)
        else:
            context = None
            dropped_prefilter = {}
            synthetic_graphs, molecular_graphs_by_domain = self._partition_generated_graphs(
                generated_graphs
            )

        # Fast path: pure synthetic batch.
        if len(synthetic_graphs) > 0 and len(molecular_graphs_by_domain) == 0:
            return self.synthetic_evaluator.forward(
                synthetic_graphs,
                ref_metrics=ref_metrics,
                name=name,
                current_epoch=current_epoch,
                val_counter=val_counter,
                local_rank=local_rank,
                test=test,
                labels=labels,
            )

        to_log = {}

        for domain_name, reasons in dropped_prefilter.items():
            total = float(sum(reasons.values()))
            if total > 0:
                to_log[f"cond/domain/{domain_name}/dropped_invalid_samples"] = total
            for reason, count in reasons.items():
                to_log[f"cond/domain/{domain_name}/dropped/{reason}"] = float(count)

        if len(synthetic_graphs) > 0:
            synth_log = self.synthetic_evaluator.forward(
                synthetic_graphs,
                ref_metrics=ref_metrics,
                name=name,
                current_epoch=current_epoch,
                val_counter=val_counter,
                local_rank=local_rank,
                test=test,
                labels=labels,
            )
            to_log.update(synth_log)

        for domain_name, molecules in molecular_graphs_by_domain.items():
            evaluator = self.molecular_evaluator_by_domain.get(domain_name)
            if evaluator is None:
                continue
            # Generic defensive filtering by evaluator-declared constraints.
            filtered_molecules = []
            dropped = 0
            dropped_by_reason = {}
            dataset_info = getattr(evaluator, "dataset_info", None)
            max_n_nodes = getattr(dataset_info, "max_n_nodes", None)

            for mol in molecules:
                n_nodes = self._safe_num_nodes(mol)
                if n_nodes is None:
                    dropped += 1
                    dropped_by_reason["malformed_graph"] = dropped_by_reason.get("malformed_graph", 0) + 1
                    continue
                if max_n_nodes is not None and n_nodes > int(max_n_nodes):
                    dropped += 1
                    dropped_by_reason["max_n_nodes"] = dropped_by_reason.get("max_n_nodes", 0) + 1
                    continue

                is_decoder_valid, reason = self._molecular_graph_decoder_validity(mol, evaluator)
                if not is_decoder_valid:
                    dropped += 1
                    dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
                    continue

                filtered_molecules.append(mol)

            if len(filtered_molecules) == 0:
                if dropped > 0:
                    to_log[f"cond/domain/{domain_name}/dropped_invalid_samples"] = float(dropped)
                    for reason, count in dropped_by_reason.items():
                        to_log[f"cond/domain/{domain_name}/dropped/{reason}"] = float(count)
                continue

            mol_log = evaluator.forward(
                filtered_molecules,
                ref_metrics=ref_metrics,
                name=name,
                current_epoch=current_epoch,
                val_counter=val_counter,
                local_rank=local_rank,
                test=test,
                labels=None,
                prevalidated=bool(self.use_unified_metrics and self.shared_prefilter),
                context=context,
            )
            if dropped > 0:
                mol_log["dropped_invalid_samples"] = float(dropped)
                for reason, count in dropped_by_reason.items():
                    mol_log[f"dropped/{reason}"] = float(count)
            for key, value in mol_log.items():
                to_log[f"cond/domain/{domain_name}/{key}"] = value

        return to_log
