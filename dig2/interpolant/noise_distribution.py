import torch

from dig2 import utils


class NoiseDistribution:

    def __init__(self, model_transition, dataset_infos):

        self.x_num_classes = dataset_infos.output_dims["X"]
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        self.x_added_classes = 0
        self.e_added_classes = 0
        self.y_added_classes = 0
        self.transition = model_transition

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[0] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            node_types = dataset_infos.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            # only add virtual classes when there are several
            if self.x_num_classes > 1:
                # if self.x_num_classes >= 1:
                self.x_num_classes += 1
                self.x_added_classes = 1
            if self.e_num_classes > 1:
                self.e_num_classes += 1
                self.e_added_classes = 1

            x_limit = torch.zeros(self.x_num_classes)
            x_limit[-1] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[-1] = 1

        elif model_transition == "marginal":

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        # Optional per-domain priors (first pass): expected shapes are [D, X] and [D, E].
        # Keep explicit status/warnings so prior_mode=domain_specific experiments cannot
        # silently fall back to the global prior because of missing or malformed stats.
        self.domain_limit_dist = {}
        self.domain_prior_status = {}
        self.domain_prior_warnings = []
        domain_x = getattr(dataset_infos, "domain_node_types", None)
        domain_e = getattr(dataset_infos, "domain_edge_types", None)
        if torch.is_tensor(domain_x) and torch.is_tensor(domain_e):
            d_dim = min(int(domain_x.size(0)), int(domain_e.size(0)))
            for did in range(d_dim):
                x_d = domain_x[did].float()
                e_d = domain_e[did].float()
                if x_d.numel() != self.x_num_classes or e_d.numel() != self.e_num_classes:
                    msg = (
                        f"domain_idx={int(did)} shape mismatch for domain prior: "
                        f"x_numel={int(x_d.numel())} expected_x={int(self.x_num_classes)}, "
                        f"e_numel={int(e_d.numel())} expected_e={int(self.e_num_classes)}"
                    )
                    self.domain_prior_status[int(did)] = {
                        "ok": False,
                        "reason": "shape_mismatch",
                        "x_numel": int(x_d.numel()),
                        "expected_x": int(self.x_num_classes),
                        "e_numel": int(e_d.numel()),
                        "expected_e": int(self.e_num_classes),
                    }
                    self.domain_prior_warnings.append(msg)
                    continue
                x_sum = x_d.sum()
                e_sum = e_d.sum()
                if x_sum <= 0 or e_sum <= 0:
                    msg = (
                        f"domain_idx={int(did)} zero-mass domain prior: "
                        f"x_sum={float(x_sum.item())}, e_sum={float(e_sum.item())}"
                    )
                    self.domain_prior_status[int(did)] = {
                        "ok": False,
                        "reason": "zero_mass",
                        "x_sum": float(x_sum.item()),
                        "e_sum": float(e_sum.item()),
                    }
                    self.domain_prior_warnings.append(msg)
                    continue
                x_d = x_d / x_sum.clamp(min=1e-8)
                e_d = e_d / e_sum.clamp(min=1e-8)
                self.domain_limit_dist[int(did)] = utils.PlaceHolder(
                    X=x_d,
                    E=e_d,
                    y=y_limit.clone(),
                )
                self.domain_prior_status[int(did)] = {"ok": True}

        if len(self.domain_limit_dist) > 0:
            print(
                f"[NoiseDistribution] Built {len(self.domain_limit_dist)} domain-specific priors"
            )
        elif torch.is_tensor(domain_x) or torch.is_tensor(domain_e):
            print(
                "[NoiseDistribution] No valid domain-specific priors were built; "
                f"warnings={self.domain_prior_warnings}"
            )

    def update_input_output_dims(self, input_dims):
        input_dims["X"] += self.x_added_classes
        input_dims["E"] += self.e_added_classes
        input_dims["y"] += self.y_added_classes

    def update_dataset_infos(self, dataset_infos):
        if hasattr(dataset_infos, "atom_decoder"):
            dataset_infos.atom_decoder = (
                dataset_infos.atom_decoder + ["Y"] * self.x_added_classes
            )

    def get_limit_dist(self, domain_idx=None, prior_mode="default"):
        mode = str(prior_mode).strip().lower()
        if mode == "domain_specific" and domain_idx is not None:
            try:
                did = int(domain_idx)
            except Exception:
                did = None
            if did is not None and did in self.domain_limit_dist:
                return self.domain_limit_dist[did]
        return self.limit_dist

    def resolve_limit_dist(self, domain_idx=None, prior_mode="default"):
        """Return (limit_dist, source), where source is 'domain_specific' or 'global'."""
        mode = str(prior_mode).strip().lower()
        if mode == "domain_specific" and domain_idx is not None:
            try:
                did = int(domain_idx)
            except Exception:
                did = None
            if did is not None and did in self.domain_limit_dist:
                return self.domain_limit_dist[did], "domain_specific"
        return self.limit_dist, "global"



    def get_noise_dims(self):
        return {
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        if self.transition == "absorbing":
            new_X = X[..., : -self.x_added_classes]
            new_E = E[..., : -self.e_added_classes]
            new_y = y[..., : -self.y_added_classes] if y is not None else None
            return new_X, new_E, new_y
        else:
            return X, E, y

