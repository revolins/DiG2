import torch
from dig2 import utils
from dig2.debug import hotpath_asserts_enabled


# ---------------------------------------------------------------------------
# Sparse RRWP
# ---------------------------------------------------------------------------

FILL_THRESHOLD = 0.15   # switch to dense when nnz / N^2 exceeds this


class DummyExtraFeatures:
    def __init__(self):
        """This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    """Extra input features: RRWP for nodes/edges, and graph size + cycle counts for y."""

    def __init__(self, extra_features_type, rrwp_steps, dataset_info):
        if extra_features_type != "rrwp":
            raise ValueError(f"model.extra_features must be 'rrwp', got {extra_features_type!r}")
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.rrwp_steps = rrwp_steps
        self.RRWP = RRWPFeatures()

    def __call__(self, noisy_data):
        n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1).float() / self.max_n_nodes
        _, y_cycles = self.ncycles(noisy_data)  # (bs, n_cycles)

        E = noisy_data["E_t"].float()[..., 1:].sum(-1)  # bs, n, n
        # Edge RRWP from the noisy adjacency; node RRWP is its diagonal.
        rrwp_edge_attr = self.RRWP(E, k=self.rrwp_steps)
        diag_index = torch.arange(rrwp_edge_attr.shape[1], device=rrwp_edge_attr.device)
        rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]

        return utils.PlaceHolder(
            X=rrwp_node_attr,
            E=rrwp_edge_attr,
            y=torch.hstack((n, y_cycles)),
        )


class RRWPFeatures:
    def __init__(self, k=10, normalize=True):
        self.k = k
        self.normalize = normalize

    def __call__(self, E, k=None):
        k = k or self.k

        (
            bs,
            n,
            _,
        ) = E.shape
        if self.normalize:
            degree = torch.zeros(bs, n, n, device=E.device)
            to_fill = 1 / (E.sum(dim=-1).float())
            to_fill[E.sum(dim=-1).float() == 0] = 0
            degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
            E = degree @ E

        id = torch.eye(n, device=E.device).unsqueeze(0).repeat(bs, 1, 1)
        rrwp_list = [id]

        for i in range(k - 1):
            cur_rrwp = rrwp_list[-1] @ E
            rrwp_list.append(cur_rrwp)

        return torch.stack(rrwp_list, -1)


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data["E_t"][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(
            adj_matrix=adj_matrix
        )  # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data["node_mask"].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        # print("self.k1_matrix: ", self.k1_matrix.size())
        # print("self.adj_matrix: ", self.adj_matrix.size())
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        # print("self.k2_matrx: ", self.k2_matrix.size())
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        # print("self.k3_matrx: ", self.k3_matrix.size())
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix**2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()

        # .all() on CUDA tensors syncs the host; this runs every training and
        # sampling step, so the checks are gated behind general.debug_asserts.
        if hotpath_asserts_enabled():
            assert (k3x >= -0.1).all()
            assert (k4x >= -0.1).all()
            assert (k5x >= -0.1).all(), k5x
            assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
