import torch_geometric.utils

from dig2.debug import hotpath_asserts_enabled
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)

def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask

def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E

class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, collapse=False):
        # print("SANITY CHECK", flush=True)
        # Accept node_mask as [B,N] or [B,N,1]
        if node_mask.dim() == 3:
            node_mask_2d = node_mask.squeeze(-1)  # [B,N]
        else:
            node_mask_2d = node_mask              # [B,N]

        # bool/float both ok
        if node_mask_2d.dtype == torch.bool:
            nm = node_mask_2d
            nm_f = nm.to(self.X.dtype) if self.X is not None else None
        else:
            nm_f = node_mask_2d.to(self.X.dtype) if self.X is not None else node_mask_2d
            nm = node_mask_2d > 0.5

        x_mask = (nm_f if nm_f is not None else nm.to(torch.float32)).unsqueeze(-1)  # [B,N,1]
        e_mask1 = x_mask.unsqueeze(2)  # [B,N,1,1]
        e_mask2 = x_mask.unsqueeze(1)  # [B,1,N,1]

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[~nm] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            # print("Self.X: ", self.X.size())
            # print("x_mask size: ", x_mask.size(), flush=True)
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            # Ensure E is symmetric due to floating point issues
            self.E = 0.5 * (self.E + torch.transpose(self.E, 1, 2))

            # `torch.allclose` returns a Python bool => GPU->CPU sync, and `.mask()` is
            # called several times per step. Measured at 7.0 syncs/training-step and 6.2 per
            # reverse sampling step. Gated like every other hot-loop assert; the line directly
            # above it already ENFORCES the symmetry it checks.
            if hotpath_asserts_enabled():
                assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
            
        return self

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """Split a PlaceHolder representing a batch into a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list

class SparsePlaceHolder:
    """Sparse graph representation that avoids dense N×N padding.

    Stores the concatenated-batch format used by PyG — all active nodes and
    edges from the batch are packed into flat tensors without zero-padding.

    Attributes:
        X:          (N_total, dx)     — node features, no padding
        edge_index: (2, E_active)     — active edge pairs in global node indices
        edge_attr:  (E_active, de)    — edge features for class ≠ no-edge pairs
        batch:      (N_total,)        — graph index per node
        y:          (B, dy)           — global conditioning per graph
        n_nodes:    (B,)              — number of real nodes per graph
        rrwp_node:  (N_total, k)      — optional; RRWP return probs per node
        rrwp_edge:  (E_active, k)     — optional; RRWP transition probs per edge
    """

    def __init__(self, X, edge_index, edge_attr, batch, y, n_nodes=None,
                 rrwp_node=None, rrwp_edge=None):
        self.X = X
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.y = y
        self.n_nodes = n_nodes
        self.rrwp_node = rrwp_node
        self.rrwp_edge = rrwp_edge

    @property
    def B(self):
        if self.y is not None and self.y.dim() >= 1:
            return self.y.shape[0]
        if self.batch.numel() > 0:
            return int(self.batch.max().item()) + 1
        return 0

    @property
    def N_total(self):
        return self.X.shape[0]

    @property
    def E_active(self):
        return self.edge_attr.shape[0]

    def to_device(self, device):
        self.X = self.X.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.batch = self.batch.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        if self.n_nodes is not None:
            self.n_nodes = self.n_nodes.to(device)
        if self.rrwp_node is not None:
            self.rrwp_node = self.rrwp_node.to(device)
        if self.rrwp_edge is not None:
            self.rrwp_edge = self.rrwp_edge.to(device)
        return self


    def __repr__(self):
        return (
            f"SparsePlaceHolder(N_total={self.N_total}, E_active={self.E_active}, "
            f"B={self.B}, dx={self.X.shape[-1]}, de={self.edge_attr.shape[-1]})"
        )

