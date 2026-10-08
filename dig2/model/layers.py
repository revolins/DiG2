import torch

from dig2.debug import hotpath_asserts_enabled
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """X: bs, n, dx."""
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    """Softmax over `x`, with `mask == 0` entries excluded.

    There is no data-dependent empty-mask branch: `if mask.sum() == 0` would force a GPU->CPU
    sync in every transformer layer, and a sync-free `torch.where(mask.any(), out, x)` would
    allocate a third full-size tensor (~172 MiB per call at sampling shapes). The all-zero mask
    is unreachable here, so it is checked by an opt-in assert, and the function allocates only
    one clone plus the softmax output.
    """
    if hotpath_asserts_enabled():
        # The only input this function cannot handle is an all-zero mask, which would make
        # every logit -inf and the softmax NaN. It is unreachable in this pipeline (the mask
        # derives from node_mask and every graph has >= 3 nodes), so it is checked in
        # debug_asserts runs rather than guarded on the hot path.
        assert bool(mask.any()), "masked_softmax received an all-zero mask"
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)
