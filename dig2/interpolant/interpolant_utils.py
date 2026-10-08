import torch
from torch.nn import functional as F

from dig2.utils import PlaceHolder
from dig2.debug import hotpath_asserts_enabled


def assert_correctly_masked(variable, node_mask):
    # .item() forces a GPU->CPU sync; this runs per layer per forward, so it is
    # gated off by default (enable via general.debug_asserts for correctness runs).
    if not hotpath_asserts_enabled():
        return
    assert (
        variable * (1 - node_mask.long())
    ).abs().max().item() < 1e-4, "Variables not masked properly."


def sample_discrete_feature_noise(limit_dist, node_mask, generator=None):
    """Sample from the limit distribution of the diffusion process.

    `generator` (optional torch.Generator): when None (default), draws from the
    default global RNG stream -- byte-identical to pre-existing behavior. Passing
    an explicit generator (see GraphDiscreteInterpolativeModel.sample_per_domain's
    `sample.rng_safe_per_domain` opt-in) makes this draw reproducible independent
    of any other RNG consumption happening elsewhere in the process.
    """
    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    y_limit = limit_dist.y[None, :].expand(bs, -1)
    U_X = (
        x_limit.flatten(end_dim=-2)
        .multinomial(1, replacement=True, generator=generator)
        .reshape(bs, n_max)
    )
    U_E = (
        e_limit.flatten(end_dim=-2)
        .multinomial(1, replacement=True, generator=generator)
        .reshape(bs, n_max, n_max)
    )
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def _sample_discrete_nodes(probX, node_mask, generator=None):
    """Node-only half of `sample_discrete_features`, which delegates its X-sampling here."""
    probX = probX.clone()
    bs, n, _ = probX.shape
    probX[~node_mask] = 1 / probX.shape[-1]
    probX = probX.reshape(bs * n, -1)
    X_t = probX.multinomial(1, replacement=True, generator=generator)
    return X_t.reshape(bs, n)


def sample_discrete_features(
    probX, probE, node_mask, mask=False, generator=None,
    inverse_edge_mask=None, diag_mask=None,
):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    :param generator: optional torch.Generator; None (default) draws from the default
        global RNG stream, byte-identical to pre-existing behavior. See
        sample_discrete_feature_noise's docstring for the reproducibility rationale.
    :param inverse_edge_mask: optional precomputed `~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))`.
        node_mask is fixed for an entire sample_batch() denoising loop, so callers that
        invoke this once per step (e.g. sample_p_zs_given_zt) can pass a cached mask
        instead of rebuilding it from scratch every step. None (default) recomputes it
        here, unchanged from prior behavior.
    :param diag_mask: optional precomputed `torch.eye(n, device=probE.device).expand(bs, -1, -1)`,
        same rationale as `inverse_edge_mask`.
    """
    # Clone to avoid in-place mutation on autograd-tracked tensors
    # (e.g., softmax outputs from model forward passes).
    probE = probE.clone()

    bs, n, _ = probX.shape
    X_t = _sample_discrete_nodes(probX, node_mask, generator=generator)

    # Noise E
    # The masked rows should define probability distributions as well
    if inverse_edge_mask is None:
        inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    if diag_mask is None:
        diag_mask = torch.eye(n, device=probE.device).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1, replacement=True, generator=generator).reshape(bs, n, n)  # (bs, n, n)
    # E_t = Categorical(probs=probE).sample().reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    if mask:
        X_t = X_t * node_mask
        E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t))


