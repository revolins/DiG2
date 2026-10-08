import torch.nn.functional as F

from dig2.debug import hotpath_asserts_enabled


def p_xt_g_x1(X1, E1, t, limit_dist):
    # x1 (B, D)
    # t float
    # returns (B, D, S) for varying x_t value
    device = X1.device
    limit_dist.X = limit_dist.X.to(device)
    limit_dist.E = limit_dist.E.to(device)

    t_time = t.squeeze(-1)[:, None, None]
    X1_onehot = F.one_hot(X1, num_classes=len(limit_dist.X)).float()
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()

    Xt = t_time * X1_onehot + (1 - t_time) * limit_dist.X[None, None, :]
    Et = (
        t_time[:, None] * E1_onehot
        + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    )

    if hotpath_asserts_enabled():
        assert ((Xt.sum(-1) - 1).abs() < 1e-4).all() and (
            (Et.sum(-1) - 1).abs() < 1e-4
        ).all()

    return Xt.clamp(min=0.0, max=1.0), Et.clamp(min=0.0, max=1.0)

