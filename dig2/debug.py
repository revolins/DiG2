"""Global switch for hot-loop correctness asserts.

These asserts call .all()/.item() on CUDA tensors, forcing a GPU->CPU
synchronization on every invocation — per transformer layer per forward, and
per denoising step during sampling. They are disabled by default and enabled
for opt-in correctness runs via `general.debug_asserts: true`.
"""

_ENABLED = False


def hotpath_asserts_enabled() -> bool:
    return _ENABLED
