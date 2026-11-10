"""Training reset selection; geometry validity remains owned by Rust."""

from collections.abc import Callable

import numpy as np


def sample_regular_training_start(
    draw: Callable[[], complex], epsilon: float
) -> complex:
    """Reject invalid reset proposals, never invalid steps of a live rollout.

    Conditioning the training reset distribution on Regular geometry is
    explicit and bounded. Exhaustion preserves the last Rust refusal.
    """
    from runtime_core import ManifoldConfig, manifold_induced_metric

    config = ManifoldConfig(epsilon=epsilon)
    last_refusal: RuntimeError | None = None
    for _ in range(16):
        candidate = draw()
        start = complex(
            float(np.float32(candidate.real)), float(np.float32(candidate.imag))
        )
        try:
            manifold_induced_metric(start, config)
        except RuntimeError as error:
            last_refusal = error
            continue
        return start
    raise RuntimeError(
        "16 proposals failed to provide a Regular training start"
    ) from last_refusal
