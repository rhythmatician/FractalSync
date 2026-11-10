"""Static-only control: an opt-in caller can use both runtime API surfaces."""

import runtime_core


def accept_both(
    config: runtime_core.ManifoldConfig,
    rollout: runtime_core.BoundedOperationalRollout,
) -> tuple[runtime_core.ManifoldConfig, runtime_core.BoundedOperationalRollout]:
    return config, rollout
