"""Stubs for src.control_model used by tests."""

from __future__ import annotations

from typing import Sequence, Any

class AudioToControlModel:
    def __init__(
        self,
        window_frames: int,
        n_features_per_frame: int,
        hidden_dims: Sequence[int],
        k_bands: int,
    ) -> None: ...
    def parameters(self) -> Sequence[Any]: ...
