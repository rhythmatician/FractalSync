"""Minimal stubs for src.runtime_core_bridge used in tests."""

from __future__ import annotations

from typing import Any, Optional

SAMPLE_RATE: int
HOP_LENGTH: int
N_FFT: int
WINDOW_FRAMES: int

def make_feature_extractor(
    include_delta: bool = ..., include_delta_delta: bool = ...
) -> Any: ...
def make_orbit_state(*args: Any, **kwargs: Any) -> Any: ...
def make_residual_params(*args: Any, **kwargs: Any) -> Any: ...
def synthesize(state: Any, residual_params: Optional[Any] = None) -> Any: ...
