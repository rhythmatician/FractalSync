"""Minimal stubs for src.control_trainer used in tests."""

from __future__ import annotations

from typing import Any

class ControlTrainer:
    def __init__(self, model: Any) -> None: ...
    def train_step(self, batch: Any) -> float: ...
