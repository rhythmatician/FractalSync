"""Stubs for src.data_loader used in tests."""

from __future__ import annotations

from typing import Iterable, Sequence

class DataLoader:
    def __init__(self, data_dir: str) -> None: ...
    def __iter__(self) -> Iterable[Sequence[float]]: ...

def discover_audio(data_dir: str) -> list[str]: ...
