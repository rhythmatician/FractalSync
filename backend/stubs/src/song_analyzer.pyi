"""Stubs for src.song_analyzer used by tests."""

from __future__ import annotations

from typing import Any

class SongAnalyzer:
    def __init__(self, model: Any) -> None: ...
    def analyze(self, audio: Any) -> dict[str, float]: ...
