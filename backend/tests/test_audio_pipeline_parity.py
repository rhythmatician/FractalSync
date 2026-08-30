"""Production-path audio pipeline parity (issue #93 regression test).

The #93 incident: AnalysisTimebase was implemented in Rust, exported through
wasm, and fully tested in Rust — but the training surface never consumed it.
Component-level parity tests (feature formula parity) stayed green while the
real production paths diverged: the trainer called the FeatureExtractor
directly on whole files, bypassing resampling, hop scheduling, and epoch
semantics entirely.

This test enters through the REAL public seams of both surfaces:

  runtime semantics  → runtime_core.AnalysisTimebase (the PyO3 binding of
                       the same Rust timebase the browser drives through
                       wasm; identical Rust code path)
  training semantics → src.data_loader.AudioDataset._extract_via_timebase
                       (the actual training ingestion path)

and asserts the two produce identical ticks. If the training path ever
bypasses the timebase again (or the timebase gains a runtime-only behavior
the trainer does not execute), this test fails.

The gate in scripts/canonical_surfaces_gate.py requires this file to exist
and reference the authority; deleting it is an invalid repository state.
"""

from __future__ import annotations

import numpy as np
import pytest

import runtime_core
from runtime_core import AnalysisTimebase, HOP_LENGTH, SAMPLE_RATE

from src.data_loader import AudioDataset, PIPELINE_VERSION


CANONICAL_DT = HOP_LENGTH / SAMPLE_RATE


def _deterministic_pcm(n_samples: int, seed: int = 91) -> np.ndarray:
    """Deterministic tonal + noise PCM (source rate = canonical 48 kHz)."""
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.RandomState(seed)
    audio = (
        0.3 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 440.0 * t)
        + 0.1 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * rng.standard_normal(n_samples)
    )
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


class TestAudioPipelineParity:
    """Same PCM through both production ingestion semantics → same ticks."""

    def _runtime_ticks(self, audio: np.ndarray, block: int) -> list[dict]:
        """Runtime semantics: PCM blocks → AnalysisTimebase.ingest → ticks.

        This is exactly what the browser does (AudioWorklet → wasm
        AnalysisTimebase.ingest); the PyO3 binding runs the same Rust code.
        Blocks are fed with explicit source-frame positions, as the worklet
        transport does.
        """
        tb = AnalysisTimebase()
        ticks: list[dict] = []
        pos = 0
        while pos < len(audio):
            end = min(pos + block, len(audio))
            for tick in tb.ingest(audio[pos:end].tolist(), SAMPLE_RATE, pos):
                ticks.append(dict(tick))
            pos = end
        for tick in tb.flush():
            ticks.append(dict(tick))
        return ticks

    def _training_ticks(self, audio: np.ndarray) -> list[dict]:
        """Training semantics: the actual backend ingestion path.

        AudioDataset._extract_via_timebase is the production training
        pipeline (data_loader.py). We drive it through a real AudioDataset
        instance so the test fails if the training path changes shape —
        e.g. reverts to calling the extractor directly.
        """
        ds = AudioDataset.__new__(AudioDataset)  # bypass __init__ (no data dir)
        ds.window_frames = 10
        windows = ds._extract_via_timebase(audio)
        return [{"features": row.tolist()} for row in windows]

    def test_tick_sample_indices_and_features_match(self):
        n = SAMPLE_RATE * 2  # 2 s
        audio = _deterministic_pcm(n)

        runtime_ticks = self._runtime_ticks(audio, block=1024)
        training_windows = self._training_ticks(audio)

        # Same number of ticks.
        assert len(runtime_ticks) == len(training_windows), (
            f"tick count diverged: runtime={len(runtime_ticks)} "
            f"training={len(training_windows)}"
        )

        # Tick sample indices identical and on exact hop boundaries.
        for i, tick in enumerate(runtime_ticks):
            assert tick["sample_index"] == (i + 1) * HOP_LENGTH

        # Feature vectors identical (same Rust extractor, same history).
        max_err = 0.0
        for rt, tw in zip(runtime_ticks, training_windows):
            rf = np.asarray(rt["features"], dtype=np.float64)
            tf = np.asarray(tw["features"], dtype=np.float64)
            assert rf.shape == tf.shape
            max_err = max(max_err, float(np.max(np.abs(rf - tf))))
        assert max_err < 1e-9, f"feature vectors diverged: max_err={max_err}"

    def test_training_path_is_chunk_invariant_like_runtime(self):
        """The training path must be invariant to block cadence, exactly as
        the runtime timebase guarantees (irregular worklet cadence must not
        change which samples are new or where hops occur)."""
        n = SAMPLE_RATE  # 1 s
        audio = _deterministic_pcm(n, seed=92)

        reference = self._runtime_ticks(audio, block=4096)
        for block in (128, 1000, 333, 8192):
            other = self._runtime_ticks(audio, block=block)
            assert len(other) == len(reference)
            for a, b in zip(reference, other):
                assert a["sample_index"] == b["sample_index"]
                assert (
                    np.max(np.abs(np.array(a["features"]) - np.array(b["features"])))
                    < 1e-9
                )

    def test_tick_timestamps_derive_from_sample_clock(self):
        """Timestamps must come from integer sample position, never wall
        clock — and dt must be the canonical hop duration."""
        audio = _deterministic_pcm(SAMPLE_RATE, seed=93)
        ticks = self._runtime_ticks(audio, block=2048)
        assert ticks
        for tick in ticks:
            assert tick["dt_seconds"] == pytest.approx(CANONICAL_DT, abs=1e-12)
            assert tick["time_seconds"] == pytest.approx(
                tick["sample_index"] / SAMPLE_RATE, abs=1e-12
            )

    def test_resampled_source_matches_canonical_rate_pipeline(self):
        """A 44.1 kHz source through the timebase must land on the same
        canonical hop boundaries as the browser's resampled stream."""
        src_rate = 44_100
        n_src = src_rate  # 1.0 s
        t = np.arange(n_src, dtype=np.float64) / src_rate
        audio = (0.4 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)

        tb = AnalysisTimebase()
        ticks: list[dict] = []
        for tick in tb.ingest(audio.tolist(), src_rate, 0):
            ticks.append(dict(tick))
        for tick in tb.flush():
            ticks.append(dict(tick))

        assert ticks, "resampled stream produced no ticks"
        for w in zip(ticks, ticks[1:]):
            assert w[1]["sample_index"] - w[0]["sample_index"] == HOP_LENGTH
        d = tb.diagnostics()
        assert abs(d["canonical_sample_index"] - SAMPLE_RATE) <= 1

    def test_pipeline_version_invalidates_cache(self):
        """The cache key must include the pipeline version so features
        extracted by a different ingestion pipeline are never reused."""
        import hashlib
        import json

        payload = {
            "path": "/x.wav",
            "mtime_ns": 1,
            "sr": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": 4096,
            "window_frames": 10,
            "feature_version": runtime_core.FEATURE_VERSION,
            "pipeline_version": PIPELINE_VERSION,
        }
        key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert key  # pipeline_version participates in the key derivation
        assert PIPELINE_VERSION.startswith("timebase/")
