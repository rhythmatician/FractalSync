"""
Data loading utilities for audio files.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np
from numpy.typing import NDArray
from runtime_core import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    FEATURE_VERSION,
    WINDOW_FRAMES,
    FeatureExtractor,
    AnalysisTimebase,
)

import librosa

# Bump when the training ingestion pipeline changes shape (e.g. switching
# from direct extractor calls to the canonical AnalysisTimebase). Old caches
# computed by a different pipeline MUST NOT be reused — they were produced
# by a path the runtime does not execute (the #93 failure mode).
# NOTE: the authoritative pipeline version lives in Rust
# (runtime_core.ANALYSIS_PIPELINE_VERSION) and is stamped into ONNX
# metadata + checked by the browser; this local copy only keys the feature
# cache and must be kept in sync with the Rust constant.
PIPELINE_VERSION = "timebase/1"


class AudioDataset:
    """Dataset for loading and preprocessing audio files with optional disk cache."""

    def __init__(
        self,
        data_dir: str,
        feature_extractor=None,
        window_frames: int = 10,
        max_files: Optional[int] = None,
        cache_dir: Optional[str] = "data/cache",
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing audio files
            feature_extractor: Feature extractor instance
            window_frames: Number of frames per window
            max_files: Maximum number of files to load (None for all)
            cache_dir: Directory to persist extracted features (None to disable cache)
        """
        self.data_dir = Path(data_dir)
        # The canonical timebase emits WINDOW_FRAMES-frame windows (the
        # browser consumes exactly this contract via AnalysisTick.features).
        # A trainer-configurable window size would produce model inputs the
        # runtime never sees — a production-path divergence (#93 class).
        if window_frames != WINDOW_FRAMES:
            raise ValueError(
                f"window_frames={window_frames} but the canonical timebase "
                f"emits WINDOW_FRAMES={WINDOW_FRAMES} windows; training must "
                "consume the same tick contract as the runtime"
            )
        self.window_frames = window_frames
        self.max_files = max_files
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.feature_extractor = feature_extractor or FeatureExtractor(
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
        )

        # Supported audio formats
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        self.audio_files: List[Path] = self._find_audio_files()

        if len(self.audio_files) == 0:
            raise FileNotFoundError(f"No audio files found in {data_dir}")

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in data directory (non-recursive)."""
        audio_files: List[Path] = []
        seen_paths = set()

        for ext in self.supported_formats:
            for path in self.data_dir.glob(f"*{ext}"):
                normalized_path = path.resolve()
                if normalized_path not in seen_paths:
                    audio_files.append(path)
                    seen_paths.add(normalized_path)

            for path in self.data_dir.glob(f"*{ext.upper()}"):
                normalized_path = path.resolve()
                if normalized_path not in seen_paths:
                    audio_files.append(path)
                    seen_paths.add(normalized_path)

        if self.max_files:
            audio_files = audio_files[: self.max_files]

        return sorted(audio_files)

    def _cache_path(self, audio_file: Path) -> Path:
        assert self.cache_dir
        file_stat = audio_file.stat()
        cache_payload = {
            "path": str(audio_file.resolve()),
            "mtime_ns": file_stat.st_mtime_ns,
            "sr": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "window_frames": self.window_frames,
            # Extractor contract version: when the feature semantics change
            # (FEATURE_VERSION bump), cached features MUST be invalidated —
            # otherwise the trainer silently trains on stale features from
            # the old contract (observed: features/1 cache survived the
            # features/2 change and the model learned the wrong distribution).
            "feature_version": FEATURE_VERSION,
            # Ingestion-pipeline identity: features extracted through the
            # canonical AnalysisTimebase (resampling + hop scheduling +
            # epoch semantics) are NOT interchangeable with features
            # extracted by calling the extractor directly on the whole
            # file. Invalidate caches when the pipeline changes.
            "pipeline_version": PIPELINE_VERSION,
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.npy"

    def _load_features(self, audio_file: Path) -> NDArray[np.float64]:
        """Load features, using cache if available.

        For long audio files, performs chunked extraction to avoid large
        arrays causing stalls in the Rust/PyO3 bridge on Windows.

        Returns:
            A 2D array of shape (n_windows, n_features) with float64 values.
        """
        cache_file = self._cache_path(audio_file) if self.cache_dir else None

        if cache_file and cache_file.exists():
            try:
                loaded: NDArray[np.float64] = np.load(cache_file, allow_pickle=False)
                return loaded
            except Exception:
                cache_file.unlink(missing_ok=True)

        logging.info(f"Extracting features from {audio_file.name}...")
        # Decode at the file's NATIVE rate (sr=None). The browser sends its
        # actual source rate into the Rust StreamingResampler; training must
        # do the same. Resampling with librosa first (sr=SAMPLE_RATE) would
        # mean training and runtime run DIFFERENT resamplers on the same
        # source — a production-path divergence the component tests cannot
        # see (the #93 class, one layer deeper).
        audio, source_rate_raw = librosa.load(
            str(audio_file), sr=None, mono=True, duration=5 * 60
        )
        source_rate = int(source_rate_raw)

        features = self._extract_via_timebase(audio, source_rate)

        if cache_file:
            try:
                np.save(cache_file, features)
            except Exception:
                pass

        return features

    def _extract_via_timebase(
        self, audio: NDArray[np.floating], source_rate: int
    ) -> NDArray[np.float64]:
        """Extract features through the canonical AnalysisTimebase.

        Training must execute the SAME ingestion pipeline the browser
        executes (ADR 0001, issue #93): native-rate PCM → AnalysisTimebase
        (stateful Rust resampling, exactly-once validation, 1024-sample hop
        scheduling, epoch semantics) → Rust FeatureExtractor → ticks.

        ``source_rate`` is the file's NATIVE rate (from the decoder), the
        same role the browser's AudioWorklet ``sampleRate`` plays. The Rust
        StreamingResampler does ALL rate conversion — never resample in
        Python first, or training and runtime run different resamplers.

        Ticks are collected in order; each tick's feature vector is one
        training window.
        """
        timebase = AnalysisTimebase()
        all_windows: List[List[float]] = []

        # Feed in bounded blocks (mirrors the worklet's block cadence and
        # avoids large PyO3 boundary payloads on Windows). Block size is
        # expressed in SOURCE frames; cadence does not affect output
        # (chunk invariance is a timebase guarantee).
        block_size = source_rate  # ~1 s blocks
        n = len(audio)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = np.ascontiguousarray(audio[start:end], dtype=np.float32)
            ticks = timebase.ingest(block.tolist(), source_rate, start)
            for tick in ticks:
                all_windows.append(tick["features"])
        # End-of-stream: recover the deferred final sample/tick.
        for tick in timebase.flush():
            all_windows.append(tick["features"])

        if not all_windows:
            return np.empty((0, 6 * self.window_frames), dtype=np.float64)
        return np.asarray(all_windows, dtype=np.float64)

    def load_all_features(self) -> List[NDArray[np.float64]]:
        """
        Load features from all audio files.

        Returns:
            List of 2D feature arrays, one per audio file. Each array has
            shape (n_windows, n_features).
        """
        all_features: List[NDArray[np.float64]] = []

        for audio_file in self.audio_files:
            try:
                features = self._load_features(audio_file)
                all_features.append(features)
                logging.info(
                    f"Loaded features from {audio_file.name}: {features.shape}"
                )
            except Exception as e:
                logging.info(f"Error loading {audio_file}: {e}")
                continue

        return all_features

    def __len__(self) -> int:
        """Return number of audio files."""
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get features for a specific audio file.

        Args:
            idx: Index of audio file

        Returns:
            Tuple of (features, filename)
        """
        audio_file = self.audio_files[idx]
        features = self._load_features(audio_file)
        return features, audio_file.name
