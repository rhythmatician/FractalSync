"""
Data loading utilities for audio files.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np

from .runtime_core_bridge import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    make_feature_extractor,
)
import librosa


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
        self.window_frames = window_frames
        self.max_files = max_files
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.feature_extractor = feature_extractor or make_feature_extractor()

        # Supported audio formats
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        self.audio_files: List[Path] = self._find_audio_files()

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")

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
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.npy"

    def _load_features(self, audio_file: Path) -> np.ndarray:
        """Load features, using cache if available.

        For long audio files, performs chunked extraction to avoid large
        arrays causing stalls in the Rust/PyO3 bridge on Windows.
        """
        cache_file = self._cache_path(audio_file) if self.cache_dir else None

        if cache_file and cache_file.exists():
            try:
                return np.load(cache_file, allow_pickle=False)
            except Exception:
                cache_file.unlink(missing_ok=True)

        logging.info(f"Extracting features from {audio_file.name}...")
        # Limit decode time to avoid reading entire multi-hour files in quick runs
        audio, _ = librosa.load(
            str(audio_file), sr=SAMPLE_RATE, mono=True, duration=5 * 60
        )

        # Chunk very long audio to prevent large allocations
        max_total_seconds = 5 * 60  # 5 minutes threshold
        chunk_seconds = 60  # process in 60s chunks
        if len(audio) > SAMPLE_RATE * max_total_seconds:
            chunk_size = SAMPLE_RATE * chunk_seconds
            all_chunks: list[np.ndarray] = []
            for start in range(0, len(audio), chunk_size):
                end = min(start + chunk_size, len(audio))
                chunk = audio[start:end].astype(np.float32)
                chunk_features = self.feature_extractor.extract_windowed_features(
                    chunk, window_frames=self.window_frames
                )
                all_chunks.append(chunk_features)
                logging.info(f"  chunk {start//chunk_size + 1}: {chunk_features.shape}")
            features = (
                np.vstack(all_chunks)
                if all_chunks
                else np.empty((0, 6 * self.window_frames), dtype=np.float32)
            )
        else:
            features = self.feature_extractor.extract_windowed_features(
                audio.astype(np.float32), window_frames=self.window_frames
            )

        if cache_file:
            try:
                np.save(cache_file, features)
            except (OSError, IOError) as exc:
                # Best-effort cache write; non-fatal if disk write fails
                logging.getLogger(__name__).debug(
                    "Failed to write feature cache '%s': %s", cache_file, exc
                )

        return features

    def load_all_features(self) -> List[np.ndarray]:
        """
        Load features from all audio files.

        Returns:
            List of feature arrays, one per audio file
        """
        all_features: List[np.ndarray] = []

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


class SequenceAudioDataset:
    """Dataset view that yields contiguous sequences (clips) of windowed features.

    Can be constructed from an AudioDataset instance or from a preloaded list of
    feature arrays for testing convenience.

    Returns tuples: (sequence: np.ndarray shape (seq_len, feat_dim), file_index: int, start_idx: int)
    """

    def __init__(
        self,
        audio_dataset: Optional[AudioDataset] = None,
        features_list: Optional[List[np.ndarray]] = None,
        seq_len: int = 5,
        stride: int = 1,
    ):
        if audio_dataset is None and features_list is None:
            raise ValueError("Either audio_dataset or features_list must be provided")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if features_list is not None:
            self.features_list: List[np.ndarray] = features_list
        else:
            # audio_dataset must be provided when features_list is None (checked above)
            assert audio_dataset is not None
            self.features_list = audio_dataset.load_all_features()

        # Build index mapping: list of (file_idx, start_idx)
        self.index_map: List[tuple[int, int]] = []
        for fi, feats in enumerate(self.features_list):
            n_frames = int(feats.shape[0])
            max_start = n_frames - self.seq_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, self.stride):
                self.index_map.append((fi, start))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        fi, start = self.index_map[idx]
        seq = self.features_list[fi][start : start + self.seq_len].astype("float32")
        return seq, fi, start

    def to_tensor_dataset(self):
        """Convert sequences to a TensorDataset of shape (N, seq_len, feat_dim).

        Useful for DataLoader compatibility in training loops.
        """
        import torch

        seqs = [self[i][0] for i in range(len(self))]
        if not seqs:
            return torch.utils.data.TensorDataset(torch.empty((0, self.seq_len, 0)))
        arr = np.stack(seqs, axis=0)
        return torch.utils.data.TensorDataset(torch.from_numpy(arr))
