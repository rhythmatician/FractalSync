"""
Data loading utilities for audio files.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import numpy as np

from .audio_features import AudioFeatureExtractor, load_audio_file


class AudioDataset:
    """Dataset for loading and preprocessing audio files with optional disk cache."""

    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
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

        if feature_extractor is None:
            self.feature_extractor = AudioFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor

        # Supported audio formats
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        self.audio_files: List[Path] = self._find_audio_files()

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in data directory."""
        audio_files: List[Path] = []
        for ext in self.supported_formats:
            audio_files.extend(self.data_dir.rglob(f"*{ext}"))
            audio_files.extend(self.data_dir.rglob(f"*{ext.upper()}"))

        if self.max_files:
            audio_files = audio_files[: self.max_files]

        return sorted(audio_files)

    def _cache_path(self, audio_file: Path) -> Path:
        assert self.cache_dir
        file_stat = audio_file.stat()
        cache_payload = {
            "path": str(audio_file.resolve()),
            "mtime_ns": file_stat.st_mtime_ns,
            "sr": self.feature_extractor.sr,
            "hop_length": self.feature_extractor.hop_length,
            "n_fft": self.feature_extractor.n_fft,
            "window_size": self.feature_extractor.window_size,
            "window_frames": self.window_frames,
        }
        cache_key = hashlib.sha1(
            json.dumps(cache_payload, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{cache_key}.npy"

    def _load_features(self, audio_file: Path) -> np.ndarray:
        """Load features, using cache if available."""
        cache_file = self._cache_path(audio_file) if self.cache_dir else None

        if cache_file and cache_file.exists():
            try:
                return np.load(cache_file, allow_pickle=False)
            except Exception:
                cache_file.unlink(missing_ok=True)

        audio, _ = load_audio_file(str(audio_file))
        features = self.feature_extractor.extract_windowed_features(
            audio, window_frames=self.window_frames
        )

        if cache_file:
            try:
                np.save(cache_file, features)
            except Exception:
                pass

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
