"""
Data loading utilities for audio files.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from .audio_features import AudioFeatureExtractor, load_audio_file


class AudioDataset:
    """Dataset for loading and preprocessing audio files."""
    
    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        window_frames: int = 10,
        max_files: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing audio files
            feature_extractor: Feature extractor instance
            window_frames: Number of frames per window
            max_files: Maximum number of files to load (None for all)
        """
        self.data_dir = Path(data_dir)
        self.window_frames = window_frames
        self.max_files = max_files
        
        if feature_extractor is None:
            self.feature_extractor = AudioFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in data directory."""
        audio_files = []
        for ext in self.supported_formats:
            audio_files.extend(self.data_dir.rglob(f'*{ext}'))
            audio_files.extend(self.data_dir.rglob(f'*{ext.upper()}'))
        
        if self.max_files:
            audio_files = audio_files[:self.max_files]
        
        return sorted(audio_files)
    
    def load_all_features(self) -> List[np.ndarray]:
        """
        Load features from all audio files.
        
        Returns:
            List of feature arrays, one per audio file
        """
        all_features = []
        
        for audio_file in self.audio_files:
            try:
                audio, _ = load_audio_file(str(audio_file))
                features = self.feature_extractor.extract_windowed_features(
                    audio,
                    window_frames=self.window_frames
                )
                all_features.append(features)
                print(f"Loaded features from {audio_file.name}: {features.shape}")
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
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
        audio, _ = load_audio_file(str(audio_file))
        features = self.feature_extractor.extract_windowed_features(
            audio,
            window_frames=self.window_frames
        )
        return features, audio_file.name
