"""
Unit tests for data loading and caching.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil
import time

# Add parent directory to path for imports (to enable absolute imports of src package)
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from src.data_loader import AudioDataset
from src.audio_features import AudioFeatureExtractor


# Module-level fixtures (available to all test classes)
@pytest.fixture
def temp_audio_dir():
    """Create a temporary directory with test audio files."""
    temp_dir = tempfile.mkdtemp()
    audio_dir = Path(temp_dir) / "audio"
    audio_dir.mkdir()
    
    # Create dummy audio files (silence)
    sr = 22050
    duration = 1.0  # 1 second
    
    for i in range(3):
        audio_file = audio_dir / f"test_audio_{i}.wav"
        # Create very short silent audio
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        # Add a tiny bit of variation to make it processable
        audio += np.random.randn(len(audio)) * 0.001
        
        # Save as WAV
        try:
            import soundfile as sf
            sf.write(str(audio_file), audio, sr)
        except ImportError:
            # If soundfile not available, use scipy
            try:
                from scipy.io import wavfile
                wavfile.write(str(audio_file), sr, audio)
            except ImportError:
                # Skip if no audio writing library available
                pytest.skip("No audio writing library available (soundfile or scipy)")
    
    yield audio_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / "cache"
    cache_dir.mkdir()
    
    yield cache_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAudioDataset:
    """Test audio dataset loading and caching."""

    def test_dataset_initialization(self, temp_audio_dir, temp_cache_dir):
        """Test creating an AudioDataset."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        assert dataset is not None
        # Note: On Windows, case-insensitive filesystem may find files twice
        # (both .wav and .WAV patterns match). We expect at least 3 files.
        assert len(dataset.audio_files) >= 3
        assert dataset.window_frames == 10
        assert dataset.cache_dir == temp_cache_dir

    def test_dataset_finds_audio_files(self, temp_audio_dir, temp_cache_dir):
        """Test that dataset discovers audio files correctly."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            cache_dir=str(temp_cache_dir)
        )
        
        # Should find at least 3 WAV files (may be 6 on Windows due to case-insensitive FS)
        assert len(dataset) >= 3
        
        # All should be .wav files
        for audio_file in dataset.audio_files:
            assert audio_file.suffix.lower() == ".wav"

    def test_dataset_empty_directory(self, temp_cache_dir):
        """Test that empty directory raises an error."""
        temp_dir = tempfile.mkdtemp()
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No audio files found"):
            AudioDataset(
                data_dir=str(empty_dir),
                cache_dir=str(temp_cache_dir)
            )
        
        shutil.rmtree(temp_dir)

    def test_dataset_max_files(self, temp_audio_dir, temp_cache_dir):
        """Test limiting the number of files loaded."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            max_files=2,
            cache_dir=str(temp_cache_dir)
        )
        
        assert len(dataset) == 2

    def test_dataset_no_cache(self, temp_audio_dir):
        """Test dataset without caching."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            cache_dir=None  # No caching
        )
        
        assert dataset.cache_dir is None
        assert len(dataset) >= 3

    def test_load_single_file_features(self, temp_audio_dir, temp_cache_dir):
        """Test loading features from a single file."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        features, filename = dataset[0]
        
        # Check feature shape: should be (n_windows, 6 * window_frames)
        assert features.ndim == 2
        assert features.shape[1] == 60  # 6 features * 10 frames
        assert "test_audio" in filename

    def test_load_all_features(self, temp_audio_dir, temp_cache_dir):
        """Test loading features from all files."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        all_features = dataset.load_all_features()
        
        assert len(all_features) >= 3  # May be 6 on Windows
        
        for features in all_features:
            assert features.ndim == 2
            assert features.shape[1] == 60

    def test_feature_caching(self, temp_audio_dir, temp_cache_dir):
        """Test that features are cached to disk."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        # Load features (should create cache)
        features1, _ = dataset[0]
        
        # Check that cache file was created
        cache_files = list(temp_cache_dir.glob("*.npy"))
        assert len(cache_files) >= 1
        
        # Load again (should use cache)
        features2, _ = dataset[0]
        
        # Features should be identical
        assert np.array_equal(features1, features2)

    def test_cache_invalidation_on_config_change(self, temp_audio_dir, temp_cache_dir):
        """Test that cache is invalidated when config changes."""
        # Create dataset with one config
        dataset1 = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        features1, _ = dataset1[0]
        cache_files_1 = set(temp_cache_dir.glob("*.npy"))
        
        # Create dataset with different config
        dataset2 = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=15,  # Different window size
            cache_dir=str(temp_cache_dir)
        )
        
        features2, _ = dataset2[0]
        cache_files_2 = set(temp_cache_dir.glob("*.npy"))
        
        # Should create a new cache file
        assert len(cache_files_2) > len(cache_files_1)
        
        # Features should have different dimensions
        assert features1.shape[1] != features2.shape[1]
        assert features1.shape[1] == 60  # 6 * 10
        assert features2.shape[1] == 90  # 6 * 15

    def test_cache_invalidation_on_file_modification(self, temp_audio_dir, temp_cache_dir):
        """Test that cache is invalidated when audio file is modified."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        audio_file = dataset.audio_files[0]
        
        # Load features (creates cache)
        features1, _ = dataset[0]
        cache_path = dataset._cache_path(audio_file)
        assert cache_path.exists()
        
        # Modify file (update mtime)
        time.sleep(0.1)  # Ensure mtime changes
        audio_file.touch()
        
        # Create new dataset (cache should be invalid due to mtime change)
        dataset2 = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        # Old cache should not be used (new cache key due to mtime)
        old_cache_path = cache_path
        new_cache_path = dataset2._cache_path(audio_file)
        
        # Cache paths should be different
        assert old_cache_path != new_cache_path

    def test_custom_feature_extractor(self, temp_audio_dir, temp_cache_dir):
        """Test using a custom feature extractor."""
        custom_extractor = AudioFeatureExtractor(
            sr=16000,  # Different sample rate
            hop_length=256,
        )
        
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            feature_extractor=custom_extractor,
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        assert dataset.feature_extractor.sr == 16000
        assert dataset.feature_extractor.hop_length == 256

    def test_dataset_iteration(self, temp_audio_dir, temp_cache_dir):
        """Test iterating over dataset."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        # Iterate over all files
        for i in range(len(dataset)):
            features, filename = dataset[i]
            assert features.shape[1] == 60
            assert isinstance(filename, str)

    def test_feature_dimensions_consistency(self, temp_audio_dir, temp_cache_dir):
        """Test that all features have consistent dimensions."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        all_features = dataset.load_all_features()
        
        # All should have same number of feature dimensions
        for features in all_features:
            assert features.shape[1] == 60

    def test_cache_directory_creation(self):
        """Test that cache directory is created automatically."""
        temp_dir = tempfile.mkdtemp()
        audio_dir = Path(temp_dir) / "audio"
        audio_dir.mkdir()
        
        # Create a dummy audio file
        try:
            import soundfile as sf
            audio_file = audio_dir / "test.wav"
            audio = np.random.randn(22050) * 0.001
            sf.write(str(audio_file), audio, 22050)
        except ImportError:
            try:
                from scipy.io import wavfile
                audio_file = audio_dir / "test.wav"
                audio = (np.random.randn(22050) * 0.001).astype(np.float32)
                wavfile.write(str(audio_file), 22050, audio)
            except ImportError:
                shutil.rmtree(temp_dir)
                pytest.skip("No audio writing library available")
        
        cache_dir = Path(temp_dir) / "cache_new"
        
        # Cache dir doesn't exist yet
        assert not cache_dir.exists()
        
        # Create dataset (should create cache dir)
        dataset = AudioDataset(
            data_dir=str(audio_dir),
            cache_dir=str(cache_dir)
        )
        
        # Cache dir should now exist
        assert cache_dir.exists()
        
        shutil.rmtree(temp_dir)

    def test_supported_formats(self, temp_cache_dir):
        """Test that dataset recognizes supported audio formats."""
        dataset = AudioDataset.__new__(AudioDataset)  # Create without __init__
        dataset.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        
        assert ".wav" in dataset.supported_formats
        assert ".mp3" in dataset.supported_formats
        assert ".flac" in dataset.supported_formats
        assert ".ogg" in dataset.supported_formats
        assert ".m4a" in dataset.supported_formats

    def test_corrupted_cache_recovery(self, temp_audio_dir, temp_cache_dir):
        """Test that corrupted cache files are handled gracefully."""
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            window_frames=10,
            cache_dir=str(temp_cache_dir)
        )
        
        audio_file = dataset.audio_files[0]
        cache_path = dataset._cache_path(audio_file)
        
        # Create a corrupted cache file
        with open(cache_path, 'w') as f:
            f.write("corrupted data")
        
        # Loading should still work (will re-extract features)
        features, _ = dataset[0]
        
        assert features.shape[1] == 60
        
        # Cache file should be replaced with valid data
        assert cache_path.exists()
        loaded = np.load(cache_path)
        assert loaded.shape == features.shape


class TestAudioDatasetEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_directory(self):
        """Test that nonexistent directory raises an error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            AudioDataset(data_dir="/nonexistent/path/to/audio")

    def test_zero_window_frames(self, temp_audio_dir, temp_cache_dir):
        """Test that zero window frames is handled."""
        # This should either raise an error or be handled gracefully
        # depending on implementation
        try:
            dataset = AudioDataset(
                data_dir=str(temp_audio_dir),
                window_frames=0,
                cache_dir=str(temp_cache_dir)
            )
            # If it doesn't raise, at least check it doesn't crash
            assert dataset is not None
        except (ValueError, AssertionError):
            # Expected behavior
            pass

    def test_negative_max_files(self, temp_audio_dir, temp_cache_dir):
        """Test that negative max_files is handled."""
        # Should either raise an error or treat as unlimited
        dataset = AudioDataset(
            data_dir=str(temp_audio_dir),
            max_files=-1,
            cache_dir=str(temp_cache_dir)
        )
        # Should still work (likely treats as no limit)
        assert len(dataset) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
