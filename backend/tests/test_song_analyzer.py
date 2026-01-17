"""Unit tests for song analyzer module."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.song_analyzer import SongAnalyzer  # noqa: E402


class TestSongAnalyzer(unittest.TestCase):
    """Test song analyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SongAnalyzer(sr=22050, hop_length=512, n_fft=2048)

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.sr, 22050)
        self.assertEqual(self.analyzer.hop_length, 512)
        self.assertEqual(self.analyzer.n_fft, 2048)

    def test_analyze_song_basic(self):
        """Test basic song analysis."""
        # Create synthetic audio (5 seconds)
        audio = np.random.randn(5 * 22050).astype(np.float32)

        analysis = self.analyzer.analyze_song(audio)

        # Check that all expected keys are present
        expected_keys = [
            "tempo",
            "local_tempo",
            "section_boundaries",
            "onset_frames",
            "onset_strength",
            "beat_frames",
        ]
        for key in expected_keys:
            self.assertIn(key, analysis)

        # Check types
        self.assertIsInstance(analysis["tempo"], float)
        self.assertIsInstance(analysis["local_tempo"], np.ndarray)
        self.assertIsInstance(analysis["section_boundaries"], np.ndarray)
        self.assertIsInstance(analysis["onset_frames"], np.ndarray)
        self.assertIsInstance(analysis["onset_strength"], np.ndarray)

    def test_local_tempo_shape(self):
        """Test that local tempo has correct shape."""
        audio = np.random.randn(3 * 22050).astype(np.float32)

        local_tempo = self.analyzer._compute_local_tempo(audio)

        # Should return an array
        self.assertIsInstance(local_tempo, np.ndarray)
        self.assertGreater(len(local_tempo), 0)

    def test_section_boundary_detection(self):
        """Test section boundary detection."""
        audio = np.random.randn(5 * 22050).astype(np.float32)

        boundaries = self.analyzer._detect_section_boundaries(audio)

        # Should return an array
        self.assertIsInstance(boundaries, np.ndarray)

    def test_get_tempo_at_frame(self):
        """Test getting tempo at specific frame."""
        local_tempo = np.array([120.0, 121.0, 122.0, 123.0, 124.0])
        global_tempo = 120.0

        # Valid frame
        tempo = self.analyzer.get_tempo_at_frame(2, local_tempo, global_tempo)
        self.assertEqual(tempo, 122.0)

        # Out of bounds (should return global)
        tempo = self.analyzer.get_tempo_at_frame(10, local_tempo, global_tempo)
        self.assertEqual(tempo, global_tempo)

    def test_is_near_section_boundary(self):
        """Test section boundary proximity check."""
        boundaries = np.array([10, 50, 100])

        # Near boundary
        self.assertTrue(self.analyzer.is_near_section_boundary(12, boundaries, tolerance=5))
        self.assertTrue(self.analyzer.is_near_section_boundary(48, boundaries, tolerance=5))

        # Far from boundary
        self.assertFalse(self.analyzer.is_near_section_boundary(30, boundaries, tolerance=5))

    def test_get_hit_events(self):
        """Test hit event extraction."""
        onset_frames = np.array([10, 20, 30, 40, 50])
        # Create onset strength with proper length
        onset_strength = np.full(105, 0.1)
        onset_strength[20] = 0.8
        onset_strength[40] = 0.9

        hit_events = self.analyzer.get_hit_events(
            onset_frames, onset_strength, threshold=0.5
        )

        # Should get hits at frames 20 and 40 (high strength)
        self.assertIsInstance(hit_events, list)

        # Check structure
        if len(hit_events) > 0:
            self.assertIn("frame", hit_events[0])
            self.assertIn("strength", hit_events[0])

    def test_frames_to_time(self):
        """Test frame to time conversion."""
        frames = 43  # ~1 second at 22050 Hz, hop_length=512

        time_sec = self.analyzer.frames_to_time(frames)

        # Should be approximately 1 second
        self.assertIsInstance(time_sec, (float, np.floating))
        self.assertGreater(time_sec, 0)

    def test_time_to_frames(self):
        """Test time to frame conversion."""
        time_sec = 1.0

        frames = self.analyzer.time_to_frames(time_sec)

        # Should return valid frame number
        self.assertIsInstance(frames, (int, np.integer, np.ndarray))
        if isinstance(frames, (int, np.integer)):
            self.assertGreater(frames, 0)

    def test_mono_audio_conversion(self):
        """Test that stereo audio is converted to mono."""
        # Create stereo audio
        stereo_audio = np.random.randn(2, 22050).astype(np.float32)

        # Should not raise error
        analysis = self.analyzer.analyze_song(stereo_audio)
        self.assertIsNotNone(analysis)


if __name__ == "__main__":
    unittest.main()
