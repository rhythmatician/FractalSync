"""Tests for section novelty and section boundary picking."""

import unittest
import numpy as np

from src.song_analyzer import SongAnalyzer


class TestSectionNovelty(unittest.TestCase):
    def setUp(self):
        self.an = SongAnalyzer(sr=22050, hop_length=512, n_fft=2048)

    def test_compute_section_novelty_basic(self):
        # 6 seconds of white noise
        audio = np.random.randn(6 * 22050).astype(np.float32)
        feats = self.an._compute_section_novelty(audio)
        self.assertIn("flux", feats)
        self.assertIn("nov_mel", feats)
        self.assertIn("nov_chroma", feats)
        self.assertIn("nov_fused", feats)
        self.assertIsInstance(feats["nov_fused"], np.ndarray)
        self.assertGreater(feats["nov_fused"].size, 0)

    def test_pick_section_boundaries_detects_change(self):
        sr = 22050
        # synth: 8s of noise then 8s of sine tone to create a clear timbral change
        t = np.linspace(0, 16.0, int(sr * 16.0), endpoint=False)
        audio = np.zeros_like(t, dtype=np.float32)
        audio[: sr * 8] = np.random.randn(sr * 8).astype(np.float32) * 0.3
        audio[sr * 8 :] = (0.5 * np.sin(2.0 * np.pi * 440.0 * t[sr * 8 :])).astype(
            np.float32
        )

        feats = self.an._compute_section_novelty(audio)
        fused = feats["nov_fused"]
        # use a small min_gap_sec so the test runs quickly
        bounds = self.an._pick_section_boundaries(fused, min_gap_sec=1.0, q=0.6)
        # Expect at least one boundary near 8s
        self.assertIsInstance(bounds, np.ndarray)
        times = self.an.frames_to_time(bounds)
        self.assertTrue(np.any(np.abs(times - 8.0) < 1.25))

    def test_analyze_song_includes_section(self):
        audio = np.random.randn(5 * 22050).astype(np.float32)
        analysis = self.an.analyze_song(audio)
        self.assertIn("section", analysis)
        sec = analysis["section"]
        self.assertIn("novelty", sec)
        self.assertIn("components", sec)
        self.assertIn("boundaries", sec)
        self.assertIsInstance(sec["novelty"], np.ndarray)
        self.assertIsInstance(sec["components"], dict)
        self.assertIsInstance(sec["boundaries"], np.ndarray)
