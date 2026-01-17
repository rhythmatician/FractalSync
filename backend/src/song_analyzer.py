"""
Song analyzer for real-time audio analysis and section detection.

Extracts local tempo, detects section boundaries, and identifies
significant audio events (hits, transitions) for synchronization.
"""

import numpy as np
import librosa
from typing import Tuple, List, Dict, Optional


class SongAnalyzer:
    """Analyzes audio files to extract tempo, section boundaries, and hit events."""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        """
        Initialize song analyzer.

        Args:
            sr: Sample rate
            hop_length: Hop length for feature extraction
            n_fft: FFT size for spectral analysis
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

    def analyze_song(
        self, audio: np.ndarray
    ) -> Dict[str, np.ndarray | float | List[float]]:
        """
        Perform comprehensive song analysis.

        Args:
            audio: Audio signal array

        Returns:
            Dictionary containing:
                - 'tempo': Global tempo in BPM
                - 'local_tempo': Frame-wise tempo estimates
                - 'section_boundaries': Frame indices of section boundaries
                - 'onset_frames': Frame indices of detected onsets/hits
                - 'onset_strength': Onset strength envelope
                - 'beat_frames': Frame indices of detected beats
        """
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        # Extract global tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        # Ensure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item())
        else:
            tempo = float(tempo)

        # Extract onset strength envelope (useful for hit detection)
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )

        # Detect onset frames (hits/transients)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sr, hop_length=self.hop_length
        )

        # Compute local tempo variations using tempogram
        local_tempo = self._compute_local_tempo(audio)

        # Detect section boundaries based on spectral/timbral changes
        section_boundaries = self._detect_section_boundaries(audio)

        return {
            "tempo": tempo,
            "local_tempo": local_tempo,
            "section_boundaries": section_boundaries,
            "onset_frames": onset_frames,
            "onset_strength": onset_env,
            "beat_frames": beat_frames,
        }

    def _compute_local_tempo(
        self, audio: np.ndarray, win_length: int = 384
    ) -> np.ndarray:
        """
        Compute frame-wise local tempo estimates using tempogram.

        Args:
            audio: Audio signal
            win_length: Window length for tempogram (in frames)

        Returns:
            Array of local tempo estimates per frame (in BPM)
        """
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )

        # Compute tempogram (measures periodicity at different tempos)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            win_length=win_length,
        )

        # Extract dominant tempo at each frame
        # Find the tempo bin with maximum energy at each frame
        tempo_bins = librosa.tempo_frequencies(
            len(tempogram), hop_length=self.hop_length, sr=self.sr
        )

        # Get the index of max energy at each frame
        max_tempo_idx = np.argmax(tempogram, axis=0)

        # Map indices to actual tempo values
        local_tempo = tempo_bins[max_tempo_idx]

        return local_tempo

    def _detect_section_boundaries(
        self, audio: np.ndarray, n_mfcc: int = 13
    ) -> np.ndarray:
        """
        Detect section boundaries using spectral novelty.

        Uses self-similarity matrix analysis to find points where
        the audio character changes significantly.

        Args:
            audio: Audio signal
            n_mfcc: Number of MFCC coefficients to use

        Returns:
            Array of frame indices marking section boundaries
        """
        # Extract MFCCs (captures timbral information)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=n_mfcc, hop_length=self.hop_length
        )

        # Compute chroma features (captures harmonic content)
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )

        # Stack features
        features = np.vstack([mfcc, chroma])

        # Normalize features
        features = librosa.util.normalize(features, axis=0)

        # Compute self-similarity matrix
        rec_matrix = librosa.segment.recurrence_matrix(
            features, mode="affinity", metric="cosine", width=3
        )

        # Detect boundaries using structure features
        # Use default k or estimate based on audio length
        n_frames = features.shape[1]
        k = max(2, min(10, n_frames // 100))  # Roughly one segment per 100 frames
        
        boundaries = librosa.segment.agglomerative(
            features, k=k
        )

        return boundaries

    def get_tempo_at_frame(
        self, frame_idx: int, local_tempo: np.ndarray, global_tempo: float
    ) -> float:
        """
        Get tempo estimate at a specific frame.

        Falls back to global tempo if frame is out of bounds.

        Args:
            frame_idx: Frame index
            local_tempo: Array of local tempo estimates
            global_tempo: Global tempo value

        Returns:
            Tempo at the specified frame (BPM)
        """
        if 0 <= frame_idx < len(local_tempo):
            return float(local_tempo[frame_idx])
        return global_tempo

    def is_near_section_boundary(
        self, frame_idx: int, section_boundaries: np.ndarray, tolerance: int = 5
    ) -> bool:
        """
        Check if a frame is near a section boundary.

        Args:
            frame_idx: Frame index to check
            section_boundaries: Array of boundary frame indices
            tolerance: Number of frames within which to consider "near"

        Returns:
            True if frame is within tolerance of any boundary
        """
        for boundary in section_boundaries:
            if abs(frame_idx - boundary) <= tolerance:
                return True
        return False

    def get_hit_events(
        self,
        onset_frames: np.ndarray,
        onset_strength: np.ndarray,
        threshold: float = 0.5,
    ) -> List[Dict[str, int | float]]:
        """
        Extract hit events with their strengths.

        Args:
            onset_frames: Frame indices of detected onsets
            onset_strength: Onset strength envelope
            threshold: Minimum normalized strength to consider

        Returns:
            List of dicts with 'frame' and 'strength' keys
        """
        # Normalize onset strength
        onset_norm = onset_strength / (np.max(onset_strength) + 1e-8)

        hit_events = []
        for frame in onset_frames:
            if frame < len(onset_norm):
                strength = onset_norm[frame]
                if strength >= threshold:
                    hit_events.append({"frame": int(frame), "strength": float(strength)})

        return hit_events

    def frames_to_time(self, frames: np.ndarray | int) -> np.ndarray | float:
        """
        Convert frame indices to time in seconds.

        Args:
            frames: Frame index or array of frame indices

        Returns:
            Time in seconds (scalar or array)
        """
        return librosa.frames_to_time(
            frames, sr=self.sr, hop_length=self.hop_length
        )

    def time_to_frames(self, time_sec: float | np.ndarray) -> int | np.ndarray:
        """
        Convert time in seconds to frame indices.

        Args:
            time_sec: Time in seconds (scalar or array)

        Returns:
            Frame index (integer or array)
        """
        return librosa.time_to_frames(
            time_sec, sr=self.sr, hop_length=self.hop_length
        )


def analyze_audio_file(
    file_path: str, sr: int = 22050
) -> Tuple[Dict[str, np.ndarray | float | List[float]], np.ndarray]:
    """
    Convenience function to analyze an audio file.

    Args:
        file_path: Path to audio file
        sr: Sample rate

    Returns:
        Tuple of (analysis_dict, audio_array)
    """
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr)

    # Analyze
    analyzer = SongAnalyzer(sr=sr)
    analysis = analyzer.analyze_song(audio)

    return analysis, audio
