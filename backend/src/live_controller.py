"""
Live audio-reactive controller for FractalSync.

Implements the new architecture with:
- Fast impact detection (many per song)
- Slow section boundary detection (occasional lobe switches)
- Orbit state machine with carrier + residual
- Deterministic fallback behavior
"""

import numpy as np
import librosa
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from collections import deque

from .runtime_core_bridge import (
    make_orbit_state,
    step_orbit,
)


@dataclass
class FastFeatures:
    """Fast audio features computed at ~100 Hz."""

    onset_strength: float
    spectral_flux: float
    band_deltas: np.ndarray  # High-band energy deltas
    kick_delta: float  # Low-band delta
    timestamp: float


@dataclass
class SlowFeatures:
    """Slow audio feature summaries computed at ~10 Hz."""

    loudness: float  # in [0, 1]
    tonalness: float  # in [0, 1], high = pure/harmonic
    noisiness: float  # in [0, 1], high = broadband
    band_energies: np.ndarray  # Log mel or octave bands
    spectral_centroid: float
    spectral_rolloff: float
    onset_rate: float
    timestamp: float


@dataclass
class ImpactEvent:
    """Represents a detected impact event."""

    timestamp: float
    score: float
    s_overshoot: float


@dataclass
class BoundaryEvent:
    """Represents a detected section boundary."""

    timestamp: float
    novelty_score: float
    loudness: float
    chosen_lobe: int
    chosen_sub_lobe: int
    previous_dwell_time: float


class AudioFeatureStream:
    """
    Extracts audio features at two time scales:
    - Fast features (~100 Hz) for impact detection
    - Slow features (~10 Hz) for section detection and control
    """

    def __init__(
        self,
        sr: int = 22050,
        fast_hop: int = 220,  # ~100 Hz at 22050 sr
        slow_hop: int = 2205,  # ~10 Hz
        n_fft: int = 2048,
    ):
        """
        Initialize audio feature stream.

        Args:
            sr: Sample rate
            fast_hop: Hop length for fast features
            slow_hop: Hop length for slow features
            n_fft: FFT size
        """
        self.sr = sr
        self.fast_hop = fast_hop
        self.slow_hop = slow_hop
        self.n_fft = n_fft

        # Mel filterbank for octave bands
        self.n_bands = 8
        self.mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=self.n_bands)

        # State for computing deltas
        self.prev_spectrum = None
        self.prev_band_energies = None

    def compute_fast_features(
        self, audio_frame: np.ndarray, timestamp: float
    ) -> FastFeatures:
        """
        Compute fast features for impact detection.

        Args:
            audio_frame: Audio segment
            timestamp: Current timestamp

        Returns:
            FastFeatures object
        """
        # Compute STFT
        stft = librosa.stft(audio_frame, n_fft=self.n_fft, hop_length=self.fast_hop)
        mag = np.abs(stft)

        # Onset strength (spectral flux-based)
        onset_strength = float(
            librosa.onset.onset_strength(
                S=mag, sr=self.sr, hop_length=self.fast_hop, aggregate=np.mean
            )
        )

        # Spectral flux
        if self.prev_spectrum is not None:
            flux = np.sum(np.maximum(0, mag[:, -1] - self.prev_spectrum))
            spectral_flux = float(flux / (np.sum(mag[:, -1]) + 1e-8))
        else:
            spectral_flux = 0.0

        self.prev_spectrum = mag[:, -1].copy() if mag.shape[1] > 0 else None

        # Band energies via mel filterbank
        mel_spec = self.mel_fb @ (mag**2)
        band_energies = np.log1p(np.mean(mel_spec, axis=1))

        # Band deltas (high bands for transient detection)
        if self.prev_band_energies is not None:
            band_deltas = band_energies - self.prev_band_energies
        else:
            band_deltas = np.zeros_like(band_energies)

        self.prev_band_energies = band_energies.copy()

        # High-band deltas (bands 4-7 for treble/transients)
        high_band_deltas = band_deltas[4:]

        # Kick delta (low bands 0-2)
        kick_delta = float(np.mean(band_deltas[0:3]))

        return FastFeatures(
            onset_strength=onset_strength,
            spectral_flux=spectral_flux,
            band_deltas=high_band_deltas,
            kick_delta=kick_delta,
            timestamp=timestamp,
        )

    def compute_slow_features(
        self, audio_segment: np.ndarray, timestamp: float
    ) -> SlowFeatures:
        """
        Compute slow feature summaries for control and section detection.

        Args:
            audio_segment: Longer audio segment (~0.5s)
            timestamp: Current timestamp

        Returns:
            SlowFeatures object
        """
        # Compute STFT
        stft = librosa.stft(audio_segment, n_fft=self.n_fft, hop_length=self.slow_hop)
        mag = np.abs(stft)
        power = mag**2

        # Loudness (RMS energy normalized)
        rms = librosa.feature.rms(
            S=mag, frame_length=self.n_fft, hop_length=self.slow_hop
        )
        loudness = float(np.mean(rms))
        loudness = np.clip(loudness / 0.3, 0, 1)  # Normalize to [0, 1]

        # Band energies
        mel_spec = self.mel_fb @ power
        band_energies = np.log1p(np.mean(mel_spec, axis=1))

        # Spectral features
        spectral_centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    S=mag, sr=self.sr, hop_length=self.slow_hop
                )
            )
        )
        spectral_rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    S=mag, sr=self.sr, hop_length=self.slow_hop
                )
            )
        )

        # Tonalness: use spectral flatness (inverse)
        flatness = librosa.feature.spectral_flatness(S=mag, hop_length=self.slow_hop)
        tonalness = float(1.0 - np.mean(flatness))  # High = tonal
        tonalness = np.clip(tonalness, 0, 1)

        # Noisiness: spectral flatness itself
        noisiness = float(np.mean(flatness))
        noisiness = np.clip(noisiness, 0, 1)

        # Onset rate
        onset_env = librosa.onset.onset_strength(
            S=mag, sr=self.sr, hop_length=self.slow_hop
        )
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        onset_rate = float(len(onset_frames) / (len(audio_segment) / self.sr + 1e-6))

        return SlowFeatures(
            loudness=loudness,
            tonalness=tonalness,
            noisiness=noisiness,
            band_energies=band_energies,
            spectral_centroid=spectral_centroid / (self.sr / 2),  # Normalize
            spectral_rolloff=spectral_rolloff / (self.sr / 2),
            onset_rate=onset_rate,
            timestamp=timestamp,
        )


class ImpactDetector:
    """
    Detects impact events from fast audio features.
    Uses adaptive thresholding with hysteresis and refractory period.
    """

    def __init__(
        self,
        threshold_percentile: float = 70.0,
        history_duration: float = 10.0,
        refractory_ms: float = 150.0,
        hysteresis_ratio: float = 0.8,
    ):
        """
        Initialize impact detector.

        Args:
            threshold_percentile: Percentile for adaptive threshold
            history_duration: Duration for computing threshold (seconds)
            refractory_ms: Minimum time between impacts (milliseconds)
            hysteresis_ratio: Ratio for hysteresis threshold
        """
        self.threshold_percentile = threshold_percentile
        self.history_duration = history_duration
        self.refractory_ms = refractory_ms
        self.hysteresis_ratio = hysteresis_ratio

        # History buffer for adaptive threshold
        self.score_history: deque = deque(maxlen=1000)  # ~10s at 100Hz

        # State
        self.last_impact_time = -999.0
        self.in_refractory = False

    def compute_impact_score(self, features: FastFeatures) -> float:
        """
        Compute impact score from fast features.

        Args:
            features: FastFeatures object

        Returns:
            Impact score (higher = more impact-like)
        """
        # Weighted combination of features
        score = (
            0.4 * features.onset_strength
            + 0.3 * features.spectral_flux
            + 0.2 * np.max(features.band_deltas)
            + 0.1 * max(0, features.kick_delta)
        )
        return float(score)

    def detect(self, features: FastFeatures) -> Tuple[bool, float]:
        """
        Detect if current features represent an impact event.

        Args:
            features: FastFeatures object

        Returns:
            Tuple of (is_impact, impact_score)
        """
        score = self.compute_impact_score(features)
        self.score_history.append(score)

        # Check refractory period
        time_since_last = (features.timestamp - self.last_impact_time) * 1000
        if time_since_last < self.refractory_ms:
            self.in_refractory = True
            return False, score
        else:
            self.in_refractory = False

        # Adaptive threshold
        if len(self.score_history) < 10:
            return False, score

        threshold = np.percentile(list(self.score_history), self.threshold_percentile)
        hysteresis_threshold = threshold * self.hysteresis_ratio

        # Detect impact
        is_impact = bool(score > threshold) and not self.in_refractory

        if is_impact:
            self.last_impact_time = features.timestamp

        return is_impact, score


class ImpactEnvelope:
    """
    Manages impact envelope state for modulating orbit parameters.
    """

    def __init__(
        self,
        attack_ms: float = 30.0,
        decay_ms: float = 300.0,
        s_impulse_hi: float = 1.12,
        alpha_boost: float = 0.3,
    ):
        """
        Initialize impact envelope.

        Args:
            attack_ms: Attack time in milliseconds
            decay_ms: Decay time in milliseconds
            s_impulse_hi: Target s value during impact
            alpha_boost: Residual amplitude boost during impact
        """
        self.attack_ms = attack_ms
        self.decay_ms = decay_ms
        self.s_impulse_hi = s_impulse_hi
        self.alpha_boost = alpha_boost

        # State
        self.envelope_value = 0.0
        self.impact_start_time: Optional[float] = None
        self.is_active = False

    def trigger(self, timestamp: float):
        """Trigger a new impact envelope."""
        self.impact_start_time = timestamp
        self.is_active = True

    def update(self, timestamp: float) -> float:
        """
        Update envelope and return current value.

        Args:
            timestamp: Current timestamp

        Returns:
            Envelope value in [0, 1]
        """
        if not self.is_active or self.impact_start_time is None:
            self.envelope_value = 0.0
            return 0.0

        elapsed_ms = (timestamp - self.impact_start_time) * 1000

        if elapsed_ms < self.attack_ms:
            # Attack phase
            self.envelope_value = elapsed_ms / self.attack_ms
        elif elapsed_ms < self.attack_ms + self.decay_ms:
            # Decay phase
            decay_progress = (elapsed_ms - self.attack_ms) / self.decay_ms
            self.envelope_value = 1.0 - decay_progress
        else:
            # Envelope finished
            self.envelope_value = 0.0
            self.is_active = False

        return self.envelope_value

    def get_s_boost(self, s_base: float) -> float:
        """
        Get s value boost based on envelope.

        Args:
            s_base: Base s value

        Returns:
            Boosted s value
        """
        if self.envelope_value == 0.0:
            return s_base

        # Push s outward across boundary
        target = self.s_impulse_hi if s_base < 1.0 else s_base * 1.1
        return s_base + (target - s_base) * self.envelope_value

    def get_alpha_boost(self) -> float:
        """Get residual alpha boost based on envelope."""
        return self.alpha_boost * self.envelope_value


class NoveltyBoundaryDetector:
    """
    Detects section boundaries based on sustained novelty in audio features.
    Uses a rolling baseline and distance metric.
    """

    def __init__(
        self,
        baseline_window_sec: float = 12.0,
        persistence_sec: float = 1.5,
        min_dwell_sec: float = 30.0,
        cooldown_sec: float = 15.0,
        novelty_threshold: float = 0.6,
    ):
        """
        Initialize novelty boundary detector.

        Args:
            baseline_window_sec: Duration for computing baseline
            persistence_sec: Required duration above threshold
            min_dwell_sec: Minimum time in current section
            cooldown_sec: Cooldown after boundary detection
            novelty_threshold: Threshold for novelty score
        """
        self.baseline_window_sec = baseline_window_sec
        self.persistence_sec = persistence_sec
        self.min_dwell_sec = min_dwell_sec
        self.cooldown_sec = cooldown_sec
        self.novelty_threshold = novelty_threshold

        # State
        self.feature_history: deque = deque(maxlen=200)  # ~20s at 10Hz
        self.novelty_history: deque = deque(maxlen=100)
        self.last_boundary_time = -999.0
        self.current_section_start = 0.0
        self.above_threshold_duration = 0.0
        self.last_update_time = 0.0

    def _feature_to_vector(self, features: SlowFeatures) -> np.ndarray:
        """Convert SlowFeatures to normalized vector."""
        vec = np.concatenate(
            [
                features.band_energies,
                [features.spectral_centroid],
                [features.spectral_rolloff],
                [features.tonalness],
                [features.noisiness],
                [features.onset_rate / 10.0],  # Normalize
            ]
        )
        # L2 normalize
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _compute_baseline(self) -> Optional[np.ndarray]:
        """Compute baseline feature vector from history."""
        if len(self.feature_history) < 10:
            return None

        # Average over history
        vectors = [self._feature_to_vector(f) for f in self.feature_history]
        baseline = np.mean(vectors, axis=0)
        return baseline

    def _compute_novelty(self, features: SlowFeatures) -> float:
        """Compute novelty score for current features."""
        baseline = self._compute_baseline()
        if baseline is None:
            return 0.0

        current = self._feature_to_vector(features)

        # Cosine distance
        similarity = np.dot(current, baseline)
        novelty = 1.0 - similarity

        return float(np.clip(novelty, 0, 1))

    def update(self, features: SlowFeatures) -> Tuple[bool, float]:
        """
        Update detector with new features.

        Args:
            features: SlowFeatures object

        Returns:
            Tuple of (boundary_detected, novelty_score)
        """
        self.feature_history.append(features)
        novelty = self._compute_novelty(features)
        self.novelty_history.append(novelty)

        # Smooth novelty
        smoothed_novelty = float(np.mean(list(self.novelty_history)[-5:]))

        # Check timing constraints
        time_since_boundary = features.timestamp - self.last_boundary_time
        time_in_section = features.timestamp - self.current_section_start

        # Check if in cooldown or minimum dwell
        if time_since_boundary < self.cooldown_sec:
            return False, smoothed_novelty
        if time_in_section < self.min_dwell_sec:
            return False, smoothed_novelty

        # Track persistence
        dt = features.timestamp - self.last_update_time
        self.last_update_time = features.timestamp

        if smoothed_novelty > self.novelty_threshold:
            self.above_threshold_duration += dt
        else:
            self.above_threshold_duration = 0.0

        # Detect boundary if persistent
        boundary_detected = self.above_threshold_duration >= self.persistence_sec

        if boundary_detected:
            self.last_boundary_time = features.timestamp
            self.current_section_start = features.timestamp
            self.above_threshold_duration = 0.0

        return boundary_detected, smoothed_novelty


@dataclass
class LobeCharacteristics:
    """
    Visual and emotional characteristics of a Mandelbrot lobe.

    Each lobe produces Julia sets with distinct visual properties:
    - Period determines symmetry (n-fold rotational symmetry)
    - Location affects connectedness and detail level
    - Size affects transition smoothness
    """

    lobe: int
    sub_lobe: int
    name: str
    # Visual characteristics
    symmetry: int  # n-fold rotational symmetry
    smoothness: float  # 0=jagged, 1=smooth (connected Julia sets)
    complexity: float  # 0=simple, 1=intricate detail
    warmth: float  # 0=cold/angular, 1=warm/rounded
    # Emotional mapping
    tension: float  # 0=resolved, 1=tense
    energy_affinity: float  # Preferred energy level (0=quiet, 1=loud)
    # Transition costs (how "far" to shift)
    gear: int  # Like car transmission: 1=slow/smooth, 5=fast/complex


# Lobe characteristics database - the "transmission" system
LOBE_CHARACTERISTICS: Dict[Tuple[int, int], LobeCharacteristics] = {
    # Cardioid: The "home base" - smooth, warm, connected Julia sets
    # Best for: verses, quiet sections, resolution moments
    (1, 0): LobeCharacteristics(
        lobe=1,
        sub_lobe=0,
        name="Cardioid",
        symmetry=1,
        smoothness=1.0,
        complexity=0.3,
        warmth=1.0,
        tension=0.0,
        energy_affinity=0.4,
        gear=1,
    ),
    # Period-2: The "tension builder" - angular, two-fold symmetry
    # Best for: pre-chorus, building sections, mild tension
    (2, 0): LobeCharacteristics(
        lobe=2,
        sub_lobe=0,
        name="Period-2 Main",
        symmetry=2,
        smoothness=0.7,
        complexity=0.5,
        warmth=0.5,
        tension=0.4,
        energy_affinity=0.6,
        gear=2,
    ),
    # Period-3 upper: Dramatic, three-fold symmetry, complex
    # Best for: chorus, climax, high-energy moments
    (3, 0): LobeCharacteristics(
        lobe=3,
        sub_lobe=0,
        name="Period-3 Upper",
        symmetry=3,
        smoothness=0.5,
        complexity=0.7,
        warmth=0.3,
        tension=0.7,
        energy_affinity=0.8,
        gear=3,
    ),
    # Period-3 lower: Mirror of upper, same characteristics
    (3, 1): LobeCharacteristics(
        lobe=3,
        sub_lobe=1,
        name="Period-3 Lower",
        symmetry=3,
        smoothness=0.5,
        complexity=0.7,
        warmth=0.3,
        tension=0.7,
        energy_affinity=0.8,
        gear=3,
    ),
    # Period-3 mini-Mandelbrot: Very complex, breakdown territory
    (3, 2): LobeCharacteristics(
        lobe=3,
        sub_lobe=2,
        name="Period-3 Mini-M",
        symmetry=3,
        smoothness=0.3,
        complexity=0.9,
        warmth=0.2,
        tension=0.9,
        energy_affinity=0.5,
        gear=4,
    ),
    # Period-4 cascade: Off period-2, transitional feel
    (4, 0): LobeCharacteristics(
        lobe=4,
        sub_lobe=0,
        name="Period-4 Cascade",
        symmetry=4,
        smoothness=0.4,
        complexity=0.6,
        warmth=0.4,
        tension=0.5,
        energy_affinity=0.7,
        gear=3,
    ),
    # Period-4 primaries: Higher complexity, four-fold symmetry
    (4, 1): LobeCharacteristics(
        lobe=4,
        sub_lobe=1,
        name="Period-4 Primary",
        symmetry=4,
        smoothness=0.3,
        complexity=0.8,
        warmth=0.3,
        tension=0.6,
        energy_affinity=0.7,
        gear=4,
    ),
    (4, 2): LobeCharacteristics(
        lobe=4,
        sub_lobe=2,
        name="Period-4 Primary 2",
        symmetry=4,
        smoothness=0.3,
        complexity=0.8,
        warmth=0.3,
        tension=0.6,
        energy_affinity=0.7,
        gear=4,
    ),
    # Period-8: Very intricate, for special moments
    (8, 0): LobeCharacteristics(
        lobe=8,
        sub_lobe=0,
        name="Period-8 Cascade",
        symmetry=8,
        smoothness=0.2,
        complexity=0.95,
        warmth=0.1,
        tension=0.8,
        energy_affinity=0.6,
        gear=5,
    ),
}


class LobeScheduler:
    """
    Strategic lobe selection system - like a car transmission.

    Manages lobe switching based on:
    1. Section characteristics (verse/chorus/breakdown)
    2. Energy levels and musical tension
    3. Transition smoothness (avoid jarring gear shifts)
    4. History (avoid repetition and ping-ponging)

    The "gear" metaphor:
    - Gear 1 (Cardioid): Smooth, warm, low-energy sections
    - Gear 2 (Period-2): Building tension, moderate energy
    - Gear 3 (Period-3/4): High energy, chorus/climax
    - Gear 4-5 (Higher periods): Maximum complexity, breakdowns

    Rules:
    - Prefer shifting one gear at a time (smooth transitions)
    - Match lobe energy_affinity to section energy
    - Use tension to drive dramatic moments
    - Respect cooldown to avoid rapid switching
    """

    def __init__(
        self,
        available_lobes: Optional[List[Tuple[int, int]]] = None,
        max_gear_jump: int = 2,
        min_transition_sec: float = 8.0,
    ):
        """
        Initialize lobe scheduler.

        Args:
            available_lobes: List of (lobe, sub_lobe) tuples to choose from
            max_gear_jump: Maximum gear difference for transitions (default: 2)
            min_transition_sec: Minimum time between lobe switches
        """
        if available_lobes is None:
            # Default: main lobes ordered by gear
            self.available_lobes = [
                (1, 0),  # Gear 1: Cardioid
                (2, 0),  # Gear 2: Period-2
                (3, 0),  # Gear 3: Period-3 upper
                (3, 1),  # Gear 3: Period-3 lower
                (4, 0),  # Gear 3: Period-4 cascade
                (4, 1),  # Gear 4: Period-4 primary
            ]
        else:
            self.available_lobes = available_lobes

        self.max_gear_jump = max_gear_jump
        self.min_transition_sec = min_transition_sec

        # State
        self.lobe_history: deque = deque(maxlen=5)
        self.current_lobe = (1, 0)  # Start at cardioid
        self.last_transition_time = -999.0

    def get_characteristics(self, lobe: Tuple[int, int]) -> LobeCharacteristics:
        """Get characteristics for a lobe, with fallback for unknown lobes."""
        if lobe in LOBE_CHARACTERISTICS:
            return LOBE_CHARACTERISTICS[lobe]
        # Fallback for unknown lobes
        period = lobe[0]
        return LobeCharacteristics(
            lobe=lobe[0],
            sub_lobe=lobe[1],
            name=f"Period-{period}",
            symmetry=period,
            smoothness=max(0.1, 1.0 - period * 0.1),
            complexity=min(1.0, period * 0.15),
            warmth=max(0.1, 1.0 - period * 0.1),
            tension=min(1.0, period * 0.1),
            energy_affinity=0.5,
            gear=min(5, (period + 1) // 2),
        )

    def select_next_lobe(
        self,
        energy_level: float,
        novelty: float,
        timestamp: float = 0.0,
        section_type: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Select next lobe based on audio state and section context.

        Args:
            energy_level: Current energy level [0, 1]
            novelty: Current novelty score [0, 1]
            timestamp: Current time in seconds
            section_type: Optional hint ('verse', 'chorus', 'breakdown', 'build')

        Returns:
            Tuple of (lobe, sub_lobe)
        """
        # Enforce minimum transition time
        if timestamp - self.last_transition_time < self.min_transition_sec:
            return self.current_lobe

        current_chars = self.get_characteristics(self.current_lobe)
        current_gear = current_chars.gear

        # For verse/quiet sections, allow returning to cardioid even if in history
        allow_history_override = section_type in ("verse",) and energy_level < 0.4

        # Score each candidate lobe
        candidates_with_scores: List[Tuple[Tuple[int, int], float]] = []

        for lobe in self.available_lobes:
            if lobe == self.current_lobe:
                continue
            # Skip lobes in history unless override is active for this lobe
            if lobe in self.lobe_history:
                if allow_history_override and lobe == (1, 0):
                    pass  # Allow cardioid even if in history
                else:
                    continue

            chars = self.get_characteristics(lobe)
            score = self._score_lobe(
                chars, current_gear, energy_level, novelty, section_type
            )
            candidates_with_scores.append((lobe, score))

        # If no candidates, reset history and try again
        if not candidates_with_scores:
            self.lobe_history.clear()
            for lobe in self.available_lobes:
                if lobe == self.current_lobe:
                    continue
                chars = self.get_characteristics(lobe)
                score = self._score_lobe(
                    chars, current_gear, energy_level, novelty, section_type
                )
                candidates_with_scores.append((lobe, score))

        if not candidates_with_scores:
            return self.current_lobe

        # Sort by score (highest first)
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Select from top candidates with some randomness
        # Take top 3 and weighted random select
        top_candidates = candidates_with_scores[:3]
        weights = [score for _, score in top_candidates]
        total_weight = sum(weights) + 1e-8
        normalized_weights = [w / total_weight for w in weights]

        # Weighted random selection
        r = np.random.random()
        cumsum = 0.0
        selected = top_candidates[0][0]
        for (lobe, _), w in zip(top_candidates, normalized_weights):
            cumsum += w
            if r < cumsum:
                selected = lobe
                break

        # Update state
        self.lobe_history.append(self.current_lobe)
        self.current_lobe = selected
        self.last_transition_time = timestamp

        return selected

    def _score_lobe(
        self,
        chars: LobeCharacteristics,
        current_gear: int,
        energy_level: float,
        novelty: float,
        section_type: Optional[str],
    ) -> float:
        """
        Score a candidate lobe based on fit with current audio state.

        Higher score = better fit.
        """
        score = 0.0

        # 1. Gear proximity bonus (prefer smooth transitions)
        gear_diff = abs(chars.gear - current_gear)
        if gear_diff <= self.max_gear_jump:
            score += 2.0 * (1.0 - gear_diff / (self.max_gear_jump + 1))
        else:
            score -= 1.0  # Penalty for too big a jump

        # 2. Energy affinity match
        energy_match = 1.0 - abs(chars.energy_affinity - energy_level)
        score += 2.0 * energy_match

        # 3. Section type bonuses (strong influence)
        if section_type:
            if section_type == "verse":
                # Strongly prefer low gear, smooth, warm
                score += chars.smoothness * 1.0
                score += chars.warmth * 1.0
                if chars.gear == 1:
                    score += 3.0  # Strong bonus for cardioid
                elif chars.gear == 2:
                    score += 1.5
                elif chars.gear >= 3:
                    score -= 1.0  # Penalty for high gear in verse
            elif section_type == "chorus":
                # Prefer medium-high gear, more complex
                score += chars.complexity * 0.8
                score += chars.tension * 0.5
                if 2 <= chars.gear <= 4:
                    score += 2.0
            elif section_type == "breakdown":
                # Prefer high gear, maximum complexity
                score += chars.complexity * 1.0
                score += chars.tension * 0.5
                if chars.gear >= 4:
                    score += 2.0
                elif chars.gear >= 3:
                    score += 1.0
            elif section_type == "build":
                # Match tension to novelty, prefer gradual increase
                score += chars.tension * novelty * 1.5
                # Prefer incrementing gear from current
                if chars.gear == current_gear + 1:
                    score += 2.5
                elif chars.gear == current_gear:
                    score += 1.0

        # 4. High novelty = allow bigger jumps
        if novelty > 0.7:
            score += 0.5  # General bonus for any transition at high novelty

        # 5. Complexity matches energy at high levels
        if energy_level > 0.7:
            score += chars.complexity * 0.5

        return max(0.0, score)

    def suggest_transition_duration(
        self, from_lobe: Tuple[int, int], to_lobe: Tuple[int, int]
    ) -> float:
        """
        Suggest transition duration based on gear difference.

        Larger gear jumps need longer transitions for smoothness.
        """
        from_chars = self.get_characteristics(from_lobe)
        to_chars = self.get_characteristics(to_lobe)
        gear_diff = abs(to_chars.gear - from_chars.gear)

        # Base duration + extra for bigger jumps
        return 2.0 + gear_diff * 1.0  # 2-5 seconds typically


class BeatLockedOmega:
    """
    Manages beat-locked angular velocity for musically coherent orbits.

    Locks the base omega to detected BPM so orbit motion feels "on the beat".
    The model can still modulate omega_scale for expressive variation.
    """

    def __init__(
        self,
        default_bpm: float = 120.0,
        subdivision: int = 1,
        smoothing_tau: float = 2.0,
    ):
        """
        Initialize beat-locked omega manager.

        Args:
            default_bpm: Default tempo when none detected
            subdivision: Beat subdivision (1 = quarter notes, 2 = eighth notes, etc.)
            smoothing_tau: Time constant for BPM smoothing (seconds)
        """
        self.default_bpm = default_bpm
        self.subdivision = subdivision
        self.smoothing_tau = smoothing_tau

        self.current_bpm = default_bpm
        self.smoothed_bpm = default_bpm
        self.last_update_time = 0.0

    def update_tempo(self, detected_bpm: float, timestamp: float):
        """
        Update the locked tempo from detected BPM.

        Args:
            detected_bpm: Detected tempo in BPM
            timestamp: Current timestamp
        """
        # Validate BPM range (typical music is 60-180 BPM)
        if 40.0 <= detected_bpm <= 220.0:
            self.current_bpm = detected_bpm

        # Smooth the BPM to avoid jarring changes
        dt = timestamp - self.last_update_time
        if dt > 0:
            alpha = 1.0 - np.exp(-dt / self.smoothing_tau)
            self.smoothed_bpm = self.smoothed_bpm + alpha * (
                self.current_bpm - self.smoothed_bpm
            )

        self.last_update_time = timestamp

    def get_omega_base(self) -> float:
        """
        Get the beat-locked base angular velocity.

        Returns:
            Base omega in radians per second
        """
        # Convert BPM to radians/second
        # One full orbit per beat (or per subdivision)
        beats_per_second = self.smoothed_bpm / 60.0
        omega = 2 * np.pi * beats_per_second * self.subdivision
        return omega

    def get_omega(self, omega_scale: float = 1.0) -> float:
        """
        Get scaled angular velocity.

        Args:
            omega_scale: Scale factor from model prediction (typically 0.5 to 2.0)

        Returns:
            Final omega in radians per second
        """
        return self.get_omega_base() * np.clip(omega_scale, 0.25, 4.0)


class OrbitStateMachine:
    """
    Main orbit state machine managing carrier + residual synthesis.
    Outputs c(t) at render rate using OrbitSynthesizer.
    """

    def __init__(
        self,
        render_rate: float = 60.0,
        residual_k: int = 6,
        residual_cap: float = 0.5,
        s_loud: float = 1.02,
        s_quiet_noisy: float = 2.5,
        s_quiet_tonal: float = 0.4,
        s_smoothing_tau: float = 1.0,
    ):
        """
        Initialize orbit state machine.

        Args:
            render_rate: Render rate in Hz
            residual_k: Number of residual circles
            residual_cap: Maximum residual amplitude relative to lobe radius
            s_loud: Target s for loud sections
            s_quiet_noisy: Target s for quiet+noisy
            s_quiet_tonal: Target s for quiet+tonal
            s_smoothing_tau: Time constant for s smoothing (seconds)
        """
        self.render_rate = render_rate
        self.dt = 1.0 / render_rate
        self.residual_k = residual_k
        self.residual_cap = residual_cap
        self.s_loud = s_loud
        self.s_quiet_noisy = s_quiet_noisy
        self.s_quiet_tonal = s_quiet_tonal
        self.s_smoothing_tau = s_smoothing_tau

        # Initialize orbit state using runtime_core
        self.orbit_state = make_orbit_state(
            lobe=1,
            sub_lobe=0,
            theta=0.0,
            omega=1.0,
            s=1.02,
            alpha=0.3,
            k_residuals=residual_k,
            seed=42,
        )

        self.s_target = 1.02

        # Transition state
        self.in_transition = False
        self.transition_start_lobe = (1, 0)
        self.transition_end_lobe = (1, 0)
        self.transition_progress = 0.0
        self.transition_duration = 3.0  # seconds

        # Control inputs (updated by controller)
        self.control_loudness = 0.5
        self.control_tonalness = 0.5
        self.control_noisiness = 0.5

    def start_transition(self, new_lobe: int, new_sub_lobe: int, duration: float = 3.0):
        """Start transition to new lobe."""
        self.in_transition = True
        self.transition_start_lobe = (self.orbit_state.lobe, self.orbit_state.sub_lobe)
        self.transition_end_lobe = (new_lobe, new_sub_lobe)
        self.transition_progress = 0.0
        self.transition_duration = duration

    def update_control_inputs(
        self, loudness: float, tonalness: float, noisiness: float
    ):
        """Update control inputs from slow features."""
        self.control_loudness = loudness
        self.control_tonalness = tonalness
        self.control_noisiness = noisiness

    def _compute_s_target(self) -> float:
        """Compute target s based on audio state."""
        L = self.control_loudness
        T = self.control_tonalness
        N = self.control_noisiness

        # Mix based on state
        if L > 0.6:
            # Loud: near boundary
            return self.s_loud
        elif N > 0.6:
            # Quiet + noisy: push outward
            return self.s_quiet_noisy
        elif T > 0.6:
            # Quiet + tonal: pull inward
            return self.s_quiet_tonal
        else:
            # Mixed state: interpolate
            quiet_target = (
                T * self.s_quiet_tonal
                + N * self.s_quiet_noisy
                + (1 - T - N) * self.s_loud
            )
            return L * self.s_loud + (1 - L) * quiet_target

    def _smooth_s(self, s_current: float, s_target: float, dt: float) -> float:
        """Apply exponential smoothing to s."""
        alpha = 1.0 - np.exp(-dt / self.s_smoothing_tau)
        return s_current + alpha * (s_target - s_current)

    def step(self, impact_envelope_value: float = 0.0) -> complex:
        """
        Step the orbit state machine by one render frame.

        Args:
            impact_envelope_value: Current impact envelope value [0, 1]

        Returns:
            Complex Julia parameter c(t)
        """
        # Update s target and smooth
        base_s_target = self._compute_s_target()

        # Apply impact envelope override
        if impact_envelope_value > 0.0:
            # Push s outward during impact
            impact_s = 1.15
            self.s_target = (
                base_s_target + (impact_s - base_s_target) * impact_envelope_value
            )
        else:
            self.s_target = base_s_target

        smoothed_s = self._smooth_s(self.orbit_state.s, self.s_target, self.dt)

        # Update residual alpha (boosted by impact)
        base_alpha = 0.3  # Base residual strength
        impact_alpha_boost = 0.5 * impact_envelope_value
        residual_alpha = base_alpha + impact_alpha_boost

        # Handle transitions - just update lobe/sub_lobe for now
        if self.in_transition:
            self.transition_progress += self.dt / self.transition_duration
            if self.transition_progress >= 1.0:
                # Transition complete
                self.orbit_state.lobe = self.transition_end_lobe[0]
                self.orbit_state.sub_lobe = self.transition_end_lobe[1]
                self.in_transition = False

        # Update orbit state parameters (mutation happens via runtime_core)
        self.orbit_state.s = smoothed_s
        self.orbit_state.alpha = residual_alpha

        # Advance state and synthesize
        c = step_orbit(self.orbit_state, self.dt)

        return complex(c.re, c.im)

    def get_debug_info(self) -> Dict[str, Union[float, int, str]]:
        """Get debug information for HUD."""
        return {
            "lobe": self.orbit_state.lobe,
            "sub_lobe": self.orbit_state.sub_lobe,
            "theta": self.orbit_state.theta,
            "omega": self.orbit_state.omega,
            "s": self.orbit_state.s,
            "s_target": self.s_target,
            "residual_alpha": self.orbit_state.alpha,
            "in_transition": self.in_transition,
            "loudness": self.control_loudness,
            "tonalness": self.control_tonalness,
            "noisiness": self.control_noisiness,
        }


class LiveController:
    """
    Main controller coordinating all components.
    Implements the full live audio-reactive system.
    """

    def __init__(
        self,
        sr: int = 22050,
        render_rate: float = 60.0,
        control_rate: float = 10.0,
        fast_feature_rate: float = 100.0,
    ):
        """
        Initialize live controller.

        Args:
            sr: Audio sample rate
            render_rate: Render rate in Hz
            control_rate: Control update rate in Hz
            fast_feature_rate: Fast feature extraction rate in Hz
        """
        self.sr = sr
        self.render_rate = render_rate
        self.control_rate = control_rate
        self.fast_feature_rate = fast_feature_rate

        # Initialize components
        self.feature_stream = AudioFeatureStream(sr=sr)
        self.impact_detector = ImpactDetector()
        self.impact_envelope = ImpactEnvelope()
        self.boundary_detector = NoveltyBoundaryDetector()
        self.lobe_scheduler = LobeScheduler()
        self.orbit_state = OrbitStateMachine(render_rate=render_rate)

        # Event logs
        self.impact_events: List[ImpactEvent] = []
        self.boundary_events: List[BoundaryEvent] = []

        # Timing
        self.current_time = 0.0
        self.last_control_update = 0.0
        self.last_fast_update = 0.0

        # Latest features
        self.latest_slow_features: Optional[SlowFeatures] = None

    def process_audio_frame(self, audio_chunk: np.ndarray, timestamp: float) -> complex:
        """
        Process an audio chunk and generate Julia parameter.

        Args:
            audio_chunk: Audio data for this frame
            timestamp: Current timestamp

        Returns:
            Complex Julia parameter c(t)
        """
        self.current_time = timestamp

        # Fast feature extraction and impact detection
        if timestamp - self.last_fast_update >= 1.0 / self.fast_feature_rate:
            fast_features = self.feature_stream.compute_fast_features(
                audio_chunk, timestamp
            )
            is_impact, impact_score = self.impact_detector.detect(fast_features)

            if is_impact:
                self.impact_envelope.trigger(timestamp)
                # Log event
                s_overshoot = self.impact_envelope.s_impulse_hi
                event = ImpactEvent(timestamp, impact_score, s_overshoot)
                self.impact_events.append(event)

            self.last_fast_update = timestamp

        # Slow feature extraction and control updates
        if timestamp - self.last_control_update >= 1.0 / self.control_rate:
            slow_features = self.feature_stream.compute_slow_features(
                audio_chunk, timestamp
            )
            self.latest_slow_features = slow_features

            # Update orbit control inputs
            self.orbit_state.update_control_inputs(
                slow_features.loudness, slow_features.tonalness, slow_features.noisiness
            )

            # Boundary detection
            is_boundary, novelty = self.boundary_detector.update(slow_features)

            if is_boundary:
                # Select new lobe
                new_lobe, new_sub_lobe = self.lobe_scheduler.select_next_lobe(
                    slow_features.loudness, novelty
                )

                # Start transition
                self.orbit_state.start_transition(new_lobe, new_sub_lobe, duration=3.0)

                # Log event
                prev_dwell = timestamp - (
                    self.boundary_events[-1].timestamp if self.boundary_events else 0.0
                )
                event = BoundaryEvent(
                    timestamp,
                    novelty,
                    slow_features.loudness,
                    new_lobe,
                    new_sub_lobe,
                    prev_dwell,
                )
                self.boundary_events.append(event)

            self.last_control_update = timestamp

        # Update impact envelope
        envelope_value = self.impact_envelope.update(timestamp)

        # Step orbit state machine
        c = self.orbit_state.step(envelope_value)

        return c

    def get_debug_overlay(self) -> str:
        """Get debug overlay string for HUD."""
        info = self.orbit_state.get_debug_info()

        recent_impacts = len(
            [e for e in self.impact_events if self.current_time - e.timestamp < 5.0]
        )
        recent_boundaries = len(
            [e for e in self.boundary_events if self.current_time - e.timestamp < 30.0]
        )

        overlay = f"""FractalSync Live
Lobe: {info['lobe']}-{info['sub_lobe']} {'[TRANSITION]' if info['in_transition'] else ''}
s: {info['s']:.3f} → {info['s_target']:.3f}
L/T/N: {info['loudness']:.2f}/{info['tonalness']:.2f}/{info['noisiness']:.2f}
Impacts (5s): {recent_impacts}
Boundaries (30s): {recent_boundaries}
"""
        return overlay
