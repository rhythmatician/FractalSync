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
import time

from src.mandelbrot_orbits import MandelbrotGeometry, get_preset_orbit, list_preset_names


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
        self,
        audio_frame: np.ndarray,
        timestamp: float
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
        onset_strength = float(librosa.onset.onset_strength(
            S=mag, sr=self.sr, hop_length=self.fast_hop, aggregate=np.mean
        ))
        
        # Spectral flux
        if self.prev_spectrum is not None:
            flux = np.sum(np.maximum(0, mag[:, -1] - self.prev_spectrum))
            spectral_flux = float(flux / (np.sum(mag[:, -1]) + 1e-8))
        else:
            spectral_flux = 0.0
        
        self.prev_spectrum = mag[:, -1].copy() if mag.shape[1] > 0 else None
        
        # Band energies via mel filterbank
        mel_spec = self.mel_fb @ (mag ** 2)
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
            timestamp=timestamp
        )
    
    def compute_slow_features(
        self,
        audio_segment: np.ndarray,
        timestamp: float
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
        power = mag ** 2
        
        # Loudness (RMS energy normalized)
        rms = librosa.feature.rms(S=mag, frame_length=self.n_fft, hop_length=self.slow_hop)
        loudness = float(np.mean(rms))
        loudness = np.clip(loudness / 0.3, 0, 1)  # Normalize to [0, 1]
        
        # Band energies
        mel_spec = self.mel_fb @ power
        band_energies = np.log1p(np.mean(mel_spec, axis=1))
        
        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
            S=mag, sr=self.sr, hop_length=self.slow_hop
        )))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(
            S=mag, sr=self.sr, hop_length=self.slow_hop
        )))
        
        # Tonalness: use spectral flatness (inverse)
        flatness = librosa.feature.spectral_flatness(S=mag, hop_length=self.slow_hop)
        tonalness = float(1.0 - np.mean(flatness))  # High = tonal
        tonalness = np.clip(tonalness, 0, 1)
        
        # Noisiness: spectral flatness itself
        noisiness = float(np.mean(flatness))
        noisiness = np.clip(noisiness, 0, 1)
        
        # Onset rate
        onset_env = librosa.onset.onset_strength(S=mag, sr=self.sr, hop_length=self.slow_hop)
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
            timestamp=timestamp
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
            0.4 * features.onset_strength +
            0.3 * features.spectral_flux +
            0.2 * np.max(features.band_deltas) +
            0.1 * max(0, features.kick_delta)
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
        is_impact = score > threshold and not self.in_refractory
        
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
        vec = np.concatenate([
            features.band_energies,
            [features.spectral_centroid],
            [features.spectral_rolloff],
            [features.tonalness],
            [features.noisiness],
            [features.onset_rate / 10.0],  # Normalize
        ])
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


class LobeScheduler:
    """
    Manages lobe selection and transitions for section boundaries.
    Avoids repetition and ping-ponging.
    """
    
    def __init__(self, available_lobes: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize lobe scheduler.
        
        Args:
            available_lobes: List of (lobe, sub_lobe) tuples to choose from
        """
        if available_lobes is None:
            # Default: cardioid + main bulbs
            self.available_lobes = [
                (1, 0),  # Cardioid
                (2, 0),  # Period-2
                (3, 0),  # Period-3 upper
                (3, 1),  # Period-3 lower
                (4, 0),  # Period-4 cascade
                (4, 1),  # Period-4 primary 1
            ]
        else:
            self.available_lobes = available_lobes
        
        # History
        self.lobe_history: deque = deque(maxlen=5)
        self.current_lobe = (1, 0)  # Start at cardioid
    
    def select_next_lobe(
        self,
        energy_level: float,
        novelty: float
    ) -> Tuple[int, int]:
        """
        Select next lobe based on audio state.
        
        Args:
            energy_level: Current energy level [0, 1]
            novelty: Current novelty score
            
        Returns:
            Tuple of (lobe, sub_lobe)
        """
        # Filter out current and recent lobes
        candidates = [
            lobe for lobe in self.available_lobes
            if lobe != self.current_lobe and lobe not in self.lobe_history
        ]
        
        if not candidates:
            # All lobes used, reset history
            self.lobe_history.clear()
            candidates = [lobe for lobe in self.available_lobes if lobe != self.current_lobe]
        
        if not candidates:
            # Fallback
            candidates = self.available_lobes
        
        # Simple selection: random with slight bias toward period-2/3 for high energy
        if energy_level > 0.7 and novelty > 0.6:
            # Prefer bigger bulbs for high energy
            preferred = [(2, 0), (3, 0), (3, 1)]
            preferred_candidates = [c for c in candidates if c in preferred]
            if preferred_candidates:
                candidates = preferred_candidates
        
        # Select randomly from candidates
        selected = candidates[np.random.randint(len(candidates))]
        
        # Update history
        self.lobe_history.append(self.current_lobe)
        self.current_lobe = selected
        
        return selected


class OrbitStateMachine:
    """
    Main orbit state machine managing carrier + residual synthesis.
    Outputs c(t) at render rate.
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
        
        # Current orbit state
        self.current_lobe = 1
        self.current_sub_lobe = 0
        self.theta = 0.0
        self.omega = 1.0  # rad/s
        self.s = 1.02
        self.s_target = 1.02
        
        # Transition state
        self.in_transition = False
        self.transition_start_lobe = (1, 0)
        self.transition_end_lobe = (1, 0)
        self.transition_progress = 0.0
        self.transition_duration = 3.0  # seconds
        
        # Residual state
        self.residual_phases = np.random.uniform(0, 2*np.pi, residual_k)
        self.residual_omegas = np.ones(residual_k)  # Will be set by tempo
        self.residual_alpha = 0.0
        
        # Control inputs (updated by controller)
        self.control_loudness = 0.5
        self.control_tonalness = 0.5
        self.control_noisiness = 0.5
        
        # Get orbit object for geometry
        self.orbit = get_preset_orbit("cardioid_boundary")
    
    def start_transition(self, new_lobe: int, new_sub_lobe: int, duration: float = 3.0):
        """Start transition to new lobe."""
        self.in_transition = True
        self.transition_start_lobe = (self.current_lobe, self.current_sub_lobe)
        self.transition_end_lobe = (new_lobe, new_sub_lobe)
        self.transition_progress = 0.0
        self.transition_duration = duration
    
    def update_control_inputs(
        self,
        loudness: float,
        tonalness: float,
        noisiness: float
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
                T * self.s_quiet_tonal +
                N * self.s_quiet_noisy +
                (1 - T - N) * self.s_loud
            )
            return L * self.s_loud + (1 - L) * quiet_target
    
    def _smooth_s(self, s_current: float, s_target: float, dt: float) -> float:
        """Apply exponential smoothing to s."""
        alpha = 1.0 - np.exp(-dt / self.s_smoothing_tau)
        return s_current + alpha * (s_target - s_current)
    
    def _get_carrier_point(self) -> complex:
        """Get carrier orbit point at current theta."""
        return MandelbrotGeometry.lobe_point_at_angle(
            self.current_lobe,
            self.theta,
            self.s,
            self.current_sub_lobe
        )
    
    def _compute_residual(self) -> complex:
        """Compute residual texture from epicycles."""
        if self.residual_alpha == 0.0:
            return 0j
        
        # Get lobe radius for scaling
        radius = MandelbrotGeometry.period_n_bulb_radius(
            self.current_lobe, self.current_sub_lobe
        )
        if self.current_lobe == 1:
            radius = 0.25  # Cardioid reference scale
        
        # Sum of epicycles
        residual = 0j
        for k in range(self.residual_k):
            # Amplitude decreases as 1/k^2
            amplitude = self.residual_alpha * (self.s * radius) / ((k + 1) ** 2)
            
            # Simple band gating (could be enhanced with actual audio bands)
            g_k = 1.0  # Placeholder
            
            # Phasor
            residual += amplitude * g_k * np.exp(1j * self.residual_phases[k])
        
        # Cap residual magnitude
        mag = abs(residual)
        cap = self.residual_cap * radius
        if mag > cap:
            residual *= cap / mag
        
        return residual
    
    def step(self, impact_envelope_value: float = 0.0) -> complex:
        """
        Step the orbit state machine by one render frame.
        
        Args:
            impact_envelope_value: Current impact envelope value [0, 1]
            
        Returns:
            Complex Julia parameter c(t)
        """
        # Update theta
        self.theta += self.omega * self.dt
        self.theta = self.theta % (2 * np.pi)
        
        # Update residual phases
        self.residual_phases += self.residual_omegas * self.dt
        self.residual_phases = self.residual_phases % (2 * np.pi)
        
        # Update s target and smooth
        base_s_target = self._compute_s_target()
        
        # Apply impact envelope override
        if impact_envelope_value > 0.0:
            # Push s outward during impact
            impact_s = 1.15
            self.s_target = base_s_target + (impact_s - base_s_target) * impact_envelope_value
        else:
            self.s_target = base_s_target
        
        self.s = self._smooth_s(self.s, self.s_target, self.dt)
        
        # Update residual alpha (boosted by impact)
        base_alpha = 0.3  # Base residual strength
        impact_alpha_boost = 0.5 * impact_envelope_value
        self.residual_alpha = base_alpha + impact_alpha_boost
        
        # Handle transitions
        if self.in_transition:
            self.transition_progress += self.dt / self.transition_duration
            if self.transition_progress >= 1.0:
                # Transition complete
                self.current_lobe, self.current_sub_lobe = self.transition_end_lobe
                self.in_transition = False
                # Update orbit object
                preset_names = list_preset_names()
                # Try to find matching preset
                for name in preset_names:
                    orbit = get_preset_orbit(name)
                    if hasattr(orbit, 'lobe') and orbit.lobe == self.current_lobe:
                        if hasattr(orbit, 'sub_lobe') and orbit.sub_lobe == self.current_sub_lobe:
                            self.orbit = orbit
                            break
            else:
                # Interpolate between lobes (simplified - just use start lobe for now)
                self.current_lobe, self.current_sub_lobe = self.transition_start_lobe
        
        # Get carrier + residual
        c_carrier = self._get_carrier_point()
        c_residual = self._compute_residual()
        c = c_carrier + c_residual
        
        return c
    
    def get_debug_info(self) -> Dict[str, Union[float, int, str]]:
        """Get debug information for HUD."""
        return {
            "lobe": self.current_lobe,
            "sub_lobe": self.current_sub_lobe,
            "theta": self.theta,
            "omega": self.omega,
            "s": self.s,
            "s_target": self.s_target,
            "residual_alpha": self.residual_alpha,
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
    
    def process_audio_frame(
        self,
        audio_chunk: np.ndarray,
        timestamp: float
    ) -> complex:
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
            fast_features = self.feature_stream.compute_fast_features(audio_chunk, timestamp)
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
            slow_features = self.feature_stream.compute_slow_features(audio_chunk, timestamp)
            self.latest_slow_features = slow_features
            
            # Update orbit control inputs
            self.orbit_state.update_control_inputs(
                slow_features.loudness,
                slow_features.tonalness,
                slow_features.noisiness
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
                    timestamp, novelty, slow_features.loudness,
                    new_lobe, new_sub_lobe, prev_dwell
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
        
        recent_impacts = len([e for e in self.impact_events if self.current_time - e.timestamp < 5.0])
        recent_boundaries = len([e for e in self.boundary_events if self.current_time - e.timestamp < 30.0])
        
        overlay = f"""FractalSync Live
Lobe: {info['lobe']}-{info['sub_lobe']} {'[TRANSITION]' if info['in_transition'] else ''}
s: {info['s']:.3f} → {info['s_target']:.3f}
L/T/N: {info['loudness']:.2f}/{info['tonalness']:.2f}/{info['noisiness']:.2f}
Impacts (5s): {recent_impacts}
Boundaries (30s): {recent_boundaries}
"""
        return overlay
