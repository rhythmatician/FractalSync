"""
Orbit engine for synthetic trajectory generation around Mandelbrot bulbs.

Generates synthetic orbits with audio-like feature patterns for training
data augmentation. Integrates with the existing mandelbrot_orbits module
to create trajectories that correlate with synthetic audio features.
"""

import numpy as np
from typing import Tuple, Dict, List, Union, Optional
from src.mandelbrot_orbits import (
    get_preset_orbit,
    list_preset_names,
)


class OrbitEngine:
    """
    Synthetic orbit generator for creating training data.

    Generates trajectories around Mandelbrot bulbs and creates
    corresponding synthetic audio features that correlate with
    the orbital motion.
    """

    def __init__(
        self,
        n_audio_features: int = 6,
        sr: int = 22050,
        hop_length: int = 512,
    ):
        """
        Initialize orbit engine.

        Args:
            n_audio_features: Number of audio features to generate
            sr: Sample rate (for time calculations)
            hop_length: Hop length (for frame calculations)
        """
        if not isinstance(n_audio_features, int):
            raise TypeError(
                f"n_audio_features must be an int, got {type(n_audio_features).__name__}"
            )
        if n_audio_features <= 0:
            raise ValueError(
                f"n_audio_features must be a positive integer, got {n_audio_features}"
            )
        self.n_audio_features = n_audio_features
        self.sr = sr
        self.hop_length = hop_length

    def generate_synthetic_trajectory(
        self,
        orbit_name: str,
        n_samples: int,
        audio_correlation: str = "velocity",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic trajectory with correlated audio features.

        Args:
            orbit_name: Name of preset orbit to use
            n_samples: Number of trajectory points to generate
            audio_correlation: Type of correlation between audio and motion
                - 'velocity': Audio features correlate with velocity magnitude
                - 'position': Audio features correlate with position
                - 'acceleration': Audio features correlate with acceleration

        Returns:
            Tuple of (audio_features, visual_params)
            - audio_features: shape (n_samples, n_audio_features)
            - visual_params: shape (n_samples, 2) [c_real, c_imag]
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
            
        # Get the orbit
        orbit = get_preset_orbit(orbit_name)

        # Sample positions along orbit
        positions = orbit.sample(n_samples)  # Shape: (n_samples, 2)

        # Compute velocities
        velocities = orbit.compute_velocities(n_samples)  # Shape: (n_samples, 2)

        # Generate correlated audio features
        if audio_correlation == "velocity":
            audio_features = self._generate_velocity_correlated_features(
                velocities, n_samples
            )
        elif audio_correlation == "position":
            audio_features = self._generate_position_correlated_features(
                positions, n_samples
            )
        elif audio_correlation == "acceleration":
            accelerations = self._compute_accelerations(velocities)
            audio_features = self._generate_acceleration_correlated_features(
                accelerations, n_samples
            )
        else:
            raise ValueError(f"Unknown correlation type: {audio_correlation}")

        return audio_features, positions

    def _generate_velocity_correlated_features(
        self, velocities: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Generate audio features that correlate with velocity magnitude.

        Simulates the intuition that faster motion → higher energy/intensity.

        Args:
            velocities: Velocity vectors (n_samples, 2)
            n_samples: Number of samples

        Returns:
            Audio features (n_samples, n_audio_features)
        """
        # Compute velocity magnitude
        vel_mag = np.linalg.norm(velocities, axis=1)  # (n_samples,)

        # Normalize to [0, 1]
        vel_norm = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-8)

        # Generate features that scale with velocity
        features = np.zeros((n_samples, self.n_audio_features), dtype=np.float32)

        # Base feature patterns (support variable feature counts)
        if self.n_audio_features >= 1:
            # Feature 0: Spectral centroid ~ velocity magnitude (brightness increases with speed)
            features[:, 0] = vel_norm * 0.8 + 0.1  # Range [0.1, 0.9]
        
        if self.n_audio_features >= 2:
            # Feature 1: Spectral flux ~ velocity changes (transients)
            vel_diff = np.diff(vel_mag, prepend=vel_mag[0])
            features[:, 1] = np.abs(vel_diff) / (np.max(np.abs(vel_diff)) + 1e-8)
        
        if self.n_audio_features >= 3:
            # Feature 2: RMS energy ~ velocity magnitude (louder when faster)
            features[:, 2] = vel_norm * 0.7 + 0.2  # Range [0.2, 0.9]
        
        if self.n_audio_features >= 4:
            # Feature 3: Zero-crossing rate ~ add some variation
            features[:, 3] = 0.3 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
        
        if self.n_audio_features >= 5:
            # Feature 4: Onset strength ~ velocity peaks
            # High when velocity magnitude has local maxima
            onset = np.zeros(n_samples)
            for i in range(1, n_samples - 1):
                if vel_mag[i] > vel_mag[i - 1] and vel_mag[i] > vel_mag[i + 1]:
                    onset[i] = 1.0
            features[:, 4] = onset
        
        if self.n_audio_features >= 6:
            # Feature 5: Spectral rolloff ~ velocity (tonal vs noisy)
            features[:, 5] = vel_norm * 0.6 + 0.3  # Range [0.3, 0.9]
        
        # Fill any additional features with variations
        for i in range(6, self.n_audio_features):
            phase = 2 * np.pi * i / self.n_audio_features
            features[:, i] = 0.5 + 0.3 * np.sin(np.linspace(phase, phase + 2 * np.pi, n_samples))

        # Add small random noise for realism
        features += np.random.normal(0, 0.02, features.shape)
        features = np.clip(features, 0, 1)

        return features

    def _generate_position_correlated_features(
        self, positions: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Generate audio features that correlate with position in complex plane.

        Args:
            positions: Position vectors (n_samples, 2) [real, imag]
            n_samples: Number of samples

        Returns:
            Audio features (n_samples, n_audio_features)
        """
        features = np.zeros((n_samples, self.n_audio_features), dtype=np.float32)

        # Normalize positions to [0, 1] range
        real_norm = (positions[:, 0] - positions[:, 0].min()) / (
            positions[:, 0].max() - positions[:, 0].min() + 1e-8
        )
        imag_norm = (positions[:, 1] - positions[:, 1].min()) / (
            positions[:, 1].max() - positions[:, 1].min() + 1e-8
        )

        # Map position to features (support variable feature counts)
        if self.n_audio_features >= 1:
            features[:, 0] = real_norm * 0.8 + 0.1  # Real → centroid
        if self.n_audio_features >= 2:
            features[:, 1] = imag_norm * 0.8 + 0.1  # Imag → flux
        if self.n_audio_features >= 3:
            features[:, 2] = (real_norm + imag_norm) / 2  # Combined → energy
        if self.n_audio_features >= 4:
            features[:, 3] = np.abs(real_norm - imag_norm)  # Difference → ZCR
        if self.n_audio_features >= 5:
            features[:, 4] = np.sin(2 * np.pi * real_norm)  # Periodic → onsets
        if self.n_audio_features >= 6:
            features[:, 5] = np.cos(2 * np.pi * imag_norm)  # Periodic → rolloff
        
        # Fill any additional features
        for i in range(6, self.n_audio_features):
            phase = 2 * np.pi * i / self.n_audio_features
            features[:, i] = 0.5 + 0.3 * np.cos(np.linspace(phase, phase + 4 * np.pi, n_samples))

        # Ensure valid range
        features = np.clip(features, 0, 1)

        return features

    def _generate_acceleration_correlated_features(
        self, accelerations: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """
        Generate audio features that correlate with acceleration.

        Args:
            accelerations: Acceleration vectors (n_samples, 2)
            n_samples: Number of samples

        Returns:
            Audio features (n_samples, n_audio_features)
        """
        # Compute acceleration magnitude
        acc_mag = np.linalg.norm(accelerations, axis=1)

        # Normalize
        acc_norm = (acc_mag - acc_mag.min()) / (acc_mag.max() - acc_mag.min() + 1e-8)

        features = np.zeros((n_samples, self.n_audio_features), dtype=np.float32)

        # High acceleration → high intensity/transients
        if self.n_audio_features >= 1:
            features[:, 0] = acc_norm * 0.7 + 0.2  # Centroid
        if self.n_audio_features >= 2:
            features[:, 1] = acc_norm  # Flux (most correlated)
        if self.n_audio_features >= 3:
            features[:, 2] = acc_norm * 0.8 + 0.1  # Energy
        if self.n_audio_features >= 4:
            features[:, 3] = 0.5 + 0.3 * np.sin(np.linspace(0, 8 * np.pi, n_samples))  # ZCR
        if self.n_audio_features >= 5:
            features[:, 4] = (acc_norm > 0.7).astype(float)  # Onsets at high accel
        if self.n_audio_features >= 6:
            features[:, 5] = acc_norm * 0.6 + 0.3  # Rolloff
        
        # Fill any additional features
        for i in range(6, self.n_audio_features):
            phase = 2 * np.pi * i / self.n_audio_features
            features[:, i] = 0.4 + 0.4 * np.sin(np.linspace(phase, phase + 6 * np.pi, n_samples))

        features = np.clip(features, 0, 1)

        return features

    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """
        Compute accelerations from velocities using finite differences.

        Args:
            velocities: Velocity vectors (n_samples, 2)

        Returns:
            Acceleration vectors (n_samples, 2)
        """
        n_samples = len(velocities)
        accelerations = np.zeros_like(velocities)

        if n_samples > 1:
            # Forward differences
            accelerations[:-1] = velocities[1:] - velocities[:-1]
            # Wrap around for last point
            accelerations[-1] = velocities[0] - velocities[-1]

        return accelerations

    def generate_mixed_curriculum(
        self,
        n_samples: int,
        orbit_names: Optional[List[str]] = None,
        correlation_types: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Union[str, int]]]]:
        """
        Generate a mixed curriculum of synthetic trajectories.

        Samples from multiple orbits with different correlation patterns.

        Args:
            n_samples: Total number of samples to generate
            orbit_names: List of orbit names to use (None = use all)
            correlation_types: List of correlation types (None = use all)

        Returns:
            Tuple of (audio_features, visual_params, metadata)
            - audio_features: shape (n_samples, n_audio_features)
            - visual_params: shape (n_samples, 2)
            - metadata: List of dicts with orbit and correlation info per sample
        """
        if orbit_names is None:
            orbit_names = list_preset_names()

        if correlation_types is None:
            correlation_types = ["velocity", "position", "acceleration"]

        # Calculate samples per orbit
        samples_per_orbit = n_samples // len(orbit_names)
        remainder = n_samples % len(orbit_names)

        all_audio_features = []
        all_visual_params = []
        all_metadata = []

        for i, orbit_name in enumerate(orbit_names):
            # Distribute remainder samples
            n_orbit_samples = samples_per_orbit + (1 if i < remainder else 0)

            if n_orbit_samples == 0:
                continue

            # Randomly select correlation type
            correlation = np.random.choice(correlation_types)

            # Generate trajectory
            audio_feat, visual_params = self.generate_synthetic_trajectory(
                orbit_name=orbit_name,
                n_samples=n_orbit_samples,
                audio_correlation=correlation,
            )

            all_audio_features.append(audio_feat)
            all_visual_params.append(visual_params)

            # Store metadata
            for j in range(n_orbit_samples):
                all_metadata.append(
                    {
                        "orbit": orbit_name,
                        "correlation": correlation,
                        "sample_idx": j,
                    }
                )

        # Concatenate all samples
        audio_features = np.concatenate(all_audio_features, axis=0)
        visual_params = np.concatenate(all_visual_params, axis=0)

        return audio_features, visual_params, all_metadata

    def generate_windowed_features(
        self,
        audio_features: np.ndarray,
        visual_params: np.ndarray,
        window_frames: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert frame-by-frame features to windowed format for model input.

        Args:
            audio_features: Frame-by-frame audio features (n_frames, n_features)
            visual_params: Frame-by-frame visual params (n_frames, 2)
            window_frames: Number of frames per window

        Returns:
            Tuple of (windowed_audio, windowed_visual)
            - windowed_audio: (n_windows, n_features * window_frames)
            - windowed_visual: (n_windows, 2) - uses center frame of window
        """
        n_frames = len(audio_features)

        # Create sliding windows
        windowed_audio = []
        windowed_visual = []

        for i in range(n_frames - window_frames + 1):
            # Extract window
            audio_window = audio_features[i : i + window_frames].flatten()
            # Use center frame for visual target
            center_idx = i + window_frames // 2
            visual_target = visual_params[center_idx]

            windowed_audio.append(audio_window)
            windowed_visual.append(visual_target)

        if len(windowed_audio) == 0:
            # Handle edge case: not enough frames
            padded_audio = np.pad(
                audio_features,
                ((0, window_frames - n_frames), (0, 0)),
                mode="edge",
            )
            windowed_audio = [padded_audio.flatten()]
            windowed_visual = [visual_params[n_frames // 2]]

        return np.array(windowed_audio, dtype=np.float32), np.array(
            windowed_visual, dtype=np.float32
        )


def create_synthetic_dataset(
    n_samples: int = 10000,
    window_frames: int = 10,
    n_audio_features: int = 6,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Union[str, int]]]]:
    """
    Convenience function to create a complete synthetic dataset.

    Args:
        n_samples: Number of trajectory samples
        window_frames: Window size for feature extraction
        n_audio_features: Number of audio features

    Returns:
        Tuple of (windowed_audio_features, visual_params, metadata)
    """
    engine = OrbitEngine(n_audio_features=n_audio_features)

    # Generate mixed curriculum
    audio_features, visual_params, metadata = engine.generate_mixed_curriculum(
        n_samples=n_samples
    )

    # Convert to windowed format
    windowed_audio, windowed_visual = engine.generate_windowed_features(
        audio_features, visual_params, window_frames=window_frames
    )

    return windowed_audio, windowed_visual, metadata
