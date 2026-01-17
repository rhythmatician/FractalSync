#!/usr/bin/env python3
"""
Supervised data collection tool for FractalSync.

This tool allows users to manually control the Julia parameter c while audio
plays, collecting trajectory data for supervised learning. The collected data
includes c positions, timestamps, audio features, and boundary crossing events.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import librosa

from src.audio_features import (
    AudioFeatureExtractor,
    load_audio_file,
    detect_musical_transitions,
    compute_transition_score,
)
from src.visual_metrics import VisualMetrics
from src.mandelbrot_orbits import (
    compute_boundary_distance,
    detect_boundary_crossing,
    compute_crossing_score,
)


class SupervisedDataCollector:
    """Interactive tool for collecting supervised trajectory data."""

    def __init__(
        self,
        audio_path: str,
        output_dir: str = "data/supervised",
        window_size: int = 800,
        render_size: int = 512,
    ):
        """
        Initialize data collector.

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save collected data
            window_size: Size of the display window
            render_size: Size of the Julia set render
        """
        self.audio_path = Path(audio_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
        self.render_size = render_size

        # Initialize components
        self.feature_extractor = AudioFeatureExtractor(window_frames=10)
        self.visual_metrics = VisualMetrics()

        # Load audio
        print(f"Loading audio: {self.audio_path}")
        self.audio, self.sr = load_audio_file(str(self.audio_path))
        print(f"✓ Loaded audio: {len(self.audio)} samples, {self.sr} Hz")

        # Extract audio features
        print("Extracting audio features...")
        self.features = self.feature_extractor.extract_features(self.audio)
        self.n_features, self.n_frames = self.features.shape
        print(f"✓ Extracted {self.n_frames} frames")

        # Compute frame timing
        self.hop_length = self.feature_extractor.hop_length
        self.frame_duration = self.hop_length / self.sr
        self.total_duration = len(self.audio) / self.sr

        # Detect transitions
        print("Detecting musical transitions...")
        spectral_flux = self.features[1, :]
        onset_strength = self.features[4, :]
        rms_energy = self.features[2, :]
        self.transitions = detect_musical_transitions(
            spectral_flux, onset_strength, rms_energy
        )
        print(f"✓ Detected {np.sum(self.transitions)} transitions")

        # State
        self.current_c = complex(0.0, 0.0)  # Julia parameter
        self.current_frame = 0
        self.is_recording = False
        self.is_playing = False
        self.start_time = None

        # Recorded data
        self.trajectory: List[Dict[str, Any]] = []

        # UI state
        self.zoom = 1.0
        self.mouse_pos = (0, 0)
        self.show_help = True

    def _render_julia_set(
        self, c: complex, zoom: float = 1.0
    ) -> np.ndarray:
        """Render Julia set with current parameters."""
        return self.visual_metrics.render_julia_set(
            seed_real=c.real,
            seed_imag=c.imag,
            width=self.render_size,
            height=self.render_size,
            zoom=zoom,
        )

    def _draw_ui(
        self, julia_image: np.ndarray, frame_idx: int
    ) -> np.ndarray:
        """Draw UI overlays on the image."""
        # Create display canvas
        canvas = np.zeros(
            (self.window_size, self.window_size, 3), dtype=np.uint8
        )

        # Resize Julia set to fit canvas
        display_size = min(self.window_size - 100, self.render_size)
        julia_display = cv2.resize(julia_image, (display_size, display_size))

        # Center the Julia set
        offset_x = (self.window_size - display_size) // 2
        offset_y = 50
        canvas[
            offset_y : offset_y + display_size,
            offset_x : offset_x + display_size,
        ] = julia_display

        # Draw info overlay
        info_y = 20
        line_height = 25

        # Title
        cv2.putText(
            canvas,
            "FractalSync - Supervised Data Collection",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Status info
        info_y += line_height + 10
        status_color = (0, 255, 0) if self.is_recording else (100, 100, 100)
        status_text = "RECORDING" if self.is_recording else "PAUSED"
        cv2.putText(
            canvas,
            f"Status: {status_text}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            1,
        )

        # Frame info
        info_y = self.window_size - 120
        cv2.putText(
            canvas,
            f"Frame: {frame_idx}/{self.n_frames}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        info_y += line_height
        time_elapsed = frame_idx * self.frame_duration
        cv2.putText(
            canvas,
            f"Time: {time_elapsed:.2f}s / {self.total_duration:.2f}s",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Julia parameter
        info_y += line_height
        cv2.putText(
            canvas,
            f"c: {self.current_c.real:.3f} + {self.current_c.imag:.3f}i",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Boundary distance
        info_y += line_height
        boundary_dist = compute_boundary_distance(
            self.current_c.real, self.current_c.imag
        )
        crossing_score = compute_crossing_score(
            self.current_c.real, self.current_c.imag
        )
        cv2.putText(
            canvas,
            f"Boundary: {boundary_dist:.3f} | Score: {crossing_score:.3f}",
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Transition indicator
        if frame_idx < len(self.transitions) and self.transitions[frame_idx] > 0.5:
            cv2.circle(
                canvas,
                (self.window_size - 50, 30),
                20,
                (0, 255, 255),
                -1,
            )
            cv2.putText(
                canvas,
                "TRANSITION",
                (self.window_size - 120, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # Help text
        if self.show_help:
            help_y = self.window_size - 200
            help_texts = [
                "Controls:",
                "  Mouse: Move c parameter",
                "  SPACE: Play/Pause",
                "  R: Toggle recording",
                "  S: Save data",
                "  H: Toggle help",
                "  Q/ESC: Quit",
            ]
            for text in help_texts:
                cv2.putText(
                    canvas,
                    text,
                    (self.window_size - 200, help_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
                help_y += 20

        # Recorded frames counter
        cv2.putText(
            canvas,
            f"Recorded: {len(self.trajectory)} frames",
            (10, self.window_size - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0) if self.is_recording else (100, 100, 100),
            1,
        )

        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

            # Map mouse position to complex plane
            # Center of window maps to origin
            center_x = self.window_size // 2
            center_y = self.window_size // 2

            # Scale to [-2, 2] range
            scale = 2.0 / (self.window_size // 2)
            real = (x - center_x) * scale
            imag = (y - center_y) * scale

            self.current_c = complex(real, imag)

    def _record_frame(self, frame_idx: int):
        """Record current frame data."""
        if frame_idx >= self.n_frames:
            return

        # Get audio features for this frame
        frame_features = self.features[:, frame_idx]

        # Compute transition score
        spectral_flux = frame_features[1]
        onset_strength = frame_features[4]
        rms_energy = frame_features[2]

        # Compute RMS change
        rms_change = 0.0
        if frame_idx > 0:
            rms_change = abs(rms_energy - self.features[2, frame_idx - 1])

        transition_score = compute_transition_score(
            float(spectral_flux), float(onset_strength), float(rms_change)
        )

        # Check boundary crossing
        boundary_crossed = False
        if len(self.trajectory) > 0:
            prev_c = self.trajectory[-1]["c"]
            boundary_crossed = detect_boundary_crossing(
                prev_c["real"],
                prev_c["imag"],
                self.current_c.real,
                self.current_c.imag,
            )

        # Record frame data
        frame_data = {
            "frame": frame_idx,
            "timestamp": frame_idx * self.frame_duration,
            "c": {"real": self.current_c.real, "imag": self.current_c.imag},
            "zoom": self.zoom,
            "audio_features": {
                "spectral_centroid": float(frame_features[0]),
                "spectral_flux": float(frame_features[1]),
                "rms_energy": float(frame_features[2]),
                "zero_crossing_rate": float(frame_features[3]),
                "onset_strength": float(frame_features[4]),
                "spectral_rolloff": float(frame_features[5]),
            },
            "transition_score": float(transition_score),
            "boundary_crossed": bool(boundary_crossed),
            "boundary_distance": float(
                compute_boundary_distance(self.current_c.real, self.current_c.imag)
            ),
            "crossing_score": float(
                compute_crossing_score(self.current_c.real, self.current_c.imag)
            ),
        }

        self.trajectory.append(frame_data)

    def save_data(self) -> str:
        """Save collected trajectory data to JSON file."""
        if len(self.trajectory) == 0:
            print("No data to save!")
            return None

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = self.audio_path.stem
        filename = f"{audio_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Prepare metadata
        data = {
            "metadata": {
                "audio_file": str(self.audio_path),
                "audio_duration": self.total_duration,
                "sample_rate": self.sr,
                "hop_length": self.hop_length,
                "total_frames": self.n_frames,
                "recorded_frames": len(self.trajectory),
                "timestamp": timestamp,
            },
            "trajectory": self.trajectory,
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        duration = self.trajectory[-1]["timestamp"] - self.trajectory[0]["timestamp"]
        print(f"\n✓ Saved to: {filepath}")
        print(f"  Frames: {len(self.trajectory)}")
        print(f"  Duration: {duration:.1f}s")

        return str(filepath)

    def run(self):
        """Run the interactive data collection loop."""
        print("\n" + "=" * 60)
        print("FractalSync - Supervised Data Collection")
        print("=" * 60)
        print("\nInitializing GPU Julia renderer...")
        print("✓ Ready to collect data!")
        print("\nControls:")
        print("  Mouse: Move c parameter")
        print("  SPACE: Play/Pause audio playback")
        print("  R: Toggle recording")
        print("  S: Save collected data")
        print("  H: Toggle help overlay")
        print("  Q/ESC: Quit\n")

        # Create window
        window_name = "FractalSync Data Collection"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        # Main loop
        clock = cv2.getTickCount()
        fps = 30  # Target FPS

        try:
            while True:
                # Render Julia set
                julia_image = self._render_julia_set(self.current_c, self.zoom)

                # Draw UI
                display = self._draw_ui(julia_image, self.current_frame)

                # Show frame
                cv2.imshow(window_name, display)

                # Handle keyboard input
                key = cv2.waitKey(1000 // fps) & 0xFF

                if key == ord("q") or key == 27:  # Q or ESC
                    break
                elif key == ord(" "):  # SPACE
                    self.is_playing = not self.is_playing
                    if self.is_playing and self.start_time is None:
                        self.start_time = time.time()
                    print(f"Playback: {'Playing' if self.is_playing else 'Paused'}")
                elif key == ord("r"):  # R
                    self.is_recording = not self.is_recording
                    print(
                        f"Recording: {'ON' if self.is_recording else 'OFF'}"
                    )
                elif key == ord("s"):  # S
                    self.save_data()
                elif key == ord("h"):  # H
                    self.show_help = not self.show_help

                # Update frame if playing
                if self.is_playing:
                    self.current_frame += 1
                    if self.current_frame >= self.n_frames:
                        print("\n✓ Reached end of audio")
                        self.is_playing = False
                        self.current_frame = self.n_frames - 1

                    # Record frame if recording
                    if self.is_recording:
                        self._record_frame(self.current_frame)

                # Maintain FPS
                elapsed = (cv2.getTickCount() - clock) / cv2.getTickFrequency()
                if elapsed < 1.0 / fps:
                    time.sleep(1.0 / fps - elapsed)
                clock = cv2.getTickCount()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            # Cleanup
            cv2.destroyAllWindows()

            # Auto-save if data was recorded
            if len(self.trajectory) > 0:
                print("\nSaving recorded data...")
                self.save_data()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect supervised trajectory data for FractalSync"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (MP3, WAV, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/supervised",
        help="Output directory for collected data (default: data/supervised)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=800,
        help="Display window size (default: 800)",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Julia set render size (default: 512)",
    )

    args = parser.parse_args()

    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Run collector
    collector = SupervisedDataCollector(
        audio_path=str(audio_path),
        output_dir=args.output_dir,
        window_size=args.window_size,
        render_size=args.render_size,
    )

    collector.run()

    return 0


if __name__ == "__main__":
    exit(main())
