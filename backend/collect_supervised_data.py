"""
Interactive Mandelbrot explorer for collecting supervised training data.

Usage:
    python collect_supervised_data.py

Controls:
    - Move cursor over Mandelbrot set to select c values
    - SPACE: Start/pause recording
    - R: Reset recording
    - S: Save recording
    - Q: Quit
    - Scroll: Zoom in/out
    - Drag: Pan around
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
import sounddevice as sd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_features import AudioFeatureExtractor


class MandelbrotRecorder:
    """Interactive Mandelbrot explorer for supervised data collection."""

    def __init__(
        self,
        audio_path: str,
        sr: int = 22050,
        hop_length: int = 512,
        n_fft: int = 2048,
        frame_interval: int = 10,  # Record every N frames
    ):
        """
        Initialize recorder.

        Args:
            audio_path: Path to audio file
            sr: Sample rate
            hop_length: Hop length for feature extraction
            n_fft: FFT size
            frame_interval: Record every N frames (10 = ~0.2 sec @ 22050 Hz)
        """
        self.audio_path = audio_path
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.frame_interval = frame_interval

        # Load audio and extract features
        print(f"Loading audio: {audio_path}")
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = len(self.y) / sr
        print(f"Duration: {self.duration:.1f}s")

        print("Extracting audio features...")
        self.feature_extractor = AudioFeatureExtractor(
            sr=sr, hop_length=hop_length, n_fft=n_fft
        )
        self.features = self.feature_extractor.extract_features(self.y)
        self.n_frames = self.features.shape[0]
        print(f"Extracted {self.n_frames} frames")

        # Recording state
        self.recording = False
        self.recorded_frames: List[dict] = []
        self.current_frame_idx = 0

        # Audio playback state
        self.playback_thread = None
        self.stream = None
        self.is_playing = False
        self.playback_start_time = 0.0
        self.playback_position = 0.0

        # Mandelbrot view state
        self.xmin, self.xmax = -2.5, 1.0
        self.ymin, self.ymax = -1.25, 1.25
        self.zoom_history = []

        # UI state
        self.fig = None
        self.ax = None
        self.mandelbrot_image = None
        self.cursor_marker = None
        self.text_info = None

    def mandelbrot(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        width: int = 800,
        height: int = 600,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Render Mandelbrot set."""
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        Z = np.zeros_like(C, dtype=complex)
        M = np.zeros_like(C, dtype=int)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            M[mask] = i

        return M

    def pixel_to_complex(self, x_pixel: float, y_pixel: float) -> Tuple[float, float]:
        """Convert matplotlib pixel coordinates to complex plane."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        c_real = xlim[0] + (
            x_pixel - self.ax.get_position().x0 * self.fig.get_figwidth()
        ) / (self.ax.get_position().width * self.fig.get_figwidth()) * (
            xlim[1] - xlim[0]
        )

        c_imag = ylim[0] + (
            1.0
            - (y_pixel - self.ax.get_position().y0 * self.fig.get_figheight())
            / (self.ax.get_position().height * self.fig.get_figheight())
        ) * (ylim[1] - ylim[0])

        return c_real, c_imag

    def on_move(self, event):
        """Handle mouse movement."""
        if event.inaxes != self.ax:
            return

        if self.cursor_marker is not None:
            self.cursor_marker.remove()

        # Draw cursor
        self.cursor_marker = self.ax.plot(
            event.xdata, event.ydata, "r+", markersize=15, markeredgewidth=2
        )[0]

        # Get complex coordinates
        c_real, c_imag = self.pixel_to_complex(event.xdata, event.ydata)

        # Update info text
        if self.text_info is not None:
            self.text_info.remove()

        status = "RECORDING" if self.recording else "Ready to record"
        self.text_info = self.ax.text(
            0.02,
            0.98,
            f"c = {c_real:.4f} + {c_imag:.4f}i\nFrame: {self.current_frame_idx}/{self.n_frames}\n{status}",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        self.fig.canvas.draw_idle()

    def start_playback(self):
        """Start audio playback."""
        if self.is_playing:
            return

        try:
            self.is_playing = True
            self.playback_start_time = time.time()

            def play_audio():
                try:
                    sd.play(self.y, samplerate=self.sr)
                    sd.wait()
                except Exception as e:
                    print(f"✗ Playback error: {e}")
                finally:
                    self.is_playing = False

            self.playback_thread = threading.Thread(target=play_audio, daemon=True)
            self.playback_thread.start()
            print("♪ Audio playing...")
        except Exception as e:
            print(f"✗ Could not start playback: {e}")
            self.is_playing = False

    def stop_playback(self):
        """Stop audio playback."""
        if self.is_playing:
            try:
                sd.stop()
                self.is_playing = False
                print("✗ Audio stopped")
            except Exception as e:
                print(f"✗ Could not stop playback: {e}")

    def on_scroll(self, event):
        """Handle zoom."""
        if event.inaxes != self.ax:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        if event.button == "up":
            scale_factor = 0.5
        elif event.button == "down":
            scale_factor = 2.0
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.fig.canvas.draw_idle()

    def on_press(self, event):
        """Handle key press."""
        if event.key == " ":  # Space
            self.recording = not self.recording
            if self.recording:
                print("✓ Recording started")
                self.current_frame_idx = 0
                self.recorded_frames = []
                self.start_playback()
            else:
                print("✗ Recording paused")
                self.stop_playback()

        elif event.key == "r":  # Reset
            print("⟲ Reset recording")
            self.recording = False
            self.recorded_frames = []
            self.current_frame_idx = 0
            self.stop_playback()

        elif event.key == "s":  # Save
            self.save()
            self.stop_playback()

        elif event.key == "q":  # Quit
            self.stop_playback()
            plt.close("all")

    def update_animation(self, frame_num):
        """Update for animation loop."""
        if self.recording and self.current_frame_idx < self.n_frames:
            # Get current cursor position from last event
            if hasattr(self, "_last_xdata") and hasattr(self, "_last_ydata"):
                c_real, c_imag = self.pixel_to_complex(
                    self._last_xdata, self._last_ydata
                )

                # Record this frame
                self.recorded_frames.append(
                    {
                        "frame_idx": self.current_frame_idx,
                        "audio_features": self.features[
                            self.current_frame_idx
                        ].tolist(),
                        "c_real": float(c_real),
                        "c_imag": float(c_imag),
                    }
                )

            self.current_frame_idx += self.frame_interval
            if self.current_frame_idx >= self.n_frames:
                self.recording = False
                print(f"✓ Recording complete: {len(self.recorded_frames)} frames")

    def save(self):
        """Save recorded data."""
        if not self.recorded_frames:
            print("✗ No data to save")
            return

        # Create output directory
        output_dir = Path("data/supervised")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        song_name = Path(self.audio_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{song_name}_{timestamp}.json"

        # Save data
        data = {
            "song": song_name,
            "audio_path": str(self.audio_path),
            "sr": self.sr,
            "hop_length": self.hop_length,
            "n_fft": self.n_fft,
            "frame_interval": self.frame_interval,
            "total_frames_recorded": len(self.recorded_frames),
            "timestamp": datetime.now().isoformat(),
            "frames": self.recorded_frames,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved to: {output_file}")
        print(f"  Frames: {len(self.recorded_frames)}")
        print(
            f"  Duration: {len(self.recorded_frames) * self.frame_interval * self.hop_length / self.sr:.1f}s"
        )

    def run(self):
        """Run interactive explorer."""
        self.fig, self.ax = plt.subplots(figsize=(12, 9))

        # Initial Mandelbrot render
        mandel = self.mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax)
        self.mandelbrot_image = self.ax.imshow(
            mandel,
            extent=[self.xmin, self.xmax, self.ymin, self.ymax],
            cmap="hot",
            origin="lower",
            interpolation="bilinear",
        )

        self.ax.set_xlabel("Real")
        self.ax.set_ylabel("Imaginary")
        self.ax.set_title(
            "Mandelbrot Set Explorer - Draw your musical journey!\nSPACE: Record/Pause | R: Reset | S: Save | Q: Quit"
        )

        # Connect events
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)

        # Store cursor position for animation
        def store_cursor(event):
            if event.inaxes == self.ax:
                self._last_xdata = event.xdata
                self._last_ydata = event.ydata

        self.fig.canvas.mpl_connect("motion_notify_event", store_cursor)

        # Animation loop to sync with audio frames
        ani = FuncAnimation(
            self.fig,
            self.update_animation,
            frames=self.n_frames // self.frame_interval,
            interval=100,
            repeat=False,
        )

        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    # Find audio files
    audio_dir = Path("data/audio")
    audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))

    if not audio_files:
        print(f"✗ No audio files found in {audio_dir}")
        return

    print("Available audio files:")
    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {f.name}")

    choice = input(f"\nSelect file (1-{len(audio_files)}): ").strip()
    try:
        audio_path = audio_files[int(choice) - 1]
    except (ValueError, IndexError):
        print("✗ Invalid selection")
        return

    # Create recorder and run
    recorder = MandelbrotRecorder(str(audio_path))
    recorder.run()


if __name__ == "__main__":
    main()
