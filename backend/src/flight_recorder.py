"""
Flight recorder for recording per-timestep controller state and proxy frames.
Records newline-delimited JSON (`records.ndjson`) and optional PNG frames.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class FlightRecorder:
    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: str = "logs/flight_recorder",
        save_images: bool = True,
    ):
        self.run_id = run_id or f"run_{int(__import__('time').time())}"
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        self.proxy_dir = self.run_dir / "proxy_frames"
        self.save_images = save_images

        os.makedirs(self.proxy_dir, exist_ok=True)
        self.records_path = self.run_dir / "records.ndjson"
        self._file = open(self.records_path, "a", encoding="utf-8")

    def start_run(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Write metadata record as first line."""
        meta = metadata or {}
        meta_record = {"_meta": True, "metadata": meta}
        self._file.write(json.dumps(meta_record) + "\n")
        self._file.flush()

    def record_step(
        self,
        t: int,
        c: Any,
        controller: Dict[str, Any],
        h: float,
        band_energies: Any,
        audio_features: Any,
        proxy_frame: Optional[np.ndarray] = None,
        delta_v: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Record a single timestep.

        Args:
            t: step index
            c: complex seed pair or complex number
            controller: dict of controller state values
            h: transient strength
            band_energies: list of band energies
            audio_features: flattened feature vector
            proxy_frame: HxW or HxWx3 uint8 numpy array (optional)
            delta_v: computed ΔV (optional)
            notes: optional debug string
        """
        if isinstance(c, complex):
            c_pair = [float(c.real), float(c.imag)]
        elif isinstance(c, (list, tuple)) and len(c) == 2:
            c_pair = [float(c[0]), float(c[1])]
        else:
            c_pair = c

        rec: Dict[str, Any] = {
            "t": int(t),
            "c": c_pair,
            "controller": controller,
            "h": float(h) if h is not None else None,
            "band_energies": (
                list(map(float, band_energies)) if band_energies is not None else None
            ),
            "audio_features": (
                list(map(float, audio_features)) if audio_features is not None else None
            ),
            "deltaV": float(delta_v) if delta_v is not None else None,
            "notes": notes,
        }

        # Optionally save proxy frame as PNG and reference path
        if proxy_frame is not None and self.save_images:
            try:
                # Ensure grayscale or RGB uint8
                if proxy_frame.dtype != np.uint8:
                    arr = (
                        (proxy_frame * 255).astype(np.uint8)
                        if proxy_frame.max() <= 1.0
                        else proxy_frame.astype(np.uint8)
                    )
                else:
                    arr = proxy_frame
                # If grayscale, cv2.imwrite will handle it; if RGB, convert from RGB to BGR
                img_path = self.proxy_dir / f"{t:06d}.png"
                if arr.ndim == 3 and arr.shape[2] == 3:
                    cv2.imwrite(str(img_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(str(img_path), arr)
                rec["proxy_frame_path"] = str(img_path.relative_to(self.run_dir))
            except (cv2.error, OSError, ValueError) as e:  # pragma: no cover - best-effort write
                rec["proxy_frame_write_error"] = str(e)

        # Write ndjson
        self._file.write(json.dumps(rec) + "\n")
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass


# Lightweight helper to compute ΔV between successive frames (grayscale arrays)
def compute_delta_v(current: np.ndarray, previous: Optional[np.ndarray]) -> float:
    if previous is None:
        return 0.0
    # Ensure same shape
    if current.shape != previous.shape:
        import cv2 as _cv2

        prev_resized = _cv2.resize(previous, (current.shape[1], current.shape[0]))
    else:
        prev_resized = previous
    diff = np.abs(current.astype(np.float32) - prev_resized.astype(np.float32)) / 255.0
    return float(np.mean(diff))
