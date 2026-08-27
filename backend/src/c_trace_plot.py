"""c-space trajectory visualization for training runs.

Plots the c(t) path the model produces for each training song, superimposed
on the Mandelbrot set. This is the diagnostic the user asked for: little
curly-cues over the set show exactly which regions are being explored, what
the local Julia sets look like, and — most importantly — whether and where
c gets stuck.

Default behavior: one plot per training run (and optionally per epoch),
saved to checkpoints/c_traces/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Mandelbrot escape-time grid, computed lazily once per process.
_MANDEL_CACHE: Optional[Dict[str, object]] = None

# Cardioid + period-2 bulb outline for a clean structural overlay.
def _cardioid_outline(n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Main cardioid boundary: c = e^{i t}/2 - e^{2 i t}/4."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    re = np.cos(t) / 2.0 - np.cos(2.0 * t) / 4.0
    im = np.sin(t) / 2.0 - np.sin(2.0 * t) / 4.0
    return re, im


def _bulb_outline(cx: float, cy: float, r: float, n: int = 360) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def _mandel_escape_grid(
    re_min: float = -2.2,
    re_max: float = 0.8,
    im_min: float = -1.3,
    im_max: float = 1.3,
    width: int = 900,
    height: int = 650,
    max_iter: int = 120,
) -> np.ndarray:
    """Escape-time field for background shading (cached)."""
    global _MANDEL_CACHE
    key = (re_min, re_max, im_min, im_max, width, height, max_iter)
    if _MANDEL_CACHE is not None and _MANDEL_CACHE.get("key") == key:
        return _MANDEL_CACHE["grid"]  # type: ignore[return-value]

    re = np.linspace(re_min, re_max, width)
    im = np.linspace(im_min, im_max, height)
    C = re[None, :] + 1j * im[:, None]
    Z = np.zeros_like(C)
    grid = np.full(C.shape, max_iter, dtype=np.float32)
    for i in range(max_iter):
        mask = np.abs(Z) <= 2.0
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        newly = mask & (np.abs(Z) > 2.0)
        grid[newly] = i
    _MANDEL_CACHE = {"key": key, "grid": grid}
    return grid


def plot_c_traces(
    traces: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "c(t) trajectories over the Mandelbrot set",
    max_points_per_trace: int = 4000,
    linewidth: float = 0.5,
    alpha: float = 0.55,
) -> Optional[Path]:
    """Plot per-song c(t) paths over the Mandelbrot set.

    Args:
        traces: mapping song name -> complex array of c values (time-ordered).
        out_path: destination PNG path.
        title: plot title.
        max_points_per_trace: downsample long traces to this many points.

    Returns:
        The path written, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping c-trace plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=130)

    # Background: escape-time shading (interior black, exterior graded).
    grid = _mandel_escape_grid()
    ax.imshow(
        grid,
        extent=(-2.2, 0.8, -1.3, 1.3),
        origin="lower",
        cmap="magma",
        alpha=0.85,
        aspect="auto",
        interpolation="bilinear",
    )

    # Structural outlines: main cardioid + period-2 bulb.
    re_c, im_c = _cardioid_outline()
    ax.plot(re_c, im_c, color="cyan", lw=0.9, alpha=0.8, label="main cardioid")
    re_b, im_b = _bulb_outline(-1.0, 0.0, 0.25)
    ax.plot(re_b, im_b, color="lime", lw=0.9, alpha=0.8, label="period-2 bulb")

    # Traces: one color per song, time-gradient within a trace.
    cmap = plt.get_cmap("tab10")
    for idx, (name, cvals) in enumerate(sorted(traces.items())):
        c = np.asarray(cvals, dtype=np.complex128)
        if c.size < 2:
            continue
        if c.size > max_points_per_trace:
            sel = np.linspace(0, c.size - 1, max_points_per_trace).astype(int)
            c = c[sel]
        color = cmap(idx % 10)
        ax.plot(
            c.real,
            c.imag,
            color=color,
            lw=linewidth,
            alpha=alpha,
            label=f"{name} ({c.size} pts)",
        )
        # Mark start (green) and end (red) of each trace.
        ax.plot(c.real[0], c.imag[0], marker="o", ms=4, color="white", mec=color, mew=1.5)
        ax.plot(c.real[-1], c.imag[-1], marker="s", ms=4, color=color)

    ax.set_xlim(-2.2, 0.8)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.grid(True, lw=0.2, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("c-trace plot written: %s", out_path)
    return out_path


def collect_c_traces(
    model,
    feature_extractor,
    dataset_files: Sequence[Path],
    window_frames: int,
    max_frames_per_file: int = 6000,
) -> Dict[str, np.ndarray]:
    """Run the model over each training file and record its c(t) path.

    Uses the same momentum-mode controller replay the trainer supervises
    through, so the plot shows exactly the physics the runtime executes.

    Args:
        model: the trained AudioToControlModel (eval mode expected).
        feature_extractor: extractor with normalize_features().
        dataset_files: audio file paths (order matches dataset indices).
        window_frames: frames per feature window.
        max_frames_per_file: cap on recorded frames per file.

    Returns:
        mapping song name -> complex ndarray of c values.
    """
    import torch

    from .cspace_proxies import orbit_controller_momentum_sequence

    was_training = model.training
    model.eval()
    traces: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for idx, audio_file in enumerate(dataset_files):
            try:
                features = np.asarray(
                    feature_extractor.extract_windowed_features(
                        _load_audio(audio_file), window_frames
                    ),
                    dtype=np.float64,
                )
            except Exception as exc:  # noqa: BLE001 - skip unreadable files
                logger.warning("c-trace: failed to extract %s: %s", audio_file.name, exc)
                continue
            if features.ndim != 2 or features.shape[0] == 0:
                continue
            if features.shape[0] > max_frames_per_file:
                features = features[:max_frames_per_file]

            rows = [feature_extractor.normalize_features(r) for r in features.tolist()]
            # Model expects (batch, feature_dim) rows — one window per row.
            x = torch.tensor(np.vstack(rows), dtype=torch.float32)
            if next(model.parameters()).device.type != "cpu":
                x = x.to(next(model.parameters()).device)

            out = model(x)
            parsed = model.parse_output(out)
            n = x.shape[0]
            seg = torch.zeros(n, dtype=torch.int64)

            c = orbit_controller_momentum_sequence(
                s_target=parsed["s_target"],
                alpha=parsed["alpha"],
                omega=1.0,
                band_gates=parsed["band_gates"],
                segment_ids=seg,
                drag=0.90,
            )
            traces[audio_file.stem] = c.detach().cpu().numpy()

    if was_training:
        model.train()
    return traces


def _load_audio(audio_file: Path) -> list[float]:
    """Load an audio file as mono 48 kHz float samples (list for the binding)."""
    import librosa

    audio, _ = librosa.load(str(audio_file), sr=48000, mono=True)
    return audio.astype(np.float32).tolist()
