"""Evaluate a candidate causal onset-density summary offline on five local songs."""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import ceil, cos, isfinite, log2, pi, sin
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import runtime_core as rust
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
SONGS = ("Eulogy", "RightInTwo", "Stinkfist", "TheGrudge", "ThirdEye")
RATE, HOP, FFT = 48_000, 1024, 4096
WINDOW, REFRACTORY, HORIZON = 94, 4, 47
FLUX_RISE_FLOOR, ENERGY_FLOOR = 0.15, 0.02
CAPACITY = ceil(WINDOW / REFRACTORY)
WIDTH = 6
FloatArray = NDArray[np.float64]
BASE_NAMES = ("rms", "flux", "centroid", "zcr", "rolloff")


@dataclass
class Density:
    epoch: int | None = None
    previous_sample: int | None = None
    events: deque[int] = field(default_factory=lambda: deque(maxlen=WINDOW))
    in_peak: bool = False
    refractory: int = 0
    previous_flux: float | None = None

    def observe(self, epoch: int, sample: int, flux: float, rms: float) -> tuple[int, float, bool]:
        contiguous = self.previous_sample is not None and sample - self.previous_sample == HOP
        usable = isfinite(flux) and isfinite(rms)
        if self.epoch != epoch or not contiguous or not usable:
            self.epoch = epoch
            self.events.clear()
            self.in_peak = False
            self.refractory = 0
            self.previous_flux = None
        self.previous_sample = sample
        if not usable or sample < FFT:
            return 0, 0.0, False
        active = (self.previous_flux is not None
                  and flux - self.previous_flux >= FLUX_RISE_FLOOR and rms >= ENERGY_FLOOR)
        # Treat the initial zero-padded FFT frames as warmup, not as peaks.
        event = int(active and not self.in_peak and self.refractory == 0)
        self.previous_flux = flux
        self.in_peak = active
        self.refractory = REFRACTORY - 1 if event else max(0, self.refractory - 1)
        self.events.append(event)
        value = sum(self.events) / CAPACITY
        return event, value, len(self.events) == WINDOW


def timebase_ticks(pcm: NDArray[np.float32], rate: int, block: int) -> list[dict]:
    tb = rust.AnalysisTimebase()
    ticks: list[dict] = []
    for start in range(0, len(pcm), block):
        ticks.extend(tb.ingest(pcm[start:start + block], rate, start))
    # Match a running stream. End-of-file resampler flush cannot affect emitted rows.
    return ticks


def channels(tick: dict) -> tuple[float, float, float]:
    data = dict(rust.cycle_observation_channels_from_tick(tick))
    onset, flux, rms = data["onset"], data["flux"], data["energy"]
    return onset, flux, rms


@dataclass
class Track:
    name: str
    digest: str
    native_rate: int
    indices: NDArray[np.int64]
    base: FloatArray
    density: FloatArray
    regime: FloatArray
    labels: FloatArray
    event_count: int
    first_valid_seconds: float


def extract(path: Path, seconds: int) -> Track:
    with sf.SoundFile(str(path)) as audio:
        rate = audio.samplerate
        stereo = cast(NDArray[np.float32], audio.read(rate * seconds, dtype="float32", always_2d=True))
    pcm = stereo.mean(axis=1, dtype=np.float32)
    ticks = timebase_ticks(pcm, rate, rate)
    bank = rust.CycleBank()
    detector = Density()
    base: list[list[float]] = []
    events: list[int] = []
    densities: list[float] = []
    regimes: list[int] = []
    indices: list[int] = []
    valid_rows: list[int] = []
    regime = 0
    for tick in ticks:
        onset, flux, rms = channels(tick)
        assert onset == flux
        frame = tick["features"][-WIDTH:]
        assert len(frame) == WIDTH and frame[1] == flux and frame[2] == rms
        sample, epoch = int(tick["sampleIndex"]), int(tick["streamEpoch"])
        assert not indices or sample == indices[-1] + HOP
        assert not indices or epoch == int(ticks[0]["streamEpoch"])
        event, density, valid = detector.observe(epoch, sample, onset, rms)
        count = round(density * CAPACITY)
        if valid:
            if count >= 8:
                regime = 2
            elif count <= 2:
                regime = 0
            elif (regime == 0 and count >= 4) or (regime == 2 and count <= 6):
                regime = 1
            valid_rows.append(len(base))
        row = [rms, flux, frame[0], frame[3], frame[5]]
        modes = sorted(bank.observe_tick(tick), key=lambda m: m.frequency_hz)
        for slot in range(8):
            if slot < len(modes):
                mode = modes[slot]
                row.extend((cos(mode.phase), sin(mode.phase), log2(mode.frequency_hz),
                            mode.strength, mode.confidence, float(mode.missing_observations), 1.0))
            else:
                row.extend((0.0,) * 7)
        base.append(row)
        events.append(event)
        densities.append(density)
        regimes.append(regime)
        indices.append(sample)
    assert len(valid_rows) > HORIZON
    e = np.concatenate(([0.0], np.cumsum(np.asarray(events, dtype=float))))
    ix = np.asarray(valid_rows[:-HORIZON])
    y = (e[ix + HORIZON + 1] - e[ix + 1]) * (RATE / (HORIZON * HOP))
    with path.open("rb") as fd:
        hash_state = sha256()
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            hash_state.update(block)
        digest = hash_state.hexdigest()
    return Track(path.stem, digest, rate, np.asarray(indices)[ix], np.asarray(base)[ix],
                 np.asarray(densities)[ix], np.asarray(regimes, dtype=float)[ix], y,
                 sum(events), indices[ix[0]] / RATE)


def history_matrix(base: FloatArray) -> FloatArray:
    """Trailing time-scale summaries of the already exposed flux/RMS."""
    values = base[:, :2]
    integral = np.vstack((np.zeros((1, 2)), np.cumsum(values, axis=0)))
    power = np.vstack((np.zeros((1, 2)), np.cumsum(values**2, axis=0)))
    rows = np.arange(1, len(base) + 1)
    parts = [base]
    for width in (5, 24, WINDOW):
        start = np.maximum(0, rows - width)
        parts.append((integral[rows] - integral[start]) / (rows - start)[:, None])
        parts.append(np.sqrt(np.maximum(0.0, (power[rows] - power[start]) / (rows - start)[:, None])))
    return np.concatenate(parts, axis=1)


def design(track: Track, configuration: str) -> FloatArray:
    base = history_matrix(track.base) if configuration.startswith("history") else track.base
    if configuration.endswith("+density"):
        return np.column_stack((base, track.density))
    if configuration.endswith("+regime"):
        return np.column_stack((base, track.regime == 1, track.regime == 2))
    return base


def fit_predict(train: FloatArray, y: FloatArray, test: FloatArray) -> FloatArray:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-10] = 1
    x = (train - mean) / scale
    ridge = np.eye(x.shape[1]) * len(x) * 0.05
    average = float(y.mean())
    weights = np.linalg.solve(x.T @ x + ridge, x.T @ (y - average))
    return cast(FloatArray, (test - mean) / scale @ weights + average)


def held_out(tracks: Sequence[Track]) -> list[dict]:
    rows = []
    for track in tracks:
        others = [item for item in tracks if item is not track]
        # A 2 s common analysis warmup makes every history summary complete.
        eligible = slice(WINDOW - 1, None)
        y = track.labels[eligible]
        train_y = np.concatenate([item.labels[eligible] for item in others])
        variance = float(np.mean((y - np.mean(train_y))**2))
        result: dict = {"name": track.name, "evaluation_ticks": len(y)}
        persistence = track.density[eligible] * CAPACITY * RATE / (WINDOW * HOP)
        result["persistence_rmse_events_per_second"] = float(np.mean((y - persistence)**2)**0.5)
        for configuration in ("instant", "instant+density", "instant+regime",
                              "history", "history+density", "history+regime"):
            train = np.concatenate([design(item, configuration)[eligible] for item in others])
            test = design(track, configuration)[eligible]
            guess = np.maximum(0, fit_predict(train, train_y, test))
            error = float(np.mean((y - guess)**2))
            result[configuration] = {"rmse_events_per_second": error**0.5,
                                     "skill_vs_training_mean": 1 - error / variance}
        train = np.concatenate([design(item, "history")[eligible] for item in others])
        test = design(track, "history")[eligible]
        d = track.density[eligible]
        guess = fit_predict(train, np.concatenate([item.density[eligible] for item in others]), test)
        result["density_reconstruction_pearson"] = float(np.corrcoef(guess, d)[0, 1])
        rows.append(result)
    return rows


def self_check() -> dict:
    pcm = np.zeros(RATE * 5, dtype=np.float32)
    for start in range(RATE // 2, RATE * 3, RATE // 4):
        pcm[start:start + 32] = [(0.75 * sin(pi * i / 32)) for i in range(32)]
    left = timebase_ticks(pcm, RATE, 997)
    right = timebase_ticks(pcm, RATE, 8_193)
    assert left == right
    future_cut = 100 * HOP
    changed = pcm.copy()
    changed[future_cut:] = 0
    prefix = [tick for tick in left if tick["sampleIndex"] <= future_cut]
    assert prefix == [tick for tick in timebase_ticks(changed, RATE, 997)
                      if tick["sampleIndex"] <= future_cut]
    detector = Density()
    samples = [(int(tick["sampleIndex"]), detector.observe(int(tick["streamEpoch"]),
                int(tick["sampleIndex"]), *channels(tick)[1:])) for tick in left]
    first_valid = next(sample for sample, (_, _, valid) in samples if valid)
    assert first_valid == FFT + (WINDOW - 1) * HOP
    assert samples[-1][1] == (0, 0, True)
    events = [sample for sample, (event, _, _) in samples if event]
    assert events and all(a + REFRACTORY * HOP <= b for a, b in zip(events, events[1:]))
    assert detector.observe(1, HOP, 1, 1) == (0, 0, False)
    assert detector.observe(1, FFT, 0, 1) == (0, 0, False)
    assert detector.observe(1, FFT + HOP, 1, 1) == (1, 1 / CAPACITY, False)
    assert detector.observe(1, FFT + 3 * HOP, 0, 0) == (0, 0, False)
    assert detector.observe(1, FFT + 4 * HOP, float("nan"), 0) == (0, 0, False)
    tb = rust.AnalysisTimebase()
    old = tb.ingest(pcm[:RATE], RATE, 0)
    gap = tb.ingest(pcm[RATE:2 * RATE], RATE, RATE + HOP)
    tb.reset()
    reset = tb.ingest(pcm[:RATE], RATE, 0)
    assert gap[0]["streamEpoch"] != old[-1]["streamEpoch"]
    assert reset[0]["streamEpoch"] != gap[-1]["streamEpoch"]
    assert gap[0]["sampleIndex"] == reset[0]["sampleIndex"] == HOP
    return {"first_valid_sample": first_valid, "first_valid_seconds": first_valid / RATE,
            "chunks": [997, 8193], "prefix_cut_sample": future_cut,
            "events": len(events), "first_response_ms": 1000 * (events[0] - RATE / 2) / RATE,
            "explicit_and_discontinuity_epoch_reset_checked": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "backend/data/audio")
    parser.add_argument("--seconds", type=int, default=120, help="maximum prefix per whole-song group")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/onset-101/results.json")
    args = parser.parse_args()
    if args.seconds < 10:
        parser.error("--seconds must be at least 10")
    assert rust.FEATURE_VERSION == "features/2" and rust.FeatureExtractor().num_features_per_frame() == WIDTH
    synthetic = self_check()
    tracks = []
    for song in SONGS:
        track = extract(args.audio_dir / f"{song}.mp3", args.seconds)
        tracks.append(track)
        print(f"{song}: {len(track.indices)} usable ticks, {track.event_count} events", flush=True)
    holdouts = held_out(tracks)
    summary = {"contracts": [rust.FEATURE_VERSION, rust.ANALYSIS_PIPELINE_VERSION, rust.CYCLE_BANK_VERSION],
               "seconds_requested": args.seconds, "synthetic": synthetic,
               "configuration": {"rate": RATE, "hop": HOP, "fft": FFT, "window": WINDOW,
                                 "capacity": CAPACITY, "refractory": REFRACTORY, "horizon": HORIZON,
                                 "flux_rise_floor": FLUX_RISE_FLOOR, "energy_floor": ENERGY_FLOOR,
                                 "cycle_slots": 8, "cycle_order": "ascending frequency",
                                 "cycle_configuration": "native defaults",
                                 "regime_count_up": [4, 8], "regime_count_down": [2, 6]},
               "tracks": [{"name": item.name, "sha256": item.digest, "native_rate": item.native_rate,
                           "positive_events": item.event_count, "ticks": len(item.indices),
                           "first_valid_seconds": item.first_valid_seconds,
                           "density_quantiles": np.quantile(item.density, [0, 0.1, 0.5, 0.9, 1]).tolist(),
                           "regime_fraction": [float(np.mean(item.regime == i)) for i in range(3)],
                           "regime_changes": int(np.count_nonzero(np.diff(item.regime))),
                           "pearson_with_base": dict(zip(BASE_NAMES, [
                               float(np.corrcoef(item.density, value)[0, 1])
                               for value in item.base[:, :len(BASE_NAMES)].T], strict=True))}
                          for item in tracks],
               "whole_song_holdouts": holdouts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
