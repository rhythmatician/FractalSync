"""Tune peak-pick multipliers for beat and long-cycle detection.

This script loads all mp3s in backend/data/audio and evaluates detection
behavior for several multipliers. It prints a concise CSV-like report so
we can inspect appropriate multipliers for Tool tracks.

Usage: python backend/scripts/tune_peak_pick.py
"""

from __future__ import annotations
import os
from math import ceil
import sys
import pathlib

# Ensure backend is on sys.path so we can import `src` package inside backend
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import librosa
from src.song_analyzer import SongAnalyzer

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio")

# Sweep settings
BEAT_MULTS = [0.05, 0.1, 0.25, 0.5]
CYCLE_MULTS = [0.1, 0.25, 0.35, 0.5]

an = SongAnalyzer(sr=22050, hop_length=512)

files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".mp3")]
if len(files) == 0:
    print("No mp3 files found in", AUDIO_DIR)
    raise SystemExit(1)

print(
    "file,beat_mult,beat_peaks,median_beat_sec,mean_peak_plp,cycle_mult,cycle_peaks,median_cycle_sec,mean_peak_carrier"
)
for fn in files:
    path = os.path.join(AUDIO_DIR, fn)
    try:
        y, sr = librosa.load(path, sr=an.sr)
    except Exception as e:
        print(f"{fn},ERROR_LOADING,{e}")
        continue

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=an.hop_length)
    plp = librosa.beat.plp(onset_envelope=onset, sr=sr, hop_length=an.hop_length)

    # Beat tuning
    for bm in BEAT_MULTS:
        delta = float(np.std(plp) * bm) if plp.size > 0 else 0.0
        wait = an._min_beat_wait_frames(240.0)
        peaks = librosa.util.peak_pick(
            plp, pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=delta, wait=wait
        )
        n = peaks.size
        if n > 1:
            times = librosa.frames_to_time(peaks, sr=sr, hop_length=an.hop_length)
            median_beat = float(np.median(np.diff(times)))
            mean_plp = float(np.mean(plp[peaks]))
        else:
            median_beat = float("nan")
            mean_plp = float(np.mean(plp[peaks])) if n > 0 else 0.0

        # Cycle tuning (use mid-long band 16-64s as proxy)
        tmin, tmax = 16.0, 64.0
        max_lag_frames = int(ceil(tmax * sr / float(an.hop_length)))
        ds = max(1, int(ceil(max_lag_frames / 512)))
        onset_ds = an._downsample_1d(onset, ds)

        # harmonic change (hcdf)
        harm = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_cqt(y=harm, sr=sr, hop_length=an.hop_length)
        chroma = librosa.util.normalize(chroma, axis=0)
        hcdf = np.zeros(chroma.shape[1], dtype=np.float32)
        if chroma.shape[1] > 1:
            num = np.sum(chroma[:, 1:] * chroma[:, :-1], axis=0)
            den = (
                np.linalg.norm(chroma[:, 1:], axis=0)
                * np.linalg.norm(chroma[:, :-1], axis=0)
                + 1e-8
            )
            hcdf[1:] = (1.0 - (num / den)).astype(np.float32)
        hcdf_ds = an._downsample_1d(hcdf, ds)

        novelty = an._compute_novelty_env(y)
        novelty_ds = an._downsample_1d(novelty, ds)

        # Compute dominant period/strength via analyzer helper (using hop override)
        hop_override = an.hop_length * ds
        p_on, s_on = an._dominant_cycle_from_tempogram(
            onset_ds, tmin, tmax, win_length=None, hop_length_override=hop_override
        )
        p_hc, s_hc = an._dominant_cycle_from_tempogram(
            hcdf_ds, tmin, tmax, win_length=None, hop_length_override=hop_override
        )
        w_on, w_hc = s_on, s_hc
        w_sum = w_on + w_hc + 1e-8
        period = (p_on * w_on + p_hc * w_hc) / w_sum

        # Prepare carriers (robust scaling)
        onset_01 = an._robust_01(onset_ds)
        hcdf_01 = an._robust_01(hcdf_ds)
        novelty_01 = an._robust_01(novelty_ds)
        # use harmony for long bands
        carrier = (1.0 - hcdf_01).astype(np.float32)
        # blend novelty modestly
        m = min(len(novelty_01), len(carrier))
        if m > 0:
            carrier = ((1.0 - 0.65) * carrier[:m] + 0.65 * novelty_01[:m]).astype(
                np.float32
            )

        # smooth: get a representative period in frames
        hop_for_ds = hop_override
        period_frames = period * (sr / float(hop_for_ds))
        win = int(
            np.clip(
                (
                    int(np.median(period_frames[np.isfinite(period_frames)]))
                    if period_frames.size > 0
                    else 3
                ),
                3,
                301,
            )
        )
        kernel = np.ones(win, dtype=np.float32) / float(win)
        carrier_sm = np.convolve(carrier, kernel, mode="same").astype(np.float32)

        for cm in CYCLE_MULTS:
            delta_c = float(np.std(carrier_sm) * cm) if carrier_sm.size > 0 else 0.0
            pre_max = max(1, win // 4)
            post_max = max(1, win // 4)
            pre_avg = max(1, win // 2)
            post_avg = max(1, win // 2)
            wait_c = max(1, win // 2)
            peaks_c = librosa.util.peak_pick(
                carrier_sm,
                pre_max=pre_max,
                post_max=post_max,
                pre_avg=pre_avg,
                post_avg=post_avg,
                delta=delta_c,
                wait=wait_c,
            )
            n_c = peaks_c.size
            if n_c > 1:
                times_c = librosa.frames_to_time(
                    peaks_c * ds, sr=sr, hop_length=an.hop_length
                )
                median_cycle = float(np.median(np.diff(times_c)))
                mean_carrier = float(np.mean(carrier_sm[peaks_c]))
            else:
                median_cycle = float("nan")
                mean_carrier = float(np.mean(carrier_sm[peaks_c])) if n_c > 0 else 0.0

            print(
                f"{fn},{bm},{n},{median_beat:.3f},{mean_plp:.4f},{cm},{n_c},{median_cycle:.3f},{mean_carrier:.4f}"
            )

print("Done")
