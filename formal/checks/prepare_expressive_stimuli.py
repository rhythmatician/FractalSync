"""Prepare synthetic audio and blinded first-hearing inputs for issue #189."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import cos, log10, sin, sqrt, tau
from pathlib import Path
import random
import sys
import wave


SAMPLE_RATE = 48_000
HOP = 1024
CLIP_SAMPLES = 480 * HOP
PULSE_STEP = 16 * HOP
TONE_SAMPLES = 3600
PCM_SCALE = 32_767
TARGET_RMS = 0.05
PEAK_CEILING = 0.85
DEFAULT_SEED = 189_2026
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "tmp" / "expressive-189"
PITCHES = (196, 220, 247, 262, 294)
PHRASE_STARTS = tuple(range(0, CLIP_SAMPLES, 10 * PULSE_STEP))
SHARED_PULSES = tuple(range(0, CLIP_SAMPLES, PULSE_STEP))
ORDERS = ("AAB", "ABA")
ARM_NAMES = ("matched", "order_mismatch", "timing_negative", "shared_pulse", "audio_only")


@dataclass(frozen=True)
class Family:
    patterns: tuple[tuple[int, ...], tuple[int, ...]]
    accent_harmonic: int

    def accents(self, order: str) -> tuple[int, ...]:
        return tuple(
            phrase_start + offset * PULSE_STEP
            for phrase_start, token in zip(PHRASE_STARTS, order, strict=True)
            for offset in self.patterns[0 if token == "A" else 1]
        )


FAMILIES = (
    Family(((0, 2, 5, 8), (0, 3, 5, 8)), 2),
    Family(((0, 2, 4, 7), (0, 2, 5, 7)), 3),
)


@dataclass(frozen=True)
class AudioAsset:
    id: str
    family: int
    fundamental_hz: int
    order: str
    correct_slot: int
    accent_samples: tuple[int, ...]
    file: str
    sha256: str
    rms_dbfs: float
    peak_dbfs: float
    pcm_sum_of_squares: int


@dataclass(frozen=True)
class TimingPlan:
    id: str
    starts_samples: tuple[int, ...]
    pulse_width_samples: int


@dataclass(frozen=True)
class Trial:
    trial_id: str
    audio: AudioAsset
    arm: str
    planned_visual_id: str | None


@dataclass(frozen=True)
class Session:
    participant_id: str
    block: int
    trials: list[Trial]


def opaque_id(prefix: str, rng: random.Random) -> str:
    return f"{prefix}-{rng.getrandbits(64):016x}"


def tone(frequency: int, gain: float) -> list[float]:
    return [
        gain * sin(tau * frequency * i / SAMPLE_RATE)
        * (1 - cos(tau * i / (TONE_SAMPLES - 1))) / 2
        for i in range(TONE_SAMPLES)
    ]


def write_audio(path: Path, fundamental: int, harmonic: int, accents: tuple[int, ...]) -> None:
    signal = [0.0] * CLIP_SAMPLES
    for events, samples in ((SHARED_PULSES, tone(fundamental, 1.0)),
                            (accents, tone(fundamental * harmonic, 0.5))):
        for start in events:
            for offset, value in enumerate(samples):
                signal[start + offset] += value
    factor = TARGET_RMS / sqrt(sum(sample * sample for sample in signal) / CLIP_SAMPLES)
    if max(abs(sample) for sample in signal) * factor > PEAK_CEILING:
        raise ValueError("RMS target exceeds the peak ceiling")
    pcm = array("h", (round(value * factor * PCM_SCALE) for value in signal))
    if sys.byteorder != "little":
        pcm.byteswap()
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, SAMPLE_RATE, CLIP_SAMPLES, "NONE", "not compressed"))
        out.writeframes(pcm.tobytes())


def inspect_audio(path: Path, family: Family, order: str) -> tuple[int, float, float]:
    with wave.open(str(path), "rb") as clip:
        codec = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getcomptype())
        assert codec == (1, 2, SAMPLE_RATE, "NONE")
        assert clip.getnframes() == CLIP_SAMPLES
        pcm = array("h")
        pcm.frombytes(clip.readframes(CLIP_SAMPLES))
    if sys.byteorder != "little":
        pcm.byteswap()
    assert len(pcm) == CLIP_SAMPLES
    accents = family.accents(order)
    assert len(SHARED_PULSES) == 30 and len(set(accents)) == 12
    assert set(accents).issubset(SHARED_PULSES)
    for start in SHARED_PULSES:
        end = start + TONE_SAMPLES
        assert end <= CLIP_SAMPLES
        assert pcm[start] == pcm[end - 1] == 0
        assert max(abs(sample) for sample in pcm[start:end]) > 0
        assert not any(pcm[end:start + PULSE_STEP])
    energy = sum(sample * sample for sample in pcm)
    rms = sqrt(energy / CLIP_SAMPLES) / PCM_SCALE
    peak = max(abs(sample) for sample in pcm) / PCM_SCALE
    assert abs(rms - TARGET_RMS) <= 1 / PCM_SCALE
    assert peak < PEAK_CEILING
    return energy, 20 * log10(rms), 20 * log10(peak)


def make_audio(output: Path, rng: random.Random) -> dict[tuple[int, int, str], AudioAsset]:
    audio_dir = output / "playback" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[tuple[int, int, str], AudioAsset] = {}
    for family_index, family in enumerate(FAMILIES):
        for pitch in PITCHES:
            for order in ORDERS:
                asset_id = opaque_id("m", rng)
                file = f"audio/{asset_id}.wav"
                accents = family.accents(order)
                path = audio_dir / f"{asset_id}.wav"
                write_audio(path, pitch, family.accent_harmonic, accents)
                energy, rms, peak = inspect_audio(path, family, order)
                assets[(family_index, pitch, order)] = AudioAsset(
                    asset_id, family_index + 1, pitch, order, 2 if order == "AAB" else 3,
                    accents, file, sha256(path.read_bytes()).hexdigest(), rms, peak, energy,
                )
    return assets


def make_schedule(
    assets: dict[tuple[int, int, str], AudioAsset], rng: random.Random,
) -> tuple[list[Session], list[TimingPlan]]:
    plans: dict[tuple[int, str, str], TimingPlan] = {}
    for family_index, family in enumerate(FAMILIES):
        for order in ORDERS:
            accents = family.accents(order)
            plans[(family_index, order, "matched")] = TimingPlan(opaque_id("v", rng), accents, 10 * HOP)
            plans[(family_index, order, "timing_negative")] = TimingPlan(
                opaque_id("v", rng), tuple(start + 8 * HOP for start in accents), 10 * HOP,
            )
    shared_plan = TimingPlan(opaque_id("v", rng), SHARED_PULSES, 4 * HOP)
    sessions: list[Session] = []
    for block in range(4):
        rotations = [rng.sample(ARM_NAMES, len(ARM_NAMES)) for _ in FAMILIES]
        for lane in range(5):
            trials: list[Trial] = []
            for family_index, rotation in enumerate(rotations):
                for pitch_index, pitch in enumerate(PITCHES):
                    order_index = (family_index + pitch_index + block) % 2
                    order = ORDERS[order_index]
                    arm = rotation[(pitch_index + lane) % 5]
                    plan: TimingPlan | None = None
                    if arm == "order_mismatch":
                        plan = plans[(family_index, ORDERS[1 - order_index], "matched")]
                    elif arm == "shared_pulse":
                        plan = shared_plan
                    elif arm != "audio_only":
                        plan = plans[(family_index, order, arm)]
                    trials.append(Trial(
                        opaque_id("t", rng), assets[(family_index, pitch, order)], arm,
                        plan.id if plan else None,
                    ))
            rng.shuffle(trials)
            sessions.append(Session(opaque_id("p", rng), block + 1, trials))
    return sessions, [*plans.values(), shared_plan]


def check_schedule(sessions: list[Session], plans: list[TimingPlan], assets: list[AudioAsset]) -> None:
    assert len(sessions) == 20 and len(assets) == 20 and len(plans) == 9
    assert len({session.participant_id for session in sessions}) == 20
    assert len({asset.id for asset in assets}) == 20
    assert len({asset.sha256 for asset in assets}) == 20
    assert len({trial.trial_id for session in sessions for trial in session.trials}) == 200
    plan_by_id = {plan.id: plan for plan in plans}
    assert len(plan_by_id) == 9
    for plan in plans:
        assert len(plan.starts_samples) * plan.pulse_width_samples == 120 * HOP
        assert all(start % HOP == 0 and 0 <= start <= CLIP_SAMPLES - plan.pulse_width_samples
                   for start in plan.starts_samples)
    block_arms: Counter[tuple[int, int, int, str]] = Counter()
    answer_slots: Counter[tuple[int, int, str, int]] = Counter()
    for session in sessions:
        assert len({trial.audio.id for trial in session.trials}) == 10
        assert Counter(trial.arm for trial in session.trials) == dict.fromkeys(ARM_NAMES, 2)
        assert Counter(trial.audio.correct_slot for trial in session.trials) == {2: 5, 3: 5}
        for trial in session.trials:
            asset = trial.audio
            block_arms[(session.block, asset.family, asset.fundamental_hz, trial.arm)] += 1
            answer_slots[(asset.family, asset.fundamental_hz, trial.arm, asset.correct_slot)] += 1
            assert (trial.planned_visual_id is None) == (trial.arm == "audio_only")
            if trial.planned_visual_id is None:
                continue
            plan = plan_by_id[trial.planned_visual_id]
            if trial.arm == "matched":
                assert plan.starts_samples == asset.accent_samples
            elif trial.arm == "order_mismatch":
                opposite = "ABA" if asset.order == "AAB" else "AAB"
                assert plan.starts_samples == FAMILIES[asset.family - 1].accents(opposite)
            elif trial.arm == "timing_negative":
                assert plan.starts_samples == tuple(start + 8 * HOP for start in asset.accent_samples)
            else:
                assert trial.arm == "shared_pulse" and plan.starts_samples == SHARED_PULSES
    assert len(block_arms) == 200 and set(block_arms.values()) == {1}
    assert len(answer_slots) == 100 and set(answer_slots.values()) == {2}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def prepare(output: Path, seed: int) -> None:
    rng = random.Random(seed)
    assets = make_audio(output, rng)
    sessions, plans = make_schedule(assets, rng)
    audio = list(assets.values())
    check_schedule(sessions, plans, audio)
    write_json(output / "playback" / "schedule.json", {
        "status": "audio inputs only; no render, display or playback synchronization verified",
        "sessions": [{
            "participant_id": session.participant_id,
            "trials": [{"trial_id": trial.trial_id, "audio": trial.audio.file} for trial in session.trials],
        } for session in sessions],
    })
    write_json(output / "operator" / "key.json", {
        "schema": "expressive-stimuli/1",
        "seed": seed,
        "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": list(sys.version_info[:3]),
        "source": "Original synthetic sine pulses, no third-party audio, recordings or samples",
        "visual_status": "Timing plans only; Controls trajectories and Julia exports are pending",
        "sample_rate": SAMPLE_RATE,
        "sample_width_bits": 16,
        "channels": 1,
        "hop_samples": HOP,
        "clip_samples": CLIP_SAMPLES,
        "tone_samples": TONE_SAMPLES,
        "sample_index_convention": "zero-based offsets, clip end exclusive",
        "phrase_start_samples": PHRASE_STARTS,
        "phrase_end_exclusive_samples": (*PHRASE_STARTS[1:], CLIP_SAMPLES),
        "shared_pulse_samples": SHARED_PULSES,
        "rms_reference": "32767 and full-clip PCM samples; RMS control is not perceived loudness",
        "rms_target_dbfs": 20 * log10(TARGET_RMS),
        "peak_ceiling_dbfs": 20 * log10(PEAK_CEILING),
        "families": [asdict(family) for family in FAMILIES],
        "audio_assets": [asdict(asset) for asset in audio],
        "visual_timing_plans": [asdict(plan) for plan in plans],
        "sessions": [{
            "participant_id": session.participant_id,
            "block": session.block,
            "trials": [{
                "trial_id": trial.trial_id, "audio_id": trial.audio.id,
                "arm": trial.arm, "correct_slot": trial.audio.correct_slot,
                "planned_visual_id": trial.planned_visual_id,
            } for trial in session.trials],
        } for session in sessions],
    })
    rms = [asset.rms_dbfs for asset in audio]
    peak = [asset.peak_dbfs for asset in audio]
    print(f"{len(audio)} distinct WAVs, {CLIP_SAMPLES:,} samples each at {SAMPLE_RATE:,} Hz")
    print(f"PCM RMS {min(rms):.6f} to {max(rms):.6f} dBFS; "
          f"peak {min(peak):.3f} to {max(peak):.3f} dBFS")
    print("20 blinded sessions and 200 trials; operator scoring key saved separately")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    prepare(args.output, args.seed)


if __name__ == "__main__":
    main()
