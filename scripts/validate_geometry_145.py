"""Record conservative timestep-refinement evidence through the Rust bindings.

Run with the project venv after rebuilding runtime_core. This is an offline
diagnostic, not a replacement integrator or an acceptance-threshold override.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import runtime_core as rc


@dataclass
class Rollout:
    case: str
    duration_seconds: float
    dt: float
    steps: int
    completed: bool
    c: tuple[float, float]
    velocity: tuple[float, float]
    initial_energy: float
    final_energy: float
    maximum_excursion: float
    maximum_step_energy_change: float
    peak_from: tuple[float, float]
    peak_to: tuple[float, float]
    failure: str | None


def measure(
    name: str,
    c: complex,
    velocity: tuple[float, float],
    dt: float,
    duration_seconds: float,
) -> Rollout:
    config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
    initial = rc.manifold_total_energy(*velocity, c, config)
    energy = initial
    excursion = 0.0
    completed_steps = 0
    failure = None
    maximum_step_change = 0.0
    peak_from = peak_to = (c.real, c.imag)
    for _ in range(round(duration_seconds / dt)):
        previous_c = c
        try:
            x, y, vx, vy, info = rc.manifold_integrate_step(
                c.real, c.imag, *velocity, 0.0, 0.0, 0.0, dt, config
            )
        except RuntimeError as error:
            failure = str(error)
            break
        c = complex(x, y)
        velocity = vx, vy
        completed_steps += 1
        energy = rc.manifold_total_energy(*velocity, c, config)
        excursion = max(excursion, abs(energy - initial))
        if abs(info.delta_total) > maximum_step_change:
            maximum_step_change = abs(info.delta_total)
            peak_from = previous_c.real, previous_c.imag
            peak_to = c.real, c.imag
    return Rollout(
        name,
        duration_seconds,
        dt,
        completed_steps,
        failure is None,
        (c.real, c.imag),
        velocity,
        initial,
        energy,
        excursion,
        maximum_step_change,
        peak_from,
        peak_to,
        failure,
    )


def main() -> None:
    rows = [
        asdict(measure(name, start, velocity, dt, duration))
        for name, start, velocity, duration, timesteps in [
            (
                "low_curvature",
                complex(-0.5, 0.1),
                (0.01, 0.0),
                0.3,
                (0.004, 0.002, 0.001, 0.0005),
            ),
            (
                "high_curvature",
                complex(0.4, 0.05),
                (0.0, 0.02),
                0.3,
                (0.004, 0.002, 0.001, 0.0005),
            ),
            (
                "near_shore",
                complex(0.3, 0.05),
                (0.0, 0.02),
                0.3,
                (0.004, 0.002, 0.001, 0.0005),
            ),
            (
                "near_shore_inward",
                complex(0.4, 0.05),
                (-0.03, 0.0),
                0.5,
                (0.005, 0.0025, 0.00125, 0.000625),
            ),
        ]
        for dt in timesteps
    ]
    report = {
        "controller_version": rc.CONTROLLER_VERSION,
        "provider_version": rc.geometry_provider_version(),
        "acceptance_energy_excursion": 0.05,
        "acceptance_step_energy_change": 0.05,
        "rollouts": rows,
    }
    destination = (
        Path(__file__).resolve().parents[1] / "docs/validation/geometry-145-energy.json"
    )
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
