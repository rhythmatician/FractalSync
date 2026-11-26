"""Reproducible illustrations for #163; numerical checks are not proofs.

Run with Python 3.10+; standard library only.  This is a research check, not
an alternative runtime implementation or a training dependency.
"""

from __future__ import annotations

import cmath
import json
import math
import random


def phase_harmonic(z: complex, order: int) -> complex:
    radius = abs(z)
    return 0j if radius == 0 else radius * (z / radius) ** order


def weighted_norm(values: list[complex], weights: list[float]) -> float:
    return math.sqrt(sum(w * abs(z) ** 2 for w, z in zip(weights, values)))


def moment(
    audio: list[complex], visual: list[complex], transport: list[complex],
    weights: list[float], n: int, m: int,
) -> complex:
    return sum(
        w * phase_harmonic(a, n) * (p ** m * phase_harmonic(b, m)).conjugate()
        for w, a, b, p in zip(weights, audio, visual, transport)
    )


def close(actual: complex | float, expected: complex | float, tol: float = 1e-10) -> None:
    assert abs(actual - expected) <= tol * max(1.0, abs(actual), abs(expected)), (
        actual, expected
    )


def main() -> None:
    rng = random.Random(16320261124)
    trials = 4000
    max_covariance_error = 0.0
    max_pairwise_error = 0.0
    worst_lipschitz_ratio = 0.0
    worst_stability_ratio = 0.0
    for _ in range(trials):
        size = rng.randrange(2, 12)
        w_raw = [rng.random() for _ in range(size)]
        weights = [w / sum(w_raw) for w in w_raw]
        audio = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(size)]
        visual = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(size)]
        transport = [cmath.exp(1j * rng.uniform(-math.pi, math.pi)) for _ in range(size)]
        gauges = [cmath.exp(1j * rng.uniform(-math.pi, math.pi)) for _ in range(size)]
        reference_gauge = cmath.exp(1j * rng.uniform(-math.pi, math.pi))
        n = rng.choice([-3, -1, 0, 1, 2, 3])
        m = rng.choice([-3, -1, 0, 1, 2, 3])
        c = moment(audio, visual, transport, weights, n, m)
        changed_visual = [g * b for g, b in zip(gauges, visual)]
        changed_transport = [reference_gauge * p * g.conjugate() for p, g in zip(transport, gauges)]
        changed_c = moment(audio, changed_visual, changed_transport, weights, n, m)
        expected = reference_gauge.conjugate() ** m * c
        close(changed_c, expected)
        max_covariance_error = max(max_covariance_error, abs(changed_c - expected))

        pairwise = 0j
        for i in range(size):
            for j in range(size):
                pairwise += (
                    weights[i] * weights[j]
                    * phase_harmonic(audio[i], n)
                    * phase_harmonic(audio[j], n).conjugate()
                    * phase_harmonic(visual[i], m).conjugate()
                    * (transport[i].conjugate() * transport[j]) ** m
                    * phase_harmonic(visual[j], m)
                )
        close(pairwise, abs(c) ** 2)
        max_pairwise_error = max(max_pairwise_error, abs(pairwise - abs(c) ** 2))

        changed_audio = [a + .1 * complex(rng.gauss(0, 1), rng.gauss(0, 1)) for a in audio]
        noisy_visual = [b + .1 * complex(rng.gauss(0, 1), rng.gauss(0, 1)) for b in visual]
        angle_error = [rng.uniform(-.08, .08) for _ in range(size)]
        noisy_transport = [p * cmath.exp(1j * e) for p, e in zip(transport, angle_error)]
        noisy_c = moment(changed_audio, noisy_visual, noisy_transport, weights, n, m)
        a_norm = weighted_norm(audio, weights)
        b_norm = weighted_norm(visual, weights)
        a_prime = weighted_norm(changed_audio, weights)
        b_prime = weighted_norm(noisy_visual, weights)
        dx = weighted_norm(
            [phase_harmonic(ap, n) - phase_harmonic(a, n) for ap, a in zip(changed_audio, audio)], weights
        )
        dy = weighted_norm(
            [pp ** m * phase_harmonic(bp, m) - p ** m * phase_harmonic(b, m)
             for pp, bp, p, b in zip(noisy_transport, noisy_visual, transport, visual)], weights
        )
        ea = weighted_norm([ap - a for ap, a in zip(changed_audio, audio)], weights)
        eb = weighted_norm([bp - b for bp, b in zip(noisy_visual, visual)], weights)
        angle_term = math.sqrt(sum(
            w * abs(bp) ** 2 * 4 * math.sin(m * e / 2) ** 2
            for w, bp, e in zip(weights, noisy_visual, angle_error)
        ))
        assert dx <= max(1, abs(n)) * ea + 1e-11
        assert dy <= max(1, abs(m)) * eb + angle_term + 1e-11
        epsilon = 10 ** rng.uniform(-7, 0)
        d = math.sqrt(a_norm ** 2 * b_norm ** 2 + epsilon)
        dp = math.sqrt(a_prime ** 2 * b_prime ** 2 + epsilon)
        q = min(a_norm * dy + b_prime * dx, b_norm * dx + a_prime * dy)
        bound = min(2.0, 2 * q / max(d, dp))
        error = abs(c / d - noisy_c / dp)
        assert error <= bound + 1e-11
        if bound > 0:
            worst_stability_ratio = max(worst_stability_ratio, error / bound)

        z = 0j if rng.random() < .1 else complex(rng.gauss(0, 1), rng.gauss(0, 1))
        z_prime = 0j if rng.random() < .1 else complex(rng.gauss(0, 1), rng.gauss(0, 1))
        for k in [-7, -3, -1, 0, 1, 2, 7]:
            error_h = abs(phase_harmonic(z, k) - phase_harmonic(z_prime, k))
            bound_h = max(1, abs(k)) * abs(z - z_prime)
            assert error_h <= bound_h + 1e-10
            if bound_h > 0:
                worst_lipschitz_ratio = max(worst_lipschitz_ratio, error_h / bound_h)

    # A unit transport kernel with half-turn cycle holonomy is indefinite.
    kernel = [[1, 1, -1], [1, 1, 1], [-1, 1, 1]]
    v = [1, -1, 1]
    kv = [sum(row[j] * v[j] for j in range(3)) for row in kernel]
    assert kv == [-x for x in v]
    quadratic = sum(v[i] * kv[i] for i in range(3))
    assert quadratic == -3
    fake_squared_coherence = quadratic / 9
    close(fake_squared_coherence, -1 / 3)

    # Spherical tetrahedral face: all pairwise angles acos(-1/3), face area pi.
    side_cos = -1 / 3
    angle_cos = (side_cos - side_cos ** 2) / (1 - side_cos ** 2)
    area = 3 * math.acos(angle_cos) - math.pi
    close(area, math.pi)
    close(cmath.exp(1j * area), -1)

    # Same two epoch moments, arbitrary unknown phase at second epoch.
    epoch_phases = [0, math.pi / 2, math.pi]
    pooled_magnitudes = [abs(.5 + .5 * cmath.exp(1j * phi)) for phi in epoch_phases]
    close(pooled_magnitudes[0], 1)
    close(pooled_magnitudes[-1], 0)

    print(json.dumps({
        "seed": 16320261124,
        "random_trials": trials,
        "max_gauge_covariance_error": max_covariance_error,
        "max_common_reference_pairwise_identity_error": max_pairwise_error,
        "worst_phase_harmonic_bound_fraction": worst_lipschitz_ratio,
        "worst_regularized_coherence_bound_fraction": worst_stability_ratio,
        "indefinite_transport_quadratic_form": quadratic,
        "invalid_squared_coherence": fake_squared_coherence,
        "spherical_triangle_area": area,
        "epoch_pooling_magnitudes_same_signals": pooled_magnitudes,
        "status": "All finite numerical checks passed; this does not prove theorems.",
    }, indent=2))


if __name__ == "__main__":
    main()
