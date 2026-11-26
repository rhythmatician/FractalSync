"""Compare offline reset distributions using the installed Rust Physics bindings."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import cos, inf, isclose, isfinite, sin, sqrt, tau
from pathlib import Path
import random
from statistics import median
from typing import TypedDict

import runtime_core as rust

ROOT = Path(__file__).resolve().parents[2]
SEED = 110_2026
RADIUS = 1.90
BANDS = ((-inf, 0), (0, 3), (3, 6), (6, 9), (9, inf))
JITTER = 0.05
ENERGIES = (0.0, 0.0, 0.05, 0.25)
DT = 1024 / 48_000


@dataclass(frozen=True)
class Site:
    c: complex
    d: float
    grad_d: complex
    sigma: float
    estimate: float
    resolved: float
    region: tuple[int, int]

    @property
    def band(self) -> int:
        return next(i for i, (lower, upper) in enumerate(BANDS) if lower <= self.sigma < upper)

    @property
    def inside(self) -> bool:
        return self.d < 0


def angle(rng: random.Random) -> complex:
    phase = rng.random() * tau
    return complex(cos(phase), sin(phase))


def region(c: complex) -> tuple[int, int]:
    return (int((c.real + 2) / 0.45), int((c.imag + 2) / 0.45))


def query(c: complex, cfg: rust.ManifoldConfig, refused: Counter[str]) -> Site | None:
    if abs(c) >= RADIUS:
        refused["outside research disk"] += 1
        return None
    try:
        jet = json.loads(rust.geometry_provider_query(c, cfg.epsilon))
        validity = jet["validity"]
        if validity != "regular" or jet["is_bridge"]:
            refused[f'{validity}/{jet["singularity"]}'] += 1
            return None
        d, error, resolution = jet["D"], jet["estimated_error"], jet["resolved_scale"]
        if (not all(isinstance(x, (int, float)) and isfinite(x) for x in (d, error, resolution))
                or error < 0 or resolution <= 0 or abs(d) <= 4 * error):
            refused["uncertain sampled sign"] += 1
            return None
        g = complex(*jet["grad_D"])
        if not isfinite(abs(g)):
            refused["invalid gradient"] += 1
            return None
        return Site(c, d, g, rust.manifold_mandelbrot_scale(c, cfg), error, resolution, region(c))
    except (ValueError, RuntimeError) as error:
        refused[type(error).__name__] += 1
        return None


def area_point(rng: random.Random) -> complex:
    return RADIUS * sqrt(rng.random()) * angle(rng)


def candidate_pool(
    rng: random.Random, cfg: rust.ManifoldConfig, area_count: int,
) -> tuple[list[Site], list[Site], dict[str, Counter[str]]]:
    refusals: dict[str, Counter[str]] = {"area": Counter(), "guided": Counter()}
    area = [site for _ in range(area_count)
            if (site := query(area_point(rng), cfg, refusals["area"])) is not None]
    print(f"Validated {len(area)} of {area_count} area proposals", flush=True)
    proposals = [area_point(rng) for _ in range(20_000)]
    # Screen inexpensive embedded-raster samples. Each accepted row is re-queried
    # through the installed operational geometry provider before it enters a pool.
    raster = rust.sample_distance_field_py(proposals)
    anchors: dict[tuple[int, int], list[tuple[float, complex]]] = {}
    for c, d in zip(proposals, raster, strict=True):
        if isfinite(d) and 0 < d < 0.08:
            anchors.setdefault(region(c), []).append((d, c))
    guided = []
    for choices in anchors.values():
        best = sorted(choices, key=lambda pair: pair[0])[:2]
        for _, c in best:
            for _ in range(6):
                site = query(c, cfg, refusals["guided"])
                if site is None:
                    break
                guided.append(site)
                grad_sq = abs(site.grad_d)**2
                if grad_sq < 0.05 or site.band == 4:
                    break
                c -= 0.60 * site.d * site.grad_d / grad_sq
    return area, [*area, *guided], refusals


def choose_scale(
    rng: random.Random, buckets: dict[int, dict[bool, list[Site]]], bands: list[int],
) -> Site:
    cell = buckets[rng.choice(bands)]
    # A uniform occupied-region choice keeps a dense proposal cluster from
    # taking over a band/realm cell. Empty cells get no probability mass.
    pool = cell[rng.choice(list(cell))]
    regions: dict[tuple[int, int], list[Site]] = {}
    for site in pool:
        regions.setdefault(site.region, []).append(site)
    return rng.choice(regions[rng.choice(list(regions))])


def metric_heading(rng: random.Random, site: Site, cfg: rust.ManifoldConfig) -> complex:
    [[a, b], [_, d]] = rust.manifold_induced_metric(site.c, cfg)
    h = d - b * b / a
    assert a > 0 and h > 0
    direction = angle(rng)
    return direction.real / sqrt(a) + direction.imag * complex(-b / a, 1) / sqrt(h)


class Energy(TypedDict):
    k: float
    target_k: float
    u: float
    wall: float
    total: float
    shore_only_margin: float
    veuc: float
    fixed_euclidean_k: float


def measure(
    site: Site, target_k: float, rng: random.Random, cfg: rust.ManifoldConfig, crest: float,
) -> Energy:
    heading = metric_heading(rng, site, cfg)
    k_unit = rust.manifold_kinetic_energy(heading.real, heading.imag, site.c, cfg)
    assert isclose(k_unit, 0.5, rel_tol=1e-12)
    v = sqrt(2 * target_k) * heading
    k = rust.manifold_kinetic_energy(v.real, v.imag, site.c, cfg)
    score = rust.manifold_potential_energy(site.c, cfg)
    wall = rust.manifold_wall_potential(site.c, cfg)
    total = rust.manifold_total_energy(v.real, v.imag, site.c, cfg)
    fixed_v = 0.01 * heading / abs(heading)
    fixed_k = rust.manifold_kinetic_energy(fixed_v.real, fixed_v.imag, site.c, cfg)
    assert isclose(total, k + score + wall, rel_tol=1e-12, abs_tol=1e-12)
    return {"k": k, "target_k": target_k, "u": score + wall, "wall": wall, "total": total,
            "shore_only_margin": k + score - crest, "veuc": abs(v), "fixed_euclidean_k": fixed_k}


def quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[0], median(ordered), ordered[-1]]


def sample_summary(sites: list[Site], energies: list[Energy]) -> dict:
    by_band: dict[int, list[Energy]] = {}
    for site, row in zip(sites, energies, strict=True):
        by_band.setdefault(site.band, []).append(row)
    return {
        "draws": len(sites),
        "band_counts": [sum(site.band == i for site in sites) for i in range(len(BANDS))],
        "inside_band_counts": [sum(site.band == i and site.inside for site in sites)
                               for i in range(len(BANDS))],
        "occupied_regions": len({site.region for site in sites}),
        "unique_coordinates": len({site.c for site in sites}),
        "k_counts": dict(Counter(row["target_k"] for row in energies)),
        "u_min_median_max": quantiles([row["u"] for row in energies]),
        "wall_min_median_max": quantiles([row["wall"] for row in energies]),
        "e_min_median_max": quantiles([row["total"] for row in energies]),
        "by_band": {
            i: {"u": quantiles([row["u"] for row in values]),
                "e": quantiles([row["total"] for row in values]),
                "fixed_euclidean_k": quantiles([row["fixed_euclidean_k"] for row in values])}
            for i, values in by_band.items()
        },
        "shore_only_above_counts": [sum(site.band == i and row["shore_only_margin"] >= 0
                                        for site, row in zip(sites, energies, strict=True))
                                    for i in range(len(BANDS))],
        "max_k_error": max(abs(row["k"] - row["target_k"]) for row in energies),
        "veuc_moving_min_median_max": (
            quantiles(moving) if (moving := [row["veuc"] for row in energies if row["target_k"]])
            else None
        ),
    }


def jitter_check(
    seeds: list[Site], rng: random.Random, cfg: rust.ManifoldConfig, samples: int,
) -> dict:
    refused: Counter[str] = Counter()
    outcomes: dict[int, Counter[str]] = {}
    max_norm_error = 0.0
    for site in rng.sample(seeds, min(samples, len(seeds))):
        counts = outcomes.setdefault(site.band, Counter())
        delta = JITTER * metric_heading(rng, site, cfg)
        k = rust.manifold_kinetic_energy(delta.real, delta.imag, site.c, cfg)
        max_norm_error = max(max_norm_error, abs(sqrt(2 * k) - JITTER))
        new = query(site.c + delta, cfg, refused)
        if new is None:
            counts["provider refusal"] += 1
        elif new.band != site.band or new.inside != site.inside:
            counts["changed band or side"] += 1
        else:
            counts["accepted"] += 1
    return {"origin_tangent_length": JITTER, "max_norm_error": max_norm_error,
            "outcomes": outcomes, "refusals": refused}


def idle_check(
    buckets: dict[int, dict[bool, list[Site]]], rng: random.Random, cfg: rust.ManifoldConfig,
) -> dict:
    action = rust.MotionControls()
    outcomes: dict[int, dict[float, Counter[str]]] = {}
    refused: Counter[str] = Counter()
    for band, sides in buckets.items():
        pool = [site for sites in sides.values() for site in sites]
        band_outcomes = outcomes.setdefault(band, {})
        for site in rng.sample(pool, min(8, len(pool))):
            heading = metric_heading(rng, site, cfg)
            for target in sorted(set(ENERGIES)):
                trial = band_outcomes.setdefault(target, Counter())
                v = sqrt(2 * target) * heading
                try:
                    re, im, _, _, _ = rust.controls_integrate_step(
                        site.c.real, site.c.imag, v.real, v.imag, action, DT, cfg,
                    )
                except (ValueError, RuntimeError):
                    trial["step refused"] += 1
                    continue
                endpoint = query(complex(re, im), cfg, refused)
                if endpoint is None:
                    trial["endpoint refused"] += 1
                elif endpoint.inside != site.inside:
                    trial["changed side"] += 1
                else:
                    trial["same side"] += 1
    return {"dt_seconds": DT, "outcomes": outcomes, "endpoint_refusals": refused}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--area", type=int, default=640)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/reset-110/results.json")
    args = parser.parse_args()
    if args.area < 64 or args.draws < 20:
        parser.error("require --area >= 64 and --draws >= 20")
    rng = random.Random(args.seed)
    cfg = rust.ManifoldConfig()
    asset = rust.get_builtin_distance_field_py("mandelbrot_1024")
    area, candidates, refusals = candidate_pool(rng, cfg, args.area)
    assert area and candidates
    print(f"Collected {len(area)} area and {len(candidates)} combined candidates", flush=True)
    # Exported by the native binding; the installed generated stub omits it.
    crest_snapshot = rust.debug_snapshot_from_state(  # pyright: ignore[reportAttributeAccessIssue]
        area[0].c.real, area[0].c.imag, 0, 0, None, 0, 0, 0, cfg, None, 0,
    )
    crest = float(crest_snapshot["diagnostics"]["crestPotential"])
    buckets: dict[int, dict[bool, list[Site]]] = {}
    for site in candidates:
        buckets.setdefault(site.band, {}).setdefault(site.inside, []).append(site)
    bands = [band for band, sides in sorted(buckets.items())
             if all(len({site.region for site in sides.get(side, [])}) >= 2 for side in (True, False))]
    assert bands, "no scale bands with at least two occupied regions per realm"
    scale = [choose_scale(rng, buckets, bands) for _ in range(args.draws)]
    uniform = [rng.choice(area) for _ in range(args.draws)]
    mixed = [rng.choice(area) if draw % 2 == 0 else choose_scale(rng, buckets, bands)
             for draw in range(args.draws)]
    variants = {"area_rest": uniform, "scale_rest": scale, "mixed_rest": mixed, "mixed_k": mixed}
    report = {
        "seed": args.seed, "sampler": "reset-110-research/1", "area_proposals": args.area,
        "provider_version": rust.geometry_provider_version(), "controller_version": rust.CONTROLLER_VERSION,
        "asset_dimensions_and_bounds": asset,
        "physics": {name: value for name, value in
                    [("epsilon", cfg.epsilon), ("d_ref", cfg.d_ref), ("lambda_sq", cfg.lambda_sq),
                     ("kappa", cfg.kappa), ("mu", cfg.mu)]},
        "radius_cap": RADIUS, "scale_band_boundaries": [0, 3, 6, 9],
        "scale_draw_band_ids": bands, "shore_crest_potential_without_wall": crest,
        "pool": {band: {"inside" if side else "outside":
                        {"count": len(sites), "regions": len({site.region for site in sites})}
                        for side, sites in sides.items()} for band, sides in buckets.items()},
        "pool_refusals": refusals,
        "variants": {name: sample_summary(sites, [
            measure(site, rng.choice(ENERGIES) if name == "mixed_k" else 0, rng, cfg, crest)
            for site in sites]) for name, sites in variants.items()},
        "jitter": jitter_check(candidates, rng, cfg, 80),
        "idle_step": idle_check(buckets, rng, cfg),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
