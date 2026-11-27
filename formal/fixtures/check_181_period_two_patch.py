"""Enclose the retained field's binary64 jet error on two exact period-two patches."""

from __future__ import annotations

import argparse
import json
import subprocess
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as Q
from itertools import product
from math import comb, factorial, floor, inf, isqrt, nextafter
from pathlib import Path

from check_operational_patch import (
    ASSET_SHA,
    DERIVATIVES,
    FIELD_ID,
    POLICY_SHA,
    POLYNOMIALS,
    ROOT,
    SOURCE_SHA,
    Bound,
    coefficient_at,
    compiled_queries,
    exact_jet,
    grid_coordinate_bound,
    jet_bounds,
    retained_asset,
)

Interval = tuple[Q, Q]
Box = tuple[Interval, Interval]
COMPONENTS = ("value", "dx", "dy", "dxx", "dxy", "dyy")
HALF_WIDTH = Q(1, 2**15)


def box_at(x: Q, y: Q) -> Box:
    return ((x - HALF_WIDTH, x + HALF_WIDTH), (y - HALF_WIDTH, y + HALF_WIDTH))


def mul(left: Interval, right: Interval) -> Interval:
    endpoints = [a * b for a, b in product(left, right)]
    return min(endpoints), max(endpoints)


def sub(left: Interval, right: Interval) -> Interval:
    return left[0] - right[1], left[1] - right[0]


def square(bounds: Interval) -> Interval:
    lo, hi = bounds
    return (Q(0) if lo <= 0 <= hi else min(lo**2, hi**2)), max(lo**2, hi**2)


def radius_bounds(squared: Interval) -> Interval:
    # Floor sqrt on a dyadic grid using only exact integer arithmetic.
    scale = 2**80
    lo, hi = (isqrt(floor(value * scale**2)) for value in squared)
    return Q(lo, scale), Q(hi + 1, scale)


def shore_jet(box: Box) -> list[Interval]:
    x, y = (box[0][0] + 1, box[0][1] + 1), box[1]
    x2, y2 = square(x), square(y)
    r = radius_bounds((x2[0] + y2[0], x2[1] + y2[1]))
    if not 0 < r[0] <= r[1] < Q(1, 4):
        raise ValueError("box must be strictly inside the bulb and exclude its center")
    inverse = (1 / r[1], 1 / r[0])
    inverse_cube = (inverse[0] ** 3, inverse[1] ** 3)
    xy = mul(x, y)
    return [
        (r[0] - Q(1, 4), r[1] - Q(1, 4)),
        mul(x, inverse),
        mul(y, inverse),
        mul(y2, inverse_cube),
        mul((-xy[1], -xy[0]), inverse_cube),
        mul(x2, inverse_cube),
    ]


def shifted_weights(center: Q) -> list[list[Q]]:
    # Coefficients of B_i(center + t), using the binomial identity.
    return [
        [
            sum(poly[k] * comb(k, n) * center ** (k - n) for k in range(n, 6)) / 120
            for n in range(6)
        ]
        for poly in POLYNOMIALS
    ]


def power_ranges(radius: Q) -> list[Interval]:
    return [(Q(1), Q(1))] + [
        (-(radius**n) if n % 2 else Q(0), radius**n) for n in range(1, 6)
    ]


def spline_jet(fractions: Box, spacing: Q, stencil: list[Q]) -> list[Interval]:
    # Combine coefficients before bounding monomials so cancellation in the
    # actual spline survives the enclosure, including in the differentiated jet.
    wx, wy = (shifted_weights((lo + hi) / 2) for lo, hi in fractions)
    powers_x, powers_y = (power_ranges((hi - lo) / 2) for lo, hi in fractions)
    along_x = [
        [sum(stencil[y * 6 + x] * wx[x][i] for x in range(6)) for i in range(6)]
        for y in range(6)
    ]
    coefficients = [
        [sum(along_x[y][i] * wy[y][j] for y in range(6)) for j in range(6)]
        for i in range(6)
    ]
    bounds = []
    for a, b in DERIVATIVES:
        lower, upper = Q(0), Q(0)
        for i, j in product(range(a, 6), range(b, 6)):
            coefficient = (
                coefficients[i][j]
                * comb(i, a)
                * factorial(a)
                * comb(j, b)
                * factorial(b)
                / spacing ** (a + b)
            )
            lo, hi = mul(
                (coefficient, coefficient), mul(powers_x[i - a], powers_y[j - b])
            )
            lower += lo
            upper += hi
        bounds.append((lower, upper))
    return bounds


def require_halo(lo: Q, hi: Q, extent: int) -> range:
    if not (2 <= lo <= hi < extent - 3):
        raise ValueError("exact or machine coordinates can leave the full halo")
    assert extent < 2**53  # Cell indices are binary64 integers; 64-bit usize is exact.
    return range(floor(lo), floor(hi) + 1)


def patch_bounds(
    box: Box, asset: bytes, shape: tuple[int, int], origin: tuple[Q, Q], spacing: Q
) -> tuple[dict[tuple[int, int], list[Q]], list[Interval], list[Q]]:
    assert len(asset) < 2**32  # Covers sizes, strides, and all candidate addresses.
    ox, oy = origin
    xr = ((box[0][0] - ox) / spacing, (box[0][1] - ox) / spacing)
    yr = ((box[1][0] - oy) / spacing, (box[1][1] - oy) / spacing)
    dx = grid_coordinate_bound(max(map(abs, xr)), spacing).error
    dy = grid_coordinate_bound(max(map(abs, yr)), spacing).error
    nx, ny = shape
    xs = require_halo(xr[0] - dx, xr[1] + dx, nx)
    ys = require_halo(yr[0] - dy, yr[1] + dy, ny)
    stencils = {}
    for x, y in product(xs, ys):
        # The local fraction is exact by Sterbenz's lemma, since index >= 2.
        stencils[x, y] = [
            coefficient_at(asset, nx, i, j)
            for j in range(y - 2, y + 4)
            for i in range(x - 2, x + 4)
        ]
    maximum = max(abs(c) for stencil in stencils.values() for c in stencil)
    delta = spacing * max(dx, dy)
    shift = (
        8 * maximum * delta / (3 * spacing),
        68 * maximum * delta / (9 * spacing**2),
        32 * maximum * delta / spacing**3,
    )
    arithmetic = [
        jet_bounds((Bound.exact(1), Bound.exact(1)), spacing, c)
        for c in stencils.values()
    ]
    error = [
        max(v.error for v in each) + shift[a + b]
        for each, (a, b) in zip(zip(*arithmetic, strict=True), DERIVATIVES, strict=True)
    ]

    ranges = []
    for (x, y), stencil in stencils.items():
        xt = (max(Q(0), xr[0] - x), min(Q(1), xr[1] - x))
        yt = (max(Q(0), yr[0] - y), min(Q(1), yr[1] - y))
        if xt[0] <= xt[1] and yt[0] <= yt[1]:
            ranges.append(spline_jet((xt, yt), spacing, stencil))
    merged = [
        (min(lo for lo, _ in each), max(hi for _, hi in each))
        for each in zip(*ranges, strict=True)
    ]
    return stencils, merged, error


def representative_points(box: Box, origin: tuple[Q, Q], h: Q) -> list[tuple[Q, Q]]:
    samples = []
    for (lo, hi), o in zip(box, origin, strict=True):
        axis = {lo, (lo + hi) / 2, hi}
        for index in range(floor((lo - o) / h) + 1, floor((hi - o) / h) + 1):
            face = float(o + h * index)
            axis.update(
                Q(value)
                for value in (nextafter(face, -inf), face, nextafter(face, inf))
            )
        samples.append(sorted(p for p in axis if lo <= p <= hi))
    return list(product(*samples))


def anchored_errors(
    box: Box, field: list[Interval], reference: list[Interval], anchor: list[Interval]
) -> list[Interval]:
    gaps = [sub(f, r) for f, r in zip(field, reference, strict=True)]
    rx, ry = ((hi - lo) / 2 for lo, hi in box)
    # Integrating each differential on the segment from the box center gives
    # a second enclosure. Tighten the gradient before using it for the value.
    for i, j, k in ((1, 3, 4), (2, 4, 5), (0, 1, 2)):
        variation = max(map(abs, gaps[j])) * rx + max(map(abs, gaps[k])) * ry
        lo, hi = gaps[i]
        gaps[i] = (max(lo, anchor[i][0] - variation), min(hi, anchor[i][1] + variation))
    return gaps


def outward(value: Q, lower: bool) -> str:
    with localcontext() as context:
        context.prec = 8
        context.rounding = ROUND_FLOOR if lower else ROUND_CEILING
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def negative_controls(shape: tuple[int, int], origin: tuple[Q, Q], spacing: Q) -> None:
    for y in (Q(0), Q(-1, 4)):
        try:
            shore_jet(box_at(Q(-1), y))
        except ValueError:
            pass
        else:
            raise AssertionError(
                "certified a box containing the cut or crossing the Shore"
            )

    # Interior of the bulb, yet wholly beyond the retained asset's left halo.
    box = box_at(Q(-9, 8), Q(0))
    shore_jet(box)
    try:
        require_halo(
            (box[0][0] - origin[0]) / spacing,
            (box[0][1] - origin[0]) / spacing,
            shape[0],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("certified unavailable retained-field halo")


def main() -> None:
    if not __debug__:
        raise RuntimeError("Verification requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument(
        "--offline", action="store_true", help="Use cached Cargo dependencies"
    )
    args = parser.parse_args()
    asset, shape, origin, h = retained_asset(args.asset.resolve())
    negative_controls(shape, origin, h)
    patches = {
        "deep": box_at(Q(-1), Q(-3, 16)),
        "shallow": box_at(Q(-1), Q(-1, 4) + Q(1, 2**14)),
    }
    sample_groups = {
        name: representative_points(box, origin, h) for name, box in patches.items()
    }
    points = [point for group in sample_groups.values() for point in group]
    assert all(Q(float(p)) == p for point in points for p in point)
    results = iter(compiled_queries(args.asset.resolve(), points, args.offline))
    report = {}
    for name, box in patches.items():
        reference = shore_jet(box)
        stencils, field, error = patch_bounds(box, asset, shape, origin, h)
        center = ((box[0][0] + box[0][1]) / 2, (box[1][0] + box[1][1]) / 2)
        center_cell = (
            floor((center[0] - origin[0]) / h),
            floor((center[1] - origin[1]) / h),
        )
        exact_center = exact_jet(center, origin, h, center_cell, stencils[center_cell])
        reference_center = shore_jet(((center[0], center[0]), (center[1], center[1])))
        anchor = [
            sub((f, f), r) for f, r in zip(exact_center, reference_center, strict=True)
        ]
        exact_gaps = anchored_errors(box, field, reference, anchor)
        gaps = [(lo - e, hi + e) for (lo, hi), e in zip(exact_gaps, error, strict=True)]
        changed_floors = 0
        for point in sample_groups[name]:
            compiled = next(results)
            x, y = point
            cell = (floor((x - origin[0]) / h), floor((y - origin[1]) / h))
            machine_cell = (
                floor((float(x) - float(origin[0])) / float(h)),
                floor((float(y) - float(origin[1])) / float(h)),
            )
            assert machine_cell in stencils
            changed_floors += cell != machine_cell
            exact = exact_jet(point, origin, h, cell, stencils[cell])
            for f, act, ideal, allowance in zip(
                field, compiled, exact, error, strict=True
            ):
                assert f[0] <= ideal <= f[1] and abs(act - ideal) <= allowance
            exact_query_box = ((x, x), (y, y))
            for value, r, region in zip(
                compiled, shore_jet(exact_query_box), gaps, strict=True
            ):
                lo, hi = sub((value, value), r)
                assert region[0] <= lo <= hi <= region[1]

        if name == "deep":
            assert reference[0][1] < -Q(624, 10000)
            assert max(map(abs, gaps[0])) < Q(1, 10**6) and gaps[3][0] > 3
        else:
            assert gaps[0][1] < -Q(3, 10000) and gaps[2][0] > Q(4, 5)
        report[name] = {
            "box": [[str(lo), str(hi)] for lo, hi in box],
            "candidate_cells": sorted(stencils),
            "compiled_queries": len(sample_groups[name]),
            "query_floor_changes": changed_floors,
            "arithmetic_error_upper": dict(
                zip(COMPONENTS, (outward(e, False) for e in error))
            ),
            "signed_jet_error_intervals": dict(
                zip(
                    COMPONENTS,
                    ([outward(lo, True), outward(hi, False)] for lo, hi in gaps),
                )
            ),
        }
    assert next(results, None) is None
    print(
        json.dumps(
            {
                "field_id": FIELD_ID,
                "asset_sha256": ASSET_SHA,
                "policy_sha256": POLICY_SHA,
                "evaluator_source_sha256": SOURCE_SHA,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "rustc": subprocess.check_output(
                    ["rustc", "-vV"], text=True
                ).splitlines(),
                "patches": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
