"""Compare the two-ended displacement bound with flat-Shore path lengths."""

from __future__ import annotations

from heapq import heappop, heappush
from math import copysign, dist, hypot, inf, log1p, pi
from random import Random
from typing import TypeAlias

from check_shore_shortcuts import excursion_arc, excursion_height, path_length
from check_stable_radial_profile import radial_log


Point: TypeAlias = tuple[float, float]
FLOOR = 1.0
PANELS = 512
SLACK = 2e-12


def one_ended(x: Point, y: Point) -> float:
    return log1p(dist(x, y) / min(hypot(FLOOR, x[0]), hypot(FLOOR, y[0])))


def compatible_bound(r: float, s: float, d: float) -> float:
    r, s = min(r, s), max(r, s)
    # For incompatible enclosures fall back to the one-ended lower bound.
    gap = s - r
    if d < gap:
        return log1p(d / r)
    return log1p((d + gap) / (2 * r)) + log1p((d - gap) / (2 * s))


def two_ended(x: Point, y: Point) -> float:
    return compatible_bound(hypot(FLOOR, x[0]), hypot(FLOOR, y[0]), dist(x, y))


def full_bound(x: Point, y: Point, anisotropy: float, displacement: float) -> float:
    # The synthetic Shore is exactly r=0. A sign change entails a crossing.
    fx = radial_log(abs(x[0]), FLOOR, anisotropy)
    fy = radial_log(abs(y[0]), FLOOR, anisotropy)
    radial = fx + fy if x[0] * y[0] < 0 else abs(fx - fy)
    return max(radial, displacement)


def straight_length(x: Point, y: Point, anisotropy: float, panels: int) -> float:
    dr, dy = y[0] - x[0], y[1] - x[1]

    def path(t: float) -> tuple[float, float, float]:
        return x[0] + t * dr, dr, dy

    return path_length(path, 1.0, FLOOR, 1.0 + anisotropy**2, panels=panels)


def check_on_shore() -> None:
    print("span,one_ended,two_ended,excursion_arc_a0,excursion_arc_a1,excursion_arc_a3")
    for span in (0.0, 0.01, 0.5, pi, 3.2, 4.0, 6.0, 12.0, 32.0):
        start, end = (0.0, 0.0), (0.0, span)
        one, two = one_ended(start, end), two_ended(start, end)
        assert one <= two + SLACK
        references = []
        for a in (0.0, 1.0, 3.0):
            length = span
            if span > pi:
                b_sq = 1.0 + a**2
                height = excursion_height(span - pi, b_sq, PANELS)
                length = excursion_arc(height, b_sq, PANELS)
                refined_height = excursion_height(span - pi, b_sq, PANELS * 2)
                refined = excursion_arc(refined_height, b_sq, PANELS * 2)
                assert abs(length - refined) < 5e-8 and length < span
            assert two <= length + SLACK
            references.append(length)
        print(",".join(f"{value:.11g}" for value in (span, one, two, *references)))


def check_mixed_paths() -> None:
    generator = Random(1742026)
    stronger = 0
    comparisons = 0
    min_slack = float("inf")
    incompatible = 0
    for _ in range(192):
        start, end, via = [
            (generator.uniform(-4, 4), generator.uniform(-8, 8)) for _ in range(3)
        ]
        direct = two_ended(start, end)
        assert direct <= two_ended(start, via) + two_ended(via, end) + SLACK
        assert direct == two_ended(end, start)
        d_lower = 0.9 * dist(start, end)
        r_upper = hypot(FLOOR, start[0]) + generator.uniform(0, 2)
        s_upper = hypot(FLOOR, end[0]) + generator.uniform(0, 2)
        incompatible += int(abs(r_upper - s_upper) > d_lower)
        assert compatible_bound(r_upper, s_upper, d_lower) <= direct + SLACK
        for a in (0.0, 1.0, 3.0):
            old = full_bound(start, end, a, one_ended(start, end))
            new = full_bound(start, end, a, direct)
            coarse = straight_length(start, end, a, PANELS)
            fine = straight_length(start, end, a, 2 * PANELS)
            assert abs(coarse - fine) < 1e-7
            assert old <= new + SLACK and new <= fine + SLACK
            stronger += int(new > old + SLACK)
            comparisons += 1
            min_slack = min(min_slack, fine - new)
    print(f"{stronger}/{comparisons} full bounds strengthened; minimum straight-path gap {min_slack:.8g}")
    assert incompatible > 0
    print(f"{incompatible} incompatible enclosures used the one-ended fallback")


def check_finite_graph() -> None:
    # Axial edges have the exact analytic costs in the flat metric: along
    # Shore, use dy/rho; across normal depth, subtract the signed primitive.
    step = 0.25
    normal_count, tangent_count = 49, 97
    sizes = normal_count, tangent_count
    starts_and_goals = (
        ((24, 2), (24, 48)), ((24, 2), (24, 92)), ((24, 48), (24, 50)),
        ((21, 75), (21, 24)), ((39, 23), (42, 77)), ((31, 8), (31, 88)),
        ((15, 8), (32, 81)), ((37, 81), (19, 3)), ((37, 34), (5, 35)),
        ((41, 7), (32, 12)), ((13, 32), (9, 65)), ((25, 5), (26, 4)),
    )

    def point(node: tuple[int, int]) -> Point:
        return step * (node[0] - normal_count // 2), step * node[1]

    nodes = [(i, j) for i in range(normal_count) for j in range(tangent_count)]
    coordinates = {node: point(node) for node in nodes}
    totals = dict.fromkeys(("dijkstra", "one-ended", "two-ended"), 0)
    relative_to_old = dict.fromkeys(("fewer", "same", "more"), 0)

    for a in (0.0, 1.0, 3.0):
        profiles = [copysign(radial_log(abs(point((i, 0))[0]), FLOOR, a), i - normal_count // 2)
                    for i in range(normal_count)]
        edges = {}
        for i, j in nodes:
            edges[i, j] = [
                (other, abs(profiles[other[0]] - profiles[i]) if other[0] != i
                 else step / hypot(FLOOR, coordinates[i, j][0]))
                for other in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                if all(0 <= index < size for index, size in zip(other, sizes, strict=True))
            ]

        for start, goal in starts_and_goals:
            by_method = {}
            for method in ("dijkstra", "one-ended", "two-ended"):
                heuristic = {}
                for node, coordinate in coordinates.items():
                    h = 0.0
                    if method != "dijkstra":
                        displacement = one_ended if method == "one-ended" else two_ended
                        h = max(abs(profiles[node[0]] - profiles[goal[0]]),
                                displacement(coordinate, coordinates[goal]))
                    heuristic[node] = h
                assert heuristic[goal] == 0
                assert all(heuristic[node] <= cost + heuristic[other] + SLACK
                           for node in nodes for other, cost in edges[node])
                result = search_grid(start, goal, edges, heuristic)
                by_method[method] = result
                totals[method] += result[1]
            reference = by_method["dijkstra"][0]
            assert all(abs(cost - reference) < SLACK for cost, _ in by_method.values())
            one, two = by_method["one-ended"][1], by_method["two-ended"][1]
            relative_to_old["fewer" if two < one else "same" if two == one else "more"] += 1
    print(f"graph expansions across 36 searches: {totals}")
    print(f"two-ended versus one-ended expansions: {relative_to_old}")


def search_grid(
    start: tuple[int, int],
    goal: tuple[int, int],
    edges: dict[tuple[int, int], list[tuple[tuple[int, int], float]]],
    heuristic: dict[tuple[int, int], float],
) -> tuple[float, int]:
    queue = [(heuristic[start], 0.0, start)]
    best = {start: 0.0}
    expanded = 0
    while queue:
        _, cost, node = heappop(queue)
        if cost != best[node]:
            continue
        expanded += 1
        if node == goal:
            return cost, expanded
        for other, edge in edges[node]:
            value = cost + edge
            if value < best.get(other, inf):
                best[other] = value
                heappush(queue, (value + heuristic[other], value, other))
    raise AssertionError("The rectangular test graph must be connected")


if __name__ == "__main__":
    check_on_shore()
    check_mixed_paths()
    check_finite_graph()
