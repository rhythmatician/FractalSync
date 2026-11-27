"""Check the retained operational asset's single-cell and four-cell arithmetic certificates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
from math import floor, inf, nextafter
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
ASSET_SHA = "bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb"
SOURCE_SHA = "78cf1a7faae60f02c32d37c5397095ef2f08d01050963cefe3304b1666beb061"
FIELD_ID = "9721bb7647a0b8e6"
POLICY = ROOT / "docs/validation/bounded-reference-policy.json"
POLICY_SHA = "7931b4d06e7d1aab6e60ef35b065c5594b00bad216ccc4146dee55925d9dd1cd"
CELL = (2604, 1504)
CENTER = (Q(9839, 32768), Q(1647, 32768))
SCALE = Q(1, 16384)
U = Q(1, 2**53)
NORMALIZED_TOLERANCE = Q(1, 10**8)
TINY = Q(1, 2**80)  # Conservative replacement for the subnormal allowance 2^-1075.
FINITE_LIMIT = 2**1023
POLYNOMIALS = (
    (1, -5, 10, -10, 5, -1),
    (26, -50, 20, 20, -20, 5),
    (66, 0, -60, 0, 30, -10),
    (26, 50, 20, -20, -20, 10),
    (1, 5, 10, 10, 5, -5),
    (0, 0, 0, 0, 0, 1),
)
DERIVATIVES = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


@dataclass(frozen=True)
class Bound:
    """Bounds on exact magnitude and machine-versus-exact absolute error."""

    magnitude: Q
    error: Q = Q(0)

    @staticmethod
    def exact(value: Q | int) -> Bound:
        return Bound(abs(Q(value)))

    @staticmethod
    def rounded(magnitude: Q, prior: Q) -> Bound:
        assert magnitude + prior < FINITE_LIMIT
        return Bound(magnitude, prior + U * (magnitude + prior) + TINY)

    def add(self, other: Bound) -> Bound:
        return Bound.rounded(self.magnitude + other.magnitude, self.error + other.error)

    def mul(self, other: Bound) -> Bound:
        prior = self.error * (other.magnitude + other.error) + self.magnitude * other.error
        return Bound.rounded(self.magnitude * other.magnitude, prior)

    def div(self, other: Bound, exact_denominator: Q) -> Bound:
        # This denominator models a specified exact constant and its computed value.
        lower = abs(exact_denominator) - other.error
        assert lower > 0
        magnitude = self.magnitude / abs(exact_denominator)
        return Bound.rounded(magnitude, (self.error + magnitude * other.error) / lower)


def cell_coordinates(origin: Q, spacing: Q, center: Q, cell: int) -> Bound:
    q_lo = (center - SCALE - origin) / spacing
    q_hi = (center + SCALE - origin) / spacing
    rounded_q = grid_coordinate_bound(q_hi, spacing)
    assert 2 <= cell and cell < q_lo - rounded_q.error
    assert q_hi + rounded_q.error < cell + 1
    assert cell + 1 <= 2 * cell  # Sterbenz: subtracting the exact cell index is exact.
    return Bound(q_hi - cell, rounded_q.error)


def grid_coordinate_bound(upper: Q, spacing: Q) -> Bound:
    return Bound.rounded(upper * spacing, Q(0)).div(Bound.exact(spacing), spacing)


def basis_bound(t: Bound, polynomial: tuple[int, ...]) -> tuple[Bound, Bound, Bound]:
    v, d, dd = Bound.exact(polynomial[5]), Bound.exact(0), Bound.exact(0)
    for coefficient in reversed(polynomial[:5]):
        doubled = Bound(2 * d.magnitude, 2 * d.error)
        assert doubled.magnitude + doubled.error < FINITE_LIMIT
        dd = dd.mul(t).add(doubled)  # Binary64 doubling is exact in this range.
        d = d.mul(t).add(v)
        v = v.mul(t).add(Bound.exact(coefficient))
    denom = Q(120)
    divisor = Bound.exact(denom)
    return v.div(divisor, denom), d.div(divisor, denom), dd.div(divisor, denom)


def jet_bounds(coordinates: tuple[Bound, Bound], spacing: Q, coefficients: list[Q]) -> list[Bound]:
    wx, wy = ([basis_bound(t, p) for p in POLYNOMIALS] for t in coordinates)
    squared = Bound.exact(spacing).mul(Bound.exact(spacing))
    bounds: list[Bound] = []
    for a, b in DERIVATIVES:
        total = Bound.exact(0)
        for index, coefficient in enumerate(coefficients):
            y, x = divmod(index, 6)
            total = total.add(Bound.exact(coefficient).mul(wx[x][a]).mul(wy[y][b]))
        if a + b == 1:
            total = total.div(Bound.exact(spacing), spacing)
        if a + b == 2:
            total = total.div(squared, spacing**2)
        bounds.append(total)
    return bounds


def exact_jet(
    point: tuple[Q, Q],
    origin: tuple[Q, Q],
    spacing: Q,
    cell: tuple[int, int],
    coefficients: list[Q],
) -> list[Q]:
    """Independent rational oracle using differentiated monomials."""
    tx, ty = ((p - o) / spacing - i for p, o, i in zip(point, origin, cell, strict=True))
    assert 0 <= tx < 1 and 0 <= ty < 1

    def weight(t: Q, i: int, derivative: int) -> Q:
        total = Q(0)
        for power, coefficient in enumerate(POLYNOMIALS[i]):
            if power >= derivative:
                for factor in range(power - derivative + 1, power + 1):
                    coefficient *= factor
                total += coefficient * t ** (power - derivative)
        return total / 120

    return [
        sum((c * weight(tx, i % 6, a) * weight(ty, i // 6, b) for i, c in enumerate(coefficients)), Q(0))
        / spacing ** (a + b)
        for a, b in DERIVATIVES
    ]


def coefficient_at(asset: bytes, width: int, x: int, y: int) -> Q:
    return Q(struct.unpack_from("<d", asset, 56 + 8 * (y * width + x))[0])


def vertex_axis_samples(vertex: Q, radius: Q) -> list[Q]:
    # Move rounded patch endpoints inward if needed. Sample the nearest
    # representable vertex and increasingly wide nextafter neighborhoods.
    lo, hi = float(vertex - radius), float(vertex + radius)
    if Q(lo) < vertex - radius:
        lo = nextafter(lo, inf)
    if Q(hi) > vertex + radius:
        hi = nextafter(hi, -inf)
    middle = float(vertex)
    samples = [Q(lo), Q(middle), Q(hi)]
    for direction in (-inf, inf):
        value = middle
        for step in range(1, 33):
            value = nextafter(value, direction)
            if step in (1, 2, 4, 8, 16, 32):
                samples.append(Q(value))
    assert all(abs(p - vertex) <= radius for p in samples)
    return sorted(set(samples))


@dataclass(frozen=True)
class VertexPatch:
    origin: tuple[Q, Q]
    spacing: Q
    radius: Q
    points: list[tuple[Q, Q]]
    cell_coefficients: dict[tuple[int, int], list[Q]]
    coordinate_errors: tuple[Q, Q]
    maximum_coefficient: Q
    roundoff: list[Bound]
    shift_errors: tuple[Q, Q, Q]

    def check_samples(self, results: list[list[Q]]) -> tuple[list[Q], int, int]:
        maxima = [Q(0) for _ in DERIVATIVES]
        changes = 0
        vertex_hits = 0
        exact_cells: set[tuple[int, int]] = set()
        represented_cells: set[tuple[int, int]] = set()
        for point, compiled in zip(self.points, results, strict=True):
            x, y = point
            ox, oy = self.origin
            spacing = self.spacing
            exact_q = ((x - ox) / spacing, (y - oy) / spacing)
            machine_q = (
                Q((float(x) - float(ox)) / float(spacing)),
                Q((float(y) - float(oy)) / float(spacing)),
            )
            for exact, machine, error in zip(
                exact_q, machine_q, self.coordinate_errors, strict=True
            ):
                assert abs(exact - machine) <= error
            exact_cell = (floor(exact_q[0]), floor(exact_q[1]))
            represented_cell = (floor(machine_q[0]), floor(machine_q[1]))
            exact_cells.add(exact_cell)
            represented_cells.add(represented_cell)
            changes += exact_cell != represented_cell
            vertex_hits += machine_q == CELL
            exact = exact_jet(
                point, self.origin, spacing, exact_cell, self.cell_coefficients[exact_cell]
            )
            shifted = (ox + spacing * machine_q[0], oy + spacing * machine_q[1])
            shifted_exact = exact_jet(
                shifted, self.origin, spacing, represented_cell,
                self.cell_coefficients[represented_cell],
            )
            comparison = zip(DERIVATIVES, compiled, exact, shifted_exact, self.roundoff, strict=True)
            for i, ((a, b), actual, expected, shifted_expected, bound) in enumerate(comparison):
                shift = self.shift_errors[a + b]
                assert abs(actual - shifted_expected) <= bound.error, (point, i, "roundoff")
                assert abs(shifted_expected - expected) <= shift, (point, i, "coordinate shift")
                error = abs(actual - expected)
                assert error <= bound.error + shift
                maxima[i] = max(maxima[i], error)
        assert exact_cells == represented_cells == self.cell_coefficients.keys()
        assert changes > 0 and vertex_hits > 0
        return maxima, changes, vertex_hits

    def report(self, results: list[list[Q]]) -> None:
        maxima, changes, vertex_hits = self.check_samples(results)
        roundoff = self.roundoff
        total_error = (
            roundoff[0].error + self.shift_errors[0],
            max(roundoff[1].error, roundoff[2].error) + self.shift_errors[1],
            max(roundoff[3].error + roundoff[4].error, roundoff[4].error + roundoff[5].error)
            + self.shift_errors[2],
        )
        # Both anchor and evaluation are binary64 points in the patch.
        value = Bound(roundoff[0].magnitude + self.shift_errors[0], total_error[0])
        normalized_value = value.add(value).div(Bound.exact(self.radius), self.radius).error
        allocations = (normalized_value, total_error[1], self.radius * total_error[2])
        assert max(allocations) < NORMALIZED_TOLERANCE
        print("Vertex union coefficient maximum:", f"{float(self.maximum_coefficient):.9e}")
        print("Vertex normalized-coordinate error:", *(f"{float(e):.9e}" for e in self.coordinate_errors))
        print("Vertex physical-coordinate error (max norm):", f"{float(self.spacing * max(self.coordinate_errors)):.9e}")
        print("Vertex component roundoff at shifted point:", *(f"{float(b.error):.9e}" for b in roundoff))
        print("Vertex shift errors (value, gradient, Hessian):", *(f"{float(e):.9e}" for e in self.shift_errors))
        print("Vertex total errors (value, max gradient entry, max Hessian row sum):", *(f"{float(e):.9e}" for e in total_error))
        print("Vertex normalized value, gradient and scaled Hessian errors:", *(f"{float(e):.9e}" for e in allocations))
        print(f"{len(maxima)} jet components, {len(self.points)} vertex queries, "
              f"{changes} exact/machine floor differences, {vertex_hits} machine vertex hits")
        print("Vertex observed component errors:", *(f"{float(e):.9e}" for e in maxima))


def make_vertex_patch(
    asset: bytes, shape: tuple[int, int], origin: tuple[Q, Q], spacing: Q
) -> VertexPatch:
    cx, cy = CELL
    nx, ny = shape
    radius = spacing / 4
    vertex = (origin[0] + spacing * cx, origin[1] + spacing * cy)
    candidates = [(x, y) for y in (cy - 1, cy) for x in (cx - 1, cx)]
    dx = grid_coordinate_bound(cx + Q(1, 4), spacing).error
    dy = grid_coordinate_bound(cy + Q(1, 4), spacing).error
    for index, size, error in ((cx, nx, dx), (cy, ny, dy)):
        # The exact box and its rounded image floor into the same two
        # candidate cells. Each candidate has its entire six-node halo.
        assert 0 <= error < Q(3, 4)
        assert 2 <= index - 1 and index < size - 3
        assert 2 * index < 2**53
        for cell in (index - 1, index):
            assert cell + 1 <= 2 * cell  # Sterbenz for the local subtraction.
    union = {
        (x, y): coefficient_at(asset, nx, x, y)
        for y in range(cy - 3, cy + 4)
        for x in range(cx - 3, cx + 4)
    }
    assert len(union) == 49
    maximum = max(map(abs, union.values()))
    coefficients = {
        (x, y): [
            union[(i, j)]
            for j in range(y - 2, y + 4)
            for i in range(x - 2, x + 4)
        ]
        for x, y in candidates
    }
    cell_bounds = [
        jet_bounds((Bound.exact(1), Bound.exact(1)), spacing, stencil)
        for stencil in coefficients.values()
    ]
    roundoff = [
        Bound(max(b.magnitude for b in component), max(b.error for b in component))
        for component in zip(*cell_bounds, strict=True)
    ]
    displacement = spacing * max(dx, dy)
    shift_errors = (
        8 * maximum / (3 * spacing) * displacement,
        68 * maximum / (9 * spacing**2) * displacement,
        32 * maximum / spacing**3 * displacement,
    )
    xs, ys = (vertex_axis_samples(center, radius) for center in vertex)
    return VertexPatch(
        origin, spacing, radius, [(x, y) for y in ys for x in xs],
        coefficients, (dx, dy), maximum, roundoff, shift_errors,
    )


RUST_PROBE = r"""
use runtime_core::manifold::ManifoldConfig;
use runtime_core::operational_engine::{reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile};
use runtime_core::operational_field::PreparedField;
use std::{io::{self, BufRead}, sync::Arc};

fn main() {
    let mut paths = std::env::args_os().skip(1);
    let bytes = std::fs::read(paths.next().unwrap()).unwrap();
    let policy = std::fs::read_to_string(paths.next().unwrap()).unwrap();
    let field = Arc::new(PreparedField::from_asset_bytes(&bytes).unwrap());
    assert_eq!(field.asset_bytes(), bytes);
    println!("{:016x}", field.identity());
    let profile = PhysicsProfile::new(
        ManifoldConfig::default(), 1.0 / 60.0, 0.0, Integrator::Rk4FourSubsteps,
    ).unwrap();
    let cuts = reference_policy_for_field(&field, &policy).unwrap();
    let rollout = FrozenRollout::new(Arc::clone(&field), profile, cuts).unwrap();
    assert!(rollout.probe([-1.0, 0.0]).unwrap_err().contains("ReferenceSingularity"));
    for line in io::stdin().lock().lines() {
        let values = line.unwrap().split_whitespace().map(|v| {
            f64::from_bits(u64::from_str_radix(v, 16).unwrap())
        }).collect::<Vec<_>>();
        let point = [values[0], values[1]];
        let jet = field.query(point).unwrap();
        let probe = rollout.probe(point).unwrap();
        assert_eq!(probe["reference_distance_accuracy"].as_str(), Some("unknown"));
        assert_eq!(probe["operational_value"].as_f64().unwrap().to_bits(), jet.value.to_bits());
        assert_eq!(jet.hessian[1][0].to_bits(), jet.hessian[0][1].to_bits());
        for v in [jet.value, jet.gradient[0], jet.gradient[1],
                  jet.hessian[0][0], jet.hessian[0][1], jet.hessian[1][1]] {
            print!("{:016x} ", v.to_bits());
        }
        println!();
    }
}
"""


def compiled_queries(asset_path: Path, points: list[tuple[Q, Q]], offline: bool) -> list[list[Q]]:
    arguments = ["cargo", "run", "--quiet", "--release", "--target-dir", str(ROOT / "target")]
    if offline:
        arguments.append("--offline")
    inputs = "".join(" ".join(struct.pack(">d", float(p)).hex() for p in xy) + "\n" for xy in points)
    with TemporaryDirectory(prefix="fractalsync-patch-") as temporary:
        directory = Path(temporary)
        manifest = directory / "Cargo.toml"
        path = json.dumps((ROOT / "runtime-core").as_posix())
        manifest.write_text(
            '[package]\nname = "check_operational_patch"\nversion = "0.0.0"\nedition = "2021"\n'
            f'[dependencies]\nruntime_core = {{ path = {path}, features = ["experimental-operational-field"] }}\n',
            encoding="utf-8",
        )
        (directory / "src").mkdir()
        (directory / "src" / "main.rs").write_text(RUST_PROBE, encoding="utf-8")
        result = subprocess.run(
            [*arguments, "--manifest-path", str(manifest), "--", str(asset_path), str(POLICY)],
            input=inputs, text=True, stdout=subprocess.PIPE, check=True,
        )
    field_id, *rows = result.stdout.splitlines()
    assert field_id == FIELD_ID
    return [[Q(struct.unpack(">d", bytes.fromhex(word))[0]) for word in row.split()] for row in rows]


def retained_asset(asset_path: Path) -> tuple[bytes, tuple[int, int], tuple[Q, Q], Q]:
    source = (ROOT / "runtime-core/src/operational_field.rs").read_text(encoding="utf-8")
    fragment = source[source.index("    pub fn query(&self"):source.index("#[cfg(test)]")]
    assert sha256(fragment.encode()).hexdigest() == SOURCE_SHA, "review changed runtime operation graph"
    asset = asset_path.read_bytes()
    assert sha256(asset).hexdigest() == ASSET_SHA, "asset SHA-256 differs from the frozen replay asset"
    # Normalize the checkout's line endings before checking the text policy.
    assert sha256(POLICY.read_text(encoding="utf-8").encode()).hexdigest() == POLICY_SHA
    assert asset[:8] == b"FSOP0001"
    ox, oy, h, nx, ny, mask = struct.unpack_from("<3d3Q", asset, 8)
    assert mask == 0 and len(asset) == 56 + 8 * nx * ny
    origin, spacing = (Q(ox), Q(oy)), Q(h)
    assert spacing > 0
    return asset, (nx, ny), origin, spacing


def main() -> None:
    if not __debug__:
        raise RuntimeError("Verification requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--offline", action="store_true", help="Build the compiled probe with cached dependencies")
    args = parser.parse_args()
    asset_path = args.asset.resolve()
    asset, (nx, ny), origin, spacing = retained_asset(asset_path)
    assert 2 <= CELL[0] < nx - 3 and 2 <= CELL[1] < ny - 3
    coefficients = [
        coefficient_at(asset, nx, x, y)
        for y in range(CELL[1] - 2, CELL[1] + 4)
        for x in range(CELL[0] - 2, CELL[0] + 4)
    ]
    assert max(map(abs, coefficients)) < Q(19, 1000)
    coordinates = (
        cell_coordinates(origin[0], spacing, CENTER[0], CELL[0]),
        cell_coordinates(origin[1], spacing, CENTER[1], CELL[1]),
    )
    bounds = jet_bounds(coordinates, spacing, coefficients)
    value = bounds[0]
    # The same magnitude and error estimates apply to the anchor subtraction.
    normalized_error = value.add(value).div(Bound.exact(SCALE), SCALE).error
    gradient_error = max(bounds[1].error, bounds[2].error)
    hessian_error = max(bounds[3].error + bounds[4].error, bounds[4].error + bounds[5].error)
    assert max(normalized_error, gradient_error, SCALE * hessian_error) < NORMALIZED_TOLERANCE

    points = [(CENTER[0] + i * SCALE / 4, CENTER[1] + j * SCALE / 4) for j in range(-4, 5) for i in range(-4, 5)]
    maxima = [Q(0) for _ in DERIVATIVES]
    vertex_patch = make_vertex_patch(asset, (nx, ny), origin, spacing)
    results = compiled_queries(asset_path, points + vertex_patch.points, args.offline)
    for point, compiled in zip(points, results[:len(points)], strict=True):
        expected = exact_jet(point, origin, spacing, CELL, coefficients)
        for component, (actual, exact, bound) in enumerate(zip(compiled, expected, bounds, strict=True)):
            error = abs(actual - exact)
            assert error <= bound.error, (point, component, error, bound.error)
            maxima[component] = max(maxima[component], error)
    print("Regional coordinate errors:", *(f"{float(c.error):.9e}" for c in coordinates))
    print("Regional component errors (value, dx, dy, dxx, dxy, dyy):", *(f"{float(b.error):.9e}" for b in bounds))
    print("Regional normalized value, gradient and scaled Hessian errors:", *(f"{float(e):.9e}" for e in (normalized_error, gradient_error, SCALE * hessian_error)))
    print(f"{len(points)} release-mode Rust queries, observed component errors:", *(f"{float(e):.9e}" for e in maxima))
    vertex_patch.report(results[len(points):])


if __name__ == "__main__":
    main()
