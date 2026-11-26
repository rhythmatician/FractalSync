"""Check the retained operational asset's single-cell arithmetic certificate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
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
    rounded_q = Bound.rounded(q_hi * spacing, Q(0)).div(Bound.exact(spacing), spacing)
    assert 2 <= cell and cell < q_lo - rounded_q.error
    assert q_hi + rounded_q.error < cell + 1
    assert cell + 1 <= 2 * cell  # Sterbenz: subtracting the exact cell index is exact.
    return Bound(q_hi - cell, rounded_q.error)


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


def exact_jet(point: tuple[Q, Q], origin: tuple[Q, Q], spacing: Q, coefficients: list[Q]) -> list[Q]:
    """Independent rational oracle using differentiated monomials."""
    tx, ty = ((p - o) / spacing - i for p, o, i in zip(point, origin, CELL, strict=True))

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


def main() -> None:
    if not __debug__:
        raise RuntimeError("Verification requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--offline", action="store_true", help="Build the compiled probe with cached dependencies")
    args = parser.parse_args()
    asset_path = args.asset.resolve()
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
    assert 2 <= CELL[0] < nx - 3 and 2 <= CELL[1] < ny - 3
    origin, spacing = (Q(ox), Q(oy)), Q(h)
    assert spacing > 0
    coefficients = [
        Q(struct.unpack_from("<d", asset, 56 + 8 * (y * nx + x))[0])
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
    tolerance = Q(1, 10**8)
    assert max(normalized_error, gradient_error, SCALE * hessian_error) < tolerance

    points = [(CENTER[0] + i * SCALE / 4, CENTER[1] + j * SCALE / 4) for j in range(-4, 5) for i in range(-4, 5)]
    maxima = [Q(0) for _ in DERIVATIVES]
    for point, compiled in zip(points, compiled_queries(asset_path, points, args.offline), strict=True):
        for component, (actual, exact, bound) in enumerate(zip(compiled, exact_jet(point, origin, spacing, coefficients), bounds, strict=True)):
            error = abs(actual - exact)
            assert error <= bound.error, (point, component, error, bound.error)
            maxima[component] = max(maxima[component], error)
    print("Regional coordinate errors:", *(f"{float(c.error):.9e}" for c in coordinates))
    print("Regional component errors (value, dx, dy, dxx, dxy, dyy):", *(f"{float(b.error):.9e}" for b in bounds))
    print("Regional normalized value, gradient and scaled Hessian errors:", *(f"{float(e):.9e}" for e in (normalized_error, gradient_error, SCALE * hessian_error)))
    print(f"{len(points)} release-mode Rust queries, observed component errors:", *(f"{float(e):.9e}" for e in maxima))


if __name__ == "__main__":
    main()
