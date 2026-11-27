"""Check actual finite RK traces against the exact quadratic fixture dynamics."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from fractions import Fraction as Q
from math import factorial, floor, isqrt
from pathlib import Path
from typing import TypedDict

from check_operational_patch import exact_jet
from check_operational_quadratic import DATA, FixtureData, nearest_binary64


SCALE = 2**100


@dataclass(frozen=True)
class Interval:
    """Closed real enclosure with integer endpoints in units of 2^-100."""

    lo: int
    hi: int

    @staticmethod
    def point(value: Q | int) -> Interval:
        scaled = Q(value) * SCALE
        return Interval(floor(scaled), -floor(-scaled))

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> Interval:
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: Interval) -> Interval:
        return self + -other

    def __mul__(self, other: Interval) -> Interval:
        products = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(products) // SCALE, -(-max(products) // SCALE))

    def __truediv__(self, other: Interval) -> Interval:
        assert other.lo > 0
        return self * Interval(SCALE**2 // other.hi, -(-SCALE**2 // other.lo))

    def __pow__(self, exponent: int) -> Interval:
        assert exponent >= 0
        result = Interval.point(1)
        for _ in range(exponent):
            result = result * self
        return result

    def sqrt(self) -> Interval:
        assert self.lo >= 0
        return Interval(isqrt(self.lo * SCALE), isqrt(self.hi * SCALE) + 1)

    def error(self, recorded: Q) -> Q:
        return max(abs(recorded - Q(self.lo, SCALE)), abs(recorded - Q(self.hi, SCALE)))


def pi_bounds() -> Interval:
    # Machin's identity uses two alternating atan series at arguments in (0, 1).
    def atan_bounds(denominator: int) -> tuple[Q, Q]:
        lower = sum(
            (Q((-1)**k, (2 * k + 1) * denominator**(2 * k + 1)) for k in range(24)),
            Q(0),
        )
        return lower, lower + Q(1, 49 * denominator**49)

    a, b = atan_bounds(5), atan_bounds(239)
    return Interval(Interval.point(16 * a[0] - 4 * b[1]).lo, Interval.point(16 * a[1] - 4 * b[0]).hi)


PI = pi_bounds()
ONE = Interval.point(1)
TWO = Interval.point(2)
Vector = tuple[Interval, Interval]
Phase = tuple[Q, Q, Q, Q]


def dot(a: Vector, b: Vector) -> Interval:
    return a[0] * b[0] + a[1] * b[1]


def wall_factor(position: Vector) -> Interval:
    s = Interval.point(Q(1, 4)) * dot(position, position)
    angle = PI / TWO * s**4
    assert 0 <= angle.lo <= angle.hi <= SCALE

    # Alternating Taylor bounds on [0, 1]. The last included terms are
    # negative, so the partial sums are lower bounds and the next term bounds
    # the remainder. Dependency widening in interval powers is harmless.
    def trig(first_power: int) -> Interval:
        lower = Interval.point(0)
        for term in range(6):
            exponent = first_power + 2 * term
            lower += Interval.point(Q((-1)**term, factorial(exponent))) * angle**exponent
        exponent = first_power + 12
        remainder = Interval.point(Q(1, factorial(exponent))) * angle**exponent
        return Interval(lower.lo, lower.hi + remainder.hi)

    sine, cosine = trig(1), trig(0)
    # The exact fixture uses mu = 1/pi, so mu*pi cancels.
    return -(s**3 * sine / cosine**2)


class Jet(TypedDict):
    rho: Interval
    grad: Vector
    hessian: tuple[Interval, Interval, Interval]


class Field:
    def __init__(self) -> None:
        fixture: FixtureData = json.loads(DATA.read_text(encoding="utf-8"))
        assert fixture["shape"] == [16, 16] and fixture["origin"] == [0, 0]
        self.spacing = Q(*fixture["spacing_rational"])
        self.coefficients = [
            Q(n, fixture["coefficient_denominator"])
            for n in fixture["coefficient_numerators_row_major"]
        ]
        assert len(self.coefficients) == 16**2
        assert self.spacing == binary64(int(fixture["spacing_binary64"], 16))
        assert self.coefficients == [
            binary64(int(bits, 16)) for bits in fixture["coefficient_binary64_row_major"]
        ]
        start = Q(1, 2) / self.spacing
        assert floor(start) == 4 and nearest_binary64(start) == 5

    def ruler_jet(self, x: Q, y: Q) -> Jet:
        ix, iy = floor(x / self.spacing), floor(y / self.spacing)
        assert 2 <= ix < 13 and 2 <= iy < 13
        stencil = [
            self.coefficients[16 * row + column]
            for row in range(iy - 2, iy + 4)
            for column in range(ix - 2, ix + 4)
        ]
        jet = exact_jet((x, y), (Q(0), Q(0)), self.spacing, (ix, iy), stencil)
        d, dx, dy, dxx, dxy, dyy = map(Interval.point, jet)
        rho = (ONE + d * d).sqrt()
        rho_cubed = rho**3
        return {
            "rho": rho,
            "grad": (d * dx / rho, d * dy / rho),
            "hessian": (
                d * dxx / rho + dx * dx / rho_cubed,
                d * dxy / rho + dx * dy / rho_cubed,
                d * dyy / rho + dy * dy / rho_cubed,
            ),
        }


def binary64(bits: int) -> Q:
    return Q(struct.unpack(">d", bits.to_bytes(8, "big"))[0])


def drive(jet: Jet) -> Vector:
    ux, uy = jet["grad"]
    along_x = ONE + ux * ux
    denominator = jet["rho"] * along_x.sqrt()
    # G e_x / sqrt(e_x^T G e_x), with exact unit drive and a^2 = 1.
    return along_x / denominator, ux * uy / denominator


def rhs(phase: Phase, jet: Jet, command: Vector) -> tuple[Interval, Interval, Interval]:
    rho, u = jet["rho"], jet["grad"]
    hxx, hxy, hyy = jet["hessian"]
    x, y, vx, vy = map(Interval.point, phase)
    v = (vx, vy)
    uv = dot(u, v)
    w = ONE + dot(u, u)
    rho_squared = rho**2
    applied = (
        command[0] - (vx + uv * u[0]) / rho_squared,
        command[1] - (vy + uv * u[1]) / rho_squared,
    )
    wall = wall_factor((x, y))
    # Exact kappa = ln(2) cancels the ln(2) in -kappa grad(sigma).
    force = (applied[0] + u[0] / rho + wall * x, applied[1] + u[1] / rho + wall * y)
    projection = dot(u, force) / w
    vhv = hxx * vx * vx + TWO * hxy * vx * vy + hyy * vy * vy
    curvature = vhv / w + (dot(v, v) + uv * uv) / (rho * w)
    connection_x = curvature * u[0] - TWO * uv * vx / rho
    connection_y = curvature * u[1] - TWO * uv * vy / rho
    return (
        rho_squared * (force[0] - projection * u[0]) - connection_x,
        rho_squared * (force[1] - projection * u[1]) - connection_y,
        dot(v, applied),
    )


class Substep(TypedDict):
    dt: int
    stages: list[list[int]]
    slopes: list[list[int]]
    end: list[int]
    work: int


class OuterStep(TypedDict):
    command: list[int]
    substeps: list[Substep]
    stored_work: int


class Profile(TypedDict):
    substeps: int
    steps: list[OuterStep]


def phase_of(values: list[int]) -> Phase:
    x, y, vx, vy = map(binary64, values)
    return x, y, vx, vy


def phase_error(observed: Phase, expected: tuple[Q, ...]) -> Q:
    x, y, vx, vy = (abs(a - b) for a, b in zip(observed, expected, strict=True))
    # The 1-norm in each plane upper bounds the Euclidean plane norm.
    return max(x + y, vx + vy)


def check_profile(profile: Profile, field: Field) -> None:
    count = profile["substeps"]
    outer_dt = Q(1, 576 * 16)
    dt = outer_dt / count
    source = (Q(1, 2), Q(1, 2), Q(1, 4), Q(0))
    previous_work = Q(0)
    maxima: dict[str, Q] = {}

    def check(name: str, error: Q, limit: Q) -> None:
        assert error <= limit, (count, name, error, limit)
        maxima[name] = max(maxima.get(name, Q(0)), error)

    assert len(profile["steps"]) == 16
    for outer in profile["steps"]:
        exact_command = drive(field.ruler_jet(source[0], source[1]))
        command_x, command_y = map(binary64, outer["command"])
        command_error = exact_command[0].error(command_x) + exact_command[1].error(command_y)
        check("held drive", command_error, dt)
        exact_outer_work = Q(0)
        assert len(outer["substeps"]) == count
        for sub in outer["substeps"]:
            assert binary64(sub["dt"]) == nearest_binary64(dt)
            stages = [phase_of(stage) for stage in sub["stages"]]
            slopes = [tuple(map(binary64, slope)) for slope in sub["slopes"]]
            assert len(stages) == len(slopes) == 4 and stages[0] == source
            for stage, slope in zip(stages, slopes, strict=True):
                x, y, vx, vy = stage
                assert slope[:2] == (vx, vy)
                ax, ay, power = rhs(stage, field.ruler_jet(x, y), exact_command)
                check("RHS", ax.error(slope[2]) + ay.error(slope[3]), dt)
                check("power", power.error(slope[4]), dt)
            for factor, actual, prior in zip((dt / 2, dt / 2, dt), stages[1:], slopes[:3], strict=True):
                expected = tuple(source[i] + factor * prior[i] for i in range(4))
                check("stage construction", phase_error(actual, expected), dt**2)
            weighted = tuple(dt / 6 * (a + 2 * b + 2 * c + d) for a, b, c, d in zip(*slopes, strict=True))
            endpoint = phase_of(sub["end"])
            expected_end = tuple(source[i] + weighted[i] for i in range(4))
            check("weighted endpoint", phase_error(endpoint, expected_end), dt**2)
            check("substep work", abs(binary64(sub["work"]) - weighted[4]), dt**2)
            exact_outer_work += weighted[4]
            source = endpoint
        stored_work = binary64(outer["stored_work"])
        check("stored outer work", abs(stored_work - previous_work - exact_outer_work), outer_dt**2)
        previous_work = stored_work

    print(f"{count} substep(s), {16 * count} accepted updates, bounds rounded up to multiples of 1e-18")
    for name, error in maxima.items():
        print(f"  {name}: {-floor(-error * 10**18)}e-18")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    profiles: list[Profile] = json.loads(args.trace.read_text(encoding="utf-8"))
    assert [profile["substeps"] for profile in profiles] == [1, 4]
    field = Field()
    for profile in profiles:
        check_profile(profile, field)


if __name__ == "__main__":
    main()
