"""Enclose center jets using escape bounds on two parameter disks."""
from __future__ import annotations

from math import factorial
from typing import Protocol, cast

from mpmath import iv


# mpmath installs its numeric operators dynamically.
class RealInterval(Protocol):
    @property
    def a(self) -> RealInterval: ...
    @property
    def b(self) -> RealInterval: ...
    def __add__(self, other: RealInterval) -> RealInterval: ...
    def __sub__(self, other: RealInterval) -> RealInterval: ...
    def __rsub__(self, other: int) -> RealInterval: ...
    def __mul__(self, other: RealInterval) -> RealInterval: ...
    def __rmul__(self, other: int) -> RealInterval: ...
    def __truediv__(self, other: RealInterval) -> RealInterval: ...
    def __pow__(self, exponent: int) -> RealInterval: ...
    def __neg__(self) -> RealInterval: ...
    def __float__(self) -> float: ...
    def __ge__(self, other: int) -> bool | None: ...
    def __gt__(self, other: int) -> bool | None: ...
    def __le__(self, other: RealInterval) -> bool | None: ...


class ComplexInterval(Protocol):
    @property
    def real(self) -> RealInterval: ...
    @property
    def imag(self) -> RealInterval: ...
    def __abs__(self) -> RealInterval: ...
    def __add__(self, other: ComplexInterval | int) -> ComplexInterval: ...
    def __sub__(self, other: ComplexInterval) -> ComplexInterval: ...
    def __mul__(self, other: ComplexInterval) -> ComplexInterval: ...
    def __rmul__(self, other: int) -> ComplexInterval: ...
    def __truediv__(self, other: ComplexInterval | RealInterval) -> ComplexInterval: ...
    def __pow__(self, exponent: int) -> ComplexInterval: ...


class IntervalContext(Protocol):
    dps: int

    def mpf(self, value: int | str | list[RealInterval]) -> RealInterval: ...
    def mpc(self, real: int | str | RealInterval, imag: str | RealInterval = "0") -> ComplexInterval: ...
    def ln(self, value: RealInterval) -> RealInterval: ...


intervals = cast(IntervalContext, iv)
OrbitJet = tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]


def advance(c: ComplexInterval, orbit: OrbitJet) -> OrbitJet:
    z, d, q, t = orbit
    return z * z + c, 2 * z * d + 1, 2 * z * q + 2 * d * d, 2 * z * t + 6 * d * q


def guarded_disk(
    c: ComplexInterval, h: RealInterval
) -> tuple[int, OrbitJet, RealInterval, RealInterval]:
    parameter_bound = (abs(c) + h).b
    zero = intervals.mpc(0)
    orbit = (zero, zero, zero, zero)
    error = intervals.mpf(0)
    for n in range(500):
        radius = (abs(orbit[0]) - error).a
        if radius >= 4 and 4 * parameter_bound <= radius**2:
            return n, orbit, parameter_bound, radius
        error = (2 * abs(orbit[0]) * error + error**2 + h).b
        orbit = advance(c, orbit)
    raise ArithmeticError("Disk guard unresolved within 500 iterates")


def symmetric(error: RealInterval) -> RealInterval:
    return intervals.mpf([-error.b, error.b])


def enclose_complex(value: ComplexInterval, error: RealInterval) -> ComplexInterval:
    spread = symmetric(error)
    return intervals.mpc(value.real + spread, value.imag + spread)


def ruler_jet(
    orbit: OrbitJet, tail: RealInterval, h: RealInterval
) -> tuple[RealInterval, ...] | None:
    z, d, q, t = orbit
    derivative_errors = [factorial(k) * tail / h**k for k in range(1, 4)]
    raw_u = d / z
    g = intervals.ln(abs(z)) + symmetric(tail)
    u = enclose_complex(raw_u, derivative_errors[0])
    v = enclose_complex(q / z - raw_u**2, derivative_errors[1])
    w = enclose_complex(t / z - 3 * q * d / z**2 + 2 * raw_u**3, derivative_errors[2])
    if not (g > 0 and abs(u) > 0):
        return None

    rho = g / abs(u)
    logarithmic_gradient = u / g - v / u
    gx, gy = logarithmic_gradient.real, -logarithmic_gradient.imag
    gradient_g_x, gradient_g_y = u.real / g, -u.imag / g
    harmonic_hessian = v / g - w / u + (v / u)**2
    return (
        rho, rho * gx, rho * gy,
        rho * (harmonic_hessian.real - gradient_g_x**2 + gx**2),
        rho * (-harmonic_hessian.imag - gradient_g_x * gradient_g_y + gx * gy),
        rho * (-harmonic_hessian.real - gradient_g_y**2 + gy**2),
    )


def width(values: tuple[RealInterval, ...]) -> float:
    return max(float((value.b - value.a).b) for value in values)


def main() -> None:
    intervals.dps = 60
    print("c radius stage value_width gradient_max_component_width hessian_max_component_width")
    for real, imag, h_text in (("0.5", "0.5", "1e-4"), ("0.2501", "0", "1e-8")):
        c, h = intervals.mpc(real, imag), intervals.mpf(h_text)
        n, orbit, parameter_bound, radius = guarded_disk(c, h)
        for extra in range(5):
            tail = 12 * parameter_bound / (17 * radius**2)
            result = ruler_jet(orbit, tail, h)
            if result is None:
                print(f"({real},{imag}) {h_text} {n + extra} unresolved")
            else:
                widths = (width(result[:1]), width(result[1:3]), width(result[3:]))
                print(f"({real},{imag}) {h_text} {n + extra}", *(f"{value:.5g}" for value in widths))
            radius = (radius**2 - parameter_bound).a
            orbit = advance(c, orbit)


if __name__ == "__main__":
    main()
