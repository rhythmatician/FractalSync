"""Check finite critical-orbit return and trap-selection controls for #185."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from check_exterior_ruler_jets import ComplexInterval, RealInterval, advance, intervals

WINDOW = 12
RETURN_SQUARED = "0.00000001"
PERIOD_THREE_LO = "-1.754877666246692760050"
PERIOD_THREE_HI = "-1.754877666246692760049"
TRANSIENT = "0.2501"


def interval(lo: str, hi: str) -> RealInterval:
    return intervals.mpf([intervals.mpf(lo).a, intervals.mpf(hi).b])


def abs_squared(z: ComplexInterval) -> RealInterval:
    return z.real**2 + z.imag**2


def center_polynomial(c: Fraction) -> Fraction:
    # P3(c)=c(c³+2c²+c+1). This independent scalar identity brackets
    # the one real period-three center used by the test.
    return c**3 + 2 * c**2 + c + 1


def check_center_enclosure() -> None:
    lo, hi = Fraction(PERIOD_THREE_LO), Fraction(PERIOD_THREE_HI)
    assert lo < hi < Fraction(-3, 2)
    assert center_polynomial(lo) < 0 < center_polynomial(hi)
    # The polynomial derivative is (3c+1)(c+1)>0 throughout the bracket.
    # Neither the period-one center 0 nor the period-two center -1 belongs.


def candidate_indices(scores: list[RealInterval]) -> tuple[int, ...]:
    # Eliminate an index only when some competing upper bound is
    # strictly below its lower bound. Equality and overlaps survive.
    return tuple(
        i for i, score in enumerate(scores, 1)
        if all(score.a <= other.b for other in scores)
    )


def first_small_return(orbit: list[ComplexInterval]) -> int | None:
    threshold = intervals.mpf(RETURN_SQUARED)
    for i, position in enumerate(orbit, 1):
        score = abs_squared(position)
        if score.b <= threshold.a:
            return i
        if not threshold.b <= score.a:
            raise AssertionError(f"return threshold unresolved at iterate {i}")
    return None


@dataclass(frozen=True)
class Trial:
    name: str
    parameter: ComplexInterval
    expected_first_return: int | None
    expected_escape: int | None


def check(fixture: Trial) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    zero = intervals.mpc(0)
    orbit_jet = (zero, zero, zero, zero)
    values: list[ComplexInterval] = []
    escape = None
    for n in range(1, WINDOW + 1):
        orbit_jet = advance(fixture.parameter, orbit_jet)
        z = orbit_jet[0]
        if abs(z).a >= 4 and 4 * abs(fixture.parameter).b <= abs_squared(z).a:
            escape = n
            break
        values.append(z)
    observed_return = first_small_return(values)
    assert observed_return == fixture.expected_first_return
    assert escape == fixture.expected_escape
    targets = [zero, intervals.mpc(-1), intervals.mpc("0.25", "0.125")]
    selections = [
        candidate_indices([abs_squared(z - target) for z in values])
        for target in targets
    ]
    # Every real orbit gives the same ambiguous minimum for the real-axis
    # line trap: the score (Im P_n)² vanishes for every candidate.
    if fixture.name != "exterior":
        assert candidate_indices([z.imag**2 for z in values]) == tuple(range(1, WINDOW + 1))
    print(f"{fixture.name} {len(values)} {observed_return} {escape} {selections}")
    return selections[0], selections[1], selections[2]


def main() -> None:
    if not __debug__:
        raise RuntimeError("Verification requires Python assertions")
    intervals.dps = 60
    check_center_enclosure()
    # For real c>1/4 each step increases by (z-1/2)^2+c-1/4.
    # This analytic exterior control can still have no 12-step guard.
    assert Fraction(TRANSIENT) > Fraction(1, 4)
    trials = [
        Trial("period-one center", intervals.mpc(0), 1, None),
        Trial("period-two center", intervals.mpc(-1), 2, None),
        Trial("period-three center", intervals.mpc(interval(PERIOD_THREE_LO, PERIOD_THREE_HI)), 3, None),
        Trial("near period-two center", intervals.mpc("-0.99999"), 2, None),
        Trial("finite near-cusp transient", intervals.mpc(TRANSIENT), None, None),
        Trial("exterior", intervals.mpc("0.5", "0.5"), None, 6),
    ]
    print("fixture prefix first-small-return escape-guard possible-closest-indices")
    p1, p2, p3, near, transient, exterior = map(check, trials)
    assert p1[0] == tuple(range(1, WINDOW + 1))
    assert p2[0] == (2, 4, 6, 8, 10, 12)
    assert p2[1] == (1, 3, 5, 7, 9, 11)
    assert p3[0] == (3, 6, 9, 12)
    assert near[0] != near[1] and len(near[0]) == 1
    assert transient[0] == (1,) and exterior[0] == (1,) and exterior[1] == (4,)


if __name__ == "__main__":
    main()
