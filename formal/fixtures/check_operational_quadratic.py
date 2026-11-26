"""Materialize/check the existing runtime quadratic fixture's binary64 values.

This is reproducible extraction evidence, not an IEEE or Rust compiler proof.
The source operation tree is evaluated with separately rounded, non-fused
binary64 multiplication, addition, and subtraction. Every Python arithmetic
result is independently checked using exact Fraction arithmetic rounded to
nearest, ties to even. No compiler-contraction or FMA equivalence is claimed.
Lean checks the literal dyadic table and its analytic certificates.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runtime-core/src/operational_field.rs"
DATA = ROOT / "formal/fixtures/operational-quadratic.json"
LEAN = ROOT / "formal/FractalSync/ConcreteOperationalAsset.lean"
BEGIN = "-- BEGIN GENERATED QUADRATIC NUMERATORS"
END = "-- END GENERATED QUADRATIC NUMERATORS"
EXPECTED = """    fn field() -> PreparedField {
        let mut c = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                let (x, y) = (x as f64 * 0.1, y as f64 * 0.1);
                c.push(x * x + 2.0 * x * y - 3.0 * y * y);
            }
        }
        PreparedField::new([0.0; 2], 0.1, [16; 2], c).unwrap()
    }"""


class FixtureData(TypedDict):
    schema: str
    source_commit: str
    source: str
    source_function_sha256: str
    scope: str
    arithmetic_model: str
    shape: list[int]
    origin: list[int]
    spacing_binary64: str
    spacing_rational: list[int]
    coefficient_denominator: int
    coefficient_numerators_row_major: list[int]
    coefficient_binary64_row_major: list[str]
    coefficient_magnitude_bound: int
    quadratic_sample_error_bound: list[int]
    observed_max_sample_error: list[int]


def pow2(exponent: int) -> Fraction:
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)


def nearest_binary64(value: Fraction) -> Fraction:
    if value == 0:
        return Fraction(0)
    sign = 1 if value > 0 else -1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < pow2(exponent):
        exponent -= 1
    assert -1022 <= exponent <= 1023, "fixture uses only finite normal nonzero results"
    scaled = value / pow2(exponent - 52)
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
        whole += 1
    return sign * whole * pow2(exponent - 52)


def mul(x: float, y: float) -> float:
    result = x * y
    assert Fraction.from_float(result) == nearest_binary64(Fraction.from_float(x) * Fraction.from_float(y))
    return result


def add(x: float, y: float) -> float:
    result = x + y
    assert Fraction.from_float(result) == nearest_binary64(Fraction.from_float(x) + Fraction.from_float(y))
    return result


def sub(x: float, y: float) -> float:
    result = x - y
    assert Fraction.from_float(result) == nearest_binary64(Fraction.from_float(x) - Fraction.from_float(y))
    return result


def bits(value: float) -> str:
    return f"0x{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def build() -> tuple[FixtureData, str]:
    source = SOURCE.read_text()
    first = source.index("    fn field() -> PreparedField {")
    last = source.index("\n    #[test]", first)
    fragment = source[first:last]
    assert fragment == EXPECTED, "runtime fixture expression changed; review required"
    spacing = 0.1
    assert Fraction.from_float(spacing) == nearest_binary64(Fraction(1, 10))
    h = Fraction.from_float(spacing)
    values: list[float] = []
    errors: list[Fraction] = []
    for j in range(16):
        for i in range(16):
            x, y = mul(float(i), spacing), mul(float(j), spacing)
            value = sub(add(mul(x, x), mul(mul(2.0, x), y)), mul(mul(3.0, y), y))
            values.append(value)
            model = (i * h) ** 2 + 2 * (i * h) * (j * h) - 3 * (j * h) ** 2
            errors.append(abs(Fraction.from_float(value) - model))
    rational_values = [Fraction.from_float(value) for value in values]
    denominator = max(value.denominator for value in rational_values)
    numerators = [int(value * denominator) for value in rational_values]
    assert all(Fraction(n, denominator) == value for n, value in zip(numerators, rational_values))
    assert max(abs(v) for v in rational_values) <= 7
    assert max(errors) <= Fraction(1, 2**48)
    data: FixtureData = {
        "schema": "fractalsync-exact-operational-fixture/1",
        "source_commit": "57a1650fe5a2d8ce6254434252ecc0ce2e24cdd2",
        "source": "runtime-core/src/operational_field.rs::tests::field",
        "source_function_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
        "scope": "exact-value model of existing quadratic test fixture; not production or Mandelbrot fidelity; no Rust execution or IEEE kernel proof",
        "arithmetic_model": "separately rounded non-fused binary64 mul/add/sub; round to nearest, ties to even; no compiler-contraction or FMA equivalence claim",
        "shape": [16, 16],
        "origin": [0, 0],
        "spacing_binary64": bits(spacing),
        "spacing_rational": [h.numerator, h.denominator],
        "coefficient_denominator": denominator,
        "coefficient_numerators_row_major": numerators,
        "coefficient_binary64_row_major": [bits(value) for value in values],
        "coefficient_magnitude_bound": 7,
        "quadratic_sample_error_bound": [1, 2**48],
        "observed_max_sample_error": [max(errors).numerator, max(errors).denominator],
    }
    rows = ["    ![" + ", ".join(str(n) for n in numerators[j*16:(j+1)*16]) + "]" for j in range(16)]
    block = BEGIN + "\ndef coefficientNumerators : Fin 16 → Fin 16 → ℤ :=\n  ![\n" + ",\n".join(rows) + "\n  ]\n" + END
    return data, block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data, block = build()
    rendered = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.write:
        DATA.write_text(rendered)
        if LEAN.exists():
            lean = LEAN.read_text()
            before, rest = lean.split(BEGIN, 1)
            _, after = rest.split(END, 1)
            LEAN.write_text(before + block + after)
    else:
        assert DATA.read_text() == rendered, "fixture data differs from reproducible extraction"
        assert block in LEAN.read_text(), "Lean literal table differs from extracted dyadic values"
    print(json.dumps({"coefficients": 256, "coefficient_denominator": data["coefficient_denominator"],
        "magnitude_bound": 7, "sample_error_bound": "2^-48", "exact_rounding_crosscheck": True,
        "rust_execution_claimed": False, "write": args.write}))


if __name__ == "__main__":
    main()
