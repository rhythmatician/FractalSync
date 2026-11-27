"""Compare compiled retained prepared-field jets with the exact period-two disk oracle."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "docs/validation/bounded-reference-policy.json"
POLICY_SHA = "7931b4d06e7d1aab6e60ef35b065c5594b00bad216ccc4146dee55925d9dd1cd"
ASSET_SHA = {
    "fine": "bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb",
    "coarse": "3bb6bbaca940bc98bbd4d735702d79209cda7bc862559a321491dbb990a62a63",
}

RUST_PROBE = r"""
use runtime_core::manifold::ManifoldConfig;
use runtime_core::operational_engine::{reference_policy_for_field, FrozenRollout, Integrator, PhysicsProfile};
use runtime_core::operational_field::PreparedField;
use serde_json::json;
use std::{io::{self, BufRead}, sync::Arc};

fn main() {
    let mut paths = std::env::args_os().skip(1);
    let bytes = std::fs::read(paths.next().unwrap()).unwrap();
    let field = Arc::new(PreparedField::from_asset_bytes(&bytes).unwrap());
    assert_eq!(field.asset_bytes(), bytes);
    let rollout = paths.next().map(|path| {
        let policy = std::fs::read_to_string(path).unwrap();
        let cuts = reference_policy_for_field(&field, &policy).unwrap();
        let profile = PhysicsProfile::new(ManifoldConfig::default(), 1.0 / 60.0, 0.0,
            Integrator::Rk4FourSubsteps).unwrap();
        FrozenRollout::new(Arc::clone(&field), profile, cuts).unwrap()
    });
    println!("{:016x}", field.identity());
    for line in io::stdin().lock().lines() {
        let values = line.unwrap().split_whitespace().map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
            .collect::<Vec<_>>();
        let point = [values[0], values[1]];
        let jet = field.query(point).ok().map(|jet| json!({"value": jet.value, "gradient": jet.gradient, "hessian": jet.hessian}));
        let probe = rollout.as_ref().map(|r| match r.probe(point) {
            Ok(value) => {
                assert_eq!(value["reference_distance_accuracy"].as_str(), Some("unknown"));
                assert_eq!(value["operational_value"].as_f64().unwrap().to_bits(),
                    jet.as_ref().unwrap()["value"].as_f64().unwrap().to_bits());
                json!({"status": "accepted", "distance_fidelity": "unknown"})
            },
            Err(error) => json!({"status": "refused", "error": error}),
        });
        println!("{}", json!({"jet": jet, "frozen_probe": probe}));
    }
}
"""


def points() -> dict[str, list[tuple[float, float]]]:
    directions = [
        (Fraction(1), Fraction(0)),
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(-1)),
        (Fraction(3, 5), Fraction(4, 5)),
        (Fraction(3, 5), Fraction(-4, 5)),
        (Fraction(-3, 5), Fraction(4, 5)),
        (Fraction(-3, 5), Fraction(-4, 5)),
    ]

    def dyadic(value: Fraction) -> float:
        return round(value * 2**30) / 2**30

    bands = {}
    for power in (4, 6, 9, 11, 13, 15, 17):
        radial = Fraction(1, 4) - Fraction(1, 2**power)
        bands[f"depth_2^-{power}"] = [
            (dyadic(-1 + radial * x), dyadic(radial * y)) for x, y in directions
        ]
    bands["near_halo"] = [(-1 - 2**-10, 0.0), (-1 - 2**-9, 0.0)]
    bands["center_cut"] = [(-1.0, 0.0)]
    return bands


def reference(
    point: tuple[float, float],
) -> tuple[Decimal, list[Decimal] | None, list[list[Decimal]] | None]:
    x, y = map(Decimal.from_float, point)
    x += 1
    radius = (x * x + y * y).sqrt()
    assert radius < Decimal(1) / 4
    distance = radius - Decimal(1) / 4
    if radius == 0:
        return distance, None, None
    nx, ny = x / radius, y / radius
    return (
        distance,
        [nx, ny],
        [[ny * ny / radius, -nx * ny / radius], [-nx * ny / radius, nx * nx / radius]],
    )


def compare(point: tuple[float, float], row: dict[str, object]) -> dict[str, object]:
    with localcontext() as context:
        context.prec = 70
        distance, gradient, hessian = reference(point)
        errors = {}
        jet = row["jet"]
        if jet is not None:
            assert isinstance(jet, dict)
            value = Decimal.from_float(jet["value"])
            errors["D_abs"] = float(abs(value - distance))
            errors["D_rel"] = float(abs((value - distance) / distance))
            if gradient is not None and hessian is not None:
                gradient_delta = [
                    Decimal.from_float(x) - y
                    for x, y in zip(jet["gradient"], gradient, strict=True)
                ]
                hessian_delta = [
                    Decimal.from_float(jet["hessian"][i][j]) - hessian[i][j]
                    for i in range(2)
                    for j in range(2)
                ]
                errors["gradient_l2"] = float(
                    sum((delta**2 for delta in gradient_delta), Decimal(0)).sqrt()
                )
                errors["hessian_frobenius"] = float(
                    sum((delta**2 for delta in hessian_delta), Decimal(0)).sqrt()
                )
        probe = row["frozen_probe"]
        return {
            "point": point,
            "point_bits": [struct.pack(">d", x).hex() for x in point],
            "reference": {
                "D": str(distance),
                "gradient": None if gradient is None else list(map(str, gradient)),
                "hessian": None
                if hessian is None
                else [list(map(str, v)) for v in hessian],
            },
            "errors": errors,
            "raw_status": "accepted" if jet is not None else "unavailable",
            "frozen_status": probe["status"]
            if isinstance(probe, dict)
            else "no archived policy",
            **row,
        }


def prepared_header(path: Path, expected: str) -> dict[str, object]:
    contents = path.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == expected, (
        f"Unexpected retained asset digest: {path.name}"
    )
    magic, ox, oy, spacing, nx, ny, mask = struct.unpack_from("<8s3d3Q", contents)
    assert magic == b"FSOP0001" and mask == 0 and len(contents) == 56 + 8 * nx * ny
    return {
        "origin": [ox, oy],
        "spacing": spacing,
        "shape": [nx, ny],
        "mask_bytes": mask,
    }


def main() -> None:
    if not __debug__:
        raise RuntimeError("The retained field check requires Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fine", type=Path)
    parser.add_argument("coarse", type=Path)
    args = parser.parse_args()
    policy_digest = hashlib.sha256(
        POLICY.read_text(encoding="utf-8").encode()
    ).hexdigest()
    assert policy_digest == POLICY_SHA, (
        "Frozen policy SHA-256 differs from the archived replay policy"
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    bands = points()
    inputs = "".join(
        " ".join(struct.pack(">d", component).hex() for component in point) + "\n"
        for group in bands.values()
        for point in group
    )

    with tempfile.TemporaryDirectory(prefix="fractalsync-period-two-") as temporary:
        directory = Path(temporary)
        source = directory / "src"
        source.mkdir()
        manifest = directory / "Cargo.toml"
        manifest.write_text(
            '[package]\nname = "check_retained_period_two"\nversion = "0.0.0"\nedition = "2021"\n'
            f"[dependencies]\nruntime_core = {{ path = {json.dumps((ROOT / 'runtime-core').as_posix())}, "
            'features = ["experimental-operational-field"] }\nserde_json = "1"\n',
            encoding="utf-8",
        )
        (source / "main.rs").write_text(RUST_PROBE, encoding="utf-8")
        assets = {}
        jets: dict[str, list[object]] = {}
        for name, path in (("fine", args.fine), ("coarse", args.coarse)):
            header = prepared_header(path, ASSET_SHA[name])
            optional_policy = [str(POLICY)] if name == "fine" else []
            command = [
                "cargo",
                "run",
                "--offline",
                "--quiet",
                "--release",
                "--target-dir",
                str(ROOT / "target"),
                "--manifest-path",
                str(manifest),
                "--",
                str(path.resolve()),
                *optional_policy,
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                input=inputs,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            )
            identity, *rows = result.stdout.splitlines()
            assert len(rows) == sum(map(len, bands.values()))
            raw = [json.loads(row) for row in rows]
            jets[name] = [row["jet"] for row in raw]
            raw_queries = iter(raw)
            band_report = {}
            for band, group in bands.items():
                sample_rows = [compare(point, next(raw_queries)) for point in group]
                maxima: dict[str, float] = {}
                statuses: Counter[str] = Counter()
                for row in sample_rows:
                    statuses[str(row["raw_status"])] += 1
                    errors = row["errors"]
                    assert isinstance(errors, dict)
                    for field, error in errors.items():
                        maxima[field] = max(maxima.get(field, 0.0), error)
                band_report[band] = {
                    "raw_status_counts": dict(statuses),
                    "max_observed_error": maxima,
                    "rows": sample_rows,
                }
            assets[name] = {
                "sha256": ASSET_SHA[name],
                "field_id": identity,
                "header": header,
                "bands": band_report,
            }
    print(
        json.dumps(
            {
                "schema": "retained-period-two-probe/1",
                "assets": assets,
                "fine_coarse_raw_jets_equal": jets["fine"] == jets["coarse"],
                "source_commit": source_commit,
                "reference_policy_sha256": policy_digest,
                "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
                "reference_decimal_precision": 70,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
