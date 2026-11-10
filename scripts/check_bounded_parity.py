"""Compare the frozen synthetic transaction across native Python and WASM."""

import json
import math
import subprocess
import sys


def main() -> None:
    native = json.loads(
        subprocess.check_output(
            [sys.executable, "scripts/check_bounded_python.py"], text=True
        )
    )
    wasm = json.loads(
        subprocess.check_output(
            [
                "node",
                "scripts/check_bounded_wasm.cjs",
                "tmp/final-wasm/orbit_synth_wasm.js",
            ],
            text=True,
        )
    )
    assert native["field_id"] == wasm["field_id"]
    assert native["physics_profile_id"] == wasm["physics_profile_id"]
    errors = []
    for key in ("position", "velocity"):
        errors.extend(
            abs(a - b)
            for a, b in zip(native["state"][key], wasm["state"][key], strict=True)
        )
    for key in ("mechanical_time", "integrated_work", "energy"):
        errors.append(abs(native["state"][key] - wasm["state"][key]))
    assert all(math.isfinite(error) and error <= 1e-9 for error in errors)
    print(
        json.dumps(
            {
                "passed": True,
                "tolerance": 1e-9,
                "maximum_absolute_error": max(errors),
                "field_id": native["field_id"],
                "physics_profile_id": native["physics_profile_id"],
            }
        )
    )


if __name__ == "__main__":
    main()
