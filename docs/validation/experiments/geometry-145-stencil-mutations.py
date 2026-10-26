"""Protect full-stencil identity, proposal recovery and uncertainty propagation."""
from pathlib import Path
import json
import subprocess

root = Path("docs/validation/experiments/geometry-145-point-proof")
sources = {name: (root / "src" / name).read_bytes()
           for name in ["component_probe.rs", "quintic_field.rs"]}
cases = [
    ("stencil-origin", "component_probe.rs", "NodeLayout::Quintic => index.floor()", "NodeLayout::Quintic => index.round()", "quintic_grid_matches_the_entire_canonical_stencil", 1),
    ("stencil-size", "component_probe.rs", "NodeLayout::Quintic => 4", "NodeLayout::Quintic => 1", "quintic_grid_matches_the_entire_canonical_stencil", 1),
    ("multistart", "component_probe.rs", "let angle = theta + offset;", "let angle = theta;", "bounded_multistart_recovers_a_local_step_refusal_but_keeps_the_cut", 1),
    ("sample-query", "quintic_field.rs", "|| bracket.query().re.cmp(&node.re) != Some(0)", "", "sample_binding_and_declared_error_are_independent_of_witness_bounds", 1),
    ("sample-accuracy", "quintic_field.rs", "!bracket.certifies_value(value, error)", "false", "sample_binding_and_declared_error_are_independent_of_witness_bounds", 1),
    ("declared-window", "quintic_field.rs", "hi: candidate.add(&error, P, R::Up)", "hi: bracket.upper().clone()", "sample_binding_and_declared_error_are_independent_of_witness_bounds", 1),
    ("shared-value", "quintic_field.rs", "previous.value != sample.value", "false", "conflicting_shared_assignments_and_missing_nodes_refuse", 1),
    ("shared-error", "quintic_field.rs", "previous.bounds.hi.cmp(&sample.bounds.hi) != Some(0)", "false", "conflicting_shared_assignments_and_missing_nodes_refuse", 1),
    ("prefilter-sign", "quintic_field.rs", ".scale(-0.25)", ".scale(0.25)", "reproduces_quadratic_values_and_derivatives", 2),
    ("derivative-factor", "quintic_field.rs", "(0..derivative).fold(1, |product, i| product * (power - i) as i32)", "(0..derivative).fold(0, |product, i| product * (power - i) as i32)", "reproduces_quadratic_values_and_derivatives", 1),
    ("spatial-scaling", "quintic_field.rs", ".powi(dx + dy, P, R::ToEven)", ".powi(0, P, R::ToEven)", "reproduces_quadratic_values_and_derivatives", 1),
    ("negative-interval", "quintic_field.rs", "if k >= 0.", "if true", "nodal_uncertainty_reaches_every_jet_component", 1),
    ("nodal-propagation", "quintic_field.rs", "bounds = bounds.add(&weight.mul(&self.coefficients[y * 6 + x].1));", "bounds = bounds.add(&weight.mul(&self.coefficients[y * 6 + x].0));", "arbitrary_nodal_values_are_enclosed_against_independent_512_bit_reconstruction", 1),
]
rows = []
try:
    for name, filename, old, new, test, count in cases:
        source = sources[filename].decode().replace("\r\n", "\n")
        assert source.count(old) == count, (name, source.count(old))
        (root / "src" / filename).write_bytes(source.replace(old, new).encode())
        result = subprocess.run(
            ["cargo", "test", "--release", "--jobs", "1", "--manifest-path", str(root / "Cargo.toml"),
             "--target-dir", "target/geometry145-precision", "--bin", "component_probe", test, "--", "--nocapture"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        Path(f"tmp/geometry-145-stencil-mutation-{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        rows.append({"mutation": name, "exit_code": result.returncode,
                     "caught": result.returncode != 0 and "test result: FAILED" in result.stdout})
        (root / "src" / filename).write_bytes(sources[filename])
finally:
    for filename, original in sources.items():
        (root / "src" / filename).write_bytes(original)
Path("tmp/geometry-145-stencil-mutations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows))
assert rows and all(row["caught"] for row in rows)
