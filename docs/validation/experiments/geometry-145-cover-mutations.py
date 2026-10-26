"""Check closed-cover refusal guards; restore exact source bytes after each run."""
from pathlib import Path
import json
import subprocess

root = Path("docs/validation/experiments/geometry-145-point-proof")
path = root / "src/validated_cover.rs"
original = path.read_bytes()
cases = [
    ("multi-query-refinement", "for (query, minimum) in &self.extra_targets", "for (query, minimum) in self.extra_targets.iter().take(0)", "multiple_queries_require_independent_refinement_and_final_targets"),
    ("multi-query-final-target", ".cmp(minimum)\n            .is_some_and(|v| v >= 0)", ".cmp(minimum)\n            .is_some_and(|_| true)", "multiple_queries_require_independent_refinement_and_final_targets"),
    ("closed-parameter-path", "if !equal(&a.end, &b.start)", "if false", "open_or_non_enclosing_paths_refuse"),
    ("cycle-joins", "join(&a.patch, &b.patch, &a.end).map_err(|_| Refusal::UnprovedJoin)?;", "", "closed_multiplier_path_cannot_join_distinct_components"),
    ("query-separation", "if !bound.cmp(&B::new(P)).is_some_and(|v| v > 0) {\n                return Err(Refusal::QueryNotSeparated);", "if false {\n                return Err(Refusal::QueryNotSeparated);", "intersecting_query_refuses_a_closed_certified_path"),
    ("connector-separation", "for capsule in [&a.capsule, &connector]", "for capsule in [&a.capsule]", "connector_separation_is_required_independently"),
    ("nonzero-winding", "if winding == 0", "if false", "open_or_non_enclosing_paths_refuse"),
    ("reused-connector-separation", "for capsule in [&a.capsule, &connector]", "for capsule in [&a.capsule]", "reused_query_requires_winding_and_connector_separation"),
    ("reused-nonzero-winding", "if winding == 0", "if false", "reused_query_requires_winding_and_connector_separation"),
    ("orientation", "Err(Refusal::AmbiguousOrientation)\n    }", "Ok(0)\n    }", "uncertain_ray_orientation_refuses"),
    ("region-budget", "if self.attempted >= self.max_regions", "if false", "exhausted_builder_does_not_return_an_accepted_prefix"),
    ("requested-refinement", ".cmp(&self.minimum_clearance)", ".cmp(&B::new(P))", "requested_clearance_is_proved_or_refused"),
    ("requested-finite-clearance", "|| !minimum_clearance.is_finite()", "", "requested_clearance_is_proved_or_refused"),
    ("point-bridge-construction", "Some(point)", "None", "requested_clearance_is_proved_or_refused"),
    ("point-bridge-left", "if connects(a) && connects(b)", "if true && connects(b)", "point_bridge_requires_both_independent_cycle_links"),
    ("point-bridge-right", "if connects(a) && connects(b)", "if connects(a) && true", "point_bridge_requires_both_independent_cycle_links"),
    ("final-clearance", ".cmp(minimum_clearance)\n        .is_some_and(|v| v >= 0)", ".cmp(minimum_clearance)\n        .is_some_and(|_| true)", "final_clearance_checks_the_closing_geometry_independently"),
]
rows = []
try:
    for name, old, new, test in cases:
        source = original.decode().replace("\r\n", "\n")
        expected = 2 if name == "requested-refinement" else 1
        assert source.count(old) == expected, (name, source.count(old))
        path.write_bytes(source.replace(old, new).encode())
        result = subprocess.run(
            ["cargo", "test", "--release", "--jobs", "1", "--manifest-path", str(root / "Cargo.toml"),
             "--target-dir", "target/geometry145-precision", "--bin", "component_probe", test, "--", "--nocapture"],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        Path(f"tmp/geometry-145-cover-mutation-{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        rows.append({"mutation": name, "exit_code": result.returncode,
                     "caught": result.returncode != 0 and "test result: FAILED" in result.stdout})
        path.write_bytes(original)
finally:
    path.write_bytes(original)
Path("tmp/geometry-145-cover-mutations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows))
assert rows and all(row["caught"] for row in rows)
