"""Check closed-cover refusal guards; restore exact source bytes after each run."""
from pathlib import Path
import json
import subprocess

root = Path("docs/validation/experiments/geometry-145-point-proof")
path = root / "src/validated_cover.rs"
original = path.read_bytes()
cases = [
    ("closed-parameter-path", "if !equal(&a.end, &b.start)", "if false", "open_or_non_enclosing_paths_refuse"),
    ("cycle-joins", "join(&a.patch, &b.patch, &a.end).map_err(|_| Refusal::UnprovedJoin)?;", "", "closed_multiplier_path_cannot_join_distinct_components"),
    ("query-separation", "if !bound.cmp(&B::new(P)).is_some_and(|v| v > 0)", "if false", "intersecting_query_refuses_a_closed_certified_path"),
    ("connector-separation", "for capsule in [&a.capsule, &connector]", "for capsule in [&a.capsule]", "connector_separation_is_required_independently"),
    ("nonzero-winding", "if winding == 0", "if false", "open_or_non_enclosing_paths_refuse"),
    ("orientation", "Err(Refusal::AmbiguousOrientation)\n    }", "Ok(0)\n    }", "uncertain_ray_orientation_refuses"),
    ("region-budget", "if self.attempted >= self.max_regions", "if false", "exhausted_builder_does_not_return_an_accepted_prefix"),
]
rows = []
try:
    for name, old, new, test in cases:
        source = original.decode().replace("\r\n", "\n")
        assert source.count(old) == 1, (name, source.count(old))
        path.write_bytes(source.replace(old, new, 1).encode())
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
