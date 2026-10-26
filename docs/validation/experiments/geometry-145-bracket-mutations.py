"""Require the independent outside proof and directed query transport."""
from pathlib import Path
import json
import subprocess

root = Path("docs/validation/experiments/geometry-145-point-proof")
sources = {name:(root/"src"/name).read_bytes() for name in ["distance_bracket.rs","complex_ball.rs"]}
cases = [
    ("outside-proof", "distance_bracket.rs", "let proof = escape(point, precision, budget)?;", "let proof = escape([1.,0.],128,4)?;", "absence_of_escape_is_not_an_exterior_witness"),
    ("transport-distance", "distance_bracket.rs", "let lower = self.lower.sub(&travel, P, R::Down);", "let lower = self.lower.clone();", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("travel-rounding", "distance_bracket.rs", "let travel = query.distance_upper(&self.origin, P)?;", "let travel = Complex{re:query.re.sub(&self.origin.re,P,R::Down),im:query.im.sub(&self.origin.im,P,R::Down)}.norm_squared(P,R::Down).sqrt(P,R::Down);", "diagonal_transport_rounds_travel_outward_near_clearance_boundary"),
    ("subtraction-rounding", "distance_bracket.rs", "let lower = self.lower.sub(&travel, P, R::Down);", "let lower = self.lower.sub(&travel, P, R::Up);", "tiny_diagonal_transport_rounds_clearance_subtraction_downward"),
    ("transport-interior", "distance_bracket.rs", "if !lower.cmp(&B::new(P)).is_some_and(|v| v > 0)", "if false", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("query-binding", "distance_bracket.rs", "let upper = query.distance_upper(&self.outside, P)?;", "let upper = self.origin.distance_upper(&self.outside, P)?;", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("upper-rounding", "complex_ball.rs", ".norm_upper(p);\n        (!result.is_nan()", ".norm_squared(p, R::Down).sqrt(p, R::Down);\n        (!result.is_nan()", "upper_distance_is_outward_for_irrational_endpoint_norm"),
]
rows = []
try:
    for name, filename, old, new, test in cases:
        source = sources[filename].decode().replace("\r\n","\n")
        assert source.count(old) == 1, (name, source.count(old))
        (root/"src"/filename).write_bytes(source.replace(old,new,1).encode())
        result = subprocess.run(
            ["cargo","test","--release","--jobs","1","--manifest-path",str(root/"Cargo.toml"),
             "--target-dir","target/geometry145-precision","--bin","component_probe",test,"--","--nocapture"],
            capture_output=True,text=True,encoding="utf-8",timeout=120,
        )
        Path(f"tmp/geometry-145-bracket-mutation-{name}.log").write_text(result.stdout+result.stderr,encoding="utf-8")
        rows.append({"mutation":name,"exit_code":result.returncode,
                     "caught":result.returncode!=0 and "test result: FAILED" in result.stdout})
        (root/"src"/filename).write_bytes(sources[filename])
finally:
    for filename, original in sources.items():
        (root/"src"/filename).write_bytes(original)
Path("tmp/geometry-145-bracket-mutations.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(json.dumps(rows))
assert rows and all(row["caught"] for row in rows)
