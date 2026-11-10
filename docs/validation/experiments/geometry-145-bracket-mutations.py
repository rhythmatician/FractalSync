"""Require the independent outside proof and directed query transport."""
from pathlib import Path
import json
import subprocess

root = Path("docs/validation/experiments/geometry-145-point-proof")
sources = {name:(root/"src"/name).read_bytes() for name in ["distance_bracket.rs","complex_ball.rs","directed_norm.rs","validated_cover.rs"]}
cases = [
    ("candidate-lower", "distance_bracket.rs", "self.lower.cmp(&candidate_low).is_some_and(|v| v >= 0)", "true", "candidate_error_requires_both_endpoints"),
    ("candidate-upper", "distance_bracket.rs", "self.upper.cmp(&candidate_high).is_some_and(|v| v <= 0)", "true", "candidate_error_requires_both_endpoints"),
    ("dyadic-lower-bin", "distance_bracket.rs", "!self.lower.cmp(&low).is_some_and(|v| v >= 0)", "false", "dyadic_sample_is_canonical_and_refuses_bin_crossings"),
    ("dyadic-upper-bin", "distance_bracket.rs", "!self.upper.cmp(&high).is_some_and(|v| v < 0)", "false", "dyadic_sample_is_canonical_and_refuses_bin_crossings"),
    ("outside-proof", "distance_bracket.rs", "let proof = escape(point, precision, budget)?;", "let proof = escape([1.,0.],128,4)?;", "absence_of_escape_is_not_an_exterior_witness"),
    ("transport-distance", "distance_bracket.rs", "let lower = self.lower.sub(&travel, P, R::Down);", "let lower = self.lower.clone();", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("travel-rounding", "distance_bracket.rs", "let travel = query.distance_upper(&self.origin, P)?;", "let travel = Complex{re:query.re.sub(&self.origin.re,P,R::Down),im:query.im.sub(&self.origin.im,P,R::Down)}.norm_squared(P,R::Down).sqrt(P,R::Down);", "diagonal_transport_rounds_travel_outward_near_clearance_boundary"),
    ("subtraction-rounding", "distance_bracket.rs", "let lower = self.lower.sub(&travel, P, R::Down);", "let lower = self.lower.sub(&travel, P, R::Up);", "tiny_diagonal_transport_rounds_clearance_subtraction_downward"),
    ("transport-interior", "distance_bracket.rs", "if !lower.cmp(&B::new(P)).is_some_and(|v| v > 0)", "if false", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("query-binding", "distance_bracket.rs", "let upper = query.distance_upper(&self.outside, P)?;", "let upper = self.origin.distance_upper(&self.outside, P)?;", "compact_domain_transports_bounds_and_refuses_its_boundary"),
    ("upper-rounding", "complex_ball.rs", ".norm_upper_precise(p);\n        (!result.is_nan()", ".norm_squared(p, R::Down).sqrt(p, R::Down);\n        (!result.is_nan()", "upper_distance_is_outward_for_irrational_endpoint_norm"),
    ("mantissa-tail", "directed_norm.rs", "Some(truncated.next_up())", "Some(truncated)", "mantissa_tail_and_both_word_layouts_are_enclosed"),
    ("sqrt-rounding", "directed_norm.rs", "value.sqrt().next_up()", "value.sqrt()", "square_root_bound_is_outward"),
    ("cover-query-binding", "validated_cover.rs", "curve_clearance(&self.edges, query)", "curve_clearance(&self.edges, &self.query)", "reused_cover_accepts_queries_outside_original_transport_disk"),
    ("requested-width", "distance_bracket.rs", ".cmp(&B::from_f64(requested, P))\n                .is_some_and(|v| v <= 0)", ".cmp(&B::from_f64(requested, P))\n                .is_some_and(|_| true)", "independently_certified_endpoints_enclose_known_cardioid_distance"),
    ("requested-finite-width", "distance_bracket.rs", "requested.is_finite()", "true", "independently_certified_endpoints_enclose_known_cardioid_distance"),
    ("final-width", "distance_bracket.rs", "self.meets_width(requested).then_some(self)", "Some(self)", "independently_certified_endpoints_enclose_known_cardioid_distance"),
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
