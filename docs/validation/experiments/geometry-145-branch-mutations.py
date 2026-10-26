from pathlib import Path
import subprocess,json,sys
root=Path("docs/validation/experiments/geometry-145-point-proof")
p=root/"src/validated_branch.rs"
original=p.read_bytes()
cases=[
 ("self-inclusion", ".filter(|_| included && contracting)", ".filter(|_| true && contracting)", 1, "insufficient_quadratic_remainder_refuses"),
 ("contraction", ".filter(|_| included && contracting)", ".filter(|_| included && true)", 1, "self_inclusion_without_contraction_refuses"),
 ("operation-roundoff", "Some((center, abs(&C { re, im })))", "Some((center, B::new(P)))", 2, "center_roundoff_must_not_disappear"),
 ("quadratic-remainder", "&self\n                    .perturbation(radii)\n                    .mul(&b.perturbation(radii), P, R::Up)", "&B::new(P)", 1, "insufficient_quadratic_remainder_refuses"),
 ("remainder-transport", "abs(&self.center)\n            .mul(&b.remainder, P, R::Up)", "B::new(P)", 1, "multiplying_a_remainder_disk_encloses_its_boundary"),
 ("glue-parameter-domain", "if !contains(&old.model, parameter)? || !contains(&next.model, parameter)?", "if false", 1, "adjacent_cardioid_patches_glue_and_outside_parameters_refuse"),
 ("glue-c-identity", "if !finite(&c_margin) || !c_margin.cmp(&B::new(P)).is_some_and(|v| v > 0)", "if false", 1, "distinct_period_three_components_do_not_glue"),
 ("glue-cycle-phase", "if finite(&z_margin) && z_margin.cmp(&B::new(P)).is_some_and(|v| v > 0)", "if true", 1, "cycle_phase_is_checked_when_gluing_period_two"),
 ("segment-attraction", "if !bound.cmp(&B::from(1u8)).is_some_and(|v| v < 0)", "if false", 1, "attracting_segment_requires_both_domain_and_multiplier_guards"),
 ("segment-domain", "if !contains(&patch.model, start).ok_or(SegmentRefusal::NumericalFailure)?\n        || !contains(&patch.model, end).ok_or(SegmentRefusal::NumericalFailure)?", "if false", 1, "attracting_segment_requires_both_domain_and_multiplier_guards"),
]
names={case[0] for case in cases}
unknown=set(sys.argv[1:])-names
if unknown:raise SystemExit("unknown mutation name(s): "+", ".join(sorted(unknown)))
rows=[]
try:
 for name,old,new,count,test in cases:
  if sys.argv[1:] and name not in sys.argv[1:]:continue
  source=original.decode().replace("\r\n","\n");assert source.count(old)==count,(name,source.count(old))
  p.write_bytes(source.replace(old,new).encode())
  result=subprocess.run(["cargo","test","--release","--jobs","1","--manifest-path",str(root/"Cargo.toml"),"--target-dir","target/geometry145-precision","--bin","component_probe",test,"--","--nocapture"],capture_output=True,text=True,encoding="utf-8")
  Path("tmp/geometry-145-branch-mutation-"+name+".log").write_text(result.stdout+result.stderr,encoding="utf-8")
  rows.append({"mutation":name,"exit_code":result.returncode,"caught":result.returncode!=0 and "test result: FAILED" in result.stdout})
  p.write_bytes(original)
finally:p.write_bytes(original)
Path("tmp/geometry-145-branch-mutations.json").write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
assert rows, "no mutation cases were selected"
assert all(row["caught"] for row in rows)
