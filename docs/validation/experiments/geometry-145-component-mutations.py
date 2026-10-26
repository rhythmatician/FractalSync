from pathlib import Path
import subprocess,json
root=Path("docs/validation/experiments/geometry-145-point-proof")
cases=[
 ("trap-parameter-radius","point_proof.rs","bound = next_radius.add(parameter_radius, p, R::Up);","bound = next_radius;","geometry145_precision_probe","trap_must_include_parameter_uncertainty"),
 ("affine-parameter-radius","affine_box_proof.rs","let radius = d.norm_upper(p).mul(&cr, p, R::Up).add(&remainder, p, R::Up);","let radius = remainder.clone();","geometry145_precision_probe","boxes_near_known_interior_and_exterior"),
 ("quadratic-parameter-radius","quadratic_box_proof.rs","let radius = linear.add(&nonlinear, p, R::Up);","let radius = remainder.clone();","geometry145_precision_probe","boxes_near_known_interior_and_exterior"),
 ("curve-multiplier-second-derivative","component_probe.rs","let rhs_g = mu.scale(-1.).sub(&m2);","let rhs_g = mu.scale(0.).sub(&m2);","component_probe","cardioid_curve_derivatives_match_closed_form"),
 ("curve-hessian-sign","component_probe.rs","let factor = -k / (1. - k * d);","let factor = k / (1. - k * d);","component_probe","period_two_distance_jet_matches_circle"),
 ("curve-interior-orientation","component_probe.rs","|| inward_alignment <= 0.","|| false","component_probe","period_two_cut_locus_and_exterior_refuse"),
 ("screen-escape-cutoff","component_probe.rs","if x * x + y * y > 4.","if x * x + y * y >= 4.","component_probe","numerical_screen_preserves_pre_update_cutoff_without_proof_authority"),
 ("screen-point-binding","component_probe.rs","let witness = OutsideWitness::certify(point, precision, 32768);","let witness = OutsideWitness::certify([1.,0.], precision, 32768);","component_probe","fabricated_numerical_escape_metadata_cannot_issue_a_witness"),
]
original={root/"src"/case[1]:(root/"src"/case[1]).read_bytes() for case in cases}
rows=[]
try:
 for name,file,old,new,bin,test in cases:
  p=root/"src"/file;source=original[p].decode().replace("\r\n","\n");assert source.count(old)==1,(name,source.count(old))
  p.write_bytes(source.replace(old,new,1).encode())
  result=subprocess.run(["cargo","test","--release","--jobs","1","--manifest-path",str(root/"Cargo.toml"),"--target-dir","target/geometry145-precision","--bin",bin,test,"--","--nocapture"],capture_output=True,text=True,encoding="utf-8")
  Path("tmp/geometry-145-component-mutation-"+name+".log").write_text(result.stdout+result.stderr,encoding="utf-8")
  rows.append({"mutation":name,"exit_code":result.returncode,"caught":result.returncode!=0 and "test result: FAILED" in result.stdout})
  p.write_bytes(original[p])
finally:
 for p,data in original.items():p.write_bytes(data)
print(json.dumps(rows))
Path("tmp/geometry-145-component-mutations.json").write_text(json.dumps(rows,indent=2))
assert all(row["caught"] for row in rows)
