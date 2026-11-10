from pathlib import Path
import subprocess,json
root=Path('docs/validation/experiments/geometry-145-point-proof')
p=root/'src/point_proof.rs';original=p.read_bytes()
command=['cargo','test','--release','--jobs','1','--manifest-path',str(root/'Cargo.toml'),'--target-dir','target/geometry145-precision']
results=[]
try:
 for name,old,new,test in [
  ('inclusive-budget','for n in 0..budget {','for n in 0..=budget {','strict_escape_budget_and_radius'),
  ('nonstrict-radius','is_some_and(|order| order > 0)','is_some_and(|order| order >= 0)','strict_escape_budget_and_radius'),
  ('nonstrict-trap','.cmp(outer)\n        .is_some_and(|v| v < 0)','.cmp(outer)\n        .is_some_and(|v| v <= 0)','interior_requires_a_strict_trap'),
 ]:
  source=original.decode().replace("\r\n","\n");assert old in source
  p.write_bytes(source.replace(old,new,1).encode())
  result=subprocess.run(command+[test,'--','--nocapture'],capture_output=True,text=True,encoding='utf-8')
  Path(f'tmp/geometry-145-point-proof-mutation-{name}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
  results.append({'mutation':name,'exit_code':result.returncode,'caught':result.returncode!=0 and 'test result: FAILED' in result.stdout})
finally:p.write_bytes(original)
Path('tmp/geometry-145-point-proof-mutations.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))

assert all(row["caught"] for row in results), "a proof boundary mutation survived"
