"""Independently check emitted f64 root domains with the existing 128-bit proof.

The local f64 checker emits exact represented root centers, multipliers and
requested radii. This compiles the existing multiprecision implementation with
one extra test for those domains; it substitutes no proof arithmetic.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

class Options(argparse.Namespace):
    input: Path
    output: Path
    target_dir: Path

parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--target-dir',type=Path,required=True)
args=parser.parse_args(namespace=Options())
root=Path(__file__).resolve().parents[3]
record=json.loads(args.input.read_text(encoding='utf-8'))
rows=[json.loads(line.split('] ',1)[1]) for line in record['baseline'].splitlines()
      if line.startswith('[F64_BRANCH_ROW145]')]
assert len(rows)==162 and all(len(row)==21 for row in rows)
directory=root/'docs/validation/experiments/geometry-145-point-proof/src'
source=(directory/'component_probe.rs').read_text(encoding='utf-8')
for name in ['complex_ball','point_proof','proof_support','quintic_field','validated_branch']:
    source=source.replace(f'mod {name};',f'#[path="{(directory/(name+".rs")).as_posix()}"] mod {name};')
literal=',\n'.join('['+','.join(repr(float(v)) for v in row)+']' for row in rows)
source+='''
fn f64_rows()->[[f64;21];162] {[ROWS]}
fn fixed_preconditioner(z:&C,c:&C,mu:&C,period:usize)->[[C;2];2] {
    let row=f64_rows().into_iter().find(|row|
        z.f64()==[row[0],row[1]] && c.f64()==[row[2],row[3]] &&
        mu.f64()==[row[4],row[5]] && period==row[6] as usize).unwrap();
    [[C::new(row[10],row[11]),C::new(row[12],row[13])],
     [C::new(row[14],row[15]),C::new(row[16],row[17])]]
}
#[test]
fn independent_f64_branch_domains() {
    for (index,row) in f64_rows().into_iter().enumerate() {
        let proof=validated_branch::certify(&C::new(row[0],row[1]),&C::new(row[2],row[3]),
            &C::new(row[4],row[5]),row[6] as usize,row[7],row[8],row[9]);
        assert!(proof.is_ok(),"independent root domain {index}: {proof:?}");
    }
}
'''.replace('ROWS',literal)
deps=args.target_dir.resolve()/'release/deps'
library=next(deps.glob('libastro_float-*.rlib'))
with tempfile.TemporaryDirectory(prefix='geometry145-branch-hp-') as name:
    tmp=Path(name)
    path=tmp/'main.rs'
    exe=tmp/'check.exe'
    branch=(directory/'validated_branch.rs').read_text(encoding='utf-8')
    start=branch.index('        let inverse = [')
    end=branch.index('        if !inverse.iter()',start)
    branch=branch[:start]+'        let inverse = crate::fixed_preconditioner(z,c,mu,period);\n'+branch[end:]
    branch+='''
#[test]
fn f64_error_bounds_enclose_independent_multiprecision_bounds() {
    for (index,row) in crate::f64_rows().into_iter().enumerate() {
        let proof=certify(&C::new(row[0],row[1]),&C::new(row[2],row[3]),
            &C::new(row[4],row[5]),row[6] as usize,row[7],row[8],row[9]).unwrap();
        for axis in 0..2 {
            assert!(proof.model.inclusion[axis].cmp(&B::from_f64(row[18+axis],P))
                .is_some_and(|order|order<=0),"f64 error bound {index} axis {axis}");
        }
        assert!(proof.contraction<=row[20],"f64 contraction bound {index}");
    }
}
'''
    branch=branch.replace('#[path = "validated_cover.rs"]',f'#[path = "{(directory/"validated_cover.rs").as_posix()}"]')
    branch_path=tmp/'validated_branch.rs'
    branch_path.write_text(branch,encoding='utf-8')
    source=source.replace((directory/'validated_branch.rs').as_posix(),branch_path.as_posix())
    path.write_text(source,encoding='utf-8')
    build=subprocess.run(['rustc','--test','--edition=2021','-O',str(path),'-L',f'dependency={deps}',
        '--extern',f'astro_float={library}','-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe),'f64_','--nocapture'],capture_output=True,text=True)
    output={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'rows':len(rows),
            'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'astro_float_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in directory.glob('*.rs')}}
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(run.stdout)
    assert run.returncode==0,run.stderr
