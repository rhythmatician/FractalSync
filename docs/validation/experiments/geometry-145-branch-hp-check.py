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
cover_rows=[]
targets=[]
scalars=[]
queries=[]
for line in record['baseline'].splitlines():
    if line.startswith('[F64_COVER_ROW145]'):
        _,group,values=line.split(' ',2)
        row=json.loads(values)
        assert len(row)==25
        cover_rows.append(row+[float(group)])
    elif line.startswith('[F64_COVER_TARGET145]'):
        _,group,values=line.split(' ',2)
        target=json.loads(values)
        assert len(target)==5
        targets.append(target+[float(group)])
    elif line.startswith('[MP_SCALAR145]'):
        _,_,values=line.split(' ',2)
        row=json.loads(values)
        assert len(row)==10
        scalars.append(row)
    elif line.startswith('[CERTIFIED_QUERY145]'):
        _,name,values=line.split(' ',2)
        row=json.loads(values)
        assert len(row)==14
        assert name in ('name=native','name=python')
        queries.append((row,0. if name=='name=native' else 1.))
assert len(queries) in (0,2)
assert not queries or ({group for _,group in queries}=={0.,1.} and len({(row[0],row[1]) for row,_ in queries})==2)
for row,group in queries:
    scalars.append(row[:10])
    targets.append([row[0],row[1],row[4],row[4],row[13],group])
assert not cover_rows or (len(targets)==162+len(queries) and {int(row[25]) for row in cover_rows}=={0,1})
if scalars:
    assert len(scalars)==162+len(queries) and len(targets)==162+len(queries)
    lower_by_query={(row[0],row[1]):row[3] for row in targets}
    assert all(row[4]==lower_by_query[(row[0],row[1])] for row in scalars)
rows.extend(row[:21] for row in cover_rows)
directory=root/'docs/validation/experiments/geometry-145-point-proof/src'
source=(directory/'component_probe.rs').read_text(encoding='utf-8')
for name in ['complex_ball','point_proof','proof_support','quintic_field','validated_branch']:
    source=source.replace(f'mod {name};',f'#[path="{(directory/(name+".rs")).as_posix()}"] mod {name};')
literal=',\n'.join('['+','.join(repr(float(v)) for v in row)+']' for row in rows)
source+='''
static F64_ROWS:[[f64;21];COUNT]=[ROWS];
fn f64_rows()->&'static [[f64;21]] {&F64_ROWS}
fn fixed_preconditioner(z:&C,c:&C,mu:&C,period:usize)->Option<[[C;2];2]> {
    let (z,c,mu)=(z.f64(),c.f64(),mu.f64());
    let row=f64_rows().iter().find(|row|
        z==[row[0],row[1]] && c==[row[2],row[3]] &&
        mu==[row[4],row[5]] && period==row[6] as usize)?;
    Some([[C::new(row[10],row[11]),C::new(row[12],row[13])],
     [C::new(row[14],row[15]),C::new(row[16],row[17])]])
}
#[test]
fn independent_f64_branch_domains() {
    for (index,row) in f64_rows().iter().take(162).enumerate() {
        let proof=validated_branch::certify(&C::new(row[0],row[1]),&C::new(row[2],row[3]),
            &C::new(row[4],row[5]),row[6] as usize,row[7],row[8],row[9]);
        assert!(proof.is_ok(),"independent root domain {index}: {proof:?}");
    }
}
'''.replace('COUNT',str(len(rows))).replace('ROWS]',literal+']')
def rust_rows(values:list[list[float]])->str:
    return ',\n'.join('['+','.join(repr(float(v)) for v in row)+']' for row in values)
source+=f'\nstatic COVER_ROWS:[[f64;26];{len(cover_rows)}]=[{rust_rows(cover_rows)}];\n'
source+=f'\nstatic COVER_TARGETS:[[f64;6];{len(targets)}]=[{rust_rows(targets)}];\n'
source+=f'\nstatic SCALAR_ROWS:[[f64;10];{len(scalars)}]=[{rust_rows(scalars)}];\n'
if scalars:
    source+='''
#[test]
fn independent_f64_exterior_witnesses_and_complete_windows() {
    use complex_ball::Complex;
    for (index,row) in SCALAR_ROWS.iter().enumerate() {
        let proof=point_proof::escape([row[6],row[7]],512,32768).unwrap();
        assert!(proof.update<32768,"exterior budget {index}");
        let query=Complex{re:BigFloat::from_f64(row[0],512),im:BigFloat::from_f64(row[1],512)};
        let point=Complex{re:BigFloat::from_f64(row[6],512),im:BigFloat::from_f64(row[7],512)};
        let upper=query.distance_upper(&point,512).unwrap();
        assert!(upper.cmp(&BigFloat::from_f64(row[5],512)).is_some_and(|v|v<=0),"endpoint upper {index}");
        let value=BigFloat::from_f64(-row[2],512);let error=BigFloat::from_f64(row[3],512);
        assert!(BigFloat::from_f64(row[4],512).cmp(&value.sub(&error,512,R::Up)).is_some_and(|v|v>=0));
        assert!(BigFloat::from_f64(row[5],512).cmp(&value.add(&error,512,R::Down)).is_some_and(|v|v<=0));
    }
    eprintln!("[HP_SCALAR145] witnesses={} precision=512 complete_windows=true",SCALAR_ROWS.len());
}
'''
deps=args.target_dir.resolve()/'release/deps'
library=next(deps.glob('libastro_float-*.rlib'))
with tempfile.TemporaryDirectory(prefix='geometry145-branch-hp-') as name:
    tmp=Path(name)
    path=tmp/'main.rs'
    exe=tmp/'check.exe'
    branch=(directory/'validated_branch.rs').read_text(encoding='utf-8')
    start=branch.index('        let inverse = [')
    end=branch.index('        if !inverse.iter()',start)
    original_inverse=branch[start:end].replace('let inverse =','let natural_inverse =',1)
    branch=branch[:start]+original_inverse+'        let inverse = crate::fixed_preconditioner(z,c,mu,period).unwrap_or(natural_inverse);\n'+branch[end:]
    branch+='''
#[test]
fn f64_error_bounds_enclose_independent_multiprecision_bounds() {
    let mut edges:[Vec<cover::Edge>;2]=std::array::from_fn(|_|Vec::new());
    for (index,row) in crate::f64_rows().iter().enumerate() {
        assert!(crate::fixed_preconditioner(&C::new(row[0],row[1]),&C::new(row[2],row[3]),
            &C::new(row[4],row[5]),row[6] as usize).is_some());
        let proof=certify(&C::new(row[0],row[1]),&C::new(row[2],row[3]),
            &C::new(row[4],row[5]),row[6] as usize,row[7],row[8],row[9]).unwrap();
        for axis in 0..2 {
            assert!(proof.model.inclusion[axis].cmp(&B::from_f64(row[18+axis],P))
                .is_some_and(|order|order<=0),"f64 error bound {index} axis {axis}");
        }
        assert!(proof.contraction<=row[20],"f64 contraction bound {index}");
        if index>=162 {
            let item=crate::COVER_ROWS[index-162];
            edges[item[25] as usize].push(cover::edge(&proof,&C::new(item[21],item[22]),&C::new(item[23],item[24])).unwrap());
        }
    }
    for group in 0..2 {
        if edges[group].is_empty() {continue;}
        let targets:Vec<_>=crate::COVER_TARGETS.iter().filter(|row|row[5] as usize==group).collect();
        let target_count=targets.len();
        let first=targets[0];
        let curve=cover::closed(std::mem::take(&mut edges[group]),&C::new(first[0],first[1])).unwrap();
        for row in targets {
            let clearance=curve.clearance_at(&C::new(row[0],row[1])).unwrap();
            let lower=clearance.mandelbrot_interior_distance_lower();
            assert!(lower.cmp(&B::from_f64(row[2],P)).is_some_and(|v|v>=0),"target group {group}");
            assert!(lower.cmp(&B::from_f64(row[3],P)).is_some_and(|v|v>=0),"f64 geometric bound group {group}");
            assert_eq!(clearance.winding(),row[4] as i32);
        }
        eprintln!("[HP_COVER145] group={group} edges={} targets={target_count} independently_closed=true",curve.edge_count());
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
            'cover_edges':len(cover_rows),'cover_targets':len(targets),
            'scalar_witnesses':len(scalars),
            'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'astro_float_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in directory.glob('*.rs')}}
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(run.stdout)
    assert run.returncode==0,run.stderr
