"""Compile the actual proposal, shared orbit, return proof and quintic code.

Uses the checkout's already-built num-complex rlib. No domain algorithm is
substituted; unrelated FieldJet::from_geometry conversion is excluded. Mutations
touch temporary copies only, never the source checkout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

class Options(argparse.Namespace):
    source_root: Path
    target_dir: Path
    output: Path

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--target-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args(namespace=Options())
root=args.source_root.resolve()
directory=root/'runtime-core/src/geometry_provider'
provider=(root/'runtime-core/src/geometry_provider.rs').read_text(encoding='utf-8')
visual=(root/'runtime-core/src/visual_metrics.rs').read_text(encoding='utf-8')
proposal=(directory/'periodic_boundary.rs').read_text(encoding='utf-8')

def item(text:str,marker:str)->str:
    start=text.index(marker)
    brace=text.index('{',start)
    depth=1
    end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        if text[end]=='}':depth-=1
        end+=1
    return text[start:end]

orbit='\n'.join(item(visual,marker) for marker in [
    'fn is_safely_interior(', 'pub(crate) struct MembershipOrbit',
    'impl MembershipOrbit', 'pub fn mandelbrot_membership('])
field_impl=item(provider,'impl FieldJet')
field_impl=field_impl.replace(item(field_impl,'fn from_geometry('),'')
basis=provider[provider.index('const QUINTIC_BASIS:'):provider.index('fn quintic_basis(')]
angle=next(line for line in provider.splitlines() if line.startswith('pub const DYADIC_CUT_NORMAL_DEGREES:'))
kernel='\n'.join(['#[derive(Clone,Copy)]',item(provider,'struct FieldJet'),field_impl,
                  item(provider,'fn distance_stencil('),basis,item(provider,'fn quintic_basis('),
                  item(provider,'fn quintic_field('),angle])
deps=args.target_dir.resolve()/'release/deps'
libraries=sorted(deps.glob('libnum_complex-*.rlib'),key=lambda p:p.stat().st_mtime,reverse=True)
assert libraries, 'build the source checkout first'
prefix='geometry_provider::periodic_boundary::tests::'
mutations=[
    ('fresh-root-disk', 'proposal',
     'super::uniform_box::proposal_cycle_trap([z.re, z.im], [c.re, c.im], period, radius)',
     'true', prefix+'a_small_multiplier_alone_cannot_supply_a_root_disk'),
    ('machine-period-is-not-model-period','proposal',
     '''let (period, phase) = (1..=machine_period)
        .filter(|p| *p <= 512 && machine_period % p == 0)
        .find_map(|p| attracting_root(phase, query, p).map(|z| (p, z)))?;''',
     'let (period,phase)=(machine_period,phase);',prefix+'discover_components_without_fixture_periods_or_angles'),
    ('long-machine-period','proposal','machine_period > 32768','machine_period > 512',
     prefix+'all_archived_nodes_keep_the_original_certified_error_windows'),
    ('distance-window','proposal','let distance = r.norm();','let distance = r.norm()*1.01;',
     prefix+'all_archived_nodes_keep_the_original_certified_error_windows'),
    ('coherent-gradient','kernel','g: [derivatives[dx + 1][dy], derivatives[dx][dy + 1]],',
     'g: [derivatives[dx + 1][dy]*1.01, derivatives[dx][dy + 1]],',
     prefix+'generic_scalar_nodes_keep_quintic_derivatives_and_cell_joins_coherent'),
]

with tempfile.TemporaryDirectory(prefix='geometry145-generic-') as name:
    tmp=Path(name)
    (tmp/'periodic_boundary_cases.rs').write_bytes((directory/'periodic_boundary_cases.rs').read_bytes())
    def write_sources(proposal_text:str,kernel_text:str)->None:
        (tmp/'proposal.rs').write_text(proposal_text,encoding='utf-8')
        harness='mod visual_metrics {\n'+orbit+'\n}\nmod geometry_provider {\nuse num_complex::Complex64;\n'
        for module,filename in [('interval','interval.rs'),('uniform_box','uniform_box.rs')]:
            harness+=f'#[path="{(directory/filename).as_posix()}"] mod {module};\n'
        harness+=kernel_text+'\n'+f'#[path="{(tmp/"proposal.rs").as_posix()}"] mod periodic_boundary;\n'+'}\n'
        (tmp/'main.rs').write_text(harness,encoding='utf-8')
    selected: Path | None=None
    def build(label:str)->Path:
        global selected
        exe=tmp/f'{label}.exe'
        failures=[]
        for library in [selected] if selected is not None else libraries:
            result=subprocess.run(['rustc','--test','--edition=2021','-O',str(tmp/'main.rs'),
                 '-L',f'dependency={deps}','--extern',f'num_complex={library}','-o',str(exe)],capture_output=True,text=True)
            if result.returncode==0:
                selected=library
                return exe
            failures.append(result.stderr)
        raise RuntimeError('\n'.join(failures))
    write_sources(proposal,kernel)
    exe=build('baseline')
    result=subprocess.run([str(exe),'--nocapture'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    baseline=result.stdout+result.stderr
    rows=[]
    for label,target,old,new,test in mutations:
        source=proposal if target=='proposal' else kernel
        assert source.count(old)==1,(label,source.count(old))
        altered=source.replace(old,new)
        write_sources(altered if target=='proposal' else proposal,altered if target=='kernel' else kernel)
        exe=build(label)
        result=subprocess.run([str(exe),'--exact',test,'--nocapture'],capture_output=True,text=True)
        caught=result.returncode!=0 and 'test result: FAILED' in result.stdout
        rows.append({'mutation':label,'caught':caught,'output':result.stdout+result.stderr})
        assert caught,label
    assert selected is not None
    record={'scope':'Numerical proposals and reconstructed-cell checks only; no Regular geometry authority.',
            'baseline':baseline,'mutations':rows,'num_complex_rlib':selected.name,
            'num_complex_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [root/'runtime-core/src/visual_metrics.rs',root/'runtime-core/src/geometry_provider.rs',
                           directory/'periodic_boundary.rs',directory/'periodic_boundary_cases.rs',
                           directory/'interval.rs',directory/'uniform_box.rs',directory/'affine_return.rs']}}
    args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_baseline_tests_passed':True,'mutations_caught':len(rows),'rlib':selected.name}))
