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
import os

class Options(argparse.Namespace):
    source_root: Path
    target_dir: Path
    output: Path
    emit_cover: bool

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--target-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--emit-cover',action='store_true')
args=parser.parse_args(namespace=Options())
root=args.source_root.resolve()
directory=root/'runtime-core/src/geometry_provider'
provider=(root/'runtime-core/src/geometry_provider.rs').read_text(encoding='utf-8')
visual=(root/'runtime-core/src/visual_metrics.rs').read_text(encoding='utf-8')
proposal=(directory/'periodic_boundary.rs').read_text(encoding='utf-8')
branch_path=directory/'periodic_boundary/branch.rs'
branch=branch_path.read_text(encoding='utf-8') if branch_path.exists() else ''
cover_path=directory/'periodic_boundary/branch/cover.rs'
cover=cover_path.read_text(encoding='utf-8') if cover_path.exists() else ''

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
if branch:
    branch_prefix='geometry_provider::periodic_boundary::branch::tests::'
    mutations.extend([
        ('branch-center-roundoff','branch', 'fn round(center: C, re: I, im: I) -> f64 {',
         'fn round(center: C, re: I, im: I) -> f64 { return 0.;',
         branch_prefix+'arithmetic_encloses_lost_center_bits_and_remainder_products'),
        ('branch-nonlinear-remainder','branch',
         'up_mul(self.perturbation(r), b.perturbation(r))','0.',
         branch_prefix+'exact_cardioid_predictor_certifies_and_bad_remainders_refuse'),
        ('branch-contraction','branch','&& proof.contraction < 1.','&& true',
         branch_prefix+'self_inclusion_without_contraction_refuses'),
        ('branch-inclusion','branch',
         'proof.inclusion[0] < r[1] && proof.inclusion[1] < r[2]','true',
         branch_prefix+'singular_wrong_and_invalid_inputs_refuse'),
        ('branch-parameter-domain','branch','if !self.contains(parameter) {','if false {',
         branch_prefix+'endpoint_enclosures_cover_exact_cardioid_and_glue_cycle_phases'),
        ('branch-cycle-phase','branch','0..old.period','0..1',
         branch_prefix+'endpoint_enclosures_cover_exact_cardioid_and_glue_cycle_phases'),
        ('branch-singleton-domain','branch','parameter == self.parameter ||','',
         branch_prefix+'singleton_multiplier_domains_evaluate_and_join_at_the_exact_center'),
    ])
if cover:
    cover_prefix='geometry_provider::periodic_boundary::branch::cover::tests::'
    mutations.extend([
        ('cover-attraction','cover','if norm(start) >= 1. || norm(end) >= 1. {','if false {',
         cover_prefix+'unclosed_paths_and_non_attracting_segments_refuse'),
        ('cover-closure','cover','if a.end != b.start {','if false {',
         cover_prefix+'unclosed_paths_and_non_attracting_segments_refuse'),
        ('cover-winding','cover','if winding == 0 {','if false {',
         cover_prefix+'cardioid_closed_curve_bounds_and_reuses_queries'),
        ('cover-capsule-radius','cover','box_lower.max(line_lower) - cap.radius','box_lower.max(line_lower)',
         cover_prefix+'capsule_radius_and_orientation_bounds_are_not_optional'),
        ('cover-final-target','cover','if clearance.lower < minimum {','if false {',
         cover_prefix+'exhausted_or_invalid_requests_never_return_a_prefix_cover'),
    ])

with tempfile.TemporaryDirectory(prefix='geometry145-generic-') as name:
    tmp=Path(name)
    (tmp/'periodic_boundary_cases.rs').write_bytes((directory/'periodic_boundary_cases.rs').read_bytes())
    (tmp/'periodic_boundary').mkdir()
    if cover:
        (tmp/'periodic_boundary/branch').mkdir()
    def write_sources(proposal_text:str,kernel_text:str,branch_text:str=branch,cover_text:str=cover)->None:
        if cover_text:
            (tmp/'periodic_boundary/branch/cover.rs').write_text(cover_text,encoding='utf-8')
        (tmp/'proposal.rs').write_text(proposal_text,encoding='utf-8')
        if branch_text:
            (tmp/'periodic_boundary/branch.rs').write_text(branch_text,encoding='utf-8')
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
    env=os.environ.copy()
    if args.emit_cover: env['GEOMETRY145_COVER_REFERENCE']='1'
    result=subprocess.run([str(exe),'--nocapture'],capture_output=True,text=True,env=env)
    baseline=result.stdout+result.stderr
    if result.returncode!=0:
        args.output.write_text(json.dumps({'stage':'baseline','returncode':result.returncode,
            'baseline':baseline},indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(f'Baseline failed; terminal log: {args.output}')
    rows=[]
    for label,target,old,new,test in mutations:
        source={'proposal':proposal,'kernel':kernel,'branch':branch,'cover':cover}[target]
        expected=3 if label=='branch-contraction' else 2 if label=='branch-inclusion' else 1
        assert source.count(old)==expected,(label,source.count(old))
        altered=source.replace(old,new,2 if target=='branch' else -1)
        write_sources(altered if target=='proposal' else proposal,altered if target=='kernel' else kernel,
                      altered if target=='branch' else branch,altered if target=='cover' else cover)
        exe=build(label)
        result=subprocess.run([str(exe),'--exact',test,'--nocapture'],capture_output=True,text=True)
        caught=result.returncode!=0 and 'test result: FAILED' in result.stdout
        rows.append({'mutation':label,'caught':caught,'output':result.stdout+result.stderr})
        assert caught,label
    assert selected is not None
    source_paths=[root/'runtime-core/src/visual_metrics.rs',root/'runtime-core/src/geometry_provider.rs',
                  directory/'periodic_boundary.rs',directory/'periodic_boundary_cases.rs',
                  directory/'interval.rs',directory/'uniform_box.rs',directory/'affine_return.rs']
    if branch: source_paths.append(branch_path)
    if cover: source_paths.append(cover_path)
    record={'scope':'Numerical proposals, local branch proofs and reconstructed-cell checks only; no Regular geometry authority.',
            'baseline':baseline,'mutations':rows,'num_complex_rlib':selected.name,
            'num_complex_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source_paths}}
    args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_baseline_tests_passed':True,'mutations_caught':len(rows),'rlib':selected.name}))
