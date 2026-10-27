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
import re

class Options(argparse.Namespace):
    source_root: Path
    target_dir: Path
    output: Path
    emit_cover: bool
    precision_target_dir: Path | None
    test_filter: str | None

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--target-dir',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--emit-cover',action='store_true')
parser.add_argument('--precision-target-dir',type=Path)
parser.add_argument('--test-filter')
args=parser.parse_args(namespace=Options())
precision_dir=args.precision_target_dir.resolve()/'release/deps' if args.precision_target_dir else None
precision_library=next(precision_dir.glob('libastro_float-*.rlib')) if precision_dir else None
experiment_dir=Path(__file__).resolve().parent
witness_path=experiment_dir/'geometry-145-witness-tests.rs'
witness=witness_path.read_text(encoding='utf-8') if precision_dir else ''
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
tie=next(line for line in provider.splitlines() if line.startswith('pub const DYADIC_CUT_TIE_FACTOR:'))
kernel='\n'.join(['#[derive(Clone,Copy)]',item(provider,'struct FieldJet'),field_impl,
                  item(provider,'fn distance_stencil('),basis,item(provider,'fn quintic_basis('),
                  item(provider,'fn quintic_field('),angle,tie])
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
if 'struct BoundaryDirectionCone' in cover:
    cone_test='geometry_provider::periodic_boundary::branch::cover::tests::capsule_cones_clip_only_by_certified_distance_and_require_half_angles'
    mutations.extend([
        ('cone-capsule-radius','cover','let remainder = (cap.radius * axis_norm).next_up();','let remainder = 0.;',cone_test),
        ('cone-angle-gate','cover','if lower > threshold {','if true {',cone_test),
        ('cone-prune-equality','cover','if distance_lower(cap, q).ok()? > radius {','if distance_lower(cap, q).ok()? >= radius {',cone_test),
        ('cone-work-exhaustion','cover','if *work > 65536 {','if false {',cone_test),
    ])
if witness:
    witness_prefix='geometry_provider::periodic_boundary::witness_tests::'
    mutations.extend([
        ('scalar-lower-gate','witness','lower < (value - error).next_up()','false',
         witness_prefix+'certified_scalar_requires_both_endpoint_gates_and_keeps_its_value'),
        ('scalar-upper-gate','witness','upper > (value + error).next_down()','false',
         witness_prefix+'certified_scalar_requires_both_endpoint_gates_and_keeps_its_value'),
        ('scalar-canonical-value','witness','value: -value,','value: -(lower+upper)*0.5,',
         witness_prefix+'certified_scalar_requires_both_endpoint_gates_and_keeps_its_value'),
        ('witness-proof-required','witness',
         'let proof = crate::point_proof::escape([point.re, point.im], precision, budget)?;',
         'let proof = crate::point_proof::EscapeProof { update: 0, precision, radius_exponent: None };',
         witness_prefix+'interior_nonfinite_and_exhausted_proposals_are_not_witnesses'),
        ('witness-screen-cutoff','proposal','candidates.push((point, upper, hi - 1));',
         'candidates.push((point, upper, 0));',prefix+'witness_candidates_preserve_upper_windows_and_shared_escape_budgets'),
    ])
if args.test_filter: mutations=[]

with tempfile.TemporaryDirectory(prefix='geometry145-generic-') as name:
    tmp=Path(name)
    (tmp/'periodic_boundary_cases.rs').write_bytes((directory/'periodic_boundary_cases.rs').read_bytes())
    (tmp/'periodic_boundary').mkdir()
    if cover:
        (tmp/'periodic_boundary/branch').mkdir()
    def write_sources(proposal_text:str,kernel_text:str,branch_text:str=branch,cover_text:str=cover,witness_text:str=witness)->None:
        if cover_text:
            (tmp/'periodic_boundary/branch/cover.rs').write_text(cover_text,encoding='utf-8')
        if precision_dir:
            (tmp/'witness.rs').write_text(witness_text,encoding='utf-8')
            proposal_text+=f'\n#[path="{(tmp/"witness.rs").as_posix()}"] mod witness_tests;\n'
        (tmp/'proposal.rs').write_text(proposal_text,encoding='utf-8')
        if branch_text:
            (tmp/'periodic_boundary/branch.rs').write_text(branch_text,encoding='utf-8')
        harness='mod visual_metrics {\n'+orbit+'\n}\nmod geometry_provider {\nuse num_complex::Complex64;\n'
        for module,filename in [('interval','interval.rs'),('uniform_box','uniform_box.rs')]:
            harness+=f'#[path="{(directory/filename).as_posix()}"] mod {module};\n'
        harness+=kernel_text+'\n'+f'#[path="{(tmp/"proposal.rs").as_posix()}"] mod periodic_boundary;\n'+'}\n'
        if precision_dir:
            for module in ['complex_ball','point_proof']:
                harness+=f'#[path="{(experiment_dir/"geometry-145-point-proof/src"/(module+".rs")).as_posix()}"] mod {module};\n'
        (tmp/'main.rs').write_text(harness,encoding='utf-8')
    selected: Path | None=None
    def build(label:str)->Path:
        global selected
        exe=tmp/f'{label}.exe'
        failures=[]
        for library in [selected] if selected is not None else libraries:
            command=['rustc','--test','--edition=2021','-O',str(tmp/'main.rs'),
                 '-L',f'dependency={deps}','--extern',f'num_complex={library}','-o',str(exe)]
            if precision_dir:
                command.extend(['-L',f'dependency={precision_dir}','--extern',f'astro_float={precision_library}'])
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode==0:
                selected=library
                return exe
            failures.append(result.stderr)
        raise RuntimeError('\n'.join(failures))
    write_sources(proposal,kernel)
    exe=build('baseline')
    env=os.environ.copy()
    if args.emit_cover: env['GEOMETRY145_COVER_REFERENCE']='1'
    test_command=[str(exe),'--nocapture']
    if args.test_filter: test_command.append(args.test_filter)
    result=subprocess.run(test_command,capture_output=True,text=True,env=env)
    baseline=result.stdout+result.stderr
    if result.returncode!=0:
        args.output.write_text(json.dumps({'stage':'baseline','returncode':result.returncode,
            'baseline':baseline},indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(f'Baseline failed; terminal log: {args.output}')
    count=re.search(r'test result: ok\. (\d+) passed;',result.stdout)
    executed_tests=int(count.group(1)) if count else 0
    if executed_tests==0:
        args.output.write_text(json.dumps({'stage':'baseline-selection','returncode':result.returncode,
            'test_filter':args.test_filter,'executed_tests':0,'baseline':baseline},indent=2)+'\n',encoding='utf-8')
        raise RuntimeError(f'No tests executed; selection log: {args.output}')
    rows=[]
    for label,target,old,new,test in mutations:
        source={'proposal':proposal,'kernel':kernel,'branch':branch,'cover':cover,'witness':witness}[target]
        expected=3 if label=='branch-contraction' else 2 if label=='branch-inclusion' else 1
        assert source.count(old)==expected,(label,source.count(old))
        altered=source.replace(old,new,2 if target=='branch' else -1)
        write_sources(altered if target=='proposal' else proposal,altered if target=='kernel' else kernel,
                      altered if target=='branch' else branch,altered if target=='cover' else cover,
                      altered if target=='witness' else witness)
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
            'test_filter':args.test_filter,
            'executed_tests':executed_tests,
            'baseline':baseline,'mutations':rows,'num_complex_rlib':selected.name,
            'num_complex_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
            'source_sha256':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source_paths}}
    if precision_dir and precision_library:
        proof_paths=[witness_path]+[experiment_dir/'geometry-145-point-proof/src'/name
                                   for name in ['point_proof.rs','complex_ball.rs','directed_norm.rs']]
        record['precision_sources']={str(p.relative_to(experiment_dir)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in proof_paths}
        record['astro_float_rlib']=precision_library.name
        record['astro_float_sha256']=hashlib.sha256(precision_library.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'all_baseline_tests_passed':True,'executed_tests':executed_tests,
                      'test_filter':args.test_filter,'mutations_caught':len(rows),'rlib':selected.name}))
