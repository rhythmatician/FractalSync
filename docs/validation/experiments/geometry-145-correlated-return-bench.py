"""Validate and compare the actual correlated-return implementation in a checkout.

Only temporary copies are mutated. The disabled-call comparison retains the same
affine step and existing disk fallback. It measures box proofs, not full queries.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source_root.resolve() / 'runtime-core/src/geometry_provider'
outer = (source / 'uniform_box.rs').read_text()
child = (source / 'affine_return.rs').read_text()
call = r'if periodic_affine_trap\(\s*prefix,\s*&centers,\s*c,\s*cr\s*\)'
assert len(re.findall(call, outer)) == 1
outer = re.sub(r'#\[cfg\(test\)\]\s*fn affine_box', 'fn affine_box', outer)
outer = outer.replace('#[cfg(test)]', '#[cfg(any())]')
outer = outer.replace('#[path = "affine_return.rs"]', '#[path = "child.rs"]')
outer += r'''
#[cfg(test)] mod validation {
 use super::affine_return::AffineBall;
 #[test] fn initial_disk_must_contain_the_enclosed_prefix() {
  let prefix=AffineBall{z:[0.;2],d:[1.,0.],remainder:1e-4};
  assert!(!prefix.traps([0.;2],0.,1,1e-5));
 }
 #[test] fn rounded_square_encloses_a_known_binary_remainder() {
  let z=1.+2f64.powi(-27);
  let next=AffineBall{z:[z,0.],d:[0.;2],remainder:0.}.step([0.;2],0.).unwrap();
  assert_eq!(next.z[0],1.+2f64.powi(-26));
  // The exact square contains the additional term 2^-54.
  assert!(next.remainder>=2f64.powi(-54));
 }
}
pub(super) fn benchmark() {
 let mut results=Vec::new();
 let start=std::time::Instant::now();
 for (center,first) in [([0.32049988639807619,0.04733531447355364],21),
                       ([0.35659381055433242,0.06404656757776669],24)] {
  for k in first..first+8 {
   let h=0.125*2f64.powi(-k);let width=56.*h;
   for y in -4..=4 {for x in -4..=4 {
    let re=center[0]+f64::from(x)*2.*h-width/2.;
    let im=center[1]+f64::from(y)*2.*h-width/2.;
    let value=match affine_box(re,im,width) {None=>-1,Some(false)=>0,Some(true)=>1};
    results.push(value);
   }}
  }
 }
 println!("{{\"elapsed_s\":{},\"results\":{:?}}}",start.elapsed().as_secs_f64(),results);
}
'''

def compile_rust(directory: Path, test: bool, name: str) -> Path:
    exe = directory / f'{name}.exe'
    command = ['rustc', '--edition=2021', '-O', str(directory / 'main.rs'), '-o', str(exe)]
    if test:
        command.insert(1, '--test')
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return exe

with tempfile.TemporaryDirectory(prefix='geometry145-correlated-') as name:
    directory = Path(name)
    interval = source / 'interval.rs'
    imports = f'#[path="{interval.as_posix()}"] mod interval;\n' if interval.exists() else ''
    (directory / 'main.rs').write_text(imports + 'mod uniform;\nfn main(){uniform::benchmark();}\n')
    (directory / 'uniform.rs').write_text(outer)
    (directory / 'child.rs').write_text(child)
    exe = compile_rust(directory, True, 'tests')
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    tests = result.stdout
    mutations = []
    for label, old, new in [
        ('prefix-inclusion', '|| radius < self.remainder', '|| false'),
        ('parameter-coefficient', 'upper_mul(coefficient_gap, cr)', '0.'),
        ('machine-rounding', 'let round = machine_round_error(a, cb);', 'let round = 0.;'),
    ]:
        assert child.count(old) == 1, label
        (directory / 'child.rs').write_text(child.replace(old, new))
        exe = compile_rust(directory, True, label)
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        caught = result.returncode != 0 and 'test result: FAILED' in result.stdout
        mutations.append({'mutation': label, 'caught': caught, 'output': result.stdout + result.stderr})
        assert caught, label
    (directory / 'child.rs').write_text(child)
    runs = {}
    for enabled in [False, True, 'Original512']:
        code = outer if enabled is True else re.sub(call, 'if false', outer)
        if enabled == 'Original512':
            code, count = re.subn(r'for n in 1usize\.\.=4096', 'for n in 1usize..=512', code)
            assert count == 1
        (directory / 'uniform.rs').write_text(code)
        exe = compile_rust(directory, False, f'benchmark-{enabled}')
        result = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
        runs[str(enabled)] = json.loads(result.stdout)
    before, after = runs['False']['results'], runs['True']['results']
    assert len(before) == len(after) == 1296
    assert all(a == b for a, b in zip(before, after) if a != -1)
    original = runs['Original512']['results']
    assert all(a == b for a, b in zip(original, after) if a != -1)
    record = {'production_adoption': False, 'tests': tests, 'mutations': mutations, 'runs': runs,
              'existing_certificates_unchanged': True,
              'new_certificates': sum(a == -1 and b != -1 for a, b in zip(before, after)),
              'ablation_limit': 'False retains the new 4096-prefix schedule; Original512 additionally restores the original prefix cap.',
              'source_sha256': {filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
                                for filename in ['uniform_box.rs', 'affine_return.rs', 'interval.rs']
                                if (source / filename).exists()}}
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'boxes': len(before), 'new_certificates': record['new_certificates'],
                      'old_certificates_unchanged': True,
                      'before_s': runs['False']['elapsed_s'], 'after_s': runs['True']['elapsed_s'],
                      'original512_s': runs['Original512']['elapsed_s'], 'original512_certificates': sum(x != -1 for x in original),
                      'mutations_caught': len(mutations)}))
