"""Compile the actual proof modules and test three missing-radius mutations."""
from pathlib import Path
import hashlib
import json
import subprocess

import argparse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--deps',type=Path,required=True)
parser.add_argument('--astro-float',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.candidate.resolve()
source=root/'runtime-core/src/geometry_provider'
out=args.output.resolve()
out.mkdir(exist_ok=True,parents=True)
deps=args.deps.resolve()
astro=[args.astro_float.resolve()]
assert astro[0].is_file()
files = [source / 'interval.rs', *(source / 'periodic_boundary' / name for name in
    ['complex_ball.rs', 'directed_norm.rs', 'point_proof.rs', 'fast_escape.rs'])]
hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
captured = out / 'inputs'
captured.mkdir(exist_ok=True)
original_files = files
for p in original_files:
    (captured / p.name).write_bytes(p.read_bytes())
files = [captured / p.name for p in original_files]
assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files] == list(hashes.values())
original = files[-1].read_text()
variants = {
    'baseline': original,
    'no_roundoff': original.replace('.add(I::point(rounding))', ''),
    'no_quadratic': original.replace('.add(r.square())', ''),
    'no_linear': original.replace('I::point(2.)\n            .mul(I::point(norm_upper(re, im)))\n            .mul(r)', 'I::point(0.)'),
}
results = []
for name, content in variants.items():
    assert name == 'baseline' or content != original
    variant = out / f'{name}.rs'
    variant.write_text(content)
    def attr(path: Path) -> str:
        return '#[path = ' + json.dumps(path.as_posix()) + ']'
    harness = out / f'{name}_harness.rs'
    harness.write_text('mod geometry_provider {\n' + attr(files[0]) + '\nmod interval;\n'
        'mod periodic_boundary {\n#[derive(Clone,Copy,Debug)] struct C { re:f64, im:f64 }\n'
        'impl C { fn new(re:f64,im:f64)->Self { Self{re,im} } }\n'
        'fn finite(c:C)->bool { c.re.is_finite() && c.im.is_finite() }\n'
        + attr(files[1]) + '\nmod complex_ball;\n' + attr(files[3])
        + '\nmod point_proof;\n' + attr(variant) + '\nmod fast_escape;\n}}\n')
    exe = out / f'{name}.exe'
    compile_run = subprocess.run(['rustc','--edition=2021','--test','-O', str(harness),
        '-L', f'dependency={deps}', '--extern', f'astro_float={astro[0]}', '-o',str(exe)],
        capture_output=True, text=True)
    (out / f'{name}-compile.log').write_text(compile_run.stdout + compile_run.stderr)
    assert compile_run.returncode == 0, compile_run.stderr
    run = subprocess.run([str(exe),'fast_escape::tests','--nocapture'], capture_output=True,text=True)
    log = run.stdout + run.stderr
    (out / f'{name}.log').write_text(log)
    assert 'running 4 tests' in log
    if name == 'baseline':
        assert run.returncode == 0, log
    else:
        assert run.returncode != 0 and 'exact image outside propagated disk' in log, log
    results.append({'variant':name,'exit_code':run.returncode,'log':str(out / f'{name}.log')})
    print(name, run.returncode, flush=True)
assert hashes == {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files}
assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files] == list(hashes.values())
(out / 'results.json').write_text(json.dumps({'source_sha256':hashes,'results':results},indent=2))
