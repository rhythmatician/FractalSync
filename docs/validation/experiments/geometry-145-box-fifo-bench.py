"""After applying the ranged-field and box-FIFO experiments, run with project Python.

Compile the actual uniform-box module in an isolated temporary harness. The
standalone harness uses std LazyLock in place of once_cell Lazy; no other proof
or cache implementation is substituted. Timings are diagnostic, not assertions.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[3]
proof = (root / "runtime-core/src/geometry_provider/uniform_box.rs").read_text(encoding="utf-8")
if "--mutate-existing-update" in sys.argv:
    old = "*entry=value;return;"
    assert proof.count(old) == 1
    proof = proof.replace(old, "*entry=value;", 1)
benchmark = r'''
pub(super) fn benchmark() {
 let mut memo=BoxMemo::new(2);
 let a=(1,0,0);let b=(2,0,0);let c=(3,0,0);
 memo.insert(a,None);memo.insert(b,Some(true));
 for _ in 0..100 {memo.insert(a,Some(false));}
 assert_eq!(memo.order.len(),2);assert_eq!(memo.map.len(),2);
 memo.insert(c,None);
 assert_eq!(memo.get(&a),None);assert_eq!(memo.get(&b),Some(Some(true)));
 assert_eq!(memo.get(&c),Some(None));assert_eq!(memo.order.len(),2);
 let cache=super::RwLock::new(BoxMemo::new(4096));
 for i in 0..4096 {membership_using(-0.01+i as f64*1e-6,0.,1e-8,Some(&cache));}
 for (label,p,count) in [("cheap_uniform",[-0.01+4095.*1e-6,0.,1e-8],20000),("expensive_uncertain",[0.32049988639807619,0.04733531447355364,1e-8],10)] {
  let expected=membership_using(p[0],p[1],p[2],None);
  for enabled in [false,true] {
   let start=std::time::Instant::now();
   for _ in 0..count {
    let got=membership_using(std::hint::black_box(p[0]),p[1],p[2],if enabled {Some(&cache)}else{None});
    assert_eq!(got,expected);std::hint::black_box(got);
   }
   println!("{{\"case\":\"{}\",\"cache_enabled\":{},\"calls\":{},\"elapsed_s\":{}}}",label,enabled,count,start.elapsed().as_secs_f64());
  }
 }
}
'''
with tempfile.TemporaryDirectory(prefix="geometry145-fifo-") as name:
    directory = Path(name)
    module = directory / "uniform.rs"
    module.write_text(proof + benchmark, encoding="utf-8")
    interval = (root / "runtime-core/src/geometry_provider/interval.rs").as_posix()
    harness = directory / "main.rs"
    harness.write_text(
        "use std::sync::RwLock;\ntype Lazy<T>=std::sync::LazyLock<T>;\n"
        f'#[path="{interval}"] mod interval;\n'
        '#[path="uniform.rs"] mod uniform;\nfn main(){uniform::benchmark();}\n',
        encoding="utf-8",
    )
    exe = directory / "benchmark.exe"
    subprocess.run(["rustc", "--edition=2021", "-O", str(harness), "-o", str(exe)], check=True, capture_output=True)
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    print(json.dumps({"fifo_update_and_capacity_checks_passed": True, "rows": rows}, indent=2))
