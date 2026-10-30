"""Run the original private Rust crossing body verbatim on the frozen asset.

The generated example changes setup only. The tracked test source, thresholds,
assertions and loop bodies are not modified. Generated source is removed after
the process exits; the body and runtime source hashes remain in the report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root=Path(__file__).resolve().parents[1]
    asset=(root / sys.argv[1]).resolve()
    output=root / sys.argv[2]
    source=root / "runtime-core/src/manifold.rs"
    original=source.read_bytes()
    start=original.index(b"    fn controls_drive_can_cross_shore_without_wall()")
    end=original.index(b"    #[test]",start)
    body=original[start:end].decode("utf-8")
    baseline=subprocess.check_output(["rtk","proxy","git","show","9187815:runtime-core/src/manifold.rs"],cwd=root)
    baseline_start=baseline.index(b"    fn controls_drive_can_cross_shore_without_wall()")
    baseline_end=baseline.index(b"    #[test]",baseline_start)
    if original[start:end]!=baseline[baseline_start:baseline_end]:
        raise RuntimeError("original native crossing body changed from fixed checkpoint")
    generated=root / "runtime-core/examples/__operational_original_crossing.rs"
    if generated.exists():
        raise RuntimeError("generated example path already exists")
    prefix='''use runtime_core::{manifold::*, distance_field};
use runtime_core::operational_field::{PreparedField, acceptance::{FrozenRollout, PhysicsProfile, Integrator}};
use num_complex::Complex64;
use std::sync::Arc;
// Observe successful endpoint returns only. All mechanics remain in runtime-core;
// the original test body below is copied verbatim and still stops at F=0.
mod controls {
    pub use runtime_core::controls::MotionControls;
    use super::*;
    thread_local! {static TRACES:std::cell::RefCell<[Vec<(f64,Complex64)>;2]>=const {std::cell::RefCell::new([Vec::new(),Vec::new()])};}
    pub fn integrate_motion_controls(c:Complex64,v:(f64,f64),u:&MotionControls,dt:f64,config:&ManifoldConfig)
      ->Result<(Complex64,(f64,f64),runtime_core::manifold::EnergyInfo),String> {
        let result=runtime_core::controls::integrate_motion_controls(c,v,u,dt,config);
        if let Ok((end,_,_))=&result {
            TRACES.with(|traces| {
                let mut traces=traces.borrow_mut();
                let trace=&mut traces[usize::from(u.throttle>0.0)];
                if trace.is_empty() {trace.push((0.0,c));}
                trace.push((trace.len() as f64*dt,*end));
            });
        }
        result
    }
    pub fn reference_reports()->serde_json::Value {
        use runtime_core::operational_field::acceptance::{reference_realm,Realm};
        TRACES.with(|traces| {
            let reports:Vec<_>=traces.borrow().iter().enumerate().map(|(i,trace)| {
                let mut inside=None;
                let mut pair=None;
                let evidence:Vec<_>=trace.iter().map(|(time,c)| {
                    let realm=reference_realm([c.re,c.im]);
                    let point=serde_json::json!({"mechanical_time":time,"position":[c.re,c.im],"realm":realm,"kind":if realm==Realm::Unknown {"unknown"}else{"certified"}});
                    if realm==Realm::Inside && inside.is_none() {inside=Some(point.clone());}
                    if realm==Realm::Outside && pair.is_none() {if let Some(before)=&inside {pair=Some(vec![before.clone(),point.clone()]);}}
                    point
                }).collect();
                serde_json::json!({"name":if i==0 {"undriven"}else{"driven"},"accepted_steps":trace.len().saturating_sub(1),"reference_crossing":if pair.is_some(){"passed"}else{"unverified"},"reference_pair":pair,"evidence":evidence})
            }).collect();
            serde_json::json!({"trajectories":reports,"source":"existing conservative reference_realm; point-scoped membership only; evaluated offline after the test","continuous_trajectory":"piecewise-linear reconstruction of successful accepted endpoints; not the exact ODE solution"})
        })
    }
}
fn panic_text(e:Box<dyn std::any::Any+Send>)->String {
    e.downcast_ref::<String>().cloned().or_else(||e.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_else(||"non-string panic".into())
}
'''
    suffix='''
fn main() {
    let asset=std::env::args().nth(1).expect("asset path required");
    let field=Arc::new(PreparedField::from_asset_bytes(&std::fs::read(asset).unwrap()).unwrap());
    let beta=controls::MotionControls{direction:[1.0,0.0],throttle:0.0,brake:0.0,grip:0.0,impulse:0.0}.friction_beta();
    let config=ManifoldConfig{d_ref:0.1,epsilon:1e-4,lambda_sq:1.0,kappa:0.5,mu:std::f64::consts::FRAC_1_PI};
    let cuts=std::env::args().nth(2).map(|path|runtime_core::operational_field::acceptance::reference_policy_for_field(&field,&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap_or_default();
    let frame=FrozenRollout::new(field,PhysicsProfile::new(config,0.02,beta,Integrator::Euler).unwrap(),cuts).unwrap();
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||frame.evaluate(|_|{
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(controls_drive_can_cross_shore_without_wall)).map_err(panic_text)
    })));
    let (passed,error)=match result {
        Ok(Ok(_))=>(true,None),Ok(Err(e))=>(false,Some(e)),
        Err(e)=>(false,Some(panic_text(e)))
    };
    let reference=controls::reference_reports();
    let reference_crossing=reference["trajectories"][1]["reference_crossing"].clone();
    println!("{}",serde_json::json!({"passed":passed,"error":error,"field_id":format!("{:016x}",frame.field_id()),"physics_profile":frame.profile().manifest(),"work":frame.last_evaluation_work(),"reference_crossing":reference_crossing,"reference_diagnostic":reference}));
    if !passed {std::process::exit(1);}
}
'''
    try:
        generated.write_text(prefix+body+suffix,encoding="utf-8")
        args=["rtk","cargo","run","--manifest-path","runtime-core/Cargo.toml","--release","--features","experimental-operational-field","--example","__operational_original_crossing","--",str(asset)]
        policy=(root / sys.argv[3]).resolve() if len(sys.argv)>3 else None
        if policy:
            args.append(str(policy))
        run=subprocess.run(args,cwd=root,text=True,encoding="utf-8",capture_output=True,timeout=180,check=False)
        if not run.stdout.strip():
            raise RuntimeError("native diagnostic produced no verdict: " + run.stderr.replace(str(root), "${REPO}")[-4000:])
        verdict=json.loads(run.stdout.strip())
        verdict.update({"test":"manifold::tests::controls_drive_can_cross_shore_without_wall","test_body_sha256":hashlib.sha256(original[start:end]).hexdigest(),"source_unchanged":original==source.read_bytes(),"asset_sha256":hashlib.sha256(asset.read_bytes()).hexdigest(),"exit_code":run.returncode,"setup":"verbatim original body in generated example; field/profile pinned around entire test; no assertion changes"})
        verdict["reference_policy_sha256"]=hashlib.sha256(policy.read_bytes()).hexdigest() if policy else None
        if run.returncode:
            verdict["stderr"]=run.stderr.replace(str(root),"${REPO}")[-3000:]
        output.write_text(json.dumps(verdict,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(verdict,indent=2))
        return run.returncode
    finally:
        # Only this invocation's newly created file, no recursive cleanup.
        generated.unlink()


if __name__=="__main__":
    raise SystemExit(main())
