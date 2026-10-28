// Assemble one report from an already frozen asset and independent replays.
// Failures are classified; unknown reference evidence never becomes success.
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { homedir } from 'node:os';
const readJson=async path => JSON.parse((await readFile(path,'utf8')).replace(/^\uFEFF/,''));
const sha=async path => createHash('sha256').update(await readFile(path)).digest('hex');
const required=(condition,message) => { if (!condition) throw new Error(message); };
const clean=text=>text.replaceAll(resolve('.'),'${REPO}').replaceAll(homedir(),'${HOME}');
const asset='tmp/geometry145-frozen-operational.bin';
const metadata=await readJson('tmp/operational-field-native/metadata.json');
const preparation=await readJson('tmp/operational-field-preparation-final.json');
const browser=await readJson('tmp/operational-field-browser.json');
const python=await readJson('tmp/operational-field-pytest-frozen-final.json');
const originalNative=await readJson('tmp/operational-field-original-native.json');
const runs=await Promise.all(metadata.runs.map(r => readJson('tmp/operational-field-native/'+r.stdout_file)));
const experiment=runs[0].frozen_experiment;
const assetSha=await sha(asset);
required(assetSha===await sha('tmp/geometry145-frozen-operational-repro.bin'),'Reconstruction changed the frozen field bytes');
required([metadata.asset_sha256,browser.asset_sha256,python.asset_sha256,originalNative.asset_sha256].every(v=>v===assetSha),'Asset differs between native, browser or Python');
required(originalNative.source_unchanged,'Original native test source changed');
required(preparation.field_id===runs[0].identity && runs.every(r=>r.identity===runs[0].identity),'Field identity changed');
required(runs.every(r=>JSON.stringify(r.frozen_experiment)===JSON.stringify(experiment)),'Frozen replay changed across fresh native processes');
required(browser.parity.passed && browser.results.identity_fixed && browser.results.jet_invariance,'Browser parity or invariance failed');
required(experiment.runs.length===35,'Empty or changed experiment matrix');
for (const r of [...experiment.runs,...browser.results.experiment.runs]) {
  const pairOK=r.initial.field_id===r.final.field_id && r.initial.field_id===r.field_id && r.initial.physics_profile_id===r.final.physics_profile_id && r.initial.physics_profile_id===r.physics_profile.physics_profile_id;
  required(pairOK && r.field_and_profile_fixed,'Mechanical identity changed');
  required(Math.abs(r.final.mechanical_time-r.committed_steps*r.physics_profile.dt)<1e-9,'Failed attempt reported simulated progress');
  if (r.reference_crossing==='passed') {
    required(r.reference_pair && r.reference_pair[0].mechanical_time<r.reference_pair[1].mechanical_time,'Reference crossing lacks ordered states');
    for (let i=0;i<2;i++) required(r.reference_evidence.some(e=>e.time===r.reference_pair[i].mechanical_time && e.kind==='certified' && e.realm===(i===0?'Outside':'Inside') && JSON.stringify(e.position)===JSON.stringify(r.reference_pair[i].position)),'Uncertified reference crossing');
  } else required(r.reference_crossing==='unverified','Unknown reference verdict');
}
const originalTest='backend/tests/test_manifold_physics.py';
required(await sha(originalTest)==='2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536','Original tests/thresholds changed');
required(python.test_source_unchanged && python.test_source_sha256===await sha(originalTest),'Test source drift');
const crossingNames=['test_ridge_is_barrier_without_h_gate','test_shore_ridge_reflects_below_barrier_and_crosses_above_it'];
required(crossingNames.every(name=>python.tests.some(t=>t.nodeid.endsWith(name) && t.outcome==='passed')),'Original crossing regression failed');
const allowedFailures=['test_energy_drift_bounded_no_forces','test_energy_conserved_low_curvature_rollout','test_jet_exposes_scale_and_convergence_quality'];
required(python.tests.length===13,'Retained energy/crossing/continuity selection changed');
required(python.tests.filter(t=>t.outcome==='failed').every(t=>allowedFailures.some(n=>t.nodeid.endsWith(n))),'Unclassified retained-test failure');
const coverageFailures=python.tests.filter(t=>t.outcome==='failed' && t.failure?.includes('GeometryUnavailable'));
required(coverageFailures.length===2,'Coverage failures were not classified from execution evidence');
const qualityFailures=python.tests.filter(t=>t.outcome==='failed' && t.nodeid.endsWith('test_jet_exposes_scale_and_convergence_quality'));
const command=['cargo','test','--manifest-path','runtime-core/Cargo.toml','--release','--features','experimental-operational-field','operational_field','--','--test-threads=1'];
const units=spawnSync('rtk',command,{encoding:'utf8',timeout:120000});
const summary=units.stdout+units.stderr;
required(units.status===0 && /14 passed/.test(summary),'Executable contract controls failed or ran no tests');
const defaultCheck=spawnSync('rtk',['cargo','check','--manifest-path','runtime-core/Cargo.toml'],{encoding:'utf8',timeout:120000});
required(defaultCheck.status===0,'Default native build failed');
const sourcePaths=['Cargo.lock','runtime-core/Cargo.toml','runtime-core/src/lib.rs','runtime-core/src/operational_field.rs','runtime-core/src/operational_field/acceptance.rs','runtime-core/src/geometry_provider.rs','runtime-core/src/geometry_provider/uniform_box.rs','runtime-core/src/geometry_provider/affine_return.rs','runtime-core/src/distance_field.rs','runtime-core/src/controls.rs','runtime-core/src/manifold.rs','runtime-core/src/pybindings.rs','runtime-core/examples/operational_field_report.rs','wasm-orbit/Cargo.toml','wasm-orbit/src/lib.rs','wasm-orbit/src/operational_experiment.rs',originalTest,'backend/src/cspace_proxies.py','docs/validation/geometry-145-operational-experiment.json','runtime-core/data/mandelbrot_distance_1024.bin','runtime-core/data/mandelbrot_distance_1024.json','scripts/validate_operational_field.py','scripts/validate_operational_field_browser.mjs','scripts/validate_operational_native_crossing.py'];
const hashes=Object.fromEntries(await Promise.all(sourcePaths.map(async p=>[p,await sha(p)])));
for (const p of ['scripts/collect_operational_acceptance.mjs','scripts/measure_operational_field.ps1']) hashes[p]=await sha(p);
for (const [path,hash] of Object.entries(python.rust_source_sha256)) required(hashes[path]===hash,`Python source drift: ${path}`);
required(metadata.executable_sha256===await sha('target/release/examples/operational_field_report.exe'),'Native executable changed');
required(python.wheel_sha256===await sha('tmp/operational-field-wheels/runtime_core-0.1.0-cp38-abi3-win_amd64.whl'),'Wheel changed');
required(browser.wasm_sha256===await sha('tmp/operational-field-wasm/orbit_synth_wasm_bg.wasm'),'WASM changed');
await mkdir('docs/validation/experiments/geometry-145-operational',{recursive:true});
const archive='docs/validation/experiments/geometry-145-operational/field.zip';
const profileMap=new Map([...experiment.runs.map(r=>r.physics_profile),...experiment.energy_runs.map(r=>r.physics_profile),runs[0].transactional_step_profile,originalNative.physics_profile].map(p=>[p.physics_profile_id,p]));
const assetManifest={field_id:runs[0].identity,field_sha256:assetSha,evaluator:'FSOP0001; immutable tensor-product C4 quintic; field fingerprint basis 5',
  experiment_manifest:experiment.experiment_manifest,physics_profiles:[...profileMap.values()].sort((a,b)=>a.physics_profile_id.localeCompare(b.physics_profile_id)),
  reference:{scalar_fidelity:'unknown',differential_fidelity:'unknown',singularity_information:'unknown',authoritative_refusals:[]},
  replacement_and_extension_during_rollout:'forbidden; each run owns its fixed field/profile pair'};
await writeFile('tmp/operational-field-asset-manifest.json',JSON.stringify(assetManifest,null,2)+'\n');
// Deterministic zip metadata; all filesystem arguments are separate argv entries.
const zip=spawnSync(resolve('.venv/Scripts/python.exe'),['-c',
  'import sys,zipfile; from pathlib import Path; z=zipfile.ZipFile(sys.argv[2],"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9);\nfor name,path in [("field.bin",sys.argv[1]),("manifest.json",sys.argv[3])]:\n info=zipfile.ZipInfo(name,(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16; z.writestr(info,Path(path).read_bytes())\nz.close()',asset,archive,'tmp/operational-field-asset-manifest.json'],{encoding:'utf8'});
required(zip.status===0,'Could not preserve frozen asset');
const preparedCost=runs.every(r=>r.cache_stress_queries.p99_us<=100 && r.unavailable_queries.p99_us<=100 && r.transactional_step.p99_us<=2000);
const mechanical=experiment.runs.every(r=>r.mechanical_expectation==='passed') && experiment.energy_runs.every(r=>r.passed);
const availability=experiment.runs.every(r=>r.availability==='complete');
const differentials=runs.every(r=>r.derivative_checks_pass) && experiment.join_differential_self_consistency.passed;
const outcomes={
  field_and_physics_profile_identity:'passed',
  analytic_chain_and_transaction_controls:'passed; 14 native executable controls',
  differential_self_consistency:differentials?'passed':'failed',
  reference_scalar_fidelity:'unknown; retained scalar-quality gate fails',
  reference_differential_fidelity:'unknown',
  reference_cut_policy_enforcement:'passed controls; corridor reference-singularity information unknown',
  frozen_corridor_mechanics:mechanical?'passed':'failed',
  original_native_controlled_crossing:originalNative.passed?'passed':originalNative.error?.includes('GeometryUnavailable')?'failed: coverage':'failed: mechanics',
  frozen_corridor_availability:availability?'passed':'failed',
  retained_energy_crossing_continuity_tests:{passed:python.tests.filter(t=>t.outcome==='passed').length,failed:python.tests.filter(t=>t.outcome==='failed').length,coverage_refusals:coverageFailures.length,scalar_quality_failures:qualityFailures.length},
  reference_shore_crossing:'unverified; no conclusive opposite-realm pair for the operational crossings',
  provisional_evaluation_cost:preparedCost && browser.provisional_targets.query_passed && browser.provisional_targets.step_passed?'passed':'failed',
  shipping_targets_ratified:false,retained_acceptance_passed:false,merge_ready:false,
  blockers:['coverage: two retained energy starts lie outside the frozen corridor','reference scalar and differential fidelity are unknown','reference crossing and corridor reference-singularity evidence are incomplete'],
};
if (!originalNative.passed) outcomes.blockers.unshift(originalNative.error?.includes('GeometryUnavailable')?'coverage: original native undriven trajectory leaves the frozen halo before its driven branch':'mechanics: original native controlled-crossing regression failed');
const timings=runs.map((r,i)=>({run:i,asset_loading_seconds:r.asset_loading_seconds,first_prepared_query_us:r.first_prepared_query_us,distinct_queries:r.distinct_queries,unavailable_queries:r.unavailable_queries,cache_stress_queries:r.cache_stress_queries,cache_stress_eviction_bytes:r.cache_stress_eviction_bytes,transactional_step:r.transactional_step,transactional_step_profile:r.transactional_step_profile,transactional_step_work:r.transactional_step_work,...metadata.runs[i]}));
const report={date:'2026-10-28',authorization:'ADR 0005 accepted for bounded prototype only; shipping adoption, cut-motion policy and device targets pending',
  field_id:runs[0].identity,asset:{sha256:assetSha,bytes:(await stat(asset)).size,archive,archive_sha256:await sha(archive),archive_bytes:(await stat(archive)).size,coefficient_bytes:runs[0].coefficient_bytes,manifest:assetManifest,preparation,reproduction_byte_identity_verified:true},
  source_sha256:hashes,native:{device:metadata.device,os:metadata.os,executable_sha256:metadata.executable_sha256,memory_note:metadata.memory_note,runs:timings,
    derivative_controls:{max_gradient_absolute_error:runs[0].neighbor_gradient_max_absolute_error,max_hessian_normalized_error:runs[0].neighbor_hessian_max_relative_error,tolerance:1e-4},
    measured_raster_max_scalar_difference:runs[0].measured_raster_max_scalar_difference},
  experiment,browser,retained_tests:python,original_native_crossing:originalNative,
  contract_tests:{command:['rtk',...command],exit_code:units.status,passed:14,summary:clean(summary.slice(-2500))},
  default_native_check:{exit_code:defaultCheck.status,summary:clean((defaultCheck.stdout+defaultCheck.stderr).slice(-1000))},
  outcomes};
// Keep complete ordered evidence without adding ~96,000 generated JSON lines
// to the review. The small index below links its SHA-256-checked exact payload.
const fullEvidence='tmp/operational-complete-evidence.json';
await writeFile(fullEvidence,JSON.stringify(report)+'\n');
const evidenceArchive='docs/validation/experiments/geometry-145-operational/evidence.zip';
const zipped=spawnSync(resolve('.venv/Scripts/python.exe'),['-c',
  'import sys,zipfile; from pathlib import Path; i=zipfile.ZipInfo("report.json",(1980,1,1,0,0,0)); i.compress_type=zipfile.ZIP_DEFLATED; i.external_attr=0o100644<<16; z=zipfile.ZipFile(sys.argv[2],"w"); z.writestr(i,Path(sys.argv[1]).read_bytes(),compresslevel=9); z.close()',fullEvidence,evidenceArchive],{encoding:'utf8'});
required(zipped.status===0,'Could not preserve full replay evidence');
const overview={...report,
  evidence:{archive:evidenceArchive,archive_sha256:await sha(evidenceArchive),archive_bytes:(await stat(evidenceArchive)).size,uncompressed_report_sha256:await sha(fullEvidence),note:'Exact full native/browser trajectories, reference certificates/unknowns and both endpoint ledgers are in report.json inside this archive'},
  experiment:{...experiment,runs:experiment.runs.map(r=>({name:r.name,field_id:r.field_id,physics_profile_id:r.physics_profile.physics_profile_id,dt:r.physics_profile.dt,committed_steps:r.committed_steps,mechanical_expectation:r.mechanical_expectation,availability:r.availability,operational_crossing:r.operational_crossing,reference_crossing:r.reference_crossing,crossing_time:r.crossing_time,failure:r.failure,field_and_profile_fixed:r.field_and_profile_fixed}))},
  browser:{...browser,results:{...browser.results,experiment:undefined}},
};
await writeFile('docs/validation/geometry-145-operational-prototype.json',JSON.stringify(overview,null,2)+'\n');
console.log(JSON.stringify({field_id:report.field_id,asset:report.asset.sha256,archive_bytes:report.asset.archive_bytes,outcomes},null,2));
// A well-formed report is not a green retained acceptance run.
if (!mechanical || !availability || !differentials || !originalNative.passed || python.pytest_exit_code!==0 || outcomes.provisional_evaluation_cost!=='passed') process.exitCode=1;
