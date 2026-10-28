// Browser validation only. Geometry, mechanics and reference evidence run in Rust.
import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import { createServer } from 'node:http';
import { readFile, writeFile, stat } from 'node:fs/promises';
import { resolve, basename } from 'node:path';
import { createHash } from 'node:crypto';
import { cpus } from 'node:os';

const [assetArg, nativeArg, outputArg] = process.argv.slice(2);
if (!outputArg) throw new Error('usage: node scripts/validate_operational_field_browser.mjs asset.bin native-report.json browser-report.json');
const asset = await readFile(resolve(assetArg));
const wasmDir = resolve('tmp/operational-field-wasm');
const native = JSON.parse(await readFile(resolve(nativeArg), 'utf8'));
const names = ['orbit_synth_wasm.js', 'orbit_synth_wasm_bg.wasm'];
const files = new Map(await Promise.all(names.map(async name => ['/' + name, await readFile(resolve(wasmDir, name))])));
files.set('/asset.bin', asset);
const server = createServer((req, res) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  if (req.url === '/') { res.setHeader('Content-Type', 'text/html'); res.end('<!doctype html><title>Frozen operational experiment</title>'); return; }
  const bytes = files.get(req.url);
  if (!bytes) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', req.url.endsWith('.js') ? 'text/javascript' : req.url.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
  res.end(bytes);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const results = await page.evaluate(async () => {
    const mod = await import('/orbit_synth_wasm.js');
    let started = performance.now();
    const wasm = await mod.default();
    const wasm_initialization_ms = performance.now() - started;
    started = performance.now();
    const asset = new Uint8Array(await (await fetch('/asset.bin')).arrayBuffer());
    const asset_fetch_ms = performance.now() - started;
    started = performance.now();
    const field = new mod.ExperimentalOperationalField(asset);
    const asset_decode_ms = performance.now() - started;
    const identity_before = field.identity();
    started = performance.now();
    const first = field.query(0.4, 0.05);
    const first_prepared_query_us = (performance.now() - started) * 1000;
    function stats(times) {
      const first_us = times[0]; times.sort((a,b) => a-b);
      return { samples: times.length, first_us, p50_us: times[Math.floor(times.length/2)], p99_us: times[Math.floor(times.length*0.99)], max_us: times.at(-1) };
    }
    const queries = [], misses = [];
    for (let i=0;i<20000;i++) {
      const x=0.2+(i%997)/997*0.4, y=0.025+(i%293)/293*0.15;
      started=performance.now(); field.query(x,y); queries.push((performance.now()-started)*1000);
      started=performance.now(); const miss=field.query(2,y); misses.push((performance.now()-started)*1000);
      if (miss.length) throw new Error('Unavailable lookup returned a jet');
    }
    const eviction=new Uint32Array(16*1024*1024), cold=[];
    globalThis.__operationalEviction=eviction;
    for (let i=0;i<1000;i++) {
      for (let j=0;j<eviction.length;j+=16) eviction[j] += i;
      started=performance.now(); field.query(0.25+i*0.0003,0.05); cold.push((performance.now()-started)*1000);
    }
    const after=field.query(0.4,0.05);
    const jet_invariance=Array.from(first).every((v,i) => Object.is(v,after[i]));
    const steps=[]; let last;
    for (let i=0;i<1000;i++) {
      field.reset(0.4+i*1e-6,0.05,0.001,0);
      started=performance.now(); const text=field.step(); steps.push((performance.now()-started)*1000);
      last=JSON.parse(text);
    }
    const experiment=JSON.parse(field.report());
    return { user_agent:navigator.userAgent, cross_origin_isolated:crossOriginIsolated,
      wasm_initialization_ms, asset_fetch_ms, asset_decode_ms, asset_bytes:asset.length,
      first_prepared_query_us, distinct_queries:stats(queries), unavailable_queries:stats(misses),
      cache_stress_queries:stats(cold), cache_stress_eviction_bytes:eviction.byteLength,
      transactional_step:stats(steps), transactional_step_work:last.work,
      transactional_step_profile_id:last.state.physics_profile_id,
      field_id:field.identity(), identity_fixed:identity_before===field.identity(), jet_invariance,
      wasm_linear_memory_bytes:wasm.memory.buffer.byteLength, experiment,
      timing_note:'Browser bridge and return-value allocation/serialization included. Asset fetch/decoding and WASM initialization excluded from queries. Single-call timing is quantized by the browser timer.' };
  });
  const experiment = native.frozen_experiment;
  const spec = experiment.experiment_manifest;
  const differences = [];
  if (results.field_id !== native.identity) differences.push('field identity differs for identical asset bytes');
  for (let i=0;i<experiment.parity_samples.length;i++) {
    const a=experiment.parity_samples[i], b=results.experiment.parity_samples[i];
    if (a.availability!==b.availability) {differences.push(`availability ${i}`);continue;}
    if (a.jet) for (let j=0;j<6;j++) {
      const tolerance=j===0 ? spec.parity_scalar_abs_tolerance : j<3 ? spec.parity_gradient_abs_tolerance : spec.parity_hessian_relative_tolerance*Math.max(1,Math.abs(a.jet[j]));
      if (Math.abs(a.jet[j]-b.jet[j])>tolerance) differences.push(`jet ${i}/${j}`);
    }
  }
  if (experiment.runs.length !== results.experiment.runs.length) differences.push('rollout count');
  for (let i=0;i<experiment.runs.length;i++) {
    const a=experiment.runs[i], b=results.experiment.runs[i];
    for (const key of ['availability','operational_crossing','reference_crossing','mechanical_expectation']) if (a[key]!==b[key]) differences.push(`rollout ${i} ${key}`);
    if (a.physics_profile.physics_profile_id!==b.physics_profile.physics_profile_id) differences.push(`profile ${i}`);
    for (const key of ['position','velocity']) for (let j=0;j<2;j++) if (Math.abs(a.final[key][j]-b.final[key][j])>1e-6*Math.max(1,Math.abs(a.final[key][j]))) differences.push(`state ${i}/${key}/${j}`);
    for (const key of ['energy','mechanical_time','integrated_work']) if (Math.abs(a.final[key]-b.final[key])>1e-6*Math.max(1,Math.abs(a.final[key]))) differences.push(`ledger ${i}/${key}`);
  }
  const report = { device:cpus()[0].model, browser:browser.version(), date:'2026-10-28',
    asset_sha256:createHash('sha256').update(asset).digest('hex'),
    wasm_sha256:createHash('sha256').update(files.get('/orbit_synth_wasm_bg.wasm')).digest('hex'),
    wasm_file_bytes:(await stat(resolve(wasmDir,'orbit_synth_wasm_bg.wasm'))).size,
    results, parity:{ passed:differences.length===0, differences, state_and_ledger_relative_tolerance:1e-6,
      scalar_abs_tolerance:spec.parity_scalar_abs_tolerance, gradient_abs_tolerance:spec.parity_gradient_abs_tolerance, hessian_relative_tolerance:spec.parity_hessian_relative_tolerance },
    provisional_targets:{query_p99_us:100,step_p99_us:2000,query_passed:results.cache_stress_queries.p99_us<=100,step_passed:results.transactional_step.p99_us<=2000,shipping_ratified:false} };
  await writeFile(resolve(outputArg),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({browser:report.browser,field_id:results.field_id,query:results.cache_stress_queries,step:results.transactional_step,parity:report.parity,output:basename(outputArg)},null,2));
  if (!report.parity.passed) process.exitCode=1;
} finally {
  await browser?.close();
  await new Promise(done => server.close(done));
}
