// Browser integration and timings; asset preparation and loading are separate.
import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { cpus } from 'node:os';
import { createHash } from 'node:crypto';

const [assetPath, policyPath, playwrightPath, outputPath, wasmDirectory = 'tmp/final-web', mode] = process.argv.slice(2);
if (!outputPath || (mode && mode !== '--verify-cusp-route')) {
  throw new Error('usage: asset policy playwright-module output [wasm-directory [--verify-cusp-route]]');
}
const { chromium } = await import(pathToFileURL(resolve(playwrightPath)).href);
const asset = await readFile(assetPath), policy = await readFile(policyPath);
const js = await readFile(join(wasmDirectory, 'orbit_synth_wasm.js'));
const wasm = await readFile(join(wasmDirectory, 'orbit_synth_wasm_bg.wasm'));
const files = new Map([['/asset', asset], ['/policy', policy], ['/engine.js', js], ['/orbit_synth_wasm_bg.wasm', wasm]]);
const server = createServer((req, res) => {
  if (req.url === '/') { res.end('<!doctype html><title>Bounded engine validation</title>'); return; }
  const data = files.get(req.url);
  if (!data) { res.writeHead(404); res.end(); return; }
  res.setHeader('Content-Type', req.url.endsWith('.js') ? 'text/javascript' : req.url.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
  res.end(data);
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const result = await page.evaluate(async verifyRoute => {
    const mod = await import('/engine.js');
    let t = performance.now(); await mod.default(); const wasm_init_ms = performance.now() - t;
    t = performance.now(); const asset = new Uint8Array(await (await fetch('/asset')).arrayBuffer()); const fetch_ms = performance.now() - t;
    const policy = await (await fetch('/policy')).text();
    t = performance.now(); const engine = new mod.BoundedOperationalRollout(asset, policy, 0.1, 1e-4, 1, 0.1, 1 / Math.PI, 0.001, 0); const load_ms = performance.now() - t;
    t = performance.now(); const probe = JSON.parse(engine.probe(0.4, 0.05)); const first_probe_us = (performance.now() - t) * 1000;
    const measures = [];
    for (let i = 0; i < 5000; i++) { t = performance.now(); engine.probe(0.4 + (i % 101 - 50) * 1e-6, 0.05); measures.push((performance.now() - t) * 1000); }
    engine.start(0.4, 0.05, -0.001, 0);
    let state; const steps = [];
    for (let i = 0; i < 500; i++) { t = performance.now(); state = JSON.parse(engine.step(0, 0)); steps.push((performance.now() - t) * 1000); if (i === 9) globalThis.parityState = state; }
    const first_step_us = steps[0]; measures.sort((a, b) => a - b); steps.sort((a, b) => a - b);
    const engineReport = { wasm_init_ms, fetch_ms, load_ms, first_probe_us, probe_p99_us: measures[4949], probe_max_us: measures[4999], first_step_us, step_p99_us: steps[494], step_max_us: steps[499], probe, parity_state: globalThis.parityState };
    if (!verifyRoute) return engineReport;
    const request = {
      window: [[450 / 2048, -41 / 2048], [550 / 2048, 39 / 2048]], gridSteps: 64,
      target: { center: [537 / 2048, -1 / 2048], radius: 0.001 },
      nominalTiming: { startSample: 48, deadlineSample: 48_048, kineticActionBudget: 0.05 },
    };
    engine.start(452 / 2048, -1 / 2048, 0.001, 0);
    const saved = engine.step(0, 0);
    const position = JSON.parse(saved).position;
    t = performance.now();
    const route = JSON.parse(engine.route_guidance(JSON.stringify(request)));
    const route_ms = performance.now() - t;
    if (route.status !== 'routeFound' || route.metricScope !== 'exactPreparedSplineInWindow'
        || !route.path[0] || route.path[0].some((x, i) => x !== position[i]) || engine.state() !== saved
        || route.graphLengthLower > route.routeLengthUpper || !route.nextDirection?.every(Number.isFinite)) {
      throw new Error('Missing read-only route from the committed position');
    }
    const arrived = route.path.at(-1);
    if (Math.hypot(arrived[0] - request.target.center[0], arrived[1] - request.target.center[1]) > request.target.radius) {
      throw new Error('Route ended outside the target ball');
    }
    if (route.nominalFeasibility?.kineticActionStatus !== 'ruledOut'
        || route.nominalFeasibility.deadlineStatus !== 'insufficientEvidence') {
      throw new Error('Unexpected nominal timing assessment');
    }
    return { ...engineReport, route, route_ms };
  }, mode === '--verify-cusp-route');
  const report = { browser: browser.version(), device: cpus()[0].model, asset_sha256: createHash('sha256').update(asset).digest('hex'), policy_sha256: createHash('sha256').update(policy).digest('hex'), wasm_sha256: createHash('sha256').update(wasm).digest('hex'), result, note: 'Probe includes metric, potential and JSON; not a single geometry-query timing. Timer quantization and no cache stress apply.' };
  await writeFile(outputPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await browser?.close(); await new Promise(done => server.close(done)); }
