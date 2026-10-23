// Run after building wasm-orbit and copying pkg into frontend/src/wasm.
// Exercises real bindings: mocks cannot detect undefined-vs-null drift.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { initSync, OrbitController, ManifoldConfig, debugSnapshotFromState } from '../src/wasm/orbit_synth_wasm.js';

initSync({ module: readFileSync(new URL('../src/wasm/orbit_synth_wasm_bg.wasm', import.meta.url)) });

function assertPartial(snapshot, classification) {
  assert.equal(snapshot.version, 'debug-snapshot/5');
  assert.equal(snapshot.diagnostics.valid, false);
  assert.equal(snapshot.diagnostics.geometry.validity, classification);
  assert.ok(snapshot.diagnostics.lastError);
  for (const key of ['metric', 'scaleGradient', 'upperHalf', 'kinetic', 'total', 'netAccel']) {
    assert.equal(snapshot.physics[key], null, `${key} must be explicit null`);
    assert.equal(JSON.parse(JSON.stringify(snapshot)).physics[key], null);
  }
  assert.equal(snapshot.diagnostics.geometry.hessianNorm, null);
}

const defaults = ManifoldConfig.defaults();
for (const [x, y, kind, epsilon] of [[-1, 0, 'singular', 1e-4], [0.25, 0.5, 'unresolved', 1e-100]]) {
  const config = new ManifoldConfig(defaults.d_ref, epsilon, defaults.lambda_sq, defaults.kappa, defaults.mu);
  const snapshot = debugSnapshotFromState(x, y, 0.2, 0.1, null, 0, 0, 0, config, 0, 0);
  assertPartial(snapshot, kind);
  assert.deepEqual(snapshot.physics.c, [x, y]);
  assert.deepEqual(snapshot.physics.velocity, [0.2, 0.1]);
  config.free();
}
defaults.free();

const automatic = new OrbitController(1, 1, 1);
automatic.manifold_physics = true;
assert.deepEqual(automatic.debugSnapshot().physics.c, [-0.5, 0.1]);
assert.equal(automatic.debugSnapshot().diagnostics.valid, true);
automatic.free();

const explicit = new OrbitController(1, 1, 1);
explicit.setC(0, 0);
explicit.manifold_physics = true;
assert.equal(explicit.debugSnapshot().diagnostics.geometry.validity, 'regular');
assert.deepEqual(explicit.debugSnapshot().physics.c, [0, 0]);
explicit.free();

const singular = new OrbitController(1, 1, 1);
singular.setC(-1, 0);
singular.manifold_physics = true;
assertPartial(singular.debugSnapshot(), 'singular');
assert.deepEqual(singular.debugSnapshot().physics.c, [-1, 0]);
singular.free();
console.log('Real WASM snapshot/startup contract passed');
