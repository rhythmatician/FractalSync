// Run against the separately built Node-target WASM package; no browser claim.
const assert = require("node:assert/strict");
const path = require("node:path");

const { BoundedOperationalRollout } = require(path.resolve(process.argv[2]));
const asset = Buffer.alloc(56 + 144 * 8);
asset.write("FSOP0001", 0);
asset.writeDoubleLE(-0.5, 8);
asset.writeDoubleLE(-0.5, 16);
asset.writeDoubleLE(0.1, 24);
asset.writeBigUInt64LE(12n, 32);
asset.writeBigUInt64LE(12n, 40);
asset.writeBigUInt64LE(0n, 48);
for (let i = 0; i < 144; i++) asset.writeDoubleLE(1.0, 56 + i * 8);
const fieldId = BoundedOperationalRollout.asset_field_id(asset);
const policy = JSON.stringify({ field_id: fieldId, refusals: [] });
for (const [kappa, mu] of [[-0.1, 0], [0, -0.1]]) {
  assert.throws(() => new BoundedOperationalRollout(asset, policy, 0.1, 0.02, 0, kappa, mu, 0.01, 0), /invalid experimental Physics profile/);
}
const engine = new BoundedOperationalRollout(asset, policy, 0.1, 0.02, 0, 0, 0, 0.01, 0);
const routeQuery = {
  window: [[-0.125, -0.125], [0.125, 0.125]], gridSteps: 4,
  target: { center: [0.125, 0], radius: 0 },
  nominalTiming: { startSample: 480, deadlineSample: 48_480, kineticActionBudget: 0.005 },
};
assert.throws(() => engine.route_guidance(JSON.stringify(routeQuery)), /rollout not started/);
const before = JSON.parse(engine.start(0, 0, 0.1, 0));
const after = JSON.parse(engine.step(0, 0));
assert.equal(before.field_id, fieldId);
assert.equal(after.physics_profile_id, before.physics_profile_id);
assert.equal(after.mechanical_time, 0.01);
const saved = engine.state();
const route = JSON.parse(engine.route_guidance(JSON.stringify(routeQuery)));
assert.equal(route.metricScope, "exactPreparedSplineInWindow");
assert.equal(route.status, "routeFound");
assert.equal(route.fieldId, fieldId);
assert.deepEqual(route.path[0], after.position);
assert.deepEqual(route.path.at(-1), routeQuery.target.center);
assert.ok(route.graphLengthLower <= route.routeLengthUpper && route.routeLengthUpper < 0.125);
assert.ok(route.nextDirection[0] > 0 && Math.abs(route.nextDirection[1]) < 1e-12);
assert.equal(route.nominalFeasibility.kineticActionStatus, "ruledOut");
assert.equal(route.nominalFeasibility.deadlineStatus, "insufficientEvidence");
assert.equal(engine.state(), saved);
assert.throws(() => engine.step(1e300, 0));
assert.equal(engine.state(), saved);
const cut = JSON.stringify({ field_id: fieldId, refusals: [{ bounds: [-0.01, -0.01, 0.01, 0.01], authority: "synthetic reference cut" }] });
const blocked = new BoundedOperationalRollout(asset, cut, 0.1, 0.02, 0, 0, 0, 0.01, 0);
assert.throws(() => blocked.start(0, 0, 0, 0), /ReferenceSingularity/);
console.log(JSON.stringify({ field_id: fieldId, physics_profile_id: after.physics_profile_id, state: after, route: route.status, transaction_and_reference_refusal: "passed" }));
