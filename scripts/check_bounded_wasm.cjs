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
const before = JSON.parse(engine.start(0, 0, 0.1, 0));
const after = JSON.parse(engine.step(0, 0));
assert.equal(before.field_id, fieldId);
assert.equal(after.physics_profile_id, before.physics_profile_id);
assert.equal(after.mechanical_time, 0.01);
const saved = engine.state();
assert.throws(() => engine.step(1e300, 0));
assert.equal(engine.state(), saved);
const cut = JSON.stringify({ field_id: fieldId, refusals: [{ bounds: [-0.01, -0.01, 0.01, 0.01], authority: "synthetic reference cut" }] });
const blocked = new BoundedOperationalRollout(asset, cut, 0.1, 0.02, 0, 0, 0, 0.01, 0);
assert.throws(() => blocked.start(0, 0, 0, 0), /ReferenceSingularity/);
console.log(JSON.stringify({ field_id: fieldId, physics_profile_id: after.physics_profile_id, state: after, transaction_and_reference_refusal: "passed" }));
