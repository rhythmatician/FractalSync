import { computeUpperHalfScaleA } from './hyperbolicDiagnosticFixture';
import { describe, it } from 'vitest';
import {
  canonicalizeUpperHalf,
  upperHalfToPoincareBall,
} from '../hyperbolicCamera';

describe('trace why distCrest was smaller', () => {
  it('traces coordinates', () => {
    const a = computeUpperHalfScaleA(1.0);
    // Peak at cx - 0.01, z = a * 0.001
    // Camera at cx, cz = a * 0.001 * 1.35
    const cz = a * 0.001 * 1.35;
    const p1 = { x: -0.01, y: 0, z: a * 0.001 };
    const p2 = { x: 0.01, y: 0, z: a * 0.001 };

    const c = { x: 0, y: 0, z: cz };
    const can1 = canonicalizeUpperHalf(p1, c);
    const can2 = canonicalizeUpperHalf(p2, c);

    console.log('Canonical X for p1 (rho=0.001):', can1.x, 'Z:', can1.z);
    console.log('Canonical X for p2 (rho=0.001):', can2.x, 'Z:', can2.z);

    const b1 = upperHalfToPoincareBall(can1);
    const b2 = upperHalfToPoincareBall(can2);
    console.log('b1:', b1);
    console.log('b2:', b2);
    console.log('Poincare distance between p1 and p2:', b1.distanceTo(b2));

    // Now for valley (rho=0.25):
    const cz_v = a * 0.25 * 1.35;
    const p1_v = { x: -0.01, y: 0, z: a * 0.25 };
    const p2_v = { x: 0.01, y: 0, z: a * 0.25 };
    const c_v = { x: 0, y: 0, z: cz_v };
    const can1_v = canonicalizeUpperHalf(p1_v, c_v);
    const b1_v = upperHalfToPoincareBall(can1_v);
    const b2_v_correct = upperHalfToPoincareBall(canonicalizeUpperHalf(p2_v, c_v));
    console.log('Poincare distance in valley (rho=0.25):', b1_v.distanceTo(b2_v_correct));
  });
});
