import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  canonicalizeUpperHalf,
  upperHalfToPoincareBall,
  projectPointToCameraPoincare,
  poincareHyperbolicDistance,
} from '../hyperbolicCamera';

describe('hyperbolicCamera projection math', () => {
  const a = 1.7; // Arbitrary curvature radius supplied by the geometry authority.

  it('computes Poincaré ball coordinates directly via upperHalfToPoincareBall', () => {
    const canonical = { x: 0, y: 0, z: 1 };
    const b = upperHalfToPoincareBall(canonical);
    expect(b.x).toBeCloseTo(0, 10);
    expect(b.y).toBeCloseTo(0, 10);
    expect(b.z).toBeCloseTo(0, 10);
  });

  it('camera maps to origin: P = C maps to b = (0, 0, 0)', () => {
    const cameraPos = { x: 0.123, y: -0.456, z: 0.789 };
    const p = { x: 0.123, y: -0.456, z: 0.789 };

    const b = projectPointToCameraPoincare(p, cameraPos);
    expect(b.x).toBeCloseTo(0, 10);
    expect(b.y).toBeCloseTo(0, 10);
    expect(b.z).toBeCloseTo(0, 10);
    expect(b.length()).toBeCloseTo(0, 10);
  });

  it('canonical camera C = (0, 0, 1): normalization is identity', () => {
    const c = { x: 0, y: 0, z: 1 };
    const p = { x: 0.5, y: -0.25, z: 2.0 };
    const canonical = canonicalizeUpperHalf(p, c);

    expect(canonical.x).toBeCloseTo(p.x, 10);
    expect(canonical.y).toBeCloseTo(p.y, 10);
    expect(canonical.z).toBeCloseTo(p.z, 10);
  });

  it('valid points remain inside ball: ||b|| < 1 for upper half space z > 0', () => {
    const camera = { x: 0.2, y: -0.3, z: 0.5 };
    const testPoints = [
      { x: 0.2, y: -0.3, z: 0.5 }, // at camera -> ||b|| = 0
      { x: 0.0, y: 0.0, z: 1.0 },
      { x: 10.0, y: -20.0, z: 0.01 },
      { x: -5.0, y: 15.0, z: 100.0 },
      { x: 0.2, y: -0.3, z: 0.0001 },
      { x: 100.0, y: 100.0, z: 0.001 },
    ];

    for (const pt of testPoints) {
      const b = projectPointToCameraPoincare(pt, camera);
      expect(b.length()).toBeLessThan(1.0);
      expect(Number.isFinite(b.x)).toBe(true);
      expect(Number.isFinite(b.y)).toBe(true);
      expect(Number.isFinite(b.z)).toBe(true);
    }
  });

  it('scale invariance: (x, y, z) -> (alpha*x, alpha*y, alpha*z) leaves Poincaré coordinates unchanged', () => {
    const camera = { x: -0.15, y: 0.35, z: 0.08 };
    const points = [
      { x: -0.15, y: 0.35, z: 0.08 },
      { x: 0.0, y: 0.2, z: 0.04 },
      { x: -0.3, y: 0.5, z: 0.12 },
      { x: 0.5, y: -0.2, z: 0.005 },
    ];

    const alphas = [1e-12, 0.001, 0.1, 2.5, 100.0, 1e4];

    for (const alpha of alphas) {
      const scaledCamera = { x: camera.x * alpha, y: camera.y * alpha, z: camera.z * alpha };
      for (const p of points) {
        const scaledP = { x: p.x * alpha, y: p.y * alpha, z: p.z * alpha };

        const bOrig = projectPointToCameraPoincare(p, camera);
        const bScaled = projectPointToCameraPoincare(scaledP, scaledCamera);

        expect(bScaled.x).toBeCloseTo(bOrig.x, 9);
        expect(bScaled.y).toBeCloseTo(bOrig.y, 9);
        expect(bScaled.z).toBeCloseTo(bOrig.z, 9);
        expect(bScaled.length()).toBeCloseTo(bOrig.length(), 9);
      }
    }
  });

  it('orientation: rotating the camera rotates direction without altering hyperbolic radial distance ||b||', () => {
    const camera = { x: 0.1, y: 0.2, z: 0.3 };
    const p = { x: -0.4, y: 0.8, z: 0.15 };

    const bUnrotated = projectPointToCameraPoincare(p, camera);
    const distUnrotated = bUnrotated.length();

    // Create arbitrary rotation quaternions
    const eulerAngles = [
      new THREE.Euler(0.5, 0.2, -0.7),
      new THREE.Euler(Math.PI / 4, -Math.PI / 3, 0.1),
      new THREE.Euler(-1.2, 0.0, 2.1),
    ];

    for (const euler of eulerAngles) {
      const q = new THREE.Quaternion().setFromEuler(euler);
      const bRotated = projectPointToCameraPoincare(p, camera, q);

      // Hyperbolic radial distance ||b|| must be invariant under camera rotation
      expect(bRotated.length()).toBeCloseTo(distUnrotated, 9);

      // But the vector direction itself should change
      expect(bRotated.equals(bUnrotated)).toBe(false);

      // And inverse rotation back recovers bUnrotated
      const recovered = bRotated.clone().applyQuaternion(q);
      expect(recovered.x).toBeCloseTo(bUnrotated.x, 9);
      expect(recovered.y).toBeCloseTo(bUnrotated.y, 9);
      expect(recovered.z).toBeCloseTo(bUnrotated.z, 9);
    }
  });

  it('hyperbolic distance relation: s = 2 * a * atanh(||b||)', () => {
    const camera = { x: 0, y: 0, z: 1 };
    // Geodesic along Z axis from (0,0,1) to (0,0,e^s0):
    // In upper half-space, hyperbolic distance along vertical ray is |ln(z2 / z1)| * a
    const sTarget = 1.5;
    const zTarget = Math.exp(sTarget / a);
    const p = { x: 0, y: 0, z: zTarget };

    const b = projectPointToCameraPoincare(p, camera);
    const sComputed = poincareHyperbolicDistance(b, a);

    expect(sComputed).toBeCloseTo(sTarget, 9);
  });
});
