import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  transformMeshToHyperbolic,
  transformTrailToHyperbolic,
  transformRiderToHyperbolic,
  HYPERBOLIC_VISUAL_SCALE,
} from '../hyperbolicCamera';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import { buildTerrainMesh } from '../cockpitScene';

describe('hyperbolic mesh, trail, and rider transformation', () => {
  const dummySnapshot: DebugSnapshot = {
    version: 'debug-snapshot/1',
    timeSeconds: 1.0,
    action: null,
    map: {
      pyramidLoaded: false,
      shoreProximity: null,
      minimapWindow: null,
      extent: null,
    },
    physics: {
      c: [-0.25, 0.0],
      velocity: [0.1, 0.0],
      signedDistance: 0.05,
      realm: 1,
      rho: 0.05001,
      sigma: 1.0,
      sigmaDot: 0.0,
      scaleGradient: [0.0, 0.0],
      metric: [1.0, 0.0, 1.0],
      metricSpeed: 0.1,
      kinetic: 0.005,
      potential: 1.0,
      total: 1.005,
      geodesicAccel: [0.0, 0.0],
      potentialForce: [0.0, 0.0],
      netAccel: [0.0, 0.0],
      derivativeValid: true,
    },
    diagnostics: {
      derivativeStep: 1e-4,
      valid: true,
      lastError: null,
      lastDeltaTotal: null,
      crestPotential: 10.0,
    },
  };

  const dummyPatch: TerrainPatch = {
    n: 3,
    center: [-0.25, 0.0],
    half: 0.1,
    // 9 points: row major
    // [x, y, z] with z = lambda * sigma(c)
    positions: [
      -0.35, 0.1, 1.0,  -0.25, 0.1, 1.0,  -0.15, 0.1, 1.0,
      -0.35, 0.0, 1.0,  -0.25, 0.0, 1.0,  -0.15, 0.0, 1.0,
      -0.35, -0.1, 1.0, -0.25, -0.1, 1.0, -0.15, -0.1, 1.0,
    ],
    signed: [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    realm: [1, 1, 1, 1, 1, 1, 1, 1, 1],
  };

  it('transforms terrain mesh into bounded Poincaré ball coordinates', () => {
    const mesh = buildTerrainMesh(dummyPatch, 'physical');
    transformMeshToHyperbolic(mesh, dummyPatch, dummySnapshot, 0);

    const pos = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    expect(pos).toBeDefined();

    for (let i = 0; i < dummyPatch.n * dummyPatch.n; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);

      // In Poincaré ball (unscaled), norm < 1.
      const norm = Math.hypot(x, y, z);
      expect(norm).toBeLessThan(1.0);
      expect(Number.isFinite(norm)).toBe(true);
    }
  });

  it('transforms rider to finite position inside Poincaré ball', () => {
    const rider = new THREE.Group();
    transformRiderToHyperbolic(rider, dummySnapshot, 0);

    expect(Number.isFinite(rider.position.x)).toBe(true);
    expect(Number.isFinite(rider.position.y)).toBe(true);
    expect(Number.isFinite(rider.position.z)).toBe(true);

    // Unscaled Poincaré distance from camera is < 1
    const unscaledNorm = rider.position.length() / HYPERBOLIC_VISUAL_SCALE;
    expect(unscaledNorm).toBeLessThan(1.0);
  });

  it('transforms trail into Poincaré ball coordinates', () => {
    const points = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 1, 1)];
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const trail = new THREE.Line(geom, new THREE.LineBasicMaterial());

    transformTrailToHyperbolic(trail, [dummySnapshot, dummySnapshot], dummySnapshot, 0);

    const posAttr = trail.geometry.getAttribute('position') as THREE.BufferAttribute;
    expect(posAttr).toBeDefined();
    for (let i = 0; i < 2; i++) {
      const norm = Math.hypot(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
      expect(norm).toBeLessThan(1.0);
    }
  });
});
