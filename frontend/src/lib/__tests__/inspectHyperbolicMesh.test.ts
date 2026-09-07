import { describe, it } from 'vitest';
import * as THREE from 'three';
import {
  computeHyperbolicCameraFrame,
  transformMeshToHyperbolic,
} from './hyperbolicDiagnosticFixture';
import { HYPERBOLIC_VISUAL_SCALE } from '../hyperbolicCamera';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import { buildTerrainMesh } from '../cockpitScene';

describe('inspect mesh vertices in hyperbolic mode', () => {
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
      c: [-0.744, 0.132],
      velocity: [-0.000023, 0.0],
      signedDistance: 0.0026695,
      realm: 1,
      rho: 0.0026714,
      sigma: 5.22627,
      sigmaDot: -0.00416,
      scaleGradient: [184.0, 0.0],
      metric: [1.0, 0.0, 1.0],
      metricSpeed: 0.00942,
      kinetic: 0.00004,
      potential: 5.22627,
      total: 5.22631,
      geodesicAccel: [0.0, 0.0],
      potentialForce: [-184.0, 0.0],
      netAccel: [0.001, 0.0],
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
    n: 5,
    center: [-0.744, 0.132],
    half: 0.01,
    positions: [],
    signed: [],
    realm: [],
  };

  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const x = -0.744 - 0.01 + (c / 4) * 0.02;
      const y = 0.132 - 0.01 + (r / 4) * 0.02;
      // Let center be a crest (sigma=9.9, rho=1e-4), and edges be lower (sigma=5, rho=0.003)
      const dist = Math.hypot(x - (-0.744), y - 0.132);
      const sigma = 9.9 - dist * 400;
      dummyPatch.positions.push(x, y, sigma);
      dummyPatch.signed.push(0.002);
      dummyPatch.realm.push(1);
    }
  }

  it('prints camera UpperHalf and vertex positions', () => {
    const { cameraUpperHalf } = computeHyperbolicCameraFrame(dummySnapshot, 0);
    console.log('cameraUpperHalf:', cameraUpperHalf);
    console.log('rider c:', dummySnapshot.physics.c, 'rho:', dummySnapshot.physics.rho);

    const mesh = buildTerrainMesh(dummyPatch, 'physical');
    transformMeshToHyperbolic(mesh, dummyPatch, dummySnapshot, 0);

    const pos = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const centerIdx = 2 * 5 + 2;
    console.log('Center vertex (rider position):', {
      x: pos.getX(centerIdx) * HYPERBOLIC_VISUAL_SCALE,
      y: pos.getY(centerIdx) * HYPERBOLIC_VISUAL_SCALE,
      z: pos.getZ(centerIdx) * HYPERBOLIC_VISUAL_SCALE,
    });
    console.log('Corner vertex (0,0):', {
      x: pos.getX(0) * HYPERBOLIC_VISUAL_SCALE,
      y: pos.getY(0) * HYPERBOLIC_VISUAL_SCALE,
      z: pos.getZ(0) * HYPERBOLIC_VISUAL_SCALE,
    });
  });
});
