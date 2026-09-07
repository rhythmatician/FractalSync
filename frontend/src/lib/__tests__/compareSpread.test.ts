import { describe, it } from 'vitest';
import * as THREE from 'three';
import {
  transformMeshToHyperbolic,
} from './hyperbolicDiagnosticFixture';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import { buildTerrainMesh } from '../cockpitScene';

describe('compare low sigma vs high sigma in hyperbolic mode', () => {
  function makeSnapshot(sigma: number, rho: number, cx = -0.75, cy = 0.1): DebugSnapshot {
    return {
      version: 'debug-snapshot/1',
      timeSeconds: 1.0,
      action: null,
      map: { pyramidLoaded: false, shoreProximity: null, minimapWindow: null, extent: null },
      physics: {
        c: [cx, cy],
        velocity: [0.0, 0.0],
        signedDistance: rho,
        realm: 1,
        rho: rho,
        sigma: sigma,
        sigmaDot: 0.0,
        scaleGradient: [0.0, 0.0],
        metric: [1.0, 0.0, 1.0],
        metricSpeed: 0.0,
        kinetic: 0.0,
        potential: sigma,
        total: sigma,
        geodesicAccel: [0.0, 0.0],
        potentialForce: [0.0, 0.0],
        netAccel: [0.0, 0.0],
        derivativeValid: true,
      },
      diagnostics: { derivativeStep: 1e-4, valid: true, lastError: null, lastDeltaTotal: null, crestPotential: 10.0 },
    };
  }

  it('compares mesh spread at valley (rho=0.25) vs crest (rho=0.001)', () => {
    // A patch with two peaks separated by delta_c = 0.02
    // Peak 1 at cx - 0.01, Peak 2 at cx + 0.01
    const cx = -0.75;
    const cy = 0.1;
    const half = 0.02;

    const patchValley: TerrainPatch = {
      n: 3,
      center: [cx, cy],
      half,
      positions: [
        cx - 0.01, cy, 0.0, // Peak 1
        cx, cy, -1.0,       // Center
        cx + 0.01, cy, 0.0, // Peak 2
        cx - 0.01, cy, 0.0,
        cx, cy, -1.0,
        cx + 0.01, cy, 0.0,
        cx - 0.01, cy, 0.0,
        cx, cy, -1.0,
        cx + 0.01, cy, 0.0,
      ],
      signed: [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1],
      realm: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    };

    // When rider is in valley (rho=0.25):
    const snapValley = makeSnapshot(-1.35, 0.25, cx, cy);
    const meshValley = buildTerrainMesh(patchValley, 'physical');
    transformMeshToHyperbolic(meshValley, patchValley, snapValley, 0);
    const posV = meshValley.geometry.getAttribute('position') as THREE.BufferAttribute;

    const p1_v = new THREE.Vector3(posV.getX(0), posV.getY(0), posV.getZ(0));
    const p2_v = new THREE.Vector3(posV.getX(2), posV.getY(2), posV.getZ(2));
    const distValley = p1_v.distanceTo(p2_v);

    // When rider is at crest (rho=0.001, high sigma):
    const snapCrest = makeSnapshot(6.64, 0.001, cx, cy);
    const meshCrest = buildTerrainMesh(patchValley, 'physical');
    transformMeshToHyperbolic(meshCrest, patchValley, snapCrest, 0);
    const posC = meshCrest.geometry.getAttribute('position') as THREE.BufferAttribute;

    const p1_c = new THREE.Vector3(posC.getX(0), posC.getY(0), posC.getZ(0));
    const p2_c = new THREE.Vector3(posC.getX(2), posC.getY(2), posC.getZ(2));
    const distCrest = p1_c.distanceTo(p2_c);

    console.log('Distance between peaks when rider is at rho=0.25 (valley):', distValley);
    console.log('Distance between peaks when rider is at rho=0.001 (crest):', distCrest);
  });
});
