import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { projectPointToCameraPoincare, projectUpperHalfTangent, poincareToSceneVector } from '../hyperbolicCamera';

interface Case {
  point: [number, number, number];
  camera: [number, number, number];
  direction: [number, number, number];
  projected: [number, number, number];
  tangent: [number, number, number];
}
const goldens: { hyperbolic_cases: Case[] } = JSON.parse(readFileSync(resolve(__dirname, '../../../../shared/golden_vectors.json'), 'utf8'));
const point = ([x, y, z]: number[]) => ({ x, y, z });

describe('hyperbolic rendering parity with Rust', () => {
  it('pins the CPU projection and differential over scales and heights', () => {
    expect(goldens.hyperbolic_cases.length).toBeGreaterThanOrEqual(16);
    for (const c of goldens.hyperbolic_cases) {
      const projected = projectPointToCameraPoincare(point(c.point), point(c.camera));
      projected.toArray().forEach((v, i) => expect(v).toBeCloseTo(c.projected[i], 12));
      const tangent = projectUpperHalfTangent(point(c.point), new THREE.Vector3(...c.direction), point(c.camera));
      const expected = poincareToSceneVector(new THREE.Vector3(...c.tangent)).toArray();
      tangent.toArray().forEach((v, i) => expect(v).toBeCloseTo(expected[i], 12));
    }
  });
});
