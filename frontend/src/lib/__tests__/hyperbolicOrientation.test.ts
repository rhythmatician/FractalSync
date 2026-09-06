import { describe, it, expect } from 'vitest';
import {
  projectPointToCameraPoincare,
  poincareToSceneVector,
} from '../hyperbolicCamera';

describe('hyperbolic camera orientation in Three.js coordinates', () => {
  it('maps lookAt orientation so looking at target aligns forward vector with -Z', () => {
    // Camera C at (0, 0, 1), target T at (0, 1, 1).
    // In upper-half coordinates:
    // C = (0, 0, 1)
    // T = (0, 1, 1) -> +y direction in c-space.
    // In canonical upper half space:
    // T is at (0, 1, 1).
    // In Poincaré ball:
    // b = (0, 2/5, (0+1+1-1)/5) = (0, 0.4, 0.2).
    // Three.js scene vector unrotated:
    // poincareToSceneVector(b) = (b.x, b.z, -b.y) = (0, 0.2, -0.4).
    // Note that -b.y is negative, pointing towards -Z_scene!

    const c = { x: 0, y: 0, z: 1 };
    const t = { x: 0, y: 1, z: 1 };

    const b = projectPointToCameraPoincare(t, c);
    const sceneP = poincareToSceneVector(b);

    // Without rotation, target is at (0, 0.2, -0.4) -> forward along -Z_scene
    expect(sceneP.z).toBeLessThan(0);
    expect(sceneP.x).toBeCloseTo(0);
  });
});
