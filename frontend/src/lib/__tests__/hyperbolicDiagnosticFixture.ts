/** CPU inspection for legacy, hand-authored diagnostic fixtures only.
 * Production terrain stays static and is projected by the GPU.
 */
import * as THREE from 'three';
import type { DebugSnapshot, TerrainPatch } from '../debugCockpit';
import { computeHyperbolicCameraFrame as frame, projectPointToCameraPoincare, poincareToSceneVector } from '../hyperbolicCamera';

export function computeUpperHalfScaleA(lambda: number) { return lambda / Math.LN2; }
export function rhoFromEmbeddedHeight(z: number, dRef = 0.1, lambda = 1) { return dRef * 2 ** (-z / lambda); }

function fixtureSnapshot(snap: DebugSnapshot): DebugSnapshot {
  const { rho, scaleGradient: [gx, gy], sigmaDot } = snap.physics;
  return { ...snap, physics: { ...snap.physics, upperHalf: {
    a: 1 / Math.LN2, z: rho / Math.LN2, gradient: [-rho * gx, -rho * gy], zDot: -rho * sigmaDot,
  } } };
}

export function computeHyperbolicCameraFrame(snap: DebugSnapshot, heading: number | null) {
  return frame(fixtureSnapshot(snap), heading);
}

export function transformMeshToHyperbolic(mesh: THREE.Mesh, patch: TerrainPatch, snap: DebugSnapshot, heading: number | null) {
  const { cameraUpperHalf, rotationMatrix } = computeHyperbolicCameraFrame(snap, heading);
  const inverse = rotationMatrix.clone().invert();
  const position = mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < position.count; i++) {
    const z = patch.upperZ?.[i] ?? 0.1 * 2 ** -patch.positions[3 * i + 2] / Math.LN2;
    const projected = poincareToSceneVector(projectPointToCameraPoincare({
      x: patch.positions[3 * i], y: patch.positions[3 * i + 1], z,
    }, cameraUpperHalf)).applyMatrix4(inverse);
    position.setXYZ(i, projected.x, projected.y, projected.z);
  }
}
