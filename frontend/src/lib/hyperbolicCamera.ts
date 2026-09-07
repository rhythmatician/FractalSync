/**
 * Hyperbolic camera projection for the FractalSync debug cockpit.
 *
 * Geometric background:
 * In upper half-space H^3 with metric ds^2 = a^2 (dx^2 + dy^2 + dz^2) / z^2,
 * where a = lambda / ln(2) and z = a * rho(x, y),
 * the cockpit views the 2D graph z(x, y) = a * rho(x, y).
 *
 * Pipeline:
 * 1. Upper-half-space point P = (x, y, z) with z = a * rho > 0.
 * 2. Hyperbolic isometry moving camera C = (cx, cy, cz) (with cz > 0) to canonical point (0, 0, 1):
 *      X = (x - cx) / cz
 *      Y = (y - cy) / cz
 *      Z = z / cz
 * 3. Cayley / Poincaré ball transform mapping (0, 0, 1) -> (0, 0, 0):
 *      d = X^2 + Y^2 + (Z + 1)^2
 *      bx = 2*X / d
 *      by = 2*Y / d
 *      bz = (X^2 + Y^2 + Z^2 - 1) / d
 * 4. Inverse camera orientation rotation:
 *      b_camera = R_camera^{-1} * b
 * 5. Feed b_camera into conventional perspective projection.
 *
 * Note on approximation:
 * This is a hyperbolic camera with conventional triangle rasterization,
 * not an exact geodesic surface intersection renderer.
 * Transformed vertices have exact hyperbolic viewing directions, but the GPU
 * still linearly rasterizes triangle interiors after projection.
 */

import * as THREE from 'three';
import type { DebugSnapshot, TerrainPatch } from './debugCockpit';
import { terrainProjectionUniforms } from './hyperbolicTerrainMaterial';

export interface UpperHalfPoint {
  x: number;
  y: number;
  z: number;
}

export interface CameraHyperbolicState {
  /** Camera position in upper half space (cx, cy, cz) where cz > 0. */
  position: UpperHalfPoint;
  /** Camera orientation rotation (applied to world vectors). */
  rotation: THREE.Quaternion | THREE.Matrix4 | THREE.Euler;
}

/**
 * Hyperbolic isometry moving camera C = (cx, cy, cz) to canonical (0, 0, 1):
 *   X = (x - cx) / cz
 *   Y = (y - cy) / cz
 *   Z = z / cz
 */
export function canonicalizeUpperHalf(
  p: UpperHalfPoint,
  c: UpperHalfPoint
): UpperHalfPoint {
  const cz = c.z;
  return {
    x: (p.x - c.x) / cz,
    y: (p.y - c.y) / cz,
    z: p.z / cz,
  };
}

/**
 * Map canonical upper-half-space point (X, Y, Z) to Poincaré ball b:
 *   d = X^2 + Y^2 + (Z + 1)^2
 *   bx = 2*X / d
 *   by = 2*Y / d
 *   bz = (X^2 + Y^2 + Z^2 - 1) / d
 *
 * singularity guard: since Z > 0, (Z + 1)^2 >= 1, so d >= 1 for all valid points.
 * We enforce a guard d > 1e-12.
 */
export function upperHalfToPoincareBall(
  canonical: UpperHalfPoint,
  epsilonDenom: number = 1e-12
): THREE.Vector3 {
  const { x: X, y: Y, z: Z } = canonical;
  const d = X * X + Y * Y + (Z + 1) * (Z + 1);
  const safeD = Math.max(d, epsilonDenom);
  const bx = (2 * X) / safeD;
  const by = (2 * Y) / safeD;
  const bz = (X * X + Y * Y + Z * Z - 1) / safeD;
  return new THREE.Vector3(bx, by, bz);
}

/**
 * Full transform from upper half space P=(x,y,z) to camera-centered Poincaré ball vector b_camera.
 *
 * Camera orientation R:
 * If R is the rotation transforming camera-local coordinates to world coordinates,
 * then the inverse rotation R^{-1} transforms world Poincaré ball coordinates b into
 * camera-local coordinates:
 *   b_camera = R^{-1} * b
 */
export function projectPointToCameraPoincare(
  p: UpperHalfPoint,
  cameraPos: UpperHalfPoint,
  cameraRotation?: THREE.Quaternion | THREE.Matrix4
): THREE.Vector3 {
  const canonical = canonicalizeUpperHalf(p, cameraPos);
  const b = upperHalfToPoincareBall(canonical);

  if (cameraRotation) {
    if (cameraRotation instanceof THREE.Quaternion) {
      const invQ = cameraRotation.clone().invert();
      b.applyQuaternion(invQ);
    } else if (cameraRotation instanceof THREE.Matrix4) {
      const invM = cameraRotation.clone().invert();
      b.applyMatrix4(invM);
    }
  }

  return b;
}

/** Browser presentation: +c_x is +X, +c_y is -Z, and decreasing z_H is up. */
export function poincareToSceneVector(b: THREE.Vector3): THREE.Vector3 {
  // b.x = Re(c) direction
  // b.y = Im(c) direction
  // b.z = vertical z = a*rho direction (INVERTED so Shore is up)
  return new THREE.Vector3(b.x, -b.z, -b.y);
}

/**
 * Scale applied to Poincaré ball coordinates (which have norm < 1.0)
 * so that vertices map comfortably into the Three.js near/far clipping range
 * (e.g. near 0.1, far 200) without tiny sub-unit clipping or depth precision loss.
 */
export const HYPERBOLIC_VISUAL_SCALE = 20.0;

/**
 * Compute the camera's upper-half position and orientation rotation matrix
 * from the cockpit snapshot and heading.
 */
export interface HyperbolicCameraFrame {
  cameraUpperHalf: UpperHalfPoint;
  targetUpperHalf: UpperHalfPoint;
  rotationMatrix: THREE.Matrix4;
}

export function upperHalfGeometry(snap: DebugSnapshot) {
  const geometry = snap.physics.upperHalf;
  if (!geometry || !(geometry.a > 0) || !(geometry.z > 0)) {
    throw new Error('Hyperbolic mode requires debug-snapshot/2 upperHalf geometry; rebuild wasm-orbit or re-record the trajectory.');
  }
  return geometry;
}

/** Intrinsic distance toward the torso, along presentation up, which is -z_H. */
export const HYPERBOLIC_TARGET_HEIGHT = 0.04;

export function computeHyperbolicCameraFrame(
  snap: DebugSnapshot,
  smoothedHeading: number | null
): HyperbolicCameraFrame {
  const [cx, cy] = snap.physics.c;
  const { a, z: riderZ, gradient } = upperHalfGeometry(snap);
  // This angle is in c-space, in every camera mode.
  const heading = smoothedHeading ?? 0;
  const backC = 0.48 * snap.physics.rho;
  const camUpperX = cx - Math.cos(heading) * backC;
  const camUpperY = cy - Math.sin(heading) * backC;
  // Extrapolate log(z) from the authoritative derivative. This preserves the
  // existing slope-aware clearance without reconstructing rho from sigma.
  const deltaLogZ = (gradient[0] * (camUpperX - cx) + gradient[1] * (camUpperY - cy)) / riderZ;
  const groundZ = riderZ * Math.exp(Math.max(-20, Math.min(20, deltaLogZ)));
  const cameraUpperHalf = {
    x: camUpperX,
    y: camUpperY,
    z: Math.max(Math.min(groundZ, riderZ) / 1.35, riderZ / 4),
  };
  const targetUpperHalf = { x: cx, y: cy, z: riderZ * Math.exp(-HYPERBOLIC_TARGET_HEIGHT / a) };
  const target = poincareToSceneVector(projectPointToCameraPoincare(targetUpperHalf, cameraUpperHalf));
  const rotationMatrix = new THREE.Matrix4().lookAt(
    new THREE.Vector3(), target, new THREE.Vector3(0, 1, 0)
  );
  return { cameraUpperHalf, targetUpperHalf, rotationMatrix };
}

/** Analytic differential of the projection, applied to an upper-half tangent.
 * Rendering mirror of runtime-core/src/hyperbolic.rs, pinned by golden parity.
 */
export function projectUpperHalfTangent(
  point: UpperHalfPoint, tangent: THREE.Vector3, camera: UpperHalfPoint
): THREE.Vector3 {
  const p = canonicalizeUpperHalf(point, camera);
  const v = tangent.clone().divideScalar(camera.z);
  const d = p.x * p.x + p.y * p.y + (p.z + 1) ** 2;
  const dd = 2 * (p.x * v.x + p.y * v.y + (p.z + 1) * v.z);
  const n = p.x * p.x + p.y * p.y + p.z * p.z - 1;
  const dn = 2 * (p.x * v.x + p.y * v.y + p.z * v.z);
  return poincareToSceneVector(new THREE.Vector3(
    (2 * v.x * d - 2 * p.x * dd) / (d * d),
    (2 * v.y * d - 2 * p.y * dd) / (d * d),
    (dn * d - n * dd) / (d * d)
  ));
}

/** After the first upload this does constant work, independent of mesh size. */
export function transformMeshToHyperbolic(
  mesh: THREE.Mesh, patch: TerrainPatch, snap: DebugSnapshot,
  smoothedHeading: number | null,
  frame = computeHyperbolicCameraFrame(snap, smoothedHeading)
): void {
  const uniforms = terrainProjectionUniforms(mesh, patch);
  uniforms.hyperbolicCamera.value.set(
    frame.cameraUpperHalf.x - patch.center[0],
    frame.cameraUpperHalf.y - patch.center[1],
    frame.cameraUpperHalf.z
  );
  uniforms.hyperbolicRotation.value.setFromMatrix4(frame.rotationMatrix.clone().invert());
  uniforms.hyperbolicScale.value = HYPERBOLIC_VISUAL_SCALE;
  mesh.position.set(0, 0, 0);
  mesh.scale.setScalar(1);
  mesh.quaternion.identity();
}

/** The trail is bounded to 300 samples and rebuilt only on playback ticks. */
export function transformTrailToHyperbolic(
  trail: THREE.Line, snapshots: DebugSnapshot[], snap: DebugSnapshot,
  smoothedHeading: number | null,
  frame = computeHyperbolicCameraFrame(snap, smoothedHeading)
): void {
  const invRot = frame.rotationMatrix.clone().invert();
  trail.position.set(0, 0, 0);
  trail.scale.setScalar(HYPERBOLIC_VISUAL_SCALE);
  trail.quaternion.identity();
  trail.frustumCulled = false;
  const posAttr = trail.geometry.getAttribute('position') as THREE.BufferAttribute;
  if (!posAttr) return;
  const count = Math.min(snapshots.length, posAttr.count);
  for (let i = 0; i < count; i++) {
    const s = snapshots[i];
    const z = upperHalfGeometry(s).z;
    const b = projectPointToCameraPoincare({ x: s.physics.c[0], y: s.physics.c[1], z }, frame.cameraUpperHalf);
    const projected = poincareToSceneVector(b).applyMatrix4(invRot);
    posAttr.setXYZ(i, projected.x, projected.y, projected.z);
  }
  posAttr.needsUpdate = true;
}

/** Push the authoritative trajectory tangent and surface normal through H3. */
export function transformRiderToHyperbolic(
  rider: THREE.Group, snap: DebugSnapshot, smoothedHeading: number | null,
  frame = computeHyperbolicCameraFrame(snap, smoothedHeading)
): void {
  const geometry = upperHalfGeometry(snap);
  const point = { x: snap.physics.c[0], y: snap.physics.c[1], z: geometry.z };
  const invRot = frame.rotationMatrix.clone().invert();
  const position = poincareToSceneVector(projectPointToCameraPoincare(point, frame.cameraUpperHalf)).applyMatrix4(invRot);
  rider.position.copy(position).multiplyScalar(HYPERBOLIC_VISUAL_SCALE);
  rider.scale.setScalar(0.4);

  const [vx, vy] = snap.physics.velocity;
  let forward = new THREE.Vector3(vx, vy, geometry.zDot);
  if (forward.lengthSq() < 1e-30) {
    const direction = snap.action?.effective.direction;
    const dx = direction && Math.hypot(...direction) > 1e-12 ? direction[0] : Math.cos(smoothedHeading ?? 0);
    const dy = direction && Math.hypot(...direction) > 1e-12 ? direction[1] : Math.sin(smoothedHeading ?? 0);
    forward.set(dx, dy, geometry.gradient[0] * dx + geometry.gradient[1] * dy);
  }
  forward = projectUpperHalfTangent(point, forward.normalize(), frame.cameraUpperHalf).transformDirection(invRot);
  const upperNormal = new THREE.Vector3(geometry.gradient[0], geometry.gradient[1], -1).normalize();
  const up = projectUpperHalfTangent(point, upperNormal, frame.cameraUpperHalf).transformDirection(invRot);
  // Remove roundoff and enforce a right-handed object frame after the scene reflection.
  const lateral = new THREE.Vector3().crossVectors(forward, up).normalize();
  up.crossVectors(lateral, forward).normalize();
  rider.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(forward, up, lateral));
}

/**
 * Hyperbolic distance in upper half-space / Poincaré ball:
 *   s = 2 * a * atanh(||b||)
 * For ||b|| < 1.
 */
export function poincareHyperbolicDistance(b: THREE.Vector3, a: number = 1.0): number {
  const r = Math.min(b.length(), 0.999999999);
  return 2 * a * Math.atanh(r);
}
