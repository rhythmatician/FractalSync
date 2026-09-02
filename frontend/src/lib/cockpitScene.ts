/**
 * Three.js scene builder for the Mandelbrot-manifold debug cockpit
 * (issue #111 Phase A).
 *
 * Everything geometric here comes from the Rust DebugSnapshot/TerrainPatch
 * seam (see debugCockpit.ts) — this module contains NO manifold math. It
 * only converts authoritative samples into Three.js objects:
 *
 * - Terrain mesh: Q(c) = (x, y, lambda*sigma(c)) sampled by Rust;
 *   vertex colors encode realm (inside/outside/Shore).
 * - Rider: low-poly board + capsule, heading/pitch from authoritative
 *   c, velocity, and embedding geometry.
 * - Trail: the recorded c(t) polyline lifted onto the surface.
 * - Camera modes: physical embedding and scale-stabilized treadmill.
 */

import * as THREE from 'three';
import type { DebugSnapshot, TerrainPatch, CockpitTrajectory } from './debugCockpit';

/** Camera presentation modes (issue #111). */
export type CameraMode = 'physical' | 'treadmill';

/** Scene scale: world units per c-space unit (visual magnification). */
const SCENE_SCALE = 10.0;

/**
 * Vertical scale of the sigma axis. NOTE: this is a presentation
 * exaggeration — the rendered surface is Q(c) = (x, y, Z_SCALE*lambda*sigma),
 * NOT the exact Q=(x,y,lambda*sigma) whose induced metric equals G. Set to 1
 * for the metric-exact surface; 2 keeps slopes legible at skate-game scale.
 */
const Z_SCALE = 2.0;

/**
 * Build (or rebuild) the terrain mesh from a Rust-sampled TerrainPatch.
 * Induced metric of this surface is exactly the candidate G — no invented
 * heightfield (issue #111 mathematical basis).
 */
export function buildTerrainMesh(patch: TerrainPatch): THREE.Mesh {
  const n = patch.n;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(n * n * 3);
  const colors = new Float32Array(n * n * 3);

  for (let i = 0; i < n * n; i++) {
    const x = patch.positions[i * 3];
    const y = patch.positions[i * 3 + 1];
    const z = patch.positions[i * 3 + 2];
    positions[i * 3] = x * SCENE_SCALE;
    positions[i * 3 + 1] = z * Z_SCALE;
    positions[i * 3 + 2] = -y * SCENE_SCALE;

    // Realm coloring from the authoritative signed distance: inside = deep
    // blue, outside = sand, Shore (|D| tiny) = bright band.
    const d = patch.signed[i];
    let r: number, g: number, b: number;
    if (Math.abs(d) < 0.002) {
      r = 1.0; g = 0.95; b = 0.4; // Shore band
    } else if (d < 0) {
      r = 0.12; g = 0.2; b = 0.45; // Inside M
    } else {
      r = 0.72; g = 0.62; b = 0.42; // Outside M
    }
    // Height shading: higher sigma = slightly lighter.
    const shade = Math.max(0.55, Math.min(1.15, 1.0 + z * 0.045));
    colors[i * 3] = r * shade;
    colors[i * 3 + 1] = g * shade;
    colors[i * 3 + 2] = b * shade;
  }

  const indices: number[] = [];
  for (let row = 0; row < n - 1; row++) {
    for (let col = 0; col < n - 1; col++) {
      const a = row * n + col;
      const b = row * n + col + 1;
      const c = (row + 1) * n + col;
      const dIdx = (row + 1) * n + col + 1;
      indices.push(a, c, b, b, c, dIdx);
    }
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  return new THREE.Mesh(geometry, material);
}

/**
 * Build the low-poly rider: a board (box) + body (capsule). Returns a group
 * whose +X axis is the heading direction.
 */
export function buildRider(): THREE.Group {
  const group = new THREE.Group();

  const boardGeometry = new THREE.BoxGeometry(1.4, 0.12, 0.5);
  const boardMaterial = new THREE.MeshLambertMaterial({ color: 0xcc4444 });
  const board = new THREE.Mesh(boardGeometry, boardMaterial);
  board.position.y = 0.12;
  group.add(board);

  const bodyGeometry = new THREE.CapsuleGeometry(0.18, 0.5, 4, 8);
  const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x3388cc });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.y = 0.62;
  group.add(body);

  // Velocity arrow: thin cone along +X, visible when moving.
  const arrowGeometry = new THREE.ConeGeometry(0.09, 0.5, 6);
  const arrowMaterial = new THREE.MeshBasicMaterial({ color: 0xffee44 });
  const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
  arrow.rotation.z = -Math.PI / 2;
  arrow.position.set(1.05, 0.35, 0);
  arrow.name = 'velocityArrow';
  group.add(arrow);

  return group;
}

/** Per-frame rider placement derived from authoritative state. */
export function placeRider(
  rider: THREE.Group,
  snap: DebugSnapshot,
  terrainHeightAt: (x: number, y: number) => number
): void {
  const [cx, cy] = snap.physics.c;
  const [vx, vy] = snap.physics.velocity;

  rider.position.set(cx * SCENE_SCALE, terrainHeightAt(cx, cy), -cy * SCENE_SCALE);

  // Heading from velocity; when nearly stationary keep last heading and
  // fall back to drive direction.
  const speed = Math.hypot(vx, vy);
  if (speed > 1e-7) {
    rider.rotation.y = Math.atan2(-vy, vx);
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    if (Math.hypot(dx, dy) > 1e-6) {
      rider.rotation.y = Math.atan2(-dy, dx);
    }
  }

  // Pitch from the surface slope along the heading (presentation derived
  // from authoritative geometry): rise = d(height)/dt ~ sigma_dot * Z_SCALE.
  const pitch = Math.atan2(snap.physics.sigmaDot * Z_SCALE, Math.max(speed * SCENE_SCALE, 1e-9));
  rider.rotation.x = -pitch;

  // Velocity arrow visibility scales with speed.
  const arrow = rider.getObjectByName('velocityArrow');
  if (arrow) {
    arrow.visible = speed > 1e-5;
  }
}

/** Build the trail line from a recorded trajectory up to `upTo` (inclusive). */
export function buildTrail(trajectory: CockpitTrajectory, upTo: number): THREE.Line {
  const count = Math.min(upTo + 1, trajectory.snapshots.length);
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const [cx, cy] = trajectory.snapshots[i].physics.c;
    const z = trajectory.snapshots[i].physics.sigma;
    points.push(new THREE.Vector3(cx * SCENE_SCALE, z * Z_SCALE + 0.05, -cy * SCENE_SCALE));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x66ff99 });
  return new THREE.Line(geometry, material);
}

/**
 * Apply the treadmill chart transform to the trail (same chart as the
 * terrain so the trail stays glued to the surface in treadmill mode).
 */
export function treadmillTrailTransform(trail: THREE.Line, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const magnify = 1.0 / rho0;
  trail.position.x = -cx * SCENE_SCALE * magnify;
  trail.position.z = cy * SCENE_SCALE * magnify;
  trail.position.y = -sigma * Z_SCALE;
  trail.scale.setScalar(magnify);
}

/** Reset the trail to physical coordinates. */
export function physicalTrailTransform(trail: THREE.Line): void {
  trail.position.set(0, 0, 0);
  trail.scale.setScalar(1.0);
}

/**
 * Position the third-person camera behind and above the rider.
 *
 * - physical: raw (x, y, lambda*sigma) embedding — geometry-debug mode.
 * - treadmill: scale-stabilized chart X=(x-x0)/rho0, Z=lambda*(sigma-sigma0)
 *   around the CURRENT rider position c0 — debug presentation ONLY; never
 *   feeds physics (guaranteed structurally: the camera only reads the
 *   snapshot, and this module has no path back into the recorder).
 */
export function updateCamera(
  camera: THREE.PerspectiveCamera,
  snap: DebugSnapshot,
  mode: CameraMode
): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;

  let rx: number, rz: number;
  if (mode === 'physical') {
    rx = cx * SCENE_SCALE;
    rz = -cy * SCENE_SCALE;
  } else {
    // Treadmill: the rider is pinned at the chart origin; the terrain mesh
    // carries the (x-x0)/rho0 transform (see treadmillTransform).
    rx = 0;
    rz = 0;
  }

  // Behind and above, biased along the rider's heading.
  const [vx, vy] = snap.physics.velocity;
  const speed = Math.hypot(vx, vy);
  const heading = speed > 1e-7 ? Math.atan2(-vy, vx) : Math.PI / 2;
  const back = 3.2;
  const up = 2.2;
  const targetY = (mode === 'treadmill' ? 0 : sigma * Z_SCALE) + 0.6;

  const camX = rx - Math.cos(heading) * back;
  const camZ = rz + Math.sin(heading) * back;
  const camY = (mode === 'treadmill' ? 0 : sigma * Z_SCALE) + up;

  camera.position.set(camX, camY, camZ);
  camera.lookAt(rx, targetY, rz);
}

/**
 * Scale-stabilized treadmill chart (issue #111):
 *   X = (x - x0) / rho0,  Z = lambda * (sigma(c) - sigma(c0))
 * implemented as a mesh transform: translate the patch by -c0 horizontally
 * and by -lambda*sigma0 vertically, then scale the whole patch by 1/rho0 so
 * local terrain stays visually resolvable as the rider descends into finer
 * Mandelbrot scale. Debug presentation ONLY — this module has no path back
 * into physics (the recorder never reads scene objects).
 */
export function treadmillTransform(mesh: THREE.Mesh, snap: DebugSnapshot): void {
  const [cx, cy] = snap.physics.c;
  const sigma = snap.physics.sigma;
  const rho0 = Math.max(snap.physics.rho, 1e-9);
  const magnify = 1.0 / rho0;
  mesh.position.x = -cx * SCENE_SCALE * magnify;
  mesh.position.z = cy * SCENE_SCALE * magnify;
  mesh.position.y = -sigma * Z_SCALE;
  mesh.scale.setScalar(magnify);
}

/** Reset the treadmill transform when returning to physical mode. */
export function physicalTransform(mesh: THREE.Mesh): void {
  mesh.position.set(0, 0, 0);
  mesh.scale.setScalar(1.0);
}

/** Lights + backdrop for the late-90s skate-game look. */
export function buildSceneDressing(scene: THREE.Scene): void {
  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff2cc, 0.9);
  sun.position.set(6, 10, 4);
  scene.add(sun);

  scene.background = new THREE.Color(0x0a0a14);
  scene.fog = new THREE.Fog(0x0a0a14, 18, 42);
}