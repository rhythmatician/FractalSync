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
 * - Camera: the hyperbolic camera (issue #142) — the single camera.
 *   Terrain and trail vertices are projected into the camera-centered
 *   Poincaré ball by hyperbolicCamera.ts / hyperbolicTerrainMaterial.ts.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import type { DebugSnapshot, TerrainPatch, CockpitTrajectory } from './debugCockpit';

/**
 * Scene scale: world units per c-space unit (visual magnification).
 * Exported so the LOD planner (debugCockpit) plans in the same units.
 */
export const SCENE_SCALE = 10.0;

/**
 * NOTE on lambda: the canonical embedding height is lambda*sigma(c), and
 * Rust owns lambda (ManifoldConfig.lambda_sq). TerrainPatch.positions
 * arrive with z ALREADY multiplied by lambda (runtime-core debug.rs:
 * "z = lambda * sigma(c)"), so this module must not keep its own copy of
 * the constant — a second factor would silently square it. The
 * DebugSnapshot's physics.sigma is raw sigma; it coincides with the patch
 * height under the controller-default lambda^2 = 1 config that
 * `sampleTerrainPatch` samples with.
 */

/**
 * Vertical scale of the sigma axis (static terrain-mesh build).
 *
 * The raw lambda*sigma surface is far too steep — near the Shore sigma
 * reaches ~10 while valleys sit at ~-1.5, so linear scaling turns hills
 * into cliffs. surfaceY() applies an asinh compression (logarithmic for
 * large |sigma|, linear near 0) so the crest stays dramatic but ridable.
 *
 * NOTE: this Y mapping only shapes the STATIC CPU `position` attribute
 * (bounding sphere / fallback). Hyperbolic rendering ignores it — the GPU
 * projects the authoritative `upperHalfPosition` attribute instead (see
 * hyperbolicTerrainMaterial.ts).
 */
const Z_SCALE = 2.0;

/** asinh knee: below |sigma| ~ Z_COMPRESS_LINEAR the mapping is ~linear. */
const Z_COMPRESS_LINEAR = 1.5;

/**
 * Scene Y for a sigma value — asinh(sigma / k) * k * Z_SCALE: linear for
 * |sigma| << k, logarithmic for |sigma| >> k. Preserves sign and
 * monotonicity (uphill stays uphill).
 */
export function surfaceY(sigma: number): number {
  const s = sigma / Z_COMPRESS_LINEAR;
  return Math.asinh(s) * Z_COMPRESS_LINEAR * Z_SCALE;
}

/**
 * Build (or rebuild) the terrain mesh from a Rust-sampled TerrainPatch.
 *
 * The static CPU `position` attribute carries the asinh-compressed
 * surfaceY(z) embedding of the canonical Q(c) = (x, y, lambda*sigma(c))
 * sampled by Rust — used for the bounding sphere and as a harmless
 * fallback. HYPERBOLIC RENDERING does not read this attribute: the GPU
 * vertex shader projects the authoritative `upperHalfPosition` attribute
 * (uploaded by terrainProjectionUniforms from patch.upperZ) into the
 * camera-centered Poincaré ball (see hyperbolicTerrainMaterial.ts).
 */
// Procedural grid texture generator for the fractal manifold terrain
let proceduralGridTexture: THREE.CanvasTexture | null = null;
function getGridTexture(): THREE.CanvasTexture | null {
  if (proceduralGridTexture) return proceduralGridTexture;
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  if (!canvas || typeof canvas.getContext !== 'function') return null;
  const ctx = canvas.getContext('2d');
  if (!ctx || typeof ctx.fillRect !== 'function') return null;

  canvas.width = 256;
  canvas.height = 256;
  // Clean neutral base that tints with vertexColors
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 256);
  // Subtle cyber/synthwave grid lines
  ctx.strokeStyle = 'rgba(150, 160, 190, 0.4)';
  ctx.lineWidth = 3;
  ctx.strokeRect(0, 0, 256, 256);
  // Secondary micro-grid
  ctx.strokeStyle = 'rgba(180, 190, 220, 0.2)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(128, 0); ctx.lineTo(128, 256);
  ctx.moveTo(0, 128); ctx.lineTo(256, 128);
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(32, 32);
  proceduralGridTexture = tex;
  return proceduralGridTexture;
}

export function buildTerrainMesh(patch: TerrainPatch): THREE.Mesh {
  const n = patch.n;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(n * n * 3);
  const colors = new Float32Array(n * n * 3);
  const uvs = new Float32Array(n * n * 2);

  for (let i = 0; i < n * n; i++) {
    const row = Math.floor(i / n);
    const col = i % n;
    const x = patch.positions[i * 3];
    const y = patch.positions[i * 3 + 1];
    const z = patch.positions[i * 3 + 2]; // Rust-embedded lambda*sigma(c).
    positions[i * 3] = x * SCENE_SCALE;
    positions[i * 3 + 1] = surfaceY(z);
    positions[i * 3 + 2] = -y * SCENE_SCALE;

    uvs[i * 2] = col / (n - 1);
    uvs[i * 2 + 1] = row / (n - 1);

    // Realm coloring from the authoritative signed distance: inside = deep
    // cosmic blue/indigo, outside = warm amber terrace, Shore (|D| tiny) = glowing electric gold.
    const d = patch.signed[i];
    let r: number, g: number, b: number;
    if (Math.abs(d) < 0.002) {
      r = 1.0; g = 0.96; b = 0.45; // Shore band
    } else if (d < 0) {
      r = 0.10; g = 0.18; b = 0.48; // Inside M (cosmic deep basin)
    } else {
      r = 0.68; g = 0.58; b = 0.40; // Outside M (warm sandy terrace)
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
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  const gridTex = getGridTexture();
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    ...(gridTex ? { map: gridTex } : {}),
    roughness: 0.72,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });
  return new THREE.Mesh(geometry, material);
}

/** Independent terrain overlay toggles (issue #111 terrain overlays). */
export interface TerrainOverlays {
  /** Highlight the Shore band D(c)=0. */
  shoreBand: boolean;
  /** Tint inside/outside realms. */
  realm: boolean;
  /** Shade by sigma(c) (Mandelbrot scale — distinct from Julia zoom). */
  sigma: boolean;
  /** Shade by U(c) = kappa*sigma (potential ramp). */
  potential: boolean;
  /** Tint vertices where derivative validity fails. */
  validity: boolean;
}

export const DEFAULT_OVERLAYS: TerrainOverlays = {
  shoreBand: true,
  realm: true,
  sigma: false,
  potential: false,
  validity: true,
};

/**
 * Repaint the terrain vertex colors from the patch + overlay toggles.
 * All quantities are already sampled by Rust in the TerrainPatch; this only
 * maps values to colors. sigma/U/validity overlays are mutually layered:
 * the last enabled one in the fixed order [validity, potential, sigma,
 * realm, shoreBand] wins for a vertex, keeping the mapping predictable.
 */
export function applyOverlays(mesh: THREE.Mesh, patch: TerrainPatch, overlays: TerrainOverlays): void {
  const colorAttr = mesh.geometry.getAttribute('color') as THREE.BufferAttribute;
  if (!colorAttr) return;
  const n = patch.n;
  const colors = colorAttr.array as Float32Array;

  // For the sigma/potential ramps: normalize against the patch min/max so
  // the ramp is legible regardless of the local scale range. Iterated (not
  // Math.min(...zs), which overflows the JS arg limit — and the call stack —
  // at LOD grid sizes above ~100k vertices).
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let i = 0; i < n * n; i++) {
    const z = patch.positions[i * 3 + 2];
    if (z < zMin) zMin = z;
    if (z > zMax) zMax = z;
  }
  const span = Math.max(zMax - zMin, 1e-9);

  for (let i = 0; i < n * n; i++) {
    const d = patch.signed[i];
    const z = patch.positions[i * 3 + 2];
    const t = (z - zMin) / span;
    const nearShore = Math.abs(d) < 0.002;

    let r: number, g: number, b: number;
    if (overlays.validity && !Number.isFinite(d)) {
      // Derivative-validity overlay: non-finite D marks a failed sample.
      r = 1; g = 0.3; b = 1;
    } else if (overlays.potential) {
      // U ramp: violet (low U) -> red (high U).
      r = 0.25 + 0.75 * t;
      g = 0.1 + 0.15 * (1 - t);
      b = 0.6 - 0.45 * t;
    } else if (overlays.sigma) {
      // sigma ramp: deep teal (low scale) -> yellow (high scale).
      r = 0.1 + 0.9 * t * t;
      g = 0.75 - 0.15 * t;
      b = 0.55 - 0.4 * t;
    } else if (overlays.realm) {
      if (d < 0) {
        r = 0.12; g = 0.2; b = 0.45;
      } else {
        r = 0.72; g = 0.62; b = 0.42;
      }
    } else {
      r = 0.5; g = 0.5; b = 0.55;
    }

    if (overlays.shoreBand && nearShore) {
      r = 1.0; g = 0.95; b = 0.4;
    }

    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  colorAttr.needsUpdate = true;
}

/**
 * Load the animated rider (Meshy "Tiny Titan" biped GLB) standing on the Meshy
 * skateboard GLB. Returns a group whose +X axis is the heading direction;
 * falls back to a capsule body if a model fails to load so the cockpit
 * degrades honestly instead of losing the rider.
 *
 * The rider plays a Mixamo skateboarding animation retargeted onto the Meshy
 * biped skeleton via an AnimationMixer owned by this module (see updateRiderAnimation).
 */

/** Rider yaw: +PI/2 aligns the Mixamo skateboarding animation forward (+X) with feet spread along deck length. */
const RIDER_MODEL_YAW = Math.PI / 2;

/** Rider scale: the GLB is ~1.6 units tall; the scene rider reads best ~1.4. */
const RIDER_MODEL_SCALE = 0.9;

/** Rider lift: calibrated so shoe soles rest squarely and flush on the skateboard deck. */
const RIDER_MODEL_LIFT = -0.230;

/** Skateboard scale: scaled down to 0.38 (~0.72 long, ~0.22 wide) to fit Tiny Titan's smaller body/leg proportions. */
const SKATEBOARD_SCALE = 0.4;

interface RiderAnimationState {
  mixer: THREE.AnimationMixer;
  actionCoast: THREE.AnimationAction | null;
  actionPush: THREE.AnimationAction | null;
}

const riderAnimationStates = new WeakMap<THREE.Group, RiderAnimationState>();

/**
 * Retarget Mixamo animation tracks onto the Meshy biped skeleton.
 * Mixamo bone names start with 'mixamorig'; Meshy biped uses plain names
 * ('Hips', 'Spine', 'Spine01', 'Spine02', 'neck', 'Head', etc.).
 * Root motion on X/Z is locked so the character rides in place on the board.
 */
function retargetMixamoToMeshy(
  sourceClip: THREE.AnimationClip,
  targetName: string,
  targetSkeletonBones: Set<string>
): THREE.AnimationClip {
  const tracks: THREE.KeyframeTrack[] = [];
  for (const track of sourceClip.tracks) {
    const dotIdx = track.name.indexOf('.');
    if (dotIdx === -1) continue;
    const boneRaw = track.name.slice(0, dotIdx);
    const prop = track.name.slice(dotIdx);

    let targetBone = boneRaw.replace(/^mixamorig/, '');
    if (targetBone === 'Spine1') targetBone = 'Spine01';
    else if (targetBone === 'Spine2') targetBone = 'Spine02';
    else if (targetBone === 'Neck') targetBone = 'neck';

    if (targetSkeletonBones.has(targetBone)) {
      const cloned = track.clone();
      cloned.name = targetBone + prop;
      // Lock root horizontal motion so the skater stays planted on the deck
      if (cloned.name === 'Hips.position') {
        const x0 = cloned.values[0];
        const z0 = cloned.values[2];
        for (let i = 0; i < cloned.values.length; i += 3) {
          cloned.values[i] = x0;
          cloned.values[i + 2] = z0;
        }
      }
      tracks.push(cloned);
    }
  }
  return new THREE.AnimationClip(targetName, sourceClip.duration, tracks);
}

/**
 * Advance the rider's animation mixer by dt, blending between Skateboarding
 * (coasting / cruising) and PushOff (accelerating forward) according to
 * throttle / forward driving effort, and scaling cadence with metric speed.
 */
export function updateRiderAnimation(
  rider: THREE.Group,
  dt: number,
  metricSpeed: number,
  throttle: number = 0
): void {
  const state = riderAnimationStates.get(rider);
  if (!state) return;

  const { mixer, actionCoast, actionPush } = state;

  if (actionCoast && actionPush) {
    // When throttle > 0, blend in the PushOff animation; otherwise coast.
    const pushWeight = Math.min(1.0, Math.max(0.0, throttle));
    const coastWeight = 1.0 - pushWeight;
    actionCoast.setEffectiveWeight(coastWeight);
    actionPush.setEffectiveWeight(pushWeight);
  } else if (actionCoast) {
    actionCoast.setEffectiveWeight(1.0);
  } else if (actionPush) {
    actionPush.setEffectiveWeight(1.0);
  }

  // Map metric speed to a legible cadence: ~0.6x at a crawl, ~1.8x flat-out.
  mixer.timeScale = Math.max(0.6, Math.min(1.8, 0.6 + metricSpeed * 8.0));
  mixer.update(dt);
}

export async function buildRider(): Promise<THREE.Group> {
  const group = new THREE.Group();
  const gltfLoader = new GLTFLoader();
  const fbxLoader = new FBXLoader();

  // Skateboard GLB: already X-aligned (deck ~1.9 long on X, wheels at
  // y ~ -0.155), so it needs only scaling and a lift to put the wheels'
  // contact plane at y=0.
  try {
    const boardGltf = await gltfLoader.loadAsync('/models/skateboard.glb');
    const board = boardGltf.scene;
    board.scale.setScalar(SKATEBOARD_SCALE);
    // Wheels bottom at -0.155 * scale; lift so contact plane sits at y=0.
    board.position.y = 0.155 * SKATEBOARD_SCALE;
    group.add(board);
  } catch (error) {
    console.warn('[cockpitScene] skateboard GLB unavailable, using box fallback:', error);
    const boardGeometry = new THREE.BoxGeometry(1.1, 0.12, 0.42);
    const boardMaterial = new THREE.MeshLambertMaterial({ color: 0xcc4444 });
    const board = new THREE.Mesh(boardGeometry, boardMaterial);
    board.position.y = 0.12;
    group.add(board);
  }

  try {
    // New player character without integrated duplicate skateboard/pedestal: Tiny Titan biped
    const titanPath = '/models/Meshy_AI_Tiny_Titan_biped/Meshy_AI_Tiny_Titan_biped_Character_output.glb';
    const gltf = await gltfLoader.loadAsync(titanPath);
    const model = gltf.scene;

    model.scale.setScalar(RIDER_MODEL_SCALE);
    model.position.y = RIDER_MODEL_LIFT;
    model.rotation.y = RIDER_MODEL_YAW;
    group.add(model);

    // Collect skeleton bone names for retargeting
    const boneNames = new Set<string>();
    model.traverse((o) => {
      if ((o as THREE.Bone).isBone) boneNames.add(o.name);
    });

    const mixer = new THREE.AnimationMixer(model);
    let actionCoast: THREE.AnimationAction | null = null;
    let actionPush: THREE.AnimationAction | null = null;

    // Load authentic skateboarding and push-off animations from Mixamo FBX
    try {
      const fbxSkate = await fbxLoader.loadAsync('/animations/Skateboarding.fbx');
      if (fbxSkate.animations.length > 0) {
        const skateClip = retargetMixamoToMeshy(fbxSkate.animations[0], 'Skateboarding', boneNames);
        actionCoast = mixer.clipAction(skateClip);
        actionCoast.play();
      }
    } catch (animError) {
      console.warn('[cockpitScene] skateboarding animation unavailable:', animError);
    }

    try {
      const fbxPush = await fbxLoader.loadAsync('/animations/PushOff.fbx');
      if (fbxPush.animations.length > 0) {
        const pushClip = retargetMixamoToMeshy(fbxPush.animations[0], 'PushOff', boneNames);
        actionPush = mixer.clipAction(pushClip);
        actionPush.play();
      }
    } catch (pushError) {
      console.warn('[cockpitScene] push-off animation unavailable:', pushError);
    }

    // Fallback if FBX animations fail
    if (!actionCoast && !actionPush && gltf.animations.length > 0) {
      const fallbackClip = gltf.animations[0];
      actionCoast = mixer.clipAction(fallbackClip);
      actionCoast.play();
    }

    riderAnimationStates.set(group, { mixer, actionCoast, actionPush });
  } catch (error) {
    console.warn('[cockpitScene] rider GLB unavailable, using capsule fallback:', error);
    const bodyGeometry = new THREE.CapsuleGeometry(0.18, 0.5, 4, 8);
    const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x3388cc });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.62;
    group.add(body);
  }

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

/**
 * Per-frame rider placement derived from authoritative state.
 *
 * Transient Euclidean staging only: the hyperbolic pass
 * (transformRiderToHyperbolic) immediately overrides position/quaternion/
 * scale with the projected Poincaré-ball values every frame. This keeps
 * the rider visible for the single frame before the hyperbolic transform
 * runs and preserves the velocity-arrow visibility rule.
 */
export function placeRider(
  rider: THREE.Group,
  snap: DebugSnapshot,
  terrainHeightAt: (x: number, y: number) => number
): void {
  rider.scale.setScalar(1);
  const [cx, cy] = snap.physics.c;
  const [vx, vy] = snap.physics.velocity;

  rider.position.set(cx * SCENE_SCALE, terrainHeightAt(cx, cy), -cy * SCENE_SCALE);

  // Heading direction in c-space.
  // In the scene: X = cx * SCENE_SCALE, Z = -cy * SCENE_SCALE.
  // When velocity is (vx, vy), the horizontal direction in scene space is (vx, -vy).
  // With rider group's forward axis along local +X:
  // Rotating (1, 0, 0) by theta_y around +Y gives:
  //   x' = cos(theta_y), z' = -sin(theta_y).
  // Matching (vx, -vy) requires cos(theta_y) ~ vx and -sin(theta_y) ~ -vy,
  // which means sin(theta_y) ~ vy.
  // Thus theta_y = Math.atan2(vy, vx).
  const speed = Math.hypot(vx, vy);
  let dirX = 1;
  let dirY = 0;
  let hasDir = false;

  if (speed > 1e-7) {
    dirX = vx / speed;
    dirY = vy / speed;
    hasDir = true;
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    const dSpeed = Math.hypot(dx, dy);
    if (dSpeed > 1e-6) {
      dirX = dx / dSpeed;
      dirY = dy / dSpeed;
      hasDir = true;
    }
  }

  // Slope along forward direction and lateral right direction (in scene units):
  // Forward c-space unit vector: (dirX, dirY).
  // Scene forward displacement: dX_scene = dirX * hScale, dZ_scene = -dirY * hScale.
  // Forward rise: dY_fwd = (dirX * dY_dcx + dirY * dY_dcy).
  // Slope forward = dY_fwd / hScale.
  //
  // Lateral right unit vector in c-space: (dirY, -dirX).
  // (In scene: dX_right = dirY * hScale, dZ_right = -(-dirX)*hScale = dirX * hScale,
  // which is perpendicular to scene forward (dirX, -dirY)).
  // Lateral rise: dY_lat = (dirY * dY_dcx - dirX * dY_dcy).
  // Slope lateral = dY_lat / hScale.
  const hScale = SCENE_SCALE;

  const [gx, gy] = snap.physics.scaleGradient;
  const dYdSigma =
    (surfaceY(snap.physics.sigma + 0.01) - surfaceY(snap.physics.sigma - 0.01)) / 0.02;

  // Scene vertical gradient: d(sceneY)/d(cx) and d(sceneY)/d(cy)
  const dY_dcx = gx * dYdSigma;
  const dY_dcy = gy * dYdSigma;

  const slopeFwd = hasDir ? (dirX * dY_dcx + dirY * dY_dcy) / Math.max(hScale, 1e-9) : 0;
  const slopeLat = hasDir ? (dirY * dY_dcx - dirX * dY_dcy) / Math.max(hScale, 1e-9) : 0;

  if (hasDir) {
    const thetaY = Math.atan2(dirY, dirX);
    const pitch = Math.atan(slopeFwd);
    const roll = Math.atan(slopeLat);

    // Forward tangent vector T (in scene coordinates):
    const T = new THREE.Vector3(
      Math.cos(pitch) * Math.cos(thetaY),
      Math.sin(pitch),
      -Math.cos(pitch) * Math.sin(thetaY)
    ).normalize();

    // Lateral right vector R_raw:
    const R_raw = new THREE.Vector3(
      Math.cos(roll) * Math.sin(thetaY),
      Math.sin(roll),
      Math.cos(roll) * Math.cos(thetaY)
    );

    // Surface normal N (points UP: R x T in right-handed coords):
    const N = new THREE.Vector3().crossVectors(R_raw, T).normalize();
    // Exact orthonormal lateral vector B = T x N:
    const B = new THREE.Vector3().crossVectors(T, N).normalize();

    // Construct basis matrix: local +X -> T (forward), local +Y -> N (up), local +Z -> B (right):
    const mat = new THREE.Matrix4().makeBasis(T, N, B);
    rider.quaternion.setFromRotationMatrix(mat);
  }

  // Velocity arrow visibility scales with speed.
  const arrow = rider.getObjectByName('velocityArrow');
  if (arrow) {
    arrow.visible = speed > 1e-5;
  }
}

/**
 * Build the trail line from a recorded trajectory up to `upTo` (inclusive).
 *
 * The static CPU `position` attribute carries the asinh-compressed
 * surfaceY(sigma) embedding with a small lift so the trail draws just
 * above the surface fallback. HYPERBOLIC RENDERING does not read this
 * attribute: transformTrailToHyperbolic (hyperbolicCamera.ts) projects
 * each authoritative snapshot's upper-half point into the camera-centered
 * Poincaré ball and overwrites the attribute every playback tick.
 */
export function buildTrail(trajectory: CockpitTrajectory, upTo: number): THREE.Line {
  const count = Math.min(upTo + 1, trajectory.snapshots.length);
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const [cx, cy] = trajectory.snapshots[i].physics.c;
    const z = trajectory.snapshots[i].physics.sigma;
    const yCoord = surfaceY(z) + 0.05;
    points.push(new THREE.Vector3(cx * SCENE_SCALE, yCoord, -cy * SCENE_SCALE));
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0x66ff99 });
  return new THREE.Line(geometry, material);
}

/**
 * Hyperbolic third-person camera (issue #142) — the one and only camera.
 *
 * The terrain and trail vertices are projected into the camera-centered
 * Poincaré ball (origin = (0, 0, 0)), with inverse camera orientation
 * already applied (hyperbolicCamera.ts / hyperbolicTerrainMaterial.ts).
 * The Three.js camera therefore sits at (0, 0, 0) with default
 * orientation (looking down -Z with +Y up) — all view geometry is baked
 * into the vertex positions by the hyperbolic projection.
 */
// Persistent smoothed heading for the camera so it stays smoothly behind without whipping
let smoothedCamHeading: number | null = null;

export function getSmoothedCamHeading(): number | null {
  return smoothedCamHeading;
}

export function resetCameraSmoothing(): void {
  smoothedCamHeading = null;
}

export function updateSmoothedHeading(snap: DebugSnapshot, dt: number): number {
  // Behind and above, biased along the rider's heading or control direction.
  const [vx, vy] = snap.physics.velocity;
  const speed = Math.hypot(vx, vy);

  // Heading stays in c-space; scene placement explicitly maps y to -Z.
  let targetHeading = smoothedCamHeading ?? 0;
  if (speed > 0) {
    targetHeading = Math.atan2(vy, vx);
  } else if (snap.action) {
    const [dx, dy] = snap.action.effective.direction;
    if (Math.hypot(dx, dy) > 1e-6) {
      targetHeading = Math.atan2(dy, dx);
    }
  }

  if (smoothedCamHeading === null) {
    smoothedCamHeading = targetHeading;
  } else {
    // Smooth angle interpolation handling wrap-around
    let diff = targetHeading - smoothedCamHeading;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    // Damped follow: ~3.5 rad/s keeps camera behind the board without jarring snaps
    const blend = 1 - Math.exp(-3.5 * Math.max(0, dt));
    smoothedCamHeading += diff * blend;
  }

  return smoothedCamHeading;
}

export function updateCamera(
  camera: THREE.PerspectiveCamera,
  snap: DebugSnapshot,
  dt: number = 0.016
): void {
  updateSmoothedHeading(snap, dt);
  // Hyperbolic camera: the terrain and trail vertices are projected into the
  // camera-centered Poincaré ball (origin = (0, 0, 0)), with inverse camera
  // orientation already applied. The Three.js camera sits at (0, 0, 0) with
  // default orientation (looking down -Z with +Y up).
  camera.position.set(0, 0, 0);
  camera.quaternion.set(0, 0, 0, 1);
}

/** Lights + atmospheric backdrop for an immersive game environment. */
export function buildSceneDressing(scene: THREE.Scene): void {
  const ambient = new THREE.AmbientLight(0xdde8ff, 0.65);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff4d6, 1.25);
  sun.position.set(12, 22, 10);
  scene.add(sun);

  // Rim / fill light for character silhouette
  const rimLight = new THREE.DirectionalLight(0x66aaff, 0.45);
  rimLight.position.set(-10, 8, -12);
  scene.add(rimLight);

  scene.background = new THREE.Color(0x070714);
  // Fog distances are set per-terrain-rebuild by applyRenderDistance so the
  // mesh edge always hides inside the fog wall at every LOD level.
  scene.fog = new THREE.Fog(0x070714, 20, 50);

  // Distant starfield particles for celestial atmosphere
  const starGeo = new THREE.BufferGeometry();
  const starCount = 350;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const r = 80 + Math.random() * 40;
    const theta = Math.random() * Math.PI * 2;
    const phi = (Math.random() * 0.4 + 0.1) * Math.PI; // Upper hemisphere
    starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPositions[i * 3 + 1] = r * Math.cos(phi);
    starPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  const starMat = new THREE.PointsMaterial({
    color: 0xccddff,
    size: 1.2,
    transparent: true,
    opacity: 0.75,
  });
  const stars = new THREE.Points(starGeo, starMat);
  scene.add(stars);
}

/**
 * Apply the LOD render distance: camera near/far planes + fog wall for the
 * hyperbolic camera. Call on every terrain rebuild.
 *
 * In hyperbolic mode, coordinates live in the scaled Poincaré ball with
 * radius ~HYPERBOLIC_VISUAL_SCALE (20 units), so the clipping range and
 * fog wall are fixed constants — the projection itself keeps the rider
 * framed regardless of Mandelbrot scale.
 */
export function applyRenderDistance(
  camera: THREE.PerspectiveCamera,
  scene: THREE.Scene
): void {
  camera.near = 0.1;
  camera.far = 100.0;
  camera.updateProjectionMatrix();
  const fog = scene.fog;
  if (fog && 'near' in fog && 'far' in fog) {
    (fog as THREE.Fog).near = 12.0;
    (fog as THREE.Fog).far = 19.0;
  }
}
