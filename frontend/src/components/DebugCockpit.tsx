/**
 * DebugCockpit — third-person Mandelbrot-manifold training spectator
 * (issue #111 Phase A).
 *
 * A visual debugging instrument, not a game: replays hand-authored Controls
 * v2 trajectories through the Rust destination physics and renders the
 * rider over the canonical manifold terrain, with full provenance-badged
 * HUD telemetry and a tick-keyed timeline.
 *
 * Provenance invariant: every displayed quantity is badged
 *   STATE  authoritative world/Physics truth (not visible to the Player)
 *   ACTION emitted Controls v2 (raw + effective)
 *   DIAG   human-only derived diagnostic
 * Phase A has no PlayerObservation contract yet (#108) so there are no OBS
 * rows; the Player View toggle hides everything except ACTION (and future
 * OBS) panels.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  CockpitRecorder,
  CANONICAL_DT,
  DEFAULT_TERRAIN_HALF,
  planTerrainLod,
  riderSurfaceHeight,
  sampleTerrainPatch,
  type CockpitTrajectory,
  type DebugSnapshot,
} from '../lib/debugCockpit';
import {
  ManualDriver,
  type KeyboardState,
} from '../lib/manualDriving';
import { baselineVariants, explorationVariants } from '../lib/shoreCrossingVariants';
import { initOrbitSynth } from '../lib/orbitSynthesizer';
import { JuliaRenderer } from '../lib/juliaRenderer';
import juliaFragment from '../../../shared/shaders/julia.frag?raw';
import { JuliaSensitivityBookmark } from './JuliaSensitivityBookmark';
import {
  MINIMAP_SIZE,
  minimapView,
  paintMinimapImage,
  type MinimapView,
  type MinimapPaintInput,
} from '../lib/cockpitMinimap';
import {
  buildTerrainMesh,
  buildRider,
  buildTrail,
  placeRider,
  updateCamera,
  updateRiderAnimation,
  applyOverlays,
  applyRenderDistance,
  buildSceneDressing,
  surfaceY,
  DEFAULT_OVERLAYS,
  type TerrainOverlays,
  getSmoothedCamHeading,
} from '../lib/cockpitScene';
import {
  transformMeshToHyperbolic,
  computeHyperbolicCameraFrame,
  transformTrailToHyperbolic,
  transformRiderToHyperbolic,
  projectPointToCameraPoincare,
  poincareToSceneVector,
  HYPERBOLIC_VISUAL_SCALE,
} from '../lib/hyperbolicCamera';

export function EnergyLedgerPanel({ frame }: { frame: DebugSnapshot }) {
  return (
  <Panel title="WORK / ENERGY LEDGER">
    {frame.energyLedger ? <>
      <Row label="step validity" value={frame.energyLedger.valid ? 'valid endpoints' : 'INVALID'} kind="DIAG" />
      <Row label="drive work" value={frame.energyLedger.controlWork} kind="DIAG" />
      <Row label="friction work" value={frame.energyLedger.frictionWork} kind="DIAG" />
      <Row label="impulse work" value={frame.energyLedger.impulseWork} kind="DIAG" />
      <Row label="step residual" value={frame.energyLedger.residual} kind="DIAG" />
      <Row label="provider" value={`${frame.energyLedger.endGeometry.provider_version} / ${frame.energyLedger.endGeometry.validity}`} kind="DIAG" />
    </> : <Row label="step ledger" value="unavailable" kind="DIAG" />}
    {frame.accumulatedEnergyLedger && <>
      <Row label="run validity" value={frame.accumulatedEnergyLedger.valid ? 'valid' : 'INVALID'} kind="DIAG" />
      <Row label="run steps" value={frame.accumulatedEnergyLedger.steps} kind="DIAG" />
      <Row label="run drive work" value={frame.accumulatedEnergyLedger.controlWork} kind="DIAG" />
      <Row label="run friction work" value={frame.accumulatedEnergyLedger.frictionWork} kind="DIAG" />
      <Row label="run impulse work" value={frame.accumulatedEnergyLedger.impulseWork} kind="DIAG" />
      <Row label="run residual" value={frame.accumulatedEnergyLedger.residual} kind="DIAG" />
    </>}
  </Panel>
  );
}

type Badge = 'STATE' | 'ACTION' | 'DIAG';

function BadgeTag({ kind }: { kind: Badge }) {
  const color =
    kind === 'STATE' ? '#7fb0ff' : kind === 'ACTION' ? '#ffd479' : '#9fe8b0';
  return (
    <span
      style={{
        fontSize: 9,
        padding: '1px 4px',
        borderRadius: 3,
        background: '#1d1d2c',
        color,
        border: `1px solid ${color}44`,
        marginRight: 6,
      }}
    >
      {kind}
    </span>
  );
}

function Row({
  label,
  value,
  kind,
  digits = 5,
}: {
  label: string;
  value: number | string | null | undefined;
  kind: Badge;
  digits?: number;
}) {
  const text =
    typeof value === 'number'
      ? Number.isFinite(value)
        ? value.toFixed(digits)
        : '—'
      : value ?? '—';
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, lineHeight: 1.6 }}>
      <span style={{ color: '#889' }}>
        <BadgeTag kind={kind} />
        {label}
      </span>
      <span style={{ color: '#dde', fontFamily: 'monospace' }}>{text}</span>
    </div>
  );
}

function Panel({ title, children, translucent = false }: { title: string; children: React.ReactNode; translucent?: boolean }) {
  return (
    <div
      style={{
        background: translucent ? 'rgba(10,10,22,0.38)' : 'rgba(10,10,22,0.82)',
        border: '1px solid #262640',
        borderRadius: 8,
        padding: '8px 10px',
        minWidth: 250,
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: 1, color: translucent ? '#eef' : '#778', marginBottom: 4 }}>{title}</div>
      {children}
    </div>
  );
}

/** Human-readable realm name from the authoritative realm field. */
function realmName(realm: number): string {
  return realm < 0 ? 'INSIDE (connected)' : realm > 0 ? 'OUTSIDE (dust)' : 'ON SHORE';
}

/**
 * K/U/E history sparkline keyed to the hop-clock timeline (issue #111
 * timeline: "K,U,E history"). Values come straight from the recorded
 * snapshots — no recomputation. The current frame is marked.
 */
function EnergySparkline({
  trajectory,
  frameIdx,
}: {
  trajectory: CockpitTrajectory | undefined;
  frameIdx: number;
}): JSX.Element {
  const W = 220;
  const H = 40;
  if (!trajectory || trajectory.snapshots.length === 0) {
    return <svg width={W} height={H} />;
  }
  // Sample at most ~400 points across the trajectory for the polyline.
  const total = trajectory.snapshots.length;
  const stride = Math.max(1, Math.floor(total / 400));
  const pts: Array<{ k: number; u: number; e: number }> = [];
  for (let i = 0; i < total; i += stride) {
    const s = trajectory.snapshots[i];
    pts.push({ k: s.physics.kinetic, u: s.physics.potential, e: s.physics.total });
  }
  const eMin = Math.min(...pts.map((p) => p.e));
  const eMax = Math.max(...pts.map((p) => p.e));
  const span = Math.max(eMax - eMin, 1e-9);
  const toY = (v: number): number => H - 2 - ((v - eMin) / span) * (H - 4);
  const toX = (i: number): number => (i / Math.max(pts.length - 1, 1)) * (W - 2) + 1;

  const poly = (key: 'k' | 'u' | 'e'): string =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p[key]).toFixed(1)}`).join(' ');

  // Marker for the current frame.
  const curIdx = Math.min(Math.floor(frameIdx / stride), pts.length - 1);
  const cur = pts[curIdx];

  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <path d={poly('e')} fill="none" stroke="#9fe8b0" strokeWidth={1.2} />
      <path d={poly('u')} fill="none" stroke="#7fb0ff" strokeWidth={1} />
      <path d={poly('k')} fill="none" stroke="#ffd479" strokeWidth={1} />
      {cur && <circle cx={toX(curIdx)} cy={toY(cur.e)} r={2.5} fill="#fff" />}
    </svg>
  );
}

const INITIAL_TETHER_POSITIONS = [0, 0, 2.5, 0];

export function DebugCockpit(): JSX.Element {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const minimapRef = useRef<HTMLCanvasElement | null>(null);
  const juliaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [riderSpace, setRiderSpace] = useState(false);
  const projectedTether = useRef<{ hand: THREE.Vector3; ball: THREE.Vector3 }>();
  const tetherRef = useRef({ x: 0, y: 0, held: false, reel: false, active: false });
  const updateTetherPointer = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const tether = tetherRef.current;
    tether.x = (18 * (event.clientX - rect.left) / rect.width - 9) / 1.4;
    tether.y = (9 - 18 * (event.clientY - rect.top) / rect.height) / 1.4;
    tether.held = (event.buttons & 1) !== 0;
    tether.reel = (event.buttons & 2) !== 0;
  };
  const juliaRendererRef = useRef<JuliaRenderer | null>(null);
  const previewScale = useRef(1);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(4);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [riderReady, setRiderReady] = useState(false);
  const [cockpitMode, setCockpitMode] = useState<'REPLAY' | 'MANUAL'>('REPLAY');
  const cockpitModeRef = useRef<'REPLAY' | 'MANUAL'>('REPLAY');
  cockpitModeRef.current = cockpitMode;
  const [playerView, setPlayerView] = useState(false);
  const [overlays, setOverlays] = useState<TerrainOverlays>(DEFAULT_OVERLAYS);
  const [runs, setRuns] = useState<CockpitTrajectory[] | null>(null);
  const [manualRun, setManualRun] = useState<CockpitTrajectory | null>(null);

  const manualDriverRef = useRef<ManualDriver | null>(null);
  const keysRef = useRef<KeyboardState>({
    up: false,
    down: false,
    left: false,
    right: false,
    drift: false,
    impulse: false,
  });

  const variants = useMemo(
    () => [...baselineVariants(), ...explorationVariants()],
    []
  );
  const sceneRefs = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    rider?: THREE.Group;
    terrain?: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
    trail?: THREE.Line;
    terrainCenter?: [number, number];
    terrainPatch?: ReturnType<typeof sampleTerrainPatch>;
    /** Latest authoritative metric speed, consumed by the rAF loop. */
    lastMetricSpeed?: number;
    /** Latest effective throttle (0..1), consumed by the rAF loop to drive PushOff blending. */
    lastThrottle?: number;
    /** Half-extent of the currently built terrain patch (LOD tracking). */
    lodHalf?: number;
    terrainOverlays?: TerrainOverlays;
  }>({});

  // Load wasm + record all variant trajectories once.
  useEffect(() => {
    (async () => {
      try {
        await initOrbitSynth();
        const recorder = new CockpitRecorder();
        const all = variants.map((v) => recorder.recordVariant(v));
        // Debug hook (read-only): lets external tooling inspect the
        // recorded snapshots without mutating any runtime state.
        (window as unknown as { __cockpitRuns?: CockpitTrajectory[] }).__cockpitRuns = all;
        setRuns(all);
        setReady(true);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, [variants]);

  // Three.js scene lifecycle.
  useEffect(() => {
    if (!ready || !mountRef.current || sceneRefs.current.renderer) return;
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    previewScale.current = 1;
    renderer.setPixelRatio(pixelRatio);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    buildSceneDressing(scene);
    const tetherView = new THREE.Group();
    tetherView.name = 'cockpit tether';
    tetherView.visible = false;
    const tetherGeometry = new THREE.BufferGeometry();
    const tetherVertices = new THREE.BufferAttribute(new Float32Array(6), 3);
    tetherGeometry.setAttribute('position', tetherVertices);
    const tetherLine = new THREE.Line(tetherGeometry, new THREE.LineBasicMaterial({ color: 0xffcb76 }));
    tetherLine.frustumCulled = false;
    const yoYo = new THREE.Mesh(
      new THREE.SphereGeometry(0.11, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffb646 }),
    );
    tetherView.add(tetherLine, yoYo);
    scene.add(tetherView);

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 200);
    camera.position.set(0, 6, -10);

    sceneRefs.current = { renderer, scene, camera };

    // Rider is async (GLB load); add it to the scene once ready. The
    // fallback capsule keeps the rider present even if the model 404s.
    let disposed = false;
    buildRider().then((rider) => {
      if (disposed) return;
      scene.add(rider);
      sceneRefs.current.rider = rider;
      setRiderReady(true);
    });

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    // Animation clock: advances the rider's GLB mixer every frame (rAF runs
    // continuously, so the gait plays even while the timeline is paused).
    let last = performance.now();
    let sampleStart = last;
    let sampleFrames = 0;
    let animId: number;
    const loop = () => {
      animId = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;

      // Adjust both preview buffers together. Keep the physics clock and
      // the CSS sizes (including the pointer coordinate space) independent.
      sampleFrames++;
      if (now - sampleStart >= 500) {
        const msPerFrame = (now - sampleStart) / sampleFrames;
        const scale = previewScale.current;
        let nextScale = scale;
        if (msPerFrame > 24) nextScale = Math.max(0.5, scale * 0.75);
        if (msPerFrame < 18) nextScale = Math.min(1, scale + 0.05);
        if (nextScale !== scale) {
          previewScale.current = nextScale;
          renderer.setPixelRatio(pixelRatio * nextScale);
          juliaRendererRef.current?.setResolutionScale(nextScale);
        }
        sampleStart = now;
        sampleFrames = 0;
      }

      // In MANUAL mode, advance the authoritative driver on canonical ticks
      const driver = manualDriverRef.current;
      if (cockpitModeRef.current === 'MANUAL' && driver) {
        const tether = tetherRef.current;
        const steps = driver.update(dt, keysRef.current, tether.active ? tether : undefined);
        if (steps > 0) {
          const snaps = driver.trajectory.snapshots;
          const latestSnap = snaps[snaps.length - 1];
          sceneRefs.current.lastMetricSpeed = latestSnap.physics.metricSpeed;
          sceneRefs.current.lastThrottle = latestSnap.action?.effective.throttle ?? 0;
          setManualRun(driver.trajectory);
          setFrameIdx(snaps.length - 1);
        }
      }

      const rider = sceneRefs.current.rider;
      if (rider) {
        const speed = sceneRefs.current.lastMetricSpeed ?? 0;
        const throttle = sceneRefs.current.lastThrottle ?? 0;
        updateRiderAnimation(rider, dt, speed, throttle);
        tetherView.visible = cockpitModeRef.current === 'MANUAL';
        const projected = projectedTether.current;
        if (tetherView.visible && projected) {
          tetherView.position.set(0, 0, 0);
          yoYo.position.copy(projected.ball);
          tetherVertices.setXYZ(0, projected.hand.x, projected.hand.y, projected.hand.z);
          tetherVertices.setXYZ(1, projected.ball.x, projected.ball.y, projected.ball.z);
          tetherVertices.needsUpdate = true;
        } else if (tetherView.visible) {
          const [handX, handY, ballX, ballY] = driver?.tetherPositions ?? INITIAL_TETHER_POSITIONS;
          // Display the top-down motion in a horizontal plane at hand height.
          const displayScale = 0.42;
          tetherView.position.copy(rider.position);
          tetherView.position.y += 0.46;
          yoYo.position.set(ballX * displayScale, 0, -ballY * displayScale);
          tetherVertices.setXYZ(0, handX * displayScale, 0, -handY * displayScale);
          tetherVertices.setXYZ(1, yoYo.position.x, yoYo.position.y, yoYo.position.z);
          tetherVertices.needsUpdate = true;
        }
      }
      renderer.render(scene, camera);
    };
    loop();

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', onResize);
      sceneRefs.current.terrain?.geometry.dispose();
      sceneRefs.current.terrain?.material.dispose();
      tetherGeometry.dispose();
      tetherLine.material.dispose();
      yoYo.geometry.dispose();
      yoYo.material.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRefs.current = {};
    };
  }, [ready]);

  // Keyboard input listeners for standard PC controls in MANUAL mode (#121).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Ignore when focused inside text inputs
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;

      if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') keysRef.current.up = true;
      if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') keysRef.current.down = true;
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keysRef.current.left = true;
      if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keysRef.current.right = true;
      if (e.key === 'Shift') keysRef.current.drift = true;
      if (e.key === ' ' || e.code === 'Space') {
        keysRef.current.impulse = true;
        e.preventDefault();
      }
      if (e.key === 'r' || e.key === 'R') {
        if (cockpitMode === 'MANUAL' && manualDriverRef.current) {
          manualDriverRef.current.reset();
          setManualRun({ ...manualDriverRef.current.trajectory });
          setFrameIdx(0);
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') keysRef.current.up = false;
      if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') keysRef.current.down = false;
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') keysRef.current.left = false;
      if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') keysRef.current.right = false;
      if (e.key === 'Shift') keysRef.current.drift = false;
      if (e.key === ' ' || e.code === 'Space') keysRef.current.impulse = false;
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [cockpitMode]);

  // Handle switching into MANUAL mode: initialize a fresh ManualDriver
  useEffect(() => {
    if (cockpitMode === 'MANUAL') {
      // Use selected variant start or [0, 0] as seed
      const currentSelectedRun = runs?.[selected];
      const startC = currentSelectedRun?.spec.initialC ?? [0, 0];
      const startV = currentSelectedRun?.spec.initialV ?? [0, 0];
      const driver = new ManualDriver(startC, startV);
      manualDriverRef.current = driver;
      setManualRun({ ...driver.trajectory });
      setFrameIdx(0);
      setPlaying(false);
    } else {
      manualDriverRef.current = null;
    }
  }, [cockpitMode, runs, selected]);

  // Rebuild terrain when the selected frame's patch center moves far from
  // the current mesh center (terrain follows the rider), and when the LOD
  // plan changes (scale shifted enough to re-plan fidelity vs performance).
  const [handX, handY, ballX, ballY] = manualDriverRef.current?.tetherPositions ?? INITIAL_TETHER_POSITIONS;
  const run = cockpitMode === 'MANUAL' ? manualRun : runs?.[selected];
  const frame = run?.snapshots[Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)];

  const mappedTether = useMemo(
    () => cockpitMode === 'MANUAL' && riderSpace && frame ? manualDriverRef.current?.riderSpaceTether(frame) : undefined,
    [cockpitMode, riderSpace, frame],
  );
  const tetherProbe = mappedTether
    ? { real: mappedTether.ball.x, imag: mappedTether.ball.y }
    : cockpitMode === 'MANUAL' ? manualDriverRef.current?.probe : undefined;
  const terrainWorker = useRef<Worker | null>(null);
  const terrainPending = useRef(false);
  const [terrainPatch, setTerrainPatch] = useState<ReturnType<typeof sampleTerrainPatch>>();
  useEffect(() => {
    if (!ready) return;
    const worker = new Worker(new URL('../lib/cockpitTerrain.worker.ts', import.meta.url), { type: 'module' });
    terrainWorker.current = worker;
    worker.onmessage = ({ data }: MessageEvent<
      { patch: ReturnType<typeof sampleTerrainPatch> } | { error: string }
    >) => {
      terrainPending.current = false;
      if ('patch' in data) setTerrainPatch(data.patch);
      else {
        terrainWorker.current = null;
        console.warn('[debugCockpit] terrain sampling failed:', data.error);
      }
    };
    return () => {
      worker.terminate();
      terrainWorker.current = null;
      terrainPending.current = false;
    };
  }, [ready]);

  useEffect(() => {
    const refs = sceneRefs.current;
    if (!refs.scene || !frame) return;
    if (terrainPatch && terrainPatch !== refs.terrainPatch) {
      const previous = refs.terrain;
      const mesh = buildTerrainMesh(terrainPatch, previous?.material);
      if (previous) {
        refs.scene.remove(previous);
        previous.geometry.dispose();
      }
      refs.scene.add(mesh);
      refs.terrain = mesh;
      refs.terrainPatch = terrainPatch;
      refs.terrainCenter = terrainPatch.center;
      refs.lodHalf = terrainPatch.half;
      refs.terrainOverlays = undefined;
      if (refs.camera) applyRenderDistance(refs.camera, refs.scene);
    }
    if (refs.terrain && refs.terrainPatch && refs.terrainOverlays !== overlays) {
      applyOverlays(refs.terrain, refs.terrainPatch, overlays);
      refs.terrainOverlays = overlays;
    }

    const [cx, cy] = frame.physics.c;
    const lod = planTerrainLod(frame.physics.rho, Math.hypot(...frame.physics.velocity));
    const center = refs.terrainCenter;
    const lodChanged = refs.lodHalf === undefined || Math.abs(refs.lodHalf - lod.half) > lod.half * 0.35;
    const moved = !center || Math.hypot(cx - center[0], cy - center[1]) > lod.half * 0.6;
    if ((moved || lodChanged) && terrainWorker.current && !terrainPending.current) {
      terrainPending.current = true;
      // Bound the preview mesh to at most 74k triangles.
      terrainWorker.current.postMessage([cx, cy, lod.half, Math.min(lod.grid, 193)]);
    }
  }, [frame, overlays, terrainPatch]);

  // Per-frame updates: rider, trail, camera.
  useEffect(() => {
    const refs = sceneRefs.current;
    const trajectory = cockpitMode === 'MANUAL' ? manualRun : runs?.[selected];
    if (!refs.scene || !refs.rider || !refs.camera || !trajectory || !frame) return;

    // Rider height: sampled per-position through the authoritative Rust
    // embedding seam — never the patch-center sigma, which diverges from
    // the surface near the Shore (|grad sigma| ~ 1e3) and let the rider
    // fly off the mountains. surfaceY applies the shared asinh compression
    // matching buildTerrainMesh.
    const heightAt = (x: number, y: number): number =>
      surfaceY(riderSurfaceHeight(frame, x, y));

    // Heading must advance before the hyperbolic frame consumes it.
    updateCamera(refs.camera, frame, CANONICAL_DT);
    placeRider(refs.rider, frame, heightAt);
    // Feed the animation gait from authoritative metric speed and forward throttle.
    refs.lastMetricSpeed = frame.physics.metricSpeed;
    refs.lastThrottle = frame.action?.effective.throttle ?? frame.action?.raw.throttle ?? 0;

    if (refs.trail) {
      refs.scene.remove(refs.trail);
      refs.trail.geometry.dispose();
      const materials = Array.isArray(refs.trail.material) ? refs.trail.material : [refs.trail.material];
      materials.forEach(material => material.dispose());
    }
    // Trail window: last 300 steps for legibility.
    const from = Math.max(0, frameIdx - 300);
    const windowTraj: CockpitTrajectory = {
      ...trajectory,
      snapshots: trajectory.snapshots.slice(from, frameIdx + 1),
    };
    // Build the trail: the static positions are a fallback; the hyperbolic
    // pass below projects the authoritative snapshots into the Poincaré
    // ball. The trail is rebuilt every frame, so the window is always fresh.
    refs.trail = buildTrail(windowTraj, windowTraj.snapshots.length - 1);
    refs.scene.add(refs.trail);

    // Hyperbolic camera (the only camera): project terrain mesh and trail
    // into the Poincaré ball centered at the camera in H^3 with orientation
    // applied.
    const smoothedHeading = getSmoothedCamHeading();
    const hyperbolicFrame = computeHyperbolicCameraFrame(frame, smoothedHeading);
    if (refs.terrain && refs.terrainPatch) {
      transformMeshToHyperbolic(refs.terrain, refs.terrainPatch, frame, smoothedHeading, hyperbolicFrame);
    }
    transformTrailToHyperbolic(refs.trail, windowTraj.snapshots, frame, smoothedHeading, hyperbolicFrame);
    transformRiderToHyperbolic(refs.rider, frame, smoothedHeading, hyperbolicFrame);
    if (mappedTether) {
      const inverseRotation = hyperbolicFrame.rotationMatrix.clone().invert();
      const project = (point: typeof mappedTether.ball) =>
        poincareToSceneVector(projectPointToCameraPoincare(point, hyperbolicFrame.cameraUpperHalf))
          .applyMatrix4(inverseRotation).multiplyScalar(HYPERBOLIC_VISUAL_SCALE);
      projectedTether.current = { hand: project(mappedTether.hand), ball: project(mappedTether.ball) };
    } else {
      projectedTether.current = undefined;
    }
  }, [runs, selected, manualRun, cockpitMode, frameIdx, frame, riderReady, terrainPatch, mappedTether]);

  const updateMinimap = useRef<((input: MinimapPaintInput) => void) | null>(null);
  useEffect(() => {
    const canvas = minimapRef.current;
    if (!ready || !canvas) return;
    const worker = new Worker(new URL('../lib/cockpitMinimap.worker.ts', import.meta.url), { type: 'module' });
    type Background = { view: MinimapView; pixels: Uint8ClampedArray };
    let latest: MinimapPaintInput | undefined;
    let background: Background | undefined;
    let pending = false;
    updateMinimap.current = input => {
      latest = input;
      if (background) paintMinimapImage(canvas, input, background.view, background.pixels);
      const view = minimapView(canvas.width, input);
      const old = background?.view;
      if (pending || (old && old.half === view.half && old.center[0] === view.center[0] && old.center[1] === view.center[1])) return;
      pending = true;
      worker.postMessage(view);
    };
    worker.onmessage = ({ data }: MessageEvent<Background>) => {
      pending = false;
      background = data;
      if (latest) updateMinimap.current?.(latest);
    };
    return () => {
      updateMinimap.current = null;
      worker.terminate();
    };
  }, [ready]);

  // Overlay the latest positions on the most recent completed map.
  useEffect(() => {
    const canvas = minimapRef.current;
    const trajectory = cockpitMode === 'MANUAL' ? manualRun : runs?.[selected];
    if (!canvas || !trajectory || !frame) return;
    // Trail window: last 200 steps for panel legibility.
    const from = Math.max(0, frameIdx - 200);
    const trail = trajectory.snapshots
      .slice(from, frameIdx + 1)
      .map((s) => [s.physics.c[0], s.physics.c[1]] as [number, number]);
    const input: MinimapPaintInput = {
      yoYoC: tetherProbe && [tetherProbe.real, tetherProbe.imag],
      trail,
      currentC: [frame.physics.c[0], frame.physics.c[1]],
      // Zoom-with-scale: the window tracks the current LOD patch extent.
      footprintHalf: sceneRefs.current.lodHalf ?? DEFAULT_TERRAIN_HALF,
      // FOV triangle: heading from the authoritative planar velocity (the
      // same rule placeRider uses), FOV from the cockpit camera.
      heading:
        Math.hypot(frame.physics.velocity[0], frame.physics.velocity[1]) > 1e-7
          ? Math.atan2(-frame.physics.velocity[1], frame.physics.velocity[0])
          : 0,
      fovDeg: 55,
    };
    updateMinimap.current?.(input);
  }, [runs, selected, manualRun, cockpitMode, frameIdx, frame, terrainPatch, tetherProbe?.real, tetherProbe?.imag]);

  // Julia panel: the ACTUAL audience-facing view (issue #111). The existing
  // JuliaRenderer stays the authoritative presentation surface.
  useEffect(() => {
    if (!ready || !juliaCanvasRef.current || juliaRendererRef.current) return;
    let cancelled = false;
    const renderer = new JuliaRenderer(juliaCanvasRef.current);
    renderer.setResolutionScale(previewScale.current);
    (async () => {
      try {
        await renderer.init(juliaFragment);
        if (cancelled) return;
        juliaRendererRef.current = renderer;
        renderer.updateParameters({
          juliaSeed: { real: 0, imag: 0 },
          colorHue: 0.58,
          colorSat: 0.75,
          colorBright: 0.6,
          // zoom=1.0 shows the whole ~2.7-span Julia set; deeper zooms land
          // the panel inside the connected interior when the seed is inside M.
          zoom: 1.0,
          speed: 1.0,
        });
        renderer.start();
      } catch (e) {
        // Backend API not available (e.g. CI, standalone mode): skip Julia panel.
        if (!cancelled) console.warn('[debugCockpit] julia panel init failed (backend API unavailable?):', e);
      }
    })();
    return () => {
      cancelled = true;
      if (juliaRendererRef.current) {
        juliaRendererRef.current.stop();
        juliaRendererRef.current = null;
      }
    };
  }, [ready]);

  // Push the current frame's c into the fixed-view Julia renderer. Only the
  // seed follows physics; zoom/rotation/palette stay FIXED so the audience
  // view's presentation deltas remain distinct from Mandelbrot scale.
  useEffect(() => {
    const renderer = juliaRendererRef.current;
    if (!renderer || !frame) return;
    const current = renderer.getCurrentParameters();
    renderer.updateParameters({
      ...current,
      juliaSeed: tetherProbe
        ?? { real: frame.physics.c[0], imag: frame.physics.c[1] },
    });
  }, [frame, cockpitMode, tetherProbe?.real, tetherProbe?.imag]);

  // Playback loop.
  useEffect(() => {
    if (!playing || !run) return;
    const timer = window.setInterval(() => {
      setFrameIdx((i) => {
        const next = i + 1;
        if (next >= run.snapshots.length) {
          setPlaying(false);
          return i;
        }
        return next;
      });
    }, 16);
    return () => window.clearInterval(timer);
  }, [playing, run]);

  const physics = frame?.physics;
  const diag = frame?.diagnostics;

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, monospace', color: '#dde', background: '#06060c' }}>
      {/* LEFT: OBSERVATION-slot + variant list (Phase A: STATE/DIAG panels) */}
      <div style={{ width: 330, padding: 12, overflowY: 'auto', borderRight: '1px solid #1c1c30' }}>
        <h1 style={{ fontSize: 15, margin: '0 0 2px' }}>Mandelbrot-manifold Debug Cockpit</h1>
        <p style={{ fontSize: 10, color: '#667', margin: '0 0 10px' }}>
          issue #111 Phase A — read-only DebugSnapshot · Rust-authoritative physics
        </p>
        {error && <p style={{ color: '#f66', fontSize: 12 }}>{error}</p>}
        {!ready && !error && <p style={{ fontSize: 12 }}>initializing wasm…</p>}

        <div hidden={playerView}><JuliaSensitivityBookmark /></div>

        {runs && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 11, color: '#778', marginBottom: 4 }}>RECORDED TRAJECTORIES (Controls v2 replays)</div>
            {runs.map((r, i) => (
              <button
                key={r.spec.name}
                onClick={() => {
                  setSelected(i);
                  setFrameIdx(0);
                  setPlaying(false);
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  background: i === selected ? '#20203a' : '#141424',
                  color: '#dde',
                  border: '1px solid #2c2c48',
                  borderRadius: 6,
                  padding: '5px 8px',
                  marginBottom: 4,
                  cursor: 'pointer',
                  fontSize: 11,
                }}
              >
                <strong>{r.spec.name}</strong>{' '}
                <span style={{ color: r.crossed ? '#7f7' : '#f96' }}>
                  {r.crossed
                    ? r.crossingStep === 0
                      ? 'starts outside'
                      : `crossed @ ${r.crossingStep}`
                    : 'no crossing'}
                </span>
                <br />
                <span style={{ color: '#667', fontSize: 10 }}>
                  max U {r.maxPotential.toFixed(2)} {r.crestedRidge ? '· crested' : ''}
                </span>
              </button>
            ))}
          </div>
        )}

        {!playerView && physics && (
          <Panel title="WORLD / PHYSICS">
            <Row label="c = (x, y)" value={`${physics.c[0].toFixed(6)}, ${physics.c[1].toFixed(6)}`} kind="STATE" digits={6} />
            <Row label="v (planar)" value={`${physics.velocity[0].toFixed(6)}, ${physics.velocity[1].toFixed(6)}`} kind="STATE" digits={6} />
            <Row label="D(c) signed" value={physics.signedDistance} kind="STATE" digits={7} />
            <Row label="realm" value={realmName(physics.realm)} kind="STATE" />
            <Row label="rho" value={physics.rho} kind="STATE" digits={7} />
            <Row label="sigma (scale)" value={physics.sigma} kind="STATE" digits={5} />
            <Row label="sigma_dot" value={physics.sigmaDot} kind="STATE" digits={5} />
            <Row label="|grad sigma|" value={Math.hypot(physics.scaleGradient[0], physics.scaleGradient[1])} kind="STATE" />
            <Row label="metric speed" value={physics.metricSpeed} kind="STATE" />
            <Row label="K kinetic" value={physics.kinetic} kind="STATE" />
            <Row label="U potential" value={physics.potential} kind="STATE" />
            <Row label="E total" value={physics.total} kind="STATE" />
            <Row label="geodesic |a|" value={Math.hypot(physics.geodesicAccel[0], physics.geodesicAccel[1])} kind="STATE" />
            <Row label="potential force" value={`${physics.potentialForce[0].toFixed(3)}, ${physics.potentialForce[1].toFixed(3)}`} kind="STATE" />
            <Row label="net accel" value={Math.hypot(physics.netAccel[0], physics.netAccel[1])} kind="STATE" />
            <Row label="derivative valid" value={physics.derivativeValid ? 'yes' : 'NO'} kind="DIAG" />
          </Panel>
        )}

        {!playerView && frame && <EnergyLedgerPanel frame={frame} />}
{!playerView && diag && (
          <Panel title="DIAGNOSTICS">
            <Row label="step clock (s)" value={frame?.timeSeconds} kind="STATE" digits={4} />
            <Row label="fd step" value={diag.derivativeStep} kind="DIAG" digits={9} />
            <Row label="last dE" value={diag.lastDeltaTotal} kind="DIAG" />
            <Row label="crest U (ceiling)" value={diag.crestPotential} kind="STATE" />
            <Row label="integrator" value={diag.valid ? 'ok' : `FAIL: ${diag.lastError ?? ''}`} kind="DIAG" />
          </Panel>
        )}

        {!playerView && (
          <Panel title="TERRAIN OVERLAYS">
            {(
              [
                ['shoreBand', 'Shore D(c)=0 band'],
                ['realm', 'Inside/Outside realm'],
                ['sigma', 'sigma(c) scale ramp'],
                ['potential', 'U(c) potential ramp'],
                ['validity', 'derivative validity'],
              ] as Array<[keyof TerrainOverlays, string]>
            ).map(([key, label]) => (
              <label key={key} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, lineHeight: 1.7, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={overlays[key]}
                  onChange={(e) => setOverlays((o) => ({ ...o, [key]: e.target.checked }))}
                />
                {label}
              </label>
            ))}
          </Panel>
        )}

        {!playerView && (
          <Panel title="MANDELBROT MINIMAP / TRAIL">
            <canvas
              ref={minimapRef}
              width={MINIMAP_SIZE}
              height={MINIMAP_SIZE}
              style={{ width: '100%', borderRadius: 4, border: '1px solid #262640', display: 'block' }}
            />
            <div style={{ fontSize: 10, color: '#667', marginTop: 4 }}>
              mandelbrot deep zoom · zoomed to scale · trail (green) · c (red) · FOV (triangle)
            </div>
          </Panel>
        )}
      </div>

      {/* CENTER: 3D SKATER + MANIFOLD */}
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

        {physics && (
          <div
            style={{
              position: 'absolute',
              top: 10,
              left: 12,
              padding: '6px 10px',
              borderRadius: 6,
              background: 'rgba(0,0,0,0.6)',
              fontSize: 12,
              border: '1px solid #333',
            }}
          >
            <span style={{ color: physics.realm < 0 ? '#6af' : physics.realm > 0 ? '#7f7' : '#ff4' }}>
              {realmName(physics.realm)}
            </span>
            <span style={{ color: '#889' }}> · step {frameIdx} · t={((frame?.timeSeconds ?? 0)).toFixed(3)}s</span>
          </div>
        )}

        {/* Mode Toggle (REPLAY vs MANUAL per #121), Camera + Player View controls */}
        <div style={{ position: 'absolute', top: 10, right: 12, display: 'flex', gap: 8 }}>
          <div style={{ display: 'flex', borderRadius: 6, overflow: 'hidden', border: '1px solid #2c2c48' }}>
            <button
              onClick={() => setCockpitMode('REPLAY')}
              style={{
                background: cockpitMode === 'REPLAY' ? '#3d3d6b' : '#141424',
                color: '#dde',
                border: 'none',
                padding: '6px 12px',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: cockpitMode === 'REPLAY' ? 'bold' : 'normal',
              }}
            >
              REPLAY
            </button>
            <button
              onClick={() => setCockpitMode('MANUAL')}
              style={{
                background: cockpitMode === 'MANUAL' ? '#2e6b3e' : '#141424',
                color: '#dde',
                border: 'none',
                padding: '6px 12px',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: cockpitMode === 'MANUAL' ? 'bold' : 'normal',
              }}
            >
              MANUAL (WASD)
            </button>
          </div>

          <button
            onClick={() => setPlayerView((p) => !p)}
            style={{
              background: playerView ? '#2b4a2b' : '#20203a',
              color: '#dde',
              border: '1px solid #2c2c48',
              borderRadius: 6,
              padding: '6px 10px',
              cursor: 'pointer',
              fontSize: 11,
            }}
          >
            {playerView ? 'player view: ON' : 'player view'}
          </button>
        </div>

        {/* Controls HUD for MANUAL mode */}
        {cockpitMode === 'MANUAL' && (
          <div
            style={{
              position: 'absolute',
              top: 56,
              left: 12,
              padding: '8px 12px',
              borderRadius: 6,
              background: 'rgba(0,0,0,0.7)',
              fontSize: 11,
              border: '1px solid #335',
              lineHeight: 1.6,
              color: '#aab',
            }}
          >
            <strong style={{ color: '#7f7' }}>MANUAL DRIVING (Controls v2)</strong>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>W</kbd> / <kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>↑</kbd> : Forward Throttle (Push)</div>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>S</kbd> / <kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>↓</kbd> : Brake (Dissipate)</div>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>A</kbd> / <kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>D</kbd> : Steer Left / Right (Yaw Heading)</div>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>Shift</kbd> : Drift (Reduce Grip)</div>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>Space</kbd> : Impulse (Jump/Pop)</div>
            <div><kbd style={{ background: '#223', padding: '1px 4px', borderRadius: 3 }}>R</kbd> : Reset trajectory</div>
          </div>
        )}

        {/* RIGHT: ACTION panel (Controls v2) */}
        {frame?.action && (
          <div style={{ position: 'absolute', right: 12, top: 56, width: 280 }}>
            <Panel title="ACTION — CONTROLS v2">
              <Row label="raw throttle" value={frame.action.raw.throttle} kind="ACTION" />
              <Row label="effective throttle" value={frame.action.effective.throttle} kind="ACTION" />
              <Row label="drive dir" value={`${frame.action.effective.direction[0].toFixed(2)}, ${frame.action.effective.direction[1].toFixed(2)}`} kind="ACTION" />
              <Row label="brake" value={frame.action.effective.brake} kind="ACTION" />
              <Row label="grip" value={frame.action.effective.grip} kind="ACTION" />
              <Row label="impulse" value={frame.action.effective.impulse} kind="ACTION" />
              <Row label="Q_drive covector" value={`${frame.action.driveCovector[0].toFixed(3)}, ${frame.action.driveCovector[1].toFixed(3)}`} kind="STATE" />
              <Row label="friction beta" value={frame.action.frictionBeta} kind="STATE" />
              <Row label="friction power" value={frame.action.frictionPower} kind="STATE" />
            </Panel>
          </div>
        )}

        {cockpitMode === 'MANUAL' && (
          <div style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: 84, width: 'min(280px, 36vw)' }}>
            <Panel title="YO-YO · TOP DOWN" translucent>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, marginBottom: 4 }}>
                Swing coordinates
                <select value={riderSpace ? 'rider' : 'complex'} onChange={event => setRiderSpace(event.target.value === 'rider')}>
                  <option value="complex">Complex plane</option>
                  <option value="rider">Rider space</option>
                </select>
              </label>
              <svg
                aria-label="Top-down yo-yo controls"
                viewBox="-9 -9 18 18"
                onContextMenu={event => event.preventDefault()}
                onPointerDown={event => {
                  if (event.button !== 0 && event.button !== 2) return;
                  event.currentTarget.setPointerCapture(event.pointerId);
                  tetherRef.current.active = true;
                  updateTetherPointer(event);
                }}
                onPointerMove={updateTetherPointer}
                onPointerUp={updateTetherPointer}
                onLostPointerCapture={() => { tetherRef.current.held = tetherRef.current.reel = false; }}
                style={{ display: 'block', width: '100%', aspectRatio: '1', background: 'rgba(17,24,36,0.18)', borderRadius: 4, cursor: 'crosshair', touchAction: 'none' }}
              >
                <path d="M-9 0H9 M0 -9V9" stroke="#9badbf" strokeOpacity=".55" strokeWidth=".06" />
                <circle r="1.4" fill="none" stroke="#adc3d9" strokeOpacity=".8" strokeWidth=".08" strokeDasharray=".2 .2" />
                <line x2={handX} y2={-handY} stroke="#80c8fa" strokeWidth=".14" />
                <line x1={handX} y1={-handY} x2={ballX} y2={-ballY} stroke="#ffcb76" strokeWidth=".12" />
                <circle r=".32" fill="#80c8fa" />
                <circle cx={handX} cy={-handY} r=".20" fill="#deefff" />
                <circle cx={ballX} cy={-ballY} r=".30" fill="#ffcb76" />
              </svg>
              <div style={{ fontSize: 11, color: '#eef', marginTop: 6, textShadow: '0 1px 3px #080812' }}>
                WASD drives. Mouse moves the hand. Hold left to catch, release to pay out, hold right to reel.
              </div>
            </Panel>
          </div>
        )}
        {/* RIGHT-BOTTOM: ACTUAL JULIA AUDIENCE VIEW (fixed presentation) */}
        <div
          style={{
            position: 'absolute',
            right: 12,
            bottom: 64,
            width: 250,
          }}
        >
          <Panel title="ACTUAL JULIA AUDIENCE VIEW (fixed)">
            <canvas ref={juliaCanvasRef} style={{ width: '100%', aspectRatio: '1 / 1', borderRadius: 4, display: 'block' }} />
          </Panel>
        </div>
      </div>

      {/* BOTTOM: timeline */}
      <div
        style={{
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          background: 'rgba(8,8,18,0.92)',
          borderTop: '1px solid #1c1c30',
          padding: '8px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}
      >
        <button
          onClick={() => setPlaying((p) => !p)}
          disabled={!run}
          style={{ background: '#20203a', color: '#dde', border: '1px solid #2c2c48', borderRadius: 6, padding: '6px 14px', cursor: 'pointer' }}
        >
          {playing ? 'pause' : 'play'}
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            setFrameIdx((i) => Math.min(i + 1, (run?.snapshots.length ?? 1) - 1));
          }}
          disabled={!run}
          style={{ background: '#20203a', color: '#dde', border: '1px solid #2c2c48', borderRadius: 6, padding: '6px 14px', cursor: 'pointer' }}
        >
          step +1 tick
        </button>
        <button
          onClick={() => {
            setPlaying(false);
            if (cockpitMode === 'MANUAL' && manualDriverRef.current) {
              manualDriverRef.current.reset();
              setManualRun(manualDriverRef.current.trajectory);
              setFrameIdx(0);
            } else {
              setFrameIdx(0);
            }
          }}
          disabled={!run}
          style={{ background: '#20203a', color: '#dde', border: '1px solid #2c2c48', borderRadius: 6, padding: '6px 14px', cursor: 'pointer' }}
        >
          reset
        </button>
        <input
          type="range"
          min={0}
          max={(run?.snapshots.length ?? 1) - 1}
          value={Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)}
          onChange={(e) => {
            setPlaying(false);
            setFrameIdx(Number(e.target.value));
          }}
          style={{ flex: 1 }}
          disabled={!run}
        />
        {!playerView && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <EnergySparkline trajectory={run ?? undefined} frameIdx={frameIdx} />
            <span style={{ fontSize: 9, color: '#667', lineHeight: 1.4 }}>
              <span style={{ color: '#ffd479' }}>K</span> ·{' '}
              <span style={{ color: '#7fb0ff' }}>U</span> ·{' '}
              <span style={{ color: '#9fe8b0' }}>E</span>
            </span>
          </div>
        )}
        <span style={{ fontSize: 11, color: '#889', minWidth: 190, textAlign: 'right' }}>
          tick {Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)} / {(run?.snapshots.length ?? 1) - 1}
          {' · '}Δt {CANONICAL_DT.toFixed(6)}s (hop clock)
        </span>
      </div>
    </div>
  );
}
