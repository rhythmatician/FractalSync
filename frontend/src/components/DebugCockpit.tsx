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
  ensureMinimapPyramid,
  type CockpitTrajectory,
} from '../lib/debugCockpit';
import { baselineVariants, explorationVariants } from '../lib/shoreCrossingVariants';
import { initOrbitSynth, getWasmModule } from '../lib/orbitSynthesizer';
import { JuliaRenderer } from '../lib/juliaRenderer';
import {
  MINIMAP_SIZE,
  paintMinimap,
  setMinimapWasmSurface,
  type MinimapPaintInput,
} from '../lib/cockpitMinimap';
import {
  buildTerrainMesh,
  buildRider,
  buildTrail,
  placeRider,
  updateCamera,
  updateRiderAnimation,
  treadmillTransform,
  physicalTransform,
  treadmillTrailTransform,
  physicalTrailTransform,
  buildSceneDressing,
  applyOverlays,
  applyRenderDistance,
  surfaceY,
  DEFAULT_OVERLAYS,
  type CameraMode,
  type TerrainOverlays,
} from '../lib/cockpitScene';

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

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      style={{
        background: 'rgba(10,10,22,0.82)',
        border: '1px solid #262640',
        borderRadius: 8,
        padding: '8px 10px',
        minWidth: 250,
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: 1, color: '#778', marginBottom: 4 }}>{title}</div>
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

export function DebugCockpit(): JSX.Element {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const minimapRef = useRef<HTMLCanvasElement | null>(null);
  const juliaCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const juliaRendererRef = useRef<JuliaRenderer | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(4);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('physical');
  const [playerView, setPlayerView] = useState(false);
  const [overlays, setOverlays] = useState<TerrainOverlays>(DEFAULT_OVERLAYS);
  const [pyramidReady, setPyramidReady] = useState(false);
  const [runs, setRuns] = useState<CockpitTrajectory[] | null>(null);

  const variants = useMemo(
    () => [...baselineVariants(), ...explorationVariants()],
    []
  );
  const sceneRefs = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    rider?: THREE.Group;
    terrain?: THREE.Mesh;
    trail?: THREE.Line;
    terrainCenter?: [number, number];
    terrainPatch?: ReturnType<typeof sampleTerrainPatch>;
    /** Latest authoritative metric speed, consumed by the rAF loop. */
    lastMetricSpeed?: number;
    /** Half-extent of the currently built terrain patch (LOD tracking). */
    lodHalf?: number;
    /** Camera mode the current terrain mesh was built for. */
    terrainMode?: CameraMode;
  }>({});

  // Load wasm + record all variant trajectories once.
  useEffect(() => {
    let disposed = false;
    (async () => {
      try {
        await initOrbitSynth();
        const meta = (getWasmModule() as unknown as {
          debugSnapshotMeta?: () => { version: string; canonicalDt: number };
        }).debugSnapshotMeta?.();
        if (!meta || meta.version !== 'debug-snapshot/1') {
          throw new Error('wasm build lacks debug-snapshot/1 seam; rebuild wasm-orbit');
        }
        if (disposed) return;
        setMinimapWasmSurface(getWasmModule() as never);
        const recorder = new CockpitRecorder();
        const all = variants.map((v) => recorder.recordVariant(v));
        setRuns(all);
        setReady(true);
        // Minimap pyramid: best-effort, off the critical path.
        ensureMinimapPyramid().then((ok) => {
          if (!disposed) setPyramidReady(ok);
        });
      } catch (e) {
        setError(String(e));
      }
    })();
    return () => {
      disposed = true;
    };
  }, [variants]);

  // Three.js scene lifecycle.
  useEffect(() => {
    if (!ready || !mountRef.current || sceneRefs.current.renderer) return;
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    buildSceneDressing(scene);

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
    const loop = () => {
      requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      const rider = sceneRefs.current.rider;
      if (rider) {
        const speed = sceneRefs.current.lastMetricSpeed ?? 0;
        updateRiderAnimation(rider, dt, speed);
      }
      renderer.render(scene, camera);
    };
    loop();

    return () => {
      disposed = true;
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRefs.current = {};
    };
  }, [ready]);

  // Rebuild terrain when the selected frame's patch center moves far from
  // the current mesh center (terrain follows the rider), and when the LOD
  // plan changes (scale shifted enough to re-plan fidelity vs performance).
  const run = runs?.[selected];
  const frame = run?.snapshots[Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)];

  useEffect(() => {
    const refs = sceneRefs.current;
    if (!refs.scene || !frame) return;
    const [cx, cy] = frame.physics.c;
    const planarSpeed = Math.hypot(frame.physics.velocity[0], frame.physics.velocity[1]);
    const lod = planTerrainLod(frame.physics.rho, planarSpeed);
    const current = refs.terrainCenter;
    const lodChanged =
      refs.lodHalf === undefined || Math.abs(refs.lodHalf - lod.half) > lod.half * 0.35;
    const moved =
      !current || Math.hypot(cx - current[0], cy - current[1]) > lod.half * 0.6;
    const modeChanged = refs.terrainMode !== cameraMode;
    if (!moved && !lodChanged && !modeChanged && refs.terrain && refs.terrainPatch) {
      applyOverlays(refs.terrain, refs.terrainPatch, overlays);
      return;
    }
    if (refs.terrain) {
      refs.scene.remove(refs.terrain);
      refs.terrain.geometry.dispose();
    }
    const patch = sampleTerrainPatch(cx, cy, lod.half, lod.grid);
    // Build the terrain mesh in CHART COORDINATES for the active camera
    // mode so the Y axis is the right surface from the start: physical
    // mode uses the asinh-compressed surfaceY(); treadmill mode uses the
    // exact linear embedding-height chart Y. modeChanged above guarantees
    // a mode toggle re-emits the geometry instead of reusing a mesh built
    // for the other mode's Y mapping.
    const mesh = buildTerrainMesh(patch, cameraMode);
    applyOverlays(mesh, patch, overlays);
    refs.scene.add(mesh);
    refs.terrain = mesh;
    refs.terrainPatch = patch;
    refs.terrainCenter = [cx, cy];
    refs.lodHalf = lod.half;
    refs.terrainMode = cameraMode;
    // Render distance + fog wall track the patch (and the treadmill chart's
    // 1/rho magnification) so the mesh edge always hides inside the fog
    // while the rider stays fog-free — fidelity balanced with performance.
    if (refs.camera) {
      applyRenderDistance(refs.camera, refs.scene, cameraMode, frame.physics.rho, lod.half);
    }
  }, [frame, overlays, cameraMode]);

  // Per-frame updates: rider, trail, camera.
  useEffect(() => {
    const refs = sceneRefs.current;
    const trajectory = runs?.[selected];
    if (!refs.scene || !refs.rider || !refs.camera || !trajectory || !frame) return;

    // Rider height: sampled per-position through the authoritative Rust
    // embedding seam — never the patch-center sigma, which diverges from
    // the surface near the Shore (|grad sigma| ~ 1e3) and let the rider
    // fly off the mountains. surfaceY applies the shared asinh compression
    // matching buildTerrainMesh.
    const heightAt = (x: number, y: number): number =>
      surfaceY(riderSurfaceHeight(frame, x, y));

    placeRider(refs.rider, frame, heightAt);
    // Feed the animation gait from authoritative metric speed.
    refs.lastMetricSpeed = frame.physics.metricSpeed;

    if (refs.trail) {
      refs.scene.remove(refs.trail);
      refs.trail.geometry.dispose();
    }
    // Trail window: last 300 steps for legibility.
    const from = Math.max(0, frameIdx - 300);
    const windowTraj: CockpitTrajectory = {
      ...trajectory,
      snapshots: trajectory.snapshots.slice(from, frameIdx + 1),
    };
    // Build the trail in CHART COORDINATES for the active camera mode:
    // physical mode uses surfaceY(); treadmill mode uses the linear
    // embedding-height chart. The trail is rebuilt every frame, so the
    // camera mode is always reflected.
    refs.trail = buildTrail(windowTraj, windowTraj.snapshots.length - 1, cameraMode);
    refs.scene.add(refs.trail);

    if (cameraMode === 'treadmill') {
      // Chart transform: terrain AND trail shift together so the trail
      // stays glued to the surface; the rider sits at the chart origin.
      if (refs.terrain) treadmillTransform(refs.terrain, frame);
      treadmillTrailTransform(refs.trail, frame);
      refs.rider.position.set(0, 0, 0);
    } else {
      if (refs.terrain) physicalTransform(refs.terrain);
      physicalTrailTransform(refs.trail);
    }
    updateCamera(refs.camera, frame, cameraMode);
  }, [runs, selected, frameIdx, frame, cameraMode]);

  // Minimap panel: repaint from the canonical pyramid when the frame moves.
  useEffect(() => {
    const canvas = minimapRef.current;
    const trajectory = runs?.[selected];
    if (!canvas || !trajectory || !frame) return;
    const extent = frame.map.extent;
    if (!extent) return;
    // Trail window: last 200 steps for panel legibility.
    const from = Math.max(0, frameIdx - 200);
    const trail = trajectory.snapshots
      .slice(from, frameIdx + 1)
      .map((s) => [s.physics.c[0], s.physics.c[1]] as [number, number]);
    const input: MinimapPaintInput = {
      extent: [extent[0], extent[1], extent[2], extent[3]],
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
    paintMinimap(canvas, input);
  }, [runs, selected, frameIdx, frame, pyramidReady]);

  // Julia panel: the ACTUAL audience-facing view (issue #111). The existing
  // JuliaRenderer stays the authoritative presentation surface.
  useEffect(() => {
    if (!ready || !juliaCanvasRef.current || juliaRendererRef.current) return;
    let cancelled = false;
    const renderer = new JuliaRenderer(juliaCanvasRef.current);
    juliaRendererRef.current = renderer;
    (async () => {
      try {
        await renderer.init();
        if (cancelled) return;
        renderer.updateParameters({
          juliaSeed: { real: 0, imag: 0 },
          colorHue: 0.58,
          colorSat: 0.75,
          colorBright: 0.6,
          zoom: 2.5,
          speed: 1.0,
        });
        renderer.start();
      } catch (e) {
        if (!cancelled) console.warn('[debugCockpit] julia panel init failed:', e);
      }
    })();
    return () => {
      cancelled = true;
      renderer.stop();
      juliaRendererRef.current = null;
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
      juliaSeed: { real: frame.physics.c[0], imag: frame.physics.c[1] },
    });
  }, [frame]);

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
                  {r.crossed ? `crossed @ ${r.crossingStep}` : 'no crossing'}
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
              {pyramidReady
                ? 'canonical mip pyramid · zoomed to scale · trail (green) · c (red) · FOV (triangle)'
                : 'mip pyramid unavailable (backend :8000) — panel degraded honestly'}
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

        {/* Camera + Player View controls */}
        <div style={{ position: 'absolute', top: 10, right: 12, display: 'flex', gap: 8 }}>
          <button
            onClick={() => setCameraMode((m) => (m === 'physical' ? 'treadmill' : 'physical'))}
            style={{ background: '#20203a', color: '#dde', border: '1px solid #2c2c48', borderRadius: 6, padding: '6px 10px', cursor: 'pointer', fontSize: 11 }}
          >
            cam: {cameraMode === 'physical' ? 'PHYSICAL (x,y,λσ)' : 'TREADMILL (debug chart)'}
          </button>
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
            <canvas
              ref={juliaCanvasRef}
              style={{ width: '100%', aspectRatio: '1 / 1', borderRadius: 4, display: 'block' }}
            />
            <div style={{ fontSize: 10, color: '#667', marginTop: 4 }}>
              seed = physics c · zoom/rotation/palette HELD FIXED — what the audience
              sees vs the terrain the Player rides
            </div>
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
            setFrameIdx(0);
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
            <EnergySparkline trajectory={run} frameIdx={frameIdx} />
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
