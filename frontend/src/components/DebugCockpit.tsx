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
  DEFAULT_TERRAIN_GRID,
  sampleTerrainPatch,
  type CockpitTrajectory,
} from '../lib/debugCockpit';
import { baselineVariants } from '../lib/shoreCrossingVariants';
import { initOrbitSynth, getWasmModule } from '../lib/orbitSynthesizer';
import {
  buildTerrainMesh,
  buildRider,
  buildTrail,
  placeRider,
  updateCamera,
  treadmillTransform,
  physicalTransform,
  treadmillTrailTransform,
  physicalTrailTransform,
  buildSceneDressing,
  type CameraMode,
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

export function DebugCockpit(): JSX.Element {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(3);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cameraMode, setCameraMode] = useState<CameraMode>('physical');
  const [playerView, setPlayerView] = useState(false);
  const [runs, setRuns] = useState<CockpitTrajectory[] | null>(null);

  const variants = useMemo(() => baselineVariants(), []);
  const sceneRefs = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    rider?: THREE.Group;
    terrain?: THREE.Mesh;
    trail?: THREE.Line;
    terrainCenter?: [number, number];
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
        const recorder = new CockpitRecorder();
        const all = variants.map((v) => recorder.recordVariant(v));
        setRuns(all);
        setReady(true);
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

    const rider = buildRider();
    scene.add(rider);

    sceneRefs.current = { renderer, scene, camera, rider };

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    const loop = () => {
      requestAnimationFrame(loop);
      renderer.render(scene, camera);
    };
    loop();

    return () => {
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRefs.current = {};
    };
  }, [ready]);

  // Rebuild terrain when the selected frame's patch center moves far from
  // the current mesh center (terrain follows the rider).
  const run = runs?.[selected];
  const frame = run?.snapshots[Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)];

  useEffect(() => {
    const refs = sceneRefs.current;
    if (!refs.scene || !frame) return;
    const [cx, cy] = frame.physics.c;
    const current = refs.terrainCenter;
    const moved = !current || Math.hypot(cx - current[0], cy - current[1]) > DEFAULT_TERRAIN_HALF * 0.6;
    if (!moved) return;
    if (refs.terrain) {
      refs.scene.remove(refs.terrain);
      refs.terrain.geometry.dispose();
    }
    const patch = sampleTerrainPatch(cx, cy, DEFAULT_TERRAIN_HALF, DEFAULT_TERRAIN_GRID);
    const mesh = buildTerrainMesh(patch);
    refs.scene.add(mesh);
    refs.terrain = mesh;
    refs.terrainCenter = [cx, cy];
  }, [frame]);

  // Per-frame updates: rider, trail, camera.
  useEffect(() => {
    const refs = sceneRefs.current;
    const trajectory = runs?.[selected];
    if (!refs.scene || !refs.rider || !refs.camera || !trajectory || !frame) return;

    // Rider height: the terrain patch is centered on the rider, so the
    // surface height under the rider is the patch-center embedding height
    // lambda*sigma (Z_SCALE applied, matching buildTerrainMesh).
    const sigma = frame.physics.sigma;
    const heightAt = (_x: number, _y: number): number => sigma * 2.0;

    placeRider(refs.rider, frame, heightAt);

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
    refs.trail = buildTrail(windowTraj, windowTraj.snapshots.length - 1);
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
        <span style={{ fontSize: 11, color: '#889', minWidth: 190, textAlign: 'right' }}>
          tick {Math.min(frameIdx, (run?.snapshots.length ?? 1) - 1)} / {(run?.snapshots.length ?? 1) - 1}
          {' · '}Δt {CANONICAL_DT.toFixed(6)}s (hop clock)
        </span>
      </div>
    </div>
  );
}
