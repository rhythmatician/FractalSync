import { useState, type FormEvent } from 'react';
import { getWasmModule } from '../lib/orbitSynthesizer';

interface ReplayRoute {
  spec: { name: string };
  snapshots: readonly {
    physics: { c: readonly [number, number]; velocity: readonly [number, number] };
    diagnostics: { valid: boolean };
  }[];
}

interface ReplayScan {
  firstArrivalRoute?: number | null;
  queriesToFirstArrival?: number | null;
  targetQueries: number;
  arrivals: (number | null | undefined)[];
}

interface NavigationReport {
  rankingFrame: number;
  startLowerBound?: number | null;
  startCrossingCertified: boolean;
  candidateIndices: number[];
  certificateQueries: number;
  baseline: ReplayScan;
  bounded: ReplayScan;
  rasterWaypointSearch?:
    | { status: 'unavailable' | 'noWitness' }
    | {
        status: 'witness';
        routeIndex: number;
        frameIndex: number;
        lengthUpper: number;
        rasterGeneration?: number;
        window?: [[number, number], [number, number]];
      };
}

interface Props {
  routes: readonly ReplayRoute[];
  selected: number;
  frameIndex: number;
  inspectArrival: (runIndex: number, frameIndex: number) => void;
}

const inputStyle = { width: '100%', boxSizing: 'border-box' } as const;

export function MapReplayNavigation({ routes, selected, frameIndex, inspectArrival }: Props) {
  const [targetX, setTargetX] = useState('0.26');
  const [targetY, setTargetY] = useState('0');
  const [radius, setRadius] = useState('0.005');
  const [report, setReport] = useState<NavigationReport>();
  const [error, setError] = useState('');

  function compare(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setReport(undefined);
    setError('');
    try {
      const sourceRoute = routes[selected];
      if (!sourceRoute?.snapshots.length) throw new Error('Select a replay with recorded frames.');

      const fields = [targetX, targetY, radius];
      const [x, y, arrivalRadius] = fields.map(Number);
      if (fields.some(value => !value.trim()) || ![x, y, arrivalRadius].every(Number.isFinite) || arrivalRadius < 0) {
        throw new Error('Enter finite target coordinates and a nonnegative arrival radius.');
      }

      const positions: number[] = [];
      const validSteps: number[] = [];
      const routeLengths: number[] = [];
      const firstVelocities: number[] = [];
      for (const route of routes) {
        routeLengths.push(route.snapshots.length);
        firstVelocities.push(...(route.snapshots[0]?.physics.velocity ?? [0, 0]));
        for (const frame of route.snapshots) {
          positions.push(...frame.physics.c);
          validSteps.push(Number(frame.diagnostics.valid));
        }
      }
      const wasm = getWasmModule();
      if (!wasm.mapReplayNavigation) throw new Error('The Map navigation comparison requires the current WASM build.');
      setReport(wasm.mapReplayNavigation(
        selected, x, y, arrivalRadius, Math.min(frameIndex, sourceRoute.snapshots.length - 1),
        Float64Array.from(positions), Uint8Array.from(validSteps), Uint32Array.from(routeLengths),
        Float64Array.from(firstVelocities),
      ) as NavigationReport);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }

  const arrival = report?.bounded.firstArrivalRoute;
  const arrivalIndex = arrival == null ? undefined : report?.candidateIndices[arrival];
  const arrivalFrame = arrival == null ? undefined : report?.bounded.arrivals[arrival];

  return (
    <details style={{ fontSize: 11, margin: '0 0 12px' }}>
      <summary>Map target comparison</summary>
      <p>Compare replays with the same first recorded position and velocity. The current frame sets the ranking cutoff.</p>
      <form onSubmit={compare}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 6 }}>
          <label>Target real<input style={inputStyle} value={targetX} onChange={e => { setTargetX(e.target.value); setReport(undefined); }} /></label>
          <label>Target imaginary<input style={inputStyle} value={targetY} onChange={e => { setTargetY(e.target.value); setReport(undefined); }} /></label>
          <label>Arrival radius<input style={inputStyle} value={radius} onChange={e => { setRadius(e.target.value); setReport(undefined); }} /></label>
        </div>
        <button type="submit">Compare recorded routes</button>
      </form>
      {error && <p role="alert">{error}</p>}
      {report && (
        <div aria-live="polite">
          <p>{report.startLowerBound == null
            ? 'Ideal length lower bound unavailable'
            : `Ideal length lower bound from launch ${report.startLowerBound.toPrecision(4)}`}
          </p>
          {report.startCrossingCertified && <p>Ideal Shore crossing certified</p>}
          <p>Ranked {report.candidateIndices.length} replays at frame {report.rankingFrame} using {report.certificateQueries} certificate queries.</p>
          {report.bounded.queriesToFirstArrival != null ? (
            <p>The first arrival took {report.baseline.queriesToFirstArrival} target checks in recording order and {report.bounded.queriesToFirstArrival} after ranking.</p>
          ) : <p>No recorded arrival</p>}
          <p>Complete scans checked {report.baseline.targetQueries} frames in recording order and {report.bounded.targetQueries} after ranking.</p>
          {report.rasterWaypointSearch?.status === 'witness' && (
            <p>
              Current raster waypoint path to {routes[report.rasterWaypointSearch.routeIndex].spec.name} frame {report.rasterWaypointSearch.frameIndex + 1}.
              {' '}Length upper bound {report.rasterWaypointSearch.lengthUpper.toPrecision(4)}.
              {report.rasterWaypointSearch.rasterGeneration !== undefined && (
                <> Raster generation {report.rasterWaypointSearch.rasterGeneration}.</>
              )}
            </p>
          )}
          {report.rasterWaypointSearch?.status === 'unavailable' && <p>Current raster waypoint certificate unavailable.</p>}
          {report.rasterWaypointSearch?.status === 'noWitness' && <p>No waypoint in the route graph certified inside the target.</p>}
          {arrivalIndex != null && arrivalFrame != null && (
            <button type="button" onClick={() => inspectArrival(arrivalIndex, arrivalFrame)}>
              Inspect arrival in {routes[arrivalIndex].spec.name}
            </button>
          )}
        </div>
      )}
    </details>
  );
}
