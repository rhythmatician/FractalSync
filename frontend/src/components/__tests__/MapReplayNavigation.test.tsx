// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { MapReplayNavigation } from '../MapReplayNavigation';

const check = vi.hoisted(() => vi.fn());
vi.mock('../../lib/orbitSynthesizer', () => ({
  getWasmModule: () => ({ mapReplayNavigation: check }),
}));

const frame = (x: number, valid = true, vx = 0) => ({
  physics: { c: [x, 0] as [number, number], velocity: [vx, 0] as [number, number] },
  diagnostics: { valid },
});
const routes = [
  { spec: { name: 'slow' }, snapshots: [frame(0), frame(0.1), frame(0.15, false)] },
  { spec: { name: 'different launch' }, snapshots: [frame(0, true, 1), frame(0.26)] },
  { spec: { name: 'fast' }, snapshots: [frame(0), frame(0.2), frame(0.26)] },
];

afterEach(() => {
  cleanup();
  check.mockReset();
});

it('uses recorded frame prefixes and opens the arrival witnessed by Rust', () => {
  check.mockReturnValue({
    rankingFrame: 1,
    startLowerBound: 2.75,
    startCrossingCertified: false,
    candidateIndices: [0, 2],
    certificateQueries: 4,
    baseline: { firstArrivalRoute: 1, queriesToFirstArrival: 5, targetQueries: 5, arrivals: [null, 2] },
    bounded: { firstArrivalRoute: 1, queriesToFirstArrival: 3, targetQueries: 5, arrivals: [null, 2] },
    rasterWaypointSearch: { status: 'witness', routeIndex: 2, frameIndex: 2, lengthUpper: 12.5, rasterGeneration: 3 },
  });
  const inspect = vi.fn();
  render(<MapReplayNavigation routes={routes} selected={0} frameIndex={1} inspectArrival={inspect} />);
  fireEvent.click(screen.getByRole('button', { name: 'Compare recorded routes' }));
  expect(check).toHaveBeenCalledWith(
    0, 0.26, 0, 0.005, 1,
    Float64Array.from([0, 0, 0.1, 0, 0.15, 0, 0, 0, 0.26, 0, 0, 0, 0.2, 0, 0.26, 0]),
    Uint8Array.from([1, 1, 0, 1, 1, 1, 1, 1]),
    Uint32Array.from([3, 2, 3]),
    Float64Array.from([0, 0, 1, 0, 0, 0]),
  );
  expect(screen.getByText(/5 target checks in recording order and 3 after ranking/)).toBeTruthy();
  expect(screen.getByText(/Current raster waypoint path to fast frame 3/)).toBeTruthy();
  expect(screen.getByText(/Raster generation 3/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Inspect arrival in fast' }));
  expect(inspect).toHaveBeenCalledWith(2, 2);
});

it('shows waypoint coverage with no ideal bound and no arrival', () => {
  check.mockReturnValue({
    rankingFrame: 0,
    startLowerBound: undefined,
    startCrossingCertified: false,
    candidateIndices: [0, 2],
    certificateQueries: 2,
    baseline: { firstArrivalRoute: undefined, queriesToFirstArrival: undefined, targetQueries: 5, arrivals: [undefined, undefined] },
    bounded: { firstArrivalRoute: undefined, queriesToFirstArrival: undefined, targetQueries: 5, arrivals: [undefined, undefined] },
    rasterWaypointSearch: { status: 'noWitness' },
  });
  render(<MapReplayNavigation routes={routes} selected={0} frameIndex={0} inspectArrival={vi.fn()} />);
  fireEvent.click(screen.getByRole('button', { name: 'Compare recorded routes' }));
  expect(screen.getByText('Ideal length lower bound unavailable')).toBeTruthy();
  expect(screen.getByText('No recorded arrival')).toBeTruthy();
  expect(screen.getByText('No waypoint in the route graph certified inside the target.')).toBeTruthy();
  expect(screen.queryByRole('button', { name: /Inspect arrival/ })).toBeNull();
});
