import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { GeometryProviderPanel } from './DebugCockpit';

describe('GeometryProviderPanel', () => {
  it('shows provider provenance, resolution, and failure classifications', () => {
    render(
      <GeometryProviderPanel
        geometry={{
          providerVersion: 'geometry-provider/3',
          providerName: 'raster-bridge',
          tileId: 'tile-17',
          requestedScale: 0.0001,
          resolvedScale: 0.0002,
          estimatedError: null,
          isBridge: true,
          validity: 'unresolved',
          singularity: 'cut_locus',
          d: -0.002,
          gradDNorm: null,
          hessianNorm: null,
          hessianEigenvalues: null,
        }}
        valid={false}
        lastError="geometry not regular: singular"
      />
    );

    expect(screen.getByText('GEOMETRY PROVIDER')).toBeTruthy();
    expect(screen.getByText('geometry-provider/3')).toBeTruthy();
    expect(screen.getByText('raster-bridge')).toBeTruthy();
    expect(screen.getByText('tile-17')).toBeTruthy();
    expect(screen.getByText('0.00010000')).toBeTruthy();
    expect(screen.getByText('0.00020000')).toBeTruthy();
    expect(screen.getAllByText('unavailable').length).toBeGreaterThanOrEqual(4);
    expect(screen.getByText('INVALID')).toBeTruthy();
    expect(screen.getByText('geometry not regular: singular')).toBeTruthy();
    expect(screen.getByText('unresolved')).toBeTruthy();
    expect(screen.getByText('cut_locus')).toBeTruthy();
  });
});
