import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { EnergyLedgerPanel } from '../DebugCockpit';
import type { DebugSnapshot } from '../../lib/debugCockpit';

describe('energy ledger panel', () => {
  it('shows signed step and run work, residuals and invalidity', () => {
    const frame = {
      energyLedger: { valid: false, controlWork: 0.25, frictionWork: -0.5, impulseWork: 0.75, residual: -0.125,
        endGeometry: { provider_version: 'provider/1', validity: 'unresolved' } },
      accumulatedEnergyLedger: { valid: false, steps: 3, controlWork: 1, frictionWork: -2, impulseWork: 3, residual: -4 },
    } as DebugSnapshot;
    const html = renderToStaticMarkup(<EnergyLedgerPanel frame={frame} />);
    for (const text of ['drive work', '-0.50000', 'impulse work', '-0.12500', 'INVALID', 'provider/1', 'unresolved', 'run residual', '-4.00000']) {
      expect(html).toContain(text);
    }
  });
  it('marks an absent step ledger unavailable', () => {
    expect(renderToStaticMarkup(<EnergyLedgerPanel frame={{} as DebugSnapshot} />)).toContain('unavailable');
  });
});
