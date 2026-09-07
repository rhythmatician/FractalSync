import { describe, it } from 'vitest';
import { rhoFromEmbeddedHeight } from './hyperbolicDiagnosticFixture';

describe('trace compareSpread failure in transformMeshToHyperbolic', () => {
  it('checks zEmbed in patch', () => {
    // In patchValley:
    // positions: [cx - 0.01, cy, 0.0, ...]
    // What does rhoFromEmbeddedHeight(0.0) return?
    console.log('rhoFromEmbeddedHeight(0.0):', rhoFromEmbeddedHeight(0.0));
    console.log('rhoFromEmbeddedHeight(-1.0):', rhoFromEmbeddedHeight(-1.0));
  });
});
