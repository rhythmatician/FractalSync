// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { JuliaSensitivityBookmark } from '../JuliaSensitivityBookmark';

const report = {
  provenance: { representation: 'RGB8 framebuffer/1', resolution: [80, 80] },
  renderer: { renderer: 'WebGL', uniforms: { u_maxIter: 500 } },
  repeatable_baseline: true,
  states: [{
    name: 'period-two', c: [-0.98, 0.09],
    geometry: { jet: { validity: 'regular', singularity: 'none', D: -0.1 }, rho: 0.1, sigma: 0 },
    metrics: [{
      name: 'G', matrix: [[1, 0], [0, 1]], basis: { columns: [[1, 0], [0, 1]] },
      steps: [
        { h: 0.02, status: 'measured', gram: [[3, 0], [0, 5]], eigenvalues: [3, 5],
          eigenvectors_intrinsic: [[1, 0], [0, 1]], rank_above_rgb8_budget: 2 },
        { h: 0.01, status: 'geometry-invalid' },
      ],
    }],
  }],
};

function chooseFile(value: unknown) {
  const data = JSON.stringify(value);
  const file = Object.assign(new File([data], 'results.json'), { text: () => Promise.resolve(data) });
  fireEvent.change(screen.getByLabelText('Open frozen Julia report'), { target: { files: [file] } });
}

afterEach(cleanup);

describe('frozen Julia bookmark', () => {
  it('shows the saved measured spectrum and distinguishes refused step sizes', async () => {
    render(<JuliaSensitivityBookmark />);
    chooseFile(report);
    expect(await screen.findByText('[[3,0],[0,5]]')).toBeTruthy();
    expect(screen.getByText('[3,5]')).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Step size'), { target: { value: '1' } });
    expect(screen.getByText('geometry-invalid')).toBeTruthy();
    expect(screen.queryByText('[3,5]')).toBeNull();
  });

  it('rejects a measured step with an invalid spectrum', async () => {
    render(<JuliaSensitivityBookmark />);
    const bad = structuredClone(report);
    bad.states[0].metrics[0].steps[0].eigenvalues = [3];
    chooseFile(bad);
    expect(await screen.findByRole('alert')).toBeTruthy();
    expect(screen.queryByLabelText('Step size')).toBeNull();
  });
});
