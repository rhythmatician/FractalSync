import assert from 'node:assert/strict';
import { sensitivitySpectrum } from './julia_sensitivity_spectrum.mjs';

const floor = 0.9;
const ruler = ([x]) => Math.hypot(x, floor);

function sampledTemplate(p, skew) {
  const rho = ruler(p);
  const dx = 0.125;
  const normalize = dx / (rho * Math.sqrt(Math.PI * (1 + skew * skew / 2)));
  const image = new Float64Array(96 ** 2);
  for (let row = 0; row < 96; row++) {
    const y = ((row - 47.5) * dx - p[1]) / rho;
    for (let col = 0; col < 96; col++) {
      const x = ((col - 47.5) * dx - p[0]) / rho;
      image[row * 96 + col] = normalize * (1 + skew * x) * Math.exp(-(x * x + y * y) / 2);
    }
  }
  return image;
}

function measuredSpectrum(evaluate, p, basis, h) {
  return sensitivitySpectrum(basis.map(e => {
    const lower = evaluate(p.map((x, i) => x - h * e[i]));
    const upper = evaluate(p.map((x, i) => x + h * e[i]));
    return upper.map((x, i) => (x - lower[i]) / (2 * h));
  }));
}

function rotatedBasis([u, v]) {
  const angle = 0.37;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [u.map((x, i) => cos * x + sin * v[i]), v.map((x, i) => cos * x - sin * u[i])];
}

export function calibrateJuliaSensitivity(basisFor) {
  const p = [0.45, -0.2];
  const rho = ruler(p);
  const g = p[0] / rho;
  const diagonal = (x, y) => [[x, 0], [0, y]];
  const profiles = { I: diagonal(1, 1),
    G0: diagonal(rho ** -2, rho ** -2),
    G: diagonal((1 + 2 * g * g) / rho ** 2, rho ** -2) };
  const templates = [0, 0.7].map(skew => {
    const q = skew * skew;
    const norm = 1 + q / 2;
    const xx = (0.5 + 0.75 * q + skew * g + (1 + q) * g * g) / (norm * rho ** 2);
    const yy = (0.5 + 0.25 * q) / (norm * rho ** 2);
    const profilesMeasured = Object.entries(profiles).map(([profile, metric]) => {
      const expected = [xx / metric[0][0], yy / metric[1][1]].sort((a, b) => a - b);
      const basis = basisFor(metric).columns;
      const rotated = rotatedBasis(basis);
      const sample = z => sampledTemplate(z, skew);
      const steps = [0.02, 0.01, 0.005].map(h => {
        const { eigenvalues } = measuredSpectrum(sample, p, basis, h);
        const rotatedSpectrum = measuredSpectrum(sample, p, rotated, h);
        return { h, eigenvalues, rotated_eigenvalues: rotatedSpectrum.eigenvalues,
          rotated_gram: rotatedSpectrum.gram,
          max_relative_error: Math.max(...[eigenvalues, rotatedSpectrum.eigenvalues]
            .flatMap(values => values.map((value, i) => Math.abs(value / expected[i] - 1)))) };
      });
      assert.ok(steps[2].max_relative_error < 1e-4);
      assert.ok(steps[2].max_relative_error < 0.3 * steps[1].max_relative_error);
      assert.ok(steps[1].max_relative_error < 0.3 * steps[0].max_relative_error);
      return { profile, expected, steps };
    });
    return { skew, position: p, rho, translation_scale_gram_x: skew / (2 * norm),
      profiles: profilesMeasured };
  });

  const centers = [6, 8, 10, 12, 14].map(n => 2 ** -n);
  const fold = centers.map(t => {
    const basis = basisFor(diagonal(t ** -2, t ** -2)).columns;
    const { eigenvalues } = measuredSpectrum(([x, y]) => [Math.sqrt(x), y], [t, 0], basis, 0.001);
    assert.ok(Math.abs(eigenvalues[0] / (t * t) - 1) < 1e-6);
    assert.ok(Math.abs(eigenvalues[1] / (t / 4) - 1) < 1e-6);
    return { distance: t, eigenvalues };
  });
  const last = fold[fold.length - 1];
  const exponents = last.eigenvalues.map((value, i) => Math.log(value / fold[0].eigenvalues[i])
    / Math.log(last.distance / fold[0].distance));
  const { eigenvalues: rankOne } = measuredSpectrum(([x]) => [x, 0], p,
    rotatedBasis(basisFor(profiles.I).columns), 0.005);
  assert.ok(Math.abs(rankOne[0]) < 1e-12 && Math.abs(rankOne[1] - 1) < 1e-12);
  return { templates, fold, fold_exponents: exponents, rank_one_eigenvalues: rankOne };
}
