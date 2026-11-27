import { dot, sensitivitySpectrum } from './julia_sensitivity_spectrum.mjs';

export const BOXES = 8;
export const RESOLUTIONS = [80, 160, 320];

export function floatTarget(gl, size) {
  const colorBuffer = gl.getExtension('WEBGL_color_buffer_float');
  if (!gl.getExtension('OES_texture_float') || !colorBuffer) {
    throw new Error('The observer comparison requires a renderable float color buffer');
  }
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.FLOAT, null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    throw new Error('Float color framebuffer is incomplete');
  }
  const componentType = gl.getFramebufferAttachmentParameter(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
    colorBuffer.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT);
  if (componentType !== gl.FLOAT) throw new Error(`Unexpected float attachment component type ${componentType}`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return framebuffer;
}

export function boxAverage(pixels, size, codeScale) {
  // A fixed equal-area partition in the zero-rotation viewport, in WebGL row order.
  if (size % BOXES !== 0) throw new Error('Expected whole pixels in every box');
  const side = size / BOXES;
  const means = new Float64Array(3 * BOXES ** 2);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sample = 4 * (y * size + x);
      const box = 3 * (Math.floor(y / side) * BOXES + Math.floor(x / side));
      for (let k = 0; k < 3; k++) means[box + k] += pixels[sample + k];
    }
  }
  return means.map(sum => sum * codeScale / side ** 2);
}

export function differenceColumns(frames, h, components, channels, codeScale) {
  const scale = codeScale / (2 * h * Math.sqrt(components));
  return [0, 2].map(k => {
    const [minus, plus] = [frames[k], frames[k + 1]];
    const column = new Float64Array(components);
    for (let i = 0, j = 0; i < plus.length; i += channels) {
      for (let ch = 0; ch < 3; ch++) column[j++] = (plus[i + ch] - minus[i + ch]) * scale;
    }
    return column;
  });
}

export function operatorDifference(a, b) {
  const error = a.map((column, j) => column.map((value, k) => value - b[j][k]));
  return Math.sqrt(sensitivitySpectrum(error).eigenvalues[1]);
}

export function observe(columns, budget, previous) {
  const spectrum = sensitivitySpectrum(columns);
  const [small, large] = spectrum.eigenvalues;
  const rank = spectrum.eigenvalues.filter(value => value > budget ** 2).length;
  return {
    ...spectrum,
    rank_above_budget: rank,
    rgb8_operator_budget: budget || null,
    jacobian_condition: small > 0 ? Math.sqrt(large / small) : null,
    step_refinement: previous ? compareSpectra(spectrum, previous, columns, previous.columns,
      rank, previous.rank_above_budget) : undefined,
    columns,
  };
}

export function compareSpectra(current, previous, columns, priorColumns, rank, priorRank) {
  const relative = (a, b) => Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) || 1);
  const eigenvalues = current.eigenvalues.map((value, j) => relative(value, previous.eigenvalues[j]));
  const axis = Math.abs(dot(current.eigenvectors_intrinsic[1], previous.eigenvectors_intrinsic[1]));
  const majorAxisAngle = Math.acos(Math.min(1, axis)) * 180 / Math.PI;
  const columnShift = columns.map((column, j) => {
    const prior = priorColumns[j];
    const squaredError = column.reduce((sum, value, k) => sum + (value - prior[k]) ** 2, 0);
    return Math.sqrt(squaredError / (Math.max(dot(column, column), dot(prior, prior)) || 1));
  });
  const split = ({ eigenvalues: [small, large] }) => large > 0 && (large - small) / large >= 0.1;
  const separated = split(current) && split(previous);
  return {
    eigenvalue_relative_shift: eigenvalues,
    column_relative_shift: columnShift,
    operator_difference: operatorDifference(columns, priorColumns),
    major_axis_angle_deg: majorAxisAngle,
    axis_separated_in_both: separated,
    stable: rank === 2 && priorRank === 2 && Math.max(...eigenvalues, ...columnShift) <= 0.15
      && (!separated || majorAxisAngle <= 10),
  };
}

export function compareBoxResolutions(panels) {
  for (let i = 1; i < panels.length; i++) {
    const coarse = panels[i - 1];
    const fine = panels[i];
    for (const [s, state] of fine.states.entries()) {
      for (const [m, metric] of state.metrics.entries()) {
        for (const [h, step] of metric.steps.entries()) {
          const priorStep = coarse.states[s].metrics[m].steps[h];
          if (step.status !== 'measured' || priorStep.status !== 'measured') continue;
          for (const kind of ['box_rgb8', 'box_float']) {
            const current = step.observations[kind];
            const previous = priorStep.observations[kind];
            current.resolution_refinement = {
              previous_resolution: coarse.resolution,
              ...compareSpectra(current, previous, current.columns, previous.columns,
                current.rank_above_budget, previous.rank_above_budget),
            };
          }
        }
      }
    }
  }
  for (const panel of panels) {
    for (const state of panel.states) {
      for (const metric of state.metrics) {
        for (const step of metric.steps) {
          if (step.status !== 'measured') continue;
          for (const observation of Object.values(step.observations)) delete observation.columns;
        }
      }
    }
  }
}

export function findJointWindows(panels) {
  const [coarse, fine] = panels.slice(-2);
  const windows = [];
  for (const [s, state] of fine.states.entries()) {
    for (const [m, metric] of state.metrics.entries()) {
      const coarseSteps = coarse.states[s].metrics[m].steps;
      for (let i = 2; i < metric.steps.length; i++) {
        const sample = metric.steps.slice(i - 2, i + 1);
        const reference = coarseSteps.slice(i - 2, i + 1);
        for (const kind of ['box_rgb8', 'box_float']) {
          const matchesResolution = sample.every(step => step.observations?.[kind].resolution_refinement?.stable);
          const matchesSteps = [sample, reference].every(row =>
            row.slice(1).every(step => step.observations?.[kind].step_refinement?.stable));
          if (matchesResolution && matchesSteps) {
            windows.push({ center: state.name, metric: metric.name, representation: kind,
              resolutions: [coarse.resolution, fine.resolution], spacings: sample.map(step => step.h) });
          }
        }
      }
    }
  }
  return windows;
}
