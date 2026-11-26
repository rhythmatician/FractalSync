import { useId, useRef, useState, type ChangeEvent } from 'react';

type Pair = [number, number];
type Matrix = [Pair, Pair];
type Refusal = 'geometry-invalid' | 'crosses-sampled-shore' | 'gpu-seed-collision' | 'gpu-seed-imprecise';

type Step = { h: number; saved: Record<string, unknown> } & (
  | { status: Refusal }
  | { status: 'measured'; gram: Matrix; spectrum: Pair; axes: Matrix; rank: 0 | 1 | 2 }
);

type Bookmark = {
  provenance: Record<string, unknown>;
  renderer: Record<string, unknown>;
  repeatable: boolean;
  representation: string;
  states: {
    name: string;
    seed: Pair;
    geometry: Record<string, unknown>;
    validity: string;
    frameHash: string | null;
    metrics: { name: 'I' | 'G0' | 'G'; matrix: Matrix; basis: Record<string, unknown>; steps: Step[] }[];
  }[];
};

function requireValue(condition: unknown, field: string): asserts condition {
  if (!condition) throw new Error(`Invalid report field: ${field}`);
}

function object(value: unknown, field: string): Record<string, unknown> {
  requireValue(typeof value === 'object' && value !== null && !Array.isArray(value), field);
  return value as Record<string, unknown>;
}

function string(value: unknown, field: string): string {
  requireValue(typeof value === 'string' && value.length > 0, field);
  return value;
}

function number(value: unknown, field: string): number {
  requireValue(typeof value === 'number' && Number.isFinite(value), field);
  return value;
}

function pair(value: unknown, field: string): Pair {
  requireValue(Array.isArray(value) && value.length === 2, field);
  return [number(value[0], field), number(value[1], field)];
}

function matrix(value: unknown, field: string): Matrix {
  requireValue(Array.isArray(value) && value.length === 2, field);
  return [pair(value[0], field), pair(value[1], field)];
}

function parseStep(value: unknown): Step {
  const saved = object(value, 'step');
  const h = number(saved.h, 'step.h');
  requireValue(h > 0, 'step.h');
  const status = saved.status;
  if (status === 'measured') {
    const rank = saved.rank_above_rgb8_budget;
    requireValue(rank === 0 || rank === 1 || rank === 2, 'step.rank_above_rgb8_budget');
    return {
      h, saved, status, rank,
      gram: matrix(saved.gram, 'step.gram'),
      spectrum: pair(saved.eigenvalues, 'step.eigenvalues'),
      axes: matrix(saved.eigenvectors_intrinsic, 'step.eigenvectors_intrinsic'),
    };
  }
  requireValue(
    status === 'geometry-invalid' || status === 'crosses-sampled-shore' ||
    status === 'gpu-seed-collision' || status === 'gpu-seed-imprecise', 'step.status',
  );
  return { h, saved, status };
}

function parseBookmark(contents: string): Bookmark {
  const report = object(JSON.parse(contents), 'report');
  const provenance = object(report.provenance, 'provenance');
  const renderer = object(report.renderer, 'renderer');
  requireValue(typeof report.repeatable_baseline === 'boolean', 'repeatable_baseline');
  requireValue(Array.isArray(report.states) && report.states.length > 0, 'states');
  const states = report.states.map((value: unknown): Bookmark['states'][number] => {
    const state = object(value, 'state');
    const geometry = object(state.geometry, 'state.geometry');
    const jet = geometry.jet === undefined ? null : object(geometry.jet, 'state.geometry.jet');
    const validity = jet ? string(jet.validity, 'state.geometry.jet.validity') :
      string(geometry.error, 'state.geometry.error');
    requireValue(Array.isArray(state.metrics), 'state.metrics');
    return {
      name: string(state.name, 'state.name'),
      seed: pair(state.c, 'state.c'),
      geometry,
      validity,
      frameHash: state.baseline_rgba_sha256 === undefined ? null :
        string(state.baseline_rgba_sha256, 'state.baseline_rgba_sha256'),
      metrics: state.metrics.map((value: unknown) => {
        const item = object(value, 'state.metric');
        const name = item.name;
        requireValue(name === 'I' || name === 'G0' || name === 'G', 'state.metric.name');
        requireValue(Array.isArray(item.steps) && item.steps.length > 0, 'state.metric.steps');
        const basis = object(item.basis, 'state.metric.basis');
        matrix(basis.columns, 'state.metric.basis.columns');
        return {
          name,
          matrix: matrix(item.matrix, 'state.metric.matrix'),
          basis,
          steps: item.steps.map(parseStep),
        };
      }),
    };
  });
  return {
    provenance, renderer, states,
    repeatable: report.repeatable_baseline,
    representation: string(provenance.representation, 'provenance.representation'),
  };
}

function savedJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return 'Saved details could not be displayed.';
  }
}

export function JuliaSensitivityBookmark() {
  const [loaded, setLoaded] = useState<{ name: string; report: Bookmark } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stateIndex, setStateIndex] = useState(0);
  const [metricIndex, setMetricIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);
  const selection = useRef(0);
  const inputId = useId();

  async function openReport(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = '';
    if (!file) return;
    const current = ++selection.current;
    setLoaded(null);
    setError(null);
    setStateIndex(0);
    setMetricIndex(0);
    setStepIndex(0);
    try {
      const report = parseBookmark(await file.text());
      if (current === selection.current) setLoaded({ name: file.name, report });
    } catch (cause) {
      if (current === selection.current) setError(cause instanceof Error ? cause.message : 'Could not open report.');
    }
  }

  const report = loaded?.report;
  const state = report?.states[stateIndex];
  const metric = state?.metrics[metricIndex];
  const step = metric?.steps[stepIndex];
  return (
    <details style={{ margin: '0 0 12px', fontSize: 12, overflowWrap: 'anywhere' }}>
      <summary>DIAG · Frozen Julia bookmark</summary>
      <label style={{ display: 'block', margin: '6px 0' }}>
        Open frozen Julia report
        <input type="file" accept=".json,application/json" onChange={openReport} style={{ maxWidth: '100%' }} />
      </label>
      {error && <p role="alert">{error}</p>}
      {loaded && report && state && <>
        <p>{loaded.name} · {report.representation}</p>
        <label htmlFor={`${inputId}-state`}>Frozen state</label>{' '}
          <select id={`${inputId}-state`} value={stateIndex} onChange={(event) => {
            setStateIndex(Number(event.target.value));
            setMetricIndex(0);
            setStepIndex(0);
          }}>
            {report.states.map((item, index) => <option key={index} value={index}>{item.name}</option>)}
          </select>
        <dl>
          <dt>Julia seed</dt><dd>{JSON.stringify(state.seed)}</dd>
          <dt>Center geometry</dt><dd>{state.validity}</dd>
          <dt>Repeated baseline</dt><dd>{report.repeatable ? 'identical' : 'different'}</dd>
        </dl>
        {metric && step ? <>
          <label htmlFor={`${inputId}-metric`}>Metric</label>{' '}
            <select id={`${inputId}-metric`} value={metricIndex} onChange={(event) => {
              setMetricIndex(Number(event.target.value));
              setStepIndex(0);
            }}>
              {state.metrics.map((item, index) => <option key={index} value={index}>{item.name}</option>)}
            </select>{' '}
          <label htmlFor={`${inputId}-step`}>Step size</label>{' '}
            <select id={`${inputId}-step`} value={stepIndex} onChange={(event) => setStepIndex(Number(event.target.value))}>
              {metric.steps.map((item, index) => <option key={index} value={index}>{item.h}</option>)}
            </select>
          <dl>
            <dt>Probe status</dt><dd>{step.status}</dd>
            {step.status === 'measured' && <>
              <dt>Raw sensitivity matrix</dt><dd>{JSON.stringify(step.gram)}</dd>
              <dt>Eigenvalues</dt><dd>{JSON.stringify(step.spectrum)}</dd>
              <dt>Intrinsic eigenvectors</dt><dd>{JSON.stringify(step.axes)}</dd>
              <dt>Rank above RGB8 budget</dt><dd>{step.rank}</dd>
            </>}
          </dl>
        </> : <p>No metric probes were recorded at this state.</p>}
        <details>
          <summary>Saved replay inputs and diagnostics</summary>
          <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{savedJson({
            provenance: report.provenance,
            renderer: report.renderer,
            baseline_rgba_sha256: state.frameHash,
            geometry: state.geometry,
            metric: metric && { name: metric.name, matrix: metric.matrix, basis: metric.basis },
            step: step?.saved,
          })}</pre>
        </details>
      </>}
    </details>
  );
}
