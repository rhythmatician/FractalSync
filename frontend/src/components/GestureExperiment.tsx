import { useEffect, useRef, useState } from 'react';
import { JuliaRenderer } from '../lib/juliaRenderer';
import { getWasmModule, initOrbitSynth } from '../lib/orbitSynthesizer';

type Schedule = { a_hz: number; b_segments: { start_seconds: number; frequency_hz: number }[] };
type Source = 'a' | 'b' | 'off';
type Settings = {
  sourceA: Source; sourceB: Source; gainA: number; gainB: number;
  excursion: number; rotating: boolean; offset: number; x: number; y: number;
  zoom: number; hue: number;
  condition: 'matched' | 'misaligned' | 'audio';
};
type Passage = {
  audio: AudioContext;
  frame: number;
  release: () => void;
  changes: { audioTime: number; settings: Settings }[];
};
type RenderedFrame = {
  elapsed: number;
  canvasSize: [number, number];
  params: Parameters<JuliaRenderer['renderFrozen']>[0];
};

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const initial: Settings = {
  sourceA: 'a', sourceB: 'b', gainA: 1, gainB: 0, excursion: 0.12,
  rotating: true, offset: 0, x: -0.7269, y: 0.1889, zoom: 1, hue: 0.5, condition: 'matched',
};

/** A reproducible known-phase trial with direct and Physics presentations. */
export function GestureExperiment() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const settingsRef = useRef(initial);
  const rendererRef = useRef<JuliaRenderer | null>(null);
  const runRef = useRef<Passage | null>(null);
  const startingRef = useRef(false);
  const recordRef = useRef<{ startedAt: string; events: unknown[]; samples: unknown[]; frames: RenderedFrame[] } | null>(null);
  const visibleFrame = useRef<RenderedFrame | null>(null);
  const replayFrame = useRef(0);
  const [replaying, setReplaying] = useState(false);
  const [hasRecord, setHasRecord] = useState(false);
  const mountedRef = useRef(false);
  const [settings, setSettings] = useState(initial);
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const [time, setTime] = useState(0);
  const [tracking, setTracking] = useState(false);

  function update(next: Partial<Settings>) {
    const value = { ...settingsRef.current, ...next };
    settingsRef.current = value;
    const run = runRef.current;
    if (run) {
      const change = { audioTime: run.audio.currentTime, settings: value };
      run.changes.push(change);
      recordRef.current?.events.push({ type: 'settings', ...change });
    }
    setSettings(value);
  }

  useEffect(() => {
    mountedRef.current = true;
    let disposed = false;
    let renderer: JuliaRenderer | null = null;
    const resize = () => renderer?.resize();
    (async () => {
      try {
        await initOrbitSynth();
        if (disposed || !canvas.current) return;
        renderer = new JuliaRenderer(canvas.current);
        await renderer.init();
        if (disposed) { renderer.dispose(); return; }
        rendererRef.current = renderer;
        renderer.renderFrozen({ juliaSeed: { real: initial.x, imag: initial.y },
          colorHue: initial.hue, colorSat: 0.8, colorBright: 0.9, zoom: initial.zoom, speed: 0 });
        window.addEventListener('resize', resize);
        setReady(true);
      } catch (cause) { if (!disposed) setError(String(cause)); }
    })();
    return () => {
      disposed = true;
      mountedRef.current = false;
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(replayFrame.current);
      if (runRef.current) {
        cancelAnimationFrame(runRef.current.frame);
        runRef.current.release();
        void runRef.current.audio.close();
        runRef.current = null;
      }
      renderer?.dispose();
      rendererRef.current = null;
    };
  }, []);

  async function stop() {
    const run = runRef.current;
    runRef.current = null;
    if (run) {
      recordRef.current?.events.push({ type: 'stop', audioTime: run.audio.currentTime });
      cancelAnimationFrame(run.frame);
      run.release();
      await run.audio.close();
    }
    setPlaying(false);
  }

  async function start() {
    if (runRef.current || startingRef.current) return;
    startingRef.current = true;
    let audio: AudioContext | null = null;
    try {
      const wasm = getWasmModule();
      if (tracking && typeof wasm.GesturePhysics !== 'function') {
        throw new Error('Rebuild the Wasm bundle to use Physics tracking.');
      }
      const schedule = JSON.parse(wasm.knownGestureScheduleJson()) as Schedule;
      audio = new AudioContext();
      await audio.resume();
      if (!mountedRef.current) { await audio.close(); return; }
      const startAt = audio.currentTime + 0.1;
      visibleFrame.current = null;
      recordRef.current = { startedAt: new Date().toISOString(), events: [{ type: 'start', audioTime: startAt, settings: settingsRef.current, tracking }], samples: [], frames: [] };
      setHasRecord(true);
      // The sine modulators and Rust phase calculation share this audio clock.
      function layer(frequency: number, modulation: number, segments?: Schedule['b_segments']) {
        const carrier = audio!.createOscillator();
        const modulator = audio!.createOscillator();
        const gain = audio!.createGain();
        const depth = audio!.createGain();
        carrier.frequency.value = frequency;
        modulator.frequency.setValueAtTime(modulation, startAt);
        segments?.forEach(segment => modulator.frequency.setValueAtTime(segment.frequency_hz, startAt + segment.start_seconds));
        gain.gain.value = 0.06;
        depth.gain.value = 0.06;
        modulator.connect(depth).connect(gain.gain);
        carrier.connect(gain).connect(audio!.destination);
        carrier.start(startAt);
        modulator.start(startAt);
      }
      layer(220, schedule.a_hz);
      layer(330, schedule.b_segments[0].frequency_hz, schedule.b_segments);
      const origin = [settingsRef.current.x, settingsRef.current.y];
      const physics = tracking ? new wasm.GesturePhysics(origin[0], origin[1]) : null;
      const run: Passage = { audio, frame: 0, release: () => physics?.free(), changes: [] };
      runRef.current = run;
      setError('');
      setPlaying(true);
      setTime(0);
      let lastDisplay = 0;
      let completedSteps = 0;
      let realized = origin;
      let physicsSettings = settingsRef.current;
      const referenceAt = (current: Settings, elapsed: number) => {
        const phases = wasm.knownGesturePhases(elapsed, current.condition === 'misaligned');
        const pick = (source: Source) => source === 'a' ? phases[0] : phases[1];
        const point = wasm.twoSourceGestureTarget(current.x, current.y,
          pick(current.sourceA), pick(current.sourceB) + current.offset,
          current.sourceA === 'off' || current.condition === 'audio' ? 0 : current.gainA,
          current.sourceB === 'off' || current.condition === 'audio' ? 0 : current.gainB, current.excursion, current.rotating);
        if (Math.hypot(point[0], point[1]) >= 2) throw new Error('Target outside the Map. Reduce excursion or move the anchor.');
        return { phases: Array.from(phases), target: Array.from(point) };
      };
      const animate = () => {
        if (runRef.current !== run) return;
        const audioTime = run.audio.currentTime;
        const elapsed = Math.max(0, audioTime - startAt);
        try {
          const current = settingsRef.current;
          let displayed: number[];
          if (physics) {
            const dt = physics.stepSeconds;
            const due = Math.floor(elapsed / dt);
            if (due - completedSteps > 50) throw new Error('Physics is more than one second behind the passage.');
            const deadline = performance.now() + 8;
            while (completedSteps < due) {
              const stepTime = (completedSteps + 1) * dt;
              while (run.changes.length && run.changes[0].audioTime <= startAt + stepTime) {
                physicsSettings = run.changes[0].settings;
                run.changes.shift();
              }
              const reference = referenceAt(physicsSettings, stepTime);
              const sample = JSON.parse(physics.step(reference.target[0], reference.target[1])) as { position: number[] };
              realized = sample.position;
              completedSteps++;
              recordRef.current?.samples.push({ stepTime, audioTime, ...reference, physics: sample, settings: physicsSettings });
              if (performance.now() >= deadline) break;
            }
            displayed = current.condition === 'audio' ? origin : realized;
            if (completedSteps < due) {
              run.frame = requestAnimationFrame(animate);
              return;
            }
          } else {
            run.changes.length = 0;
            const reference = referenceAt(current, elapsed);
            displayed = reference.target;
            recordRef.current?.samples.push({ elapsed, audioTime, ...reference, settings: current });
          }
          const frame: RenderedFrame = {
            elapsed,
            canvasSize: [canvas.current?.width ?? 0, canvas.current?.height ?? 0],
            params: { juliaSeed: { real: displayed[0], imag: displayed[1] },
            colorHue: current.hue, colorSat: 0.8, colorBright: 0.9, zoom: current.zoom, speed: 0 } };
          rendererRef.current?.renderFrozen(frame.params);
          visibleFrame.current = frame;
          recordRef.current?.frames.push(frame);
          if (elapsed - lastDisplay > 0.1) { setTime(elapsed); lastDisplay = elapsed; }
        } catch (cause) { recordRef.current?.events.push({ type: 'error', elapsed, message: String(cause) }); setError(String(cause)); void stop(); return; }
        run.frame = requestAnimationFrame(animate);
      };
      run.frame = requestAnimationFrame(animate);
    } catch (cause) {
      if (audio) await audio.close();
      if (mountedRef.current) setError(String(cause));
    } finally {
      startingRef.current = false;
    }
  }

  function replay() {
    const frames = recordRef.current?.frames;
    if (!frames?.length || runRef.current || startingRef.current) return;
    if (replaying) {
      cancelAnimationFrame(replayFrame.current);
      setReplaying(false);
      return;
    }
    setReplaying(true);
    const start = performance.now();
    let index = 0;
    const advance = (now: number) => {
      const elapsed = Math.max(0, (now - start) / 1000);
      while (index + 1 < frames.length && frames[index + 1].elapsed <= elapsed) index++;
      const frame = frames[index];
      rendererRef.current?.renderFrozen(frame.params);
      visibleFrame.current = frame;
      setTime(frame.elapsed);
      if (index + 1 < frames.length) replayFrame.current = requestAnimationFrame(advance);
      else setReplaying(false);
    };
    replayFrame.current = requestAnimationFrame(advance);
  }

  function captureFrame() {
    const frame = visibleFrame.current;
    if (!frame || !rendererRef.current || !canvas.current) return;
    rendererRef.current.renderFrozen(frame.params);
    canvas.current.toBlob(blob => {
      if (blob) download(blob, `gesture-${frame.elapsed.toFixed(3)}s.png`);
    });
  }

  function exportRecord() {
    if (!recordRef.current) return;
    const blob = new Blob([JSON.stringify({
      format: 'fractal-sync-gesture-diagnostic/1',
      source: 'Known synthetic phases',
      ...recordRef.current,
    }, null, 2)], { type: 'application/json' });
    download(blob, 'gesture-diagnostic.json');
  }

  const slider = (label: string, key: 'gainA' | 'gainB' | 'excursion' | 'offset' | 'zoom' | 'hue', min: number, max: number, step: number) =>
    <label>{label} {settings[key].toFixed(2)}<input aria-label={label} type="range" min={min} max={max} step={step} value={settings[key]} onChange={event => update({ [key]: Number(event.target.value) })} /></label>;
  const selector = (label: string, key: 'sourceA' | 'sourceB') =>
    <label>{label} <select value={settings[key]} onChange={event => update({ [key]: event.target.value as Source })}><option value="a">Pulse A</option><option value="b">Pulse B</option><option value="off">Off</option></select></label>;

  return <main style={{ background: '#171b22', color: '#eee', minHeight: '100vh', fontFamily: 'sans-serif', padding: 16 }}>
    <style>{'.gesture-output:fullscreen { width: 100vw !important; height: 100vh !important; margin: 0 !important; }'}</style>
    <h1>Two-source gesture experiment</h1>
    <p>Known synthetic pulses depart from their 3:4 relationship after 8 seconds and return at 24 seconds.</p>
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
      <button disabled={!ready || replaying} onClick={() => playing ? void stop() : void start()}>{playing ? 'Stop passage' : 'Start passage'}</button>
      <button disabled={!hasRecord} onClick={exportRecord}>Export diagnostic record</button>
      <button disabled={!hasRecord || playing} onClick={replay}>{replaying ? 'Stop replay' : 'Replay Julia motion'}</button>
      <button disabled={!hasRecord} onClick={captureFrame}>Save Julia frame</button>
      <button onClick={() => void canvas.current?.requestFullscreen()}>Show Julia only</button>
      <span>Passage: {time.toFixed(1)} s</span>
      <label><input type="checkbox" checked={tracking} disabled={playing} onChange={event => setTracking(event.target.checked)} /> Render Physics motion</label>
      <label>Comparison <select value={settings.condition} onChange={event => update({ condition: event.target.value as Settings['condition'] })}><option value="matched">Matched visual</option><option value="misaligned">Visual changes early</option><option value="audio">Audio with fixed Julia</option></select></label>
      {selector('Source 1', 'sourceA')}{slider('Gain 1', 'gainA', 0, 1, 0.01)}
      {selector('Source 2', 'sourceB')}{slider('Gain 2', 'gainB', 0, 1, 0.01)}
      {slider('Excursion', 'excursion', 0, 0.5, 0.005)}
      <label>Geometry <select value={settings.rotating ? 'rotating' : 'distinct'} onChange={event => update({ rotating: event.target.value === 'rotating' })}><option value="rotating">Rotating vectors</option><option value="distinct">Distinct directions</option></select></label>
      {slider('Phase offset 2 (radians; changes immediately)', 'offset', -3.14, 3.14, 0.01)}
      <label>Anchor real <input type="number" step="0.001" value={settings.x} onChange={event => update({ x: Number(event.target.value) })} /></label>
      <label>Anchor imaginary <input type="number" step="0.001" value={settings.y} onChange={event => update({ y: Number(event.target.value) })} /></label>
      {slider('Julia zoom', 'zoom', 0.5, 3, 0.01)}{slider('Palette hue', 'hue', 0, 1, 0.01)}
    </div>
    {error && <p role="alert">{error}</p>}
    <canvas className="gesture-output" ref={canvas} aria-label="Julia output" style={{ display: 'block', width: '100%', height: '65vh', marginTop: 16, background: '#000' }} />
  </main>;
}
