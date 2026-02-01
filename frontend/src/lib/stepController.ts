export interface StepContext {
  c_real: number;
  c_imag: number;
  prev_delta_real: number;
  prev_delta_imag: number;
  nu_norm: number;
  membership: boolean;
  grad_re: number;
  grad_im: number;
  sensitivity: number;
  patch: number[];
  mip_level: number;
  feature_vec: number[];
}

export interface StepDebug {
  mip_level: number;
  scale_g: number;
  scale_df: number;
  scale: number;
  delta_f_pred: number;
  wall_applied: boolean;
}

export interface StepResult {
  c_real: number;
  c_imag: number;
  delta_real: number;
  delta_imag: number;
  debug: StepDebug;
  context: StepContext;
}

export interface StepState {
  c_real: number;
  c_imag: number;
  prev_delta_real: number;
  prev_delta_imag: number;
}

export interface StepController {
  context(state: StepState): StepContext;
  step(state: StepState, delta_real: number, delta_imag: number): StepResult;
}

interface StepControllerConstructor {
  new (): StepController;
}

interface WasmModule {
  default: () => Promise<void>;
  StepController: StepControllerConstructor;
}

let modulePromise: Promise<WasmModule> | null = null;

async function loadModule(): Promise<WasmModule> {
  if (!modulePromise) {
    // Construct absolute URL at runtime so Vite's static import analysis does not
    // attempt to resolve the module from the `/public` directory. The file is
    // copied to `/wasm` and must be loaded via a runtime URL in dev and prod.
    // @ts-ignore dynamic import from runtime URL
    const wasmUrl = new URL('/wasm/orbit_synth_wasm.js', window.location.href).toString();
    modulePromise = import(/* @vite-ignore */ (wasmUrl as any)) as Promise<WasmModule>;
  }
  const mod = await modulePromise;
  await mod.default();
  return mod;
}

export async function loadStepController(): Promise<StepController> {
  const mod = await loadModule();
  const wasmController = new mod.StepController();

  // Wrap the raw WASM controller to present the older, JS-friendly API with
  // `.step(state, dx, dy)` and `.context(state)` methods expected by the
  // rest of the frontend. The WASM bindings expose `applyStep(...)` and
  // `contextFeatures(...)` instead.
  const wrapper: StepController = {
    step(state: StepState, delta_real: number, delta_imag: number) {
      // Call WASM applyStep and return the received result object directly.
      const res = (wasmController as any).applyStep(
        state.c_real,
        state.c_imag,
        delta_real,
        delta_imag,
        state.prev_delta_real,
        state.prev_delta_imag
      );
      return res as StepResult;
    },
    context(state: StepState) {
      // Map to WASM contextFeatures which returns a plain object with
      // feature_vec and other context fields.
      const ctx = (wasmController as any).contextFeatures(
        state.c_real,
        state.c_imag,
        state.prev_delta_real,
        state.prev_delta_imag,
        0
      );
      return ctx as StepContext;
    },
  };

  // Debug: surface wrapper shape to console (helps catch runtime mismatches)
  try {
    console.debug('[stepController] created wrapper, methods:', Object.keys(wrapper));
  } catch (e) {
    console.warn('[stepController] created wrapper, could not enumerate methods');
  }

  return wrapper;
}

export async function createInitialStepState(): Promise<StepState> {
  // The WASM bindings expose `StepController.applyStep(...)` rather than a
  // `StepState` constructor. Represent the step state as a simple JS object
  // and pass its numeric fields explicitly to `applyStep`.
  return { c_real: 0.0, c_imag: 0.0, prev_delta_real: 0.0, prev_delta_imag: 0.0 };
}
