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

interface StepStateConstructor {
  new (c_real: number, c_imag: number, prev_delta_real: number, prev_delta_imag: number): StepState;
}

interface StepControllerConstructor {
  new (): StepController;
}

interface WasmModule {
  default: () => Promise<void>;
  StepController: StepControllerConstructor;
  StepState: StepStateConstructor;
}

let modulePromise: Promise<WasmModule> | null = null;
let cachedModule: WasmModule | null = null;

async function loadModule(): Promise<WasmModule> {
  if (!modulePromise) {
    // @ts-ignore Vite serves this from /wasm at runtime.
    modulePromise = import(/* @vite-ignore */ '/wasm/orbit_synth_wasm.js') as Promise<WasmModule>;
  }
  const mod = await modulePromise;
  cachedModule = mod;
  await mod.default();
  return mod;
}

export async function loadStepController(): Promise<StepController> {
  const mod = await loadModule();
  return new mod.StepController();
}

export async function createInitialStepState(): Promise<StepState> {
  const mod = cachedModule ?? await loadModule();
  return new mod.StepState(0.0, 0.0, 0.0, 0.0);
}
