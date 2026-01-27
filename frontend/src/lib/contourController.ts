import type { Complex } from './orbitSynthesizer';

export interface HeightFieldParams {
  iterations: number;
  minMagnitude: number;
}

export interface HeightFieldSample {
  height: number;
  gradient: Complex;
  z: Complex;
  w: Complex;
  magnitude: number;
}

export interface ContourControllerParams {
  correctionGain: number;
  projectionEpsilon: number;
}

export interface ContourState {
  c: Complex;
  targetHeight: number;
  lastDelta: Complex;
}

export interface ContourStep {
  c: Complex;
  height: number;
  heightError: number;
  gradient: Complex;
  correctedDelta: Complex;
}

const DEFAULT_HEIGHT_PARAMS: HeightFieldParams = {
  iterations: 64,
  minMagnitude: 1e-6
};

const DEFAULT_CONTOUR_PARAMS: ContourControllerParams = {
  correctionGain: 0.8,
  projectionEpsilon: 1e-6
};

function complexMul(a: Complex, b: Complex): Complex {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real
  };
}

function complexAdd(a: Complex, b: Complex): Complex {
  return { real: a.real + b.real, imag: a.imag + b.imag };
}

function complexScale(a: Complex, scale: number): Complex {
  return { real: a.real * scale, imag: a.imag * scale };
}

function complexSub(a: Complex, b: Complex): Complex {
  return { real: a.real - b.real, imag: a.imag - b.imag };
}

function complexMag(a: Complex): number {
  return Math.hypot(a.real, a.imag);
}

function complexConj(a: Complex): Complex {
  return { real: a.real, imag: -a.imag };
}

function dot(a: Complex, b: Complex): number {
  return a.real * b.real + a.imag * b.imag;
}

export function sampleHeightField(
  c: Complex,
  params: HeightFieldParams = DEFAULT_HEIGHT_PARAMS
): HeightFieldSample {
  let z: Complex = { real: 0, imag: 0 };
  let w: Complex = { real: 0, imag: 0 };

  for (let i = 0; i < params.iterations; i++) {
    w = complexAdd(complexScale(complexMul(z, w), 2.0), { real: 1, imag: 0 });
    z = complexAdd(complexMul(z, z), c);
  }

  const magnitude = complexMag(z);
  const safeMag = Math.max(magnitude, params.minMagnitude);
  const height = Math.log(safeMag);
  const denom = Math.max(safeMag * safeMag, params.minMagnitude * params.minMagnitude);
  const conjZ = complexConj(z);
  const gComplex = complexScale(complexMul(conjZ, w), 1.0 / denom);
  const gradient = { real: gComplex.real, imag: -gComplex.imag };

  return {
    height,
    gradient,
    z,
    w,
    magnitude
  };
}

export function contourCorrectDelta(
  modelDelta: Complex,
  gradient: Complex,
  heightError: number,
  params: ContourControllerParams = DEFAULT_CONTOUR_PARAMS
): Complex {
  const gradNormSq = gradient.real * gradient.real + gradient.imag * gradient.imag;
  const denom = gradNormSq + params.projectionEpsilon;
  const normalScale = dot(gradient, modelDelta) / denom;
  const deltaTangent = complexSub(modelDelta, complexScale(gradient, normalScale));
  const correctionScale = (params.correctionGain * heightError) / denom;
  const correction = complexScale(gradient, correctionScale);
  return complexSub(deltaTangent, correction);
}

export class ContourController {
  private state: ContourState;
  private fieldParams: HeightFieldParams;
  private controllerParams: ContourControllerParams;

  constructor(
    initialC: Complex,
    fieldParams: HeightFieldParams = DEFAULT_HEIGHT_PARAMS,
    controllerParams: ContourControllerParams = DEFAULT_CONTOUR_PARAMS
  ) {
    const sample = sampleHeightField(initialC, fieldParams);
    this.state = {
      c: initialC,
      targetHeight: sample.height,
      lastDelta: { real: 0, imag: 0 }
    };
    this.fieldParams = fieldParams;
    this.controllerParams = controllerParams;
  }

  setTargetHeight(targetHeight: number): void {
    this.state.targetHeight = targetHeight;
  }

  getTargetHeight(): number {
    return this.state.targetHeight;
  }

  getState(): ContourState {
    return { ...this.state };
  }

  step(modelDelta: Complex): ContourStep {
    const sample = sampleHeightField(this.state.c, this.fieldParams);
    const heightError = sample.height - this.state.targetHeight;
    const correctedDelta = contourCorrectDelta(
      modelDelta,
      sample.gradient,
      heightError,
      this.controllerParams
    );
    this.state.c = complexAdd(this.state.c, correctedDelta);
    this.state.lastDelta = correctedDelta;
    return {
      c: this.state.c,
      height: sample.height,
      heightError,
      gradient: sample.gradient,
      correctedDelta
    };
  }
}
