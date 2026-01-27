/**
 * Height-field controller for Mandelbrot parameter motion.
 * Uses f(c) = log|z_N(c)| and projects model steps onto level sets.
 */

export interface Complex {
  real: number;
  imag: number;
}

export interface HeightFieldSample {
  height: number;
  gradient: Complex;
}

export interface HeightControllerParams {
  targetHeight: number;
  normalRisk: number;
  heightGain?: number;
  iterations?: number;
  epsilon?: number;
}

export interface HeightControllerStep {
  newC: Complex;
  delta: Complex;
  sample: HeightFieldSample;
}

export function iterateWithDerivative(c: Complex, iterations: number): { z: Complex; w: Complex } {
  let z: Complex = { real: 0, imag: 0 };
  let w: Complex = { real: 0, imag: 0 };

  for (let i = 0; i < iterations; i++) {
    const wReal = 2 * (z.real * w.real - z.imag * w.imag) + 1;
    const wImag = 2 * (z.real * w.imag + z.imag * w.real);

    const zReal = z.real * z.real - z.imag * z.imag + c.real;
    const zImag = 2 * z.real * z.imag + c.imag;

    w = { real: wReal, imag: wImag };
    z = { real: zReal, imag: zImag };
  }

  return { z, w };
}

export function heightField(
  c: Complex,
  iterations: number = 32,
  epsilon: number = 1e-8
): HeightFieldSample {
  const { z, w } = iterateWithDerivative(c, iterations);
  const zMagSq = Math.max(z.real * z.real + z.imag * z.imag, epsilon);
  const zMag = Math.sqrt(zMagSq);
  const height = Math.log(zMag + epsilon);

  const denom = zMagSq + epsilon;
  const aReal = (z.real * w.real + z.imag * w.imag) / denom;
  const aImag = (z.real * w.imag - z.imag * w.real) / denom;
  const gradient: Complex = { real: aReal, imag: -aImag };

  return { height, gradient };
}

export function controllerStep(
  c: Complex,
  deltaModel: Complex,
  params: HeightControllerParams
): HeightControllerStep {
  const iterations = params.iterations ?? 32;
  const epsilon = params.epsilon ?? 1e-8;
  const heightGain = params.heightGain ?? 0.15;

  const sample = heightField(c, iterations, epsilon);
  const g = sample.gradient;
  const g2 = Math.max(g.real * g.real + g.imag * g.imag, epsilon);

  const normalComponent = (g.real * deltaModel.real + g.imag * deltaModel.imag) / g2;
  const projectionScale = (1 - Math.min(Math.max(params.normalRisk, 0), 1)) * normalComponent;
  const projected: Complex = {
    real: deltaModel.real - g.real * projectionScale,
    imag: deltaModel.imag - g.imag * projectionScale,
  };

  const heightError = sample.height - params.targetHeight;
  const servoScale = (-heightGain * heightError) / g2;
  const servo: Complex = { real: g.real * servoScale, imag: g.imag * servoScale };

  const delta: Complex = { real: projected.real + servo.real, imag: projected.imag + servo.imag };
  const newC: Complex = { real: c.real + delta.real, imag: c.imag + delta.imag };

  return { newC, delta, sample };
}
