/**
 * WebGL-based Julia set renderer for real-time visualization.
 *
 * This is a full rewrite that ports the *working* Mandelbrot shading logic from `mandelbrot.py`
 * (smooth escape count + stripe average coloring + distance estimator + Blinn-Phong + overlay blending),
 * but iterates the **Julia** recurrence: z_{n+1} = z_n^2 + c, where c is the user-controlled seed.
 *
 * Key differences vs your previous renderer:
 * - Color is driven by the same cyclic post-transform used in `mandelbrot.py`:
 *     t = fract( sqrt(niter) / sqrt(ncycle) )
 *   so you don't get "outer rings" tied to MAX_ITERATIONS.
 * - Distance estimate + Blinn-Phong + optional stripe/step shading match the python logic closely.
 */

import fragSrc from './shaders/julia.frag?raw';
import type { Complex } from './orbitSynthesizer';

export interface VisualParameters {
  juliaSeed: Complex;
  colorHue: number;    // treated as palette phase shift in [0,1)
  colorSat: number;    // treated as saturation mix [0,1]
  colorBright: number; // treated as light intensity mix [0,1]
  zoom: number;
  speed: number;
}

export class JuliaRenderer {
  private gl: WebGLRenderingContext;
  private program!: WebGLProgram;
  private canvas: HTMLCanvasElement;

  private currentParams: VisualParameters;
  private targetParams: VisualParameters;

  private animationFrameId: number | null = null;
  private time = 0;
  private lastTimestamp: number | null = null;

  // Uniform locations
  private uJuliaSeedLocation: WebGLUniformLocation | null = null;
  private uResolutionLocation: WebGLUniformLocation | null = null;
  private uTimeLocation: WebGLUniformLocation | null = null;

  private uZoomLocation: WebGLUniformLocation | null = null;
  private uHueLocation: WebGLUniformLocation | null = null;
  private uSatLocation: WebGLUniformLocation | null = null;

  private uMaxIterLocation: WebGLUniformLocation | null = null;
  private uNCycleLocation: WebGLUniformLocation | null = null;

  private uStripeSLocation: WebGLUniformLocation | null = null;
  private uStripeSigLocation: WebGLUniformLocation | null = null;
  private uStepSLocation: WebGLUniformLocation | null = null;

  // Dual light system (Frax-style)
  private uLight1AnglesLocation: WebGLUniformLocation | null = null;
  private uLight1ParamsLocation: WebGLUniformLocation | null = null;
  
  private uLight2AnglesLocation: WebGLUniformLocation | null = null;
  private uLight2ParamsLocation: WebGLUniformLocation | null = null;
  private uLight2SizeLocation: WebGLUniformLocation | null = null;
  
  private uLightBalanceLocation: WebGLUniformLocation | null = null;
  private uShininessLocation: WebGLUniformLocation | null = null;

  // Gradient-normal & derivative gating uniforms
  private uDerivLowerLocation: WebGLUniformLocation | null = null;
  private uDerivUpperLocation: WebGLUniformLocation | null = null;
  private uUseGradientNormalsLocation: WebGLUniformLocation | null = null;
  private uFdIterLocation: WebGLUniformLocation | null = null;
  private uFdEpsLocation: WebGLUniformLocation | null = null;
  private uHeightScaleLocation: WebGLUniformLocation | null = null;

  // Fresnel / rim / normal preference uniform locations
  private uFresnelPowerLocation: WebGLUniformLocation | null = null;
  private uFresnelBoostLocation: WebGLUniformLocation | null = null;
  private uRimIntensityLocation: WebGLUniformLocation | null = null;
  private uPreferGradientNormalsLocation: WebGLUniformLocation | null = null;

  // Low-cost FD/gating uniform locations
  private uFdIterLowLocation: WebGLUniformLocation | null = null;
  private uFdEpsLowLocation: WebGLUniformLocation | null = null;
  private uGradDemThresholdLocation: WebGLUniformLocation | null = null;

  // Tunables
  private readonly MAX_ITER_DEFAULT = 500;
  private readonly NCYCLE_DEFAULT = 32.0;

  // FD / gradient defaults (tunable)
  private useGradientNormals: boolean = true;
  // Defaults match UI control defaults in Visualizer.tsx
  private fdIter: number = 160;         // fixed-N for potential eval (matches UI FD Iter default)
  private fdEps: number = 0.6;          // in pixels (matches UI FD ε default)
  private heightScale: number = 8.0;    // controls slope of height normal (matches UI Height default -> max)
  private derivLower: number = 1e-6;    // derivative confidence lower bound
  private derivUpper: number = 1e-3;    // derivative confidence upper bound

  // Fresnel / rim / normal behavior defaults
  private fresnelPower: number = 3.0;
  private fresnelBoost: number = 0.25;
  private rimIntensity: number = 0.08;
  // Preference: whether to prefer gradient-based normals when available.
  // This boolean toggles which normal is preferred when both derivative-based
  // gradient normals and analytic normals are available. It does not by itself
  // enable/disable gradient computation (that's controlled by `Grad Mode`).
  // true = prefer gradient normals when available; false = use analytic normals.
  private preferGradientNormals: boolean = true;

  // Gradient mode: 0=off,1=full,2=cheap
  private gradientMode: number = 1; // default to full (accurate)

  // Low-cost FD defaults
  private fdIterLow: number = 32;
  private fdEpsLow: number = 1.5;
  private gradDemThreshold: number = 0.02; // only compute cheap FD near boundary
  private readonly STRIPE_S_DEFAULT = 0.0;
  private readonly STRIPE_SIG_DEFAULT = 0.9;
  private readonly STEP_S_DEFAULT = 0.0;

  // Material property
  private readonly SHININESS = 40.0;  // Surface shininess (higher = sharper reflections)

  // Dual Frax-style lighting:
  // Light 1: Point source from upper-left (sharp directional)
  private readonly LIGHT1_AZ_DEG = 135.0;
  private readonly LIGHT1_EL_DEG = 45.0;
  private readonly LIGHT1_INTENSITY = 0.85;
  private readonly LIGHT1_KA = 0.15;
  private readonly LIGHT1_KD = 0.6;
  private readonly LIGHT1_KS = 0.7;
  
  // Light 2: Window area light from right (soft, broad with visible window frame)
  private readonly LIGHT2_AZ_DEG = -60.0;  // Right side
  private readonly LIGHT2_EL_DEG = 35.0;
  private readonly LIGHT2_INTENSITY = 0.7;
  private readonly LIGHT2_KA = 0.2;
  private readonly LIGHT2_KD = 0.5;
  private readonly LIGHT2_KS = 0.4;
  private readonly LIGHT2_WINDOW_SIZE = 0.4;  // Window angular size (larger = softer reflections)
  
  private readonly LIGHT_BALANCE = 0.5;  // 0=only Light1, 1=only Light2, 0.5=equal

  // View defaults: your old code effectively showed ~2.7 span (square) at zoom=1
  private readonly BASE_SPAN = 2.7;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false
    });
    if (!gl) throw new Error('WebGL not supported');
    this.gl = gl;

    this.currentParams = {
      juliaSeed: { real: -0.7269, imag: 0.1889 },
      colorHue: 0.0,
      colorSat: 0.85,
      colorBright: 0.9,
      zoom: 1.0,
      speed: 0.5
    };
    this.targetParams = {
      ...this.currentParams,
      juliaSeed: { ...this.currentParams.juliaSeed }
    };

    this.initWebGL();
  }

  private initWebGL(): void {
    const gl = this.gl;

    const vertexShaderSource = `
      attribute vec2 a_position;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    // Fragment shader: loaded from ` + "julia.frag" + ` as raw text; substitute placeholder for BASE_SPAN
    const fragmentShaderSource = fragSrc.replace('__BASE_SPAN__', this.BASE_SPAN.toFixed(6));

    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);

    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Failed to link program: ' + gl.getProgramInfoLog(this.program));
    }

    // Uniform locations
    this.uJuliaSeedLocation = gl.getUniformLocation(this.program, 'u_juliaSeed');
    this.uResolutionLocation = gl.getUniformLocation(this.program, 'u_resolution');
    this.uTimeLocation = gl.getUniformLocation(this.program, 'u_time');

    this.uZoomLocation = gl.getUniformLocation(this.program, 'u_zoom');
    this.uHueLocation = gl.getUniformLocation(this.program, 'u_hue');
    this.uSatLocation = gl.getUniformLocation(this.program, 'u_sat');

    this.uMaxIterLocation = gl.getUniformLocation(this.program, 'u_maxIter');
    this.uNCycleLocation = gl.getUniformLocation(this.program, 'u_ncycle');

    this.uStripeSLocation = gl.getUniformLocation(this.program, 'u_stripe_s');
    this.uStripeSigLocation = gl.getUniformLocation(this.program, 'u_stripe_sig');
    this.uStepSLocation = gl.getUniformLocation(this.program, 'u_step_s');

    // Dual light uniforms
    this.uLight1AnglesLocation = gl.getUniformLocation(this.program, 'u_light1Angles');
    this.uLight1ParamsLocation = gl.getUniformLocation(this.program, 'u_light1Params');
    
    this.uLight2AnglesLocation = gl.getUniformLocation(this.program, 'u_light2Angles');
    this.uLight2ParamsLocation = gl.getUniformLocation(this.program, 'u_light2Params');
    this.uLight2SizeLocation = gl.getUniformLocation(this.program, 'u_light2Size');
    
    this.uLightBalanceLocation = gl.getUniformLocation(this.program, 'u_lightBalance');
    this.uShininessLocation = gl.getUniformLocation(this.program, 'u_shininess');

    // Gradient-normal & gating uniforms
    this.uDerivLowerLocation = gl.getUniformLocation(this.program, 'u_derivLower');
    this.uDerivUpperLocation = gl.getUniformLocation(this.program, 'u_derivUpper');
    this.uUseGradientNormalsLocation = gl.getUniformLocation(this.program, 'u_useGradientNormals');
    this.uFdIterLocation = gl.getUniformLocation(this.program, 'u_fdIter');
    this.uFdEpsLocation = gl.getUniformLocation(this.program, 'u_fdEps');
    this.uHeightScaleLocation = gl.getUniformLocation(this.program, 'u_heightScale');

    // Low-cost FD / gating uniforms
    this.uFdIterLowLocation = gl.getUniformLocation(this.program, 'u_fdIterLow');
    this.uFdEpsLowLocation = gl.getUniformLocation(this.program, 'u_fdEpsLow');
    this.uGradDemThresholdLocation = gl.getUniformLocation(this.program, 'u_gradDemThreshold');

    // Fresnel / rim / normal-blend uniform locations
    this.uFresnelPowerLocation = gl.getUniformLocation(this.program, 'u_fresnelPower');
    this.uFresnelBoostLocation = gl.getUniformLocation(this.program, 'u_fresnelBoost');
    this.uRimIntensityLocation = gl.getUniformLocation(this.program, 'u_rimIntensity');
    this.uPreferGradientNormalsLocation = gl.getUniformLocation(this.program, 'u_preferGradientNormals');

    // Fullscreen quad
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLocation = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    this.resize();
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      console.error('Shader compilation failed:', log);
      console.error('Shader source:', source);
      throw new Error('Failed to compile shader: ' + log);
    }
    return shader;
  }

  resize(): void {
    const gl = this.gl;
    const canvas = this.canvas;

    // Use DPR for sharper output (your old code effectively ignored DPR).
    const dpr = Math.min(window.devicePixelRatio || 1, 2); // clamp for perf
    const displayWidth = Math.floor(canvas.clientWidth * dpr);
    const displayHeight = Math.floor(canvas.clientHeight * dpr);

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      gl.viewport(0, 0, displayWidth, displayHeight);
    }
  }

  updateParameters(params: VisualParameters): void {
    this.targetParams = {
      ...params,
      juliaSeed: { ...params.juliaSeed }
    };
  }

  getCurrentParameters(): VisualParameters {
    return {
      ...this.currentParams,
      juliaSeed: { ...this.currentParams.juliaSeed }
    };
  }

  private wrap01(x: number): number {
    const y = x % 1;
    return y < 0 ? y + 1 : y;
  }

  private lerpHue(current: number, target: number, alpha: number): number {
    // shortest wrap-around interpolation in [0,1)
    const c = this.wrap01(current);
    const t = this.wrap01(target);
    let d = t - c;
    if (d > 0.5) d -= 1;
    if (d < -0.5) d += 1;
    return this.wrap01(c + d * alpha);
  }

  private interpolateParams(_dt: number): void {
    const smoothing = 0.12;

    this.currentParams.juliaSeed.real += (this.targetParams.juliaSeed.real - this.currentParams.juliaSeed.real) * smoothing;
    this.currentParams.juliaSeed.imag += (this.targetParams.juliaSeed.imag - this.currentParams.juliaSeed.imag) * smoothing;

    this.currentParams.colorHue = this.lerpHue(this.currentParams.colorHue, this.targetParams.colorHue, smoothing);

    this.currentParams.colorSat += (this.targetParams.colorSat - this.currentParams.colorSat) * smoothing;
    this.currentParams.colorBright += (this.targetParams.colorBright - this.currentParams.colorBright) * smoothing;

    this.currentParams.zoom += (this.targetParams.zoom - this.currentParams.zoom) * smoothing;
    this.currentParams.speed += (this.targetParams.speed - this.currentParams.speed) * smoothing;
  }

  render(): void {
    this.resize();
    const gl = this.gl;

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    // Core uniforms
    gl.uniform2f(this.uJuliaSeedLocation!, this.currentParams.juliaSeed.real, this.currentParams.juliaSeed.imag);
    gl.uniform2f(this.uResolutionLocation!, this.canvas.width, this.canvas.height);
    gl.uniform1f(this.uTimeLocation!, this.time);

    gl.uniform1f(this.uZoomLocation!, Math.max(this.currentParams.zoom, 1e-6));
    gl.uniform1f(this.uHueLocation!, this.wrap01(this.currentParams.colorHue));
    gl.uniform1f(this.uSatLocation!, Math.max(0, Math.min(1, this.currentParams.colorSat)));

    // Ported parameters (defaults from mandelbrot.py)
    gl.uniform1i(this.uMaxIterLocation!, this.MAX_ITER_DEFAULT);
    gl.uniform1f(this.uNCycleLocation!, this.NCYCLE_DEFAULT);

    gl.uniform1f(this.uStripeSLocation!, this.STRIPE_S_DEFAULT);
    gl.uniform1f(this.uStripeSigLocation!, this.STRIPE_SIG_DEFAULT);
    gl.uniform1f(this.uStepSLocation!, this.STEP_S_DEFAULT);

    // Material property
    gl.uniform1f(this.uShininessLocation!, this.SHININESS);
    
    // Dual light setup (Frax-style)
    const bright = Math.max(0, Math.min(1, this.currentParams.colorBright));
    
    // Light 1: Point source (sharp directional)
    const az1 = (2 * Math.PI) * (this.LIGHT1_AZ_DEG / 360.0);
    const el1 = (Math.PI / 2) * (this.LIGHT1_EL_DEG / 90.0);
    const intensity1 = this.LIGHT1_INTENSITY * (0.4 + 0.6 * bright);
    
    gl.uniform2f(this.uLight1AnglesLocation!, az1, el1);
    gl.uniform4f(this.uLight1ParamsLocation!, intensity1, this.LIGHT1_KA, this.LIGHT1_KD, this.LIGHT1_KS);
    
    // Light 2: Window area light (with visible frame)
    const az2 = (2 * Math.PI) * (this.LIGHT2_AZ_DEG / 360.0);
    const el2 = (Math.PI / 2) * (this.LIGHT2_EL_DEG / 90.0);
    const intensity2 = this.LIGHT2_INTENSITY * (0.6 + 0.4 * bright);
    
    gl.uniform2f(this.uLight2AnglesLocation!, az2, el2);
    gl.uniform4f(this.uLight2ParamsLocation!, intensity2, this.LIGHT2_KA, this.LIGHT2_KD, this.LIGHT2_KS);
    gl.uniform1f(this.uLight2SizeLocation!, this.LIGHT2_WINDOW_SIZE);
    
    // Balance between lights
    gl.uniform1f(this.uLightBalanceLocation!, this.LIGHT_BALANCE);

    // Derivative gating and gradient-normal config
    gl.uniform1f(this.uDerivLowerLocation!, this.derivLower);
    gl.uniform1f(this.uDerivUpperLocation!, this.derivUpper);
    // Gradient mode: 0=off,1=full,2=cheap
    gl.uniform1i(this.uUseGradientNormalsLocation!, this.gradientMode);
    gl.uniform1i(this.uFdIterLocation!, this.fdIter);
    gl.uniform1f(this.uFdEpsLocation!, this.fdEps);
    gl.uniform1f(this.uHeightScaleLocation!, this.heightScale);

    // Low-cost FD / gating uniforms
    gl.uniform1i(this.uFdIterLowLocation!, this.fdIterLow);
    gl.uniform1f(this.uFdEpsLowLocation!, this.fdEpsLow);
    gl.uniform1f(this.uGradDemThresholdLocation!, this.gradDemThreshold);

    // Fresnel / rim / normal blend
    gl.uniform1f(this.uFresnelPowerLocation!, this.fresnelPower);
    gl.uniform1f(this.uFresnelBoostLocation!, this.fresnelBoost);
    gl.uniform1f(this.uRimIntensityLocation!, this.rimIntensity);
    // Convert boolean preference to numeric uniform (1.0 = prefer gradient normals, 0.0 = prefer analytic normals)
    gl.uniform1f(this.uPreferGradientNormalsLocation!, this.preferGradientNormals ? 1.0 : 0.0);

    // Debug: log first frame
    if (this.time === 0) {
      console.log('First render:', {
        seed: [this.currentParams.juliaSeed.real, this.currentParams.juliaSeed.imag],
        resolution: [this.canvas.width, this.canvas.height],
        zoom: this.currentParams.zoom,
        maxIter: this.MAX_ITER_DEFAULT,
        gradientNormals: this.useGradientNormals,
        fdIter: this.fdIter,
        fdEps: this.fdEps,
        fresnel: {power: this.fresnelPower, boost: this.fresnelBoost},
        rim: this.rimIntensity,
        preferGradientNormals: this.preferGradientNormals,
        uniforms: {
          uJuliaSeedLocation: this.uJuliaSeedLocation,
          uResolutionLocation: this.uResolutionLocation,
          uMaxIterLocation: this.uMaxIterLocation
        }
      });
    }

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  start(): void {
    if (this.animationFrameId !== null) return;

    const animate = (timestamp: number) => {
      if (this.lastTimestamp === null) this.lastTimestamp = timestamp;
      const dt = Math.min(0.05, (timestamp - this.lastTimestamp) / 1000);
      this.lastTimestamp = timestamp;

      this.time += dt * this.currentParams.speed;

      this.interpolateParams(dt);
      this.render();

      this.animationFrameId = requestAnimationFrame(animate);
    };

    this.animationFrameId = requestAnimationFrame(animate);
  }

  stop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
      this.lastTimestamp = null;
    }
  }

  dispose(): void {
    this.stop();
    const gl = this.gl;
    gl.deleteProgram(this.program);
  }

  // --- Gradient-normal toggles & configurables ---
  setUseGradientNormals(enabled: boolean): void {
    this.useGradientNormals = enabled;
  }

  setFdIter(n: number): void {
    this.fdIter = Math.max(1, Math.floor(n));
  }

  setFdEps(pixels: number): void {
    this.fdEps = Math.max(0.0001, pixels);
  }

  setHeightScale(s: number): void {
    this.heightScale = s;
  }

  setDerivThresholds(lower: number, upper: number): void {
    this.derivLower = lower;
    this.derivUpper = upper;
  }

  // gradient mode: 'off' | 'full' | 'cheap'
  setGradientMode(mode: 'off' | 'full' | 'cheap'): void {
    if (mode === 'off') this.gradientMode = 0;
    else if (mode === 'full') this.gradientMode = 1;
    else this.gradientMode = 2;
  }

  setFdIterLow(n: number): void {
    this.fdIterLow = Math.max(1, Math.floor(n));
  }

  setFdEpsLow(pixels: number): void {
    this.fdEpsLow = Math.max(0.0001, pixels);
  }

  setGradDemThreshold(v: number): void {
    this.gradDemThreshold = Math.max(0.0, Math.min(1.0, v));
  }

  setFresnel(power: number, boost: number): void {
    this.fresnelPower = Math.max(0.1, power);
    this.fresnelBoost = Math.max(0.0, boost);
  }

  setFresnelPower(p: number): void {
    this.fresnelPower = Math.max(0.1, p);
  }

  setFresnelBoost(b: number): void {
    this.fresnelBoost = Math.max(0.0, b);
  }

  setRimIntensity(v: number): void {
    this.rimIntensity = Math.max(0.0, v);
  }

  /**
   * Set whether to prefer gradient-derived normals when available.
   * Note: this is a preference only; enabling/disabling gradient computation
   * is done via `setGradientMode('off'|'cheap'|'full')`.
   */
  setPreferGradientNormals(enabled: boolean): void {
    this.preferGradientNormals = Boolean(enabled);
  }
}
