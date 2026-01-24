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

export interface VisualParameters {
  juliaReal: number;
  juliaImag: number;
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

  // Tunables
  private readonly MAX_ITER_DEFAULT = 500;
  private readonly NCYCLE_DEFAULT = 32.0;

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
      juliaReal: -0.7269,
      juliaImag: 0.1889,
      colorHue: 0.0,
      colorSat: 0.85,
      colorBright: 0.9,
      zoom: 1.0,
      speed: 0.5
    };
    this.targetParams = { ...this.currentParams };

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

    // Fragment shader: port of mandelbrot.py logic, adapted for Julia (z0 = pixel, c = seed).
    // Notes:
    // - We keep ESC_RADIUS^2 = 1e10 like python.
    // - We compute t = fract(sqrt(niter)/sqrt(ncycle)) like python's post-transform.
    // - Palette is the sinusoidal colormap (continuous, no table/rounding needed).
    const fragmentShaderSource = `
      precision highp float;

      uniform vec2  u_juliaSeed;
      uniform vec2  u_resolution;
      uniform float u_time;

      uniform float u_zoom;
      uniform float u_hue;   // [0,1) palette phase shift
      uniform float u_sat;   // [0,1] saturation mix (0 -> gray, 1 -> full palette)

      uniform int   u_maxIter;
      uniform float u_ncycle;

      uniform float u_stripe_s;
      uniform float u_stripe_sig;
      uniform float u_step_s;

      // Dual light system (Frax-style)
      uniform vec2  u_light1Angles;
      uniform vec4  u_light1Params;  // (intensity, kA, kD, kS)
      
      uniform vec2  u_light2Angles;
      uniform vec4  u_light2Params;
      uniform float u_light2Size;    // Window angular size
      
      uniform float u_lightBalance;  // 0=Light1 only, 1=Light2 only
      uniform float u_shininess;     // Material shininess

      const float PI = 3.141592653589793;
      const float ESC_RADIUS_2 = 1.0e10;

      // --- Complex helpers (vec2 = (re, im)) ---
      vec2 cMul(vec2 a, vec2 b) {
        return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
      }
      vec2 cDiv(vec2 a, vec2 b) {
        float d = dot(b,b);
        // avoid divide-by-zero; if d==0 the value is meaningless anyway
        if (d < 1.0e-30) return vec2(0.0);
        return vec2((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d);
      }

      float overlay(float x, float y, float gamma) {
        // Clamp because downstream assumes [0,1]
        x = clamp(x, 0.0, 1.0);
        y = clamp(y, 0.0, 1.0);
        float outv = (2.0*y < 1.0) ? (2.0*x*y) : (1.0 - 2.0*(1.0-x)*(1.0-y));
        return outv * gamma + x * (1.0 - gamma);
      }
      
      // Sample window environment (with crossbars visible at high shininess)
      float sampleWindow(vec2 reflectDir, float windowSize) {
        // reflectDir is the 2D reflection direction (normalized complex)
        // Window is centered at light2 direction with angular size
        
        // Window bounds (in normalized direction space)
        float halfSize = windowSize * 0.5;
        
        // Check if reflection is within window bounds
        if (abs(reflectDir.x) > halfSize || abs(reflectDir.y) > halfSize) {
          return 0.0;  // Outside window - no contribution
        }
        
        // Normalize to window local coordinates [-1, 1]
        vec2 windowUV = reflectDir / halfSize;
        
        // Window frame (crossbars at center)
        float frameThickness = 0.08;
        bool isFrame = (abs(windowUV.x) < frameThickness) || (abs(windowUV.y) < frameThickness);
        
        // Outer frame
        float outerThickness = 0.05;
        bool isOuterFrame = (abs(abs(windowUV.x) - 1.0) < outerThickness) || 
                            (abs(abs(windowUV.y) - 1.0) < outerThickness);
        
        // Window panes (slightly dimmer than direct light)
        float brightness = 0.8;
        
        // Frame is darker
        if (isFrame || isOuterFrame) {
          brightness = 0.2;
        }
        
        // Soft falloff at edges
        float edgeFalloff = 1.0 - pow(max(abs(windowUV.x), abs(windowUV.y)), 2.0);
        
        return brightness * edgeFalloff;
      }

      // Compute single light Blinn-Phong contribution (point source)
      float blinnPhongSingle(vec2 normalC, vec2 lightAngles, vec4 lightParams) {
        float az = lightAngles.x;
        float el = lightAngles.y;

        // normalize complex part
        float nlen = length(normalC);
        vec2 n = (nlen > 1.0e-30) ? (normalC / nlen) : vec2(0.0, 0.0);

        // Diffuse: dot(light, normal3), normalized like python
        float ldiff =
          n.x * cos(az) * cos(el) +
          n.y * sin(az) * cos(el) +
          1.0 * sin(el);

        ldiff = ldiff / (sqrt(dot(n,n) + 1.0));
        ldiff = max(ldiff, 0.0);

        // Specular: view is always (0,0,1).
        float lspec =
          n.x * cos(az) * cos(el) * 0.5 +
          n.y * sin(az) * cos(el) * 0.5 +
          (1.0 + sin(el)) * 0.5;

        lspec = lspec / (sqrt(dot(n,n) + 1.0));
        lspec = pow(max(lspec, 0.0), u_shininess);

        float intensity = lightParams.x;
        float kA = lightParams.y;
        float kD = lightParams.z;
        float kS = lightParams.w;

        return intensity * (kA + kD*ldiff + kS*lspec);
      }
      
      // Window area light with visible frame at high shininess
      float blinnPhongWindow(vec2 normalC, vec2 lightAngles, vec4 lightParams, float windowSize) {
        float az = lightAngles.x;
        float el = lightAngles.y;
        
        float nlen = length(normalC);
        vec2 n = (nlen > 1.0e-30) ? (normalC / nlen) : vec2(0.0, 0.0);
        
        // Diffuse (same as point source)
        float ldiff = n.x * cos(az) * cos(el) + n.y * sin(az) * cos(el) + 1.0 * sin(el);
        ldiff = ldiff / (sqrt(dot(n,n) + 1.0));
        ldiff = max(ldiff, 0.0);
        
        // Specular - compute reflection direction and sample window
        vec2 reflectDir = n;  // Simplified: using normal as proxy for reflection
        
        // Transform reflection to light-centered coordinates
        vec2 lightDir = vec2(cos(az) * cos(el), sin(az) * cos(el));
        vec2 toLight = reflectDir - lightDir;
        
        // Sample window at reflection direction
        float windowContrib = sampleWindow(toLight, windowSize);
        
        // Specular intensity modulated by window pattern
        float lspec = windowContrib * pow(max(dot(normalize(reflectDir), normalize(lightDir)), 0.0), u_shininess * 0.5);
        
        float intensity = lightParams.x;
        float kA = lightParams.y;
        float kD = lightParams.z;
        float kS = lightParams.w;
        
        return intensity * (kA + kD*ldiff + kS*lspec);
      }
      
      // Dual light Blinn-Phong with balance (Frax-style)
      float blinnPhong(vec2 normalC) {
        float light1 = blinnPhongSingle(normalC, u_light1Angles, u_light1Params);
        float light2 = blinnPhongWindow(normalC, u_light2Angles, u_light2Params, u_light2Size);
        
        // Balance: 0=Light1 only, 1=Light2 only
        float w1 = 1.0 - u_lightBalance;
        float w2 = u_lightBalance;
        
        return light1 * w1 + light2 * w2;
      }

      vec3 sinPalette(float x, vec3 rgb_thetas) {
        // x in [0,1), rgb_thetas are phase offsets
        vec3 ang = (x + rgb_thetas) * (2.0 * PI);
        return 0.5 + 0.5 * sin(ang);
      }

      // Returns:
      //   out_niter  : smooth escape iteration (0 if bounded)
      //   out_stripe : stripe average [0,1]
      //   out_dem    : distance estimate (un-normalized; caller normalizes by diag)
      //   out_normal : complex normal for lighting
      void smoothIterJulia(
        vec2 z0,
        vec2 c,
        int maxIter,
        float stripe_s,
        float stripe_sig,
        out float out_niter,
        out float out_stripe,
        out float out_dem,
        out vec2  out_normal
      ) {
        // Julia: z starts at pixel coordinate; c is fixed seed.
        vec2 z = z0;

        // Derivative dz/dz0 for Julia distance estimate:
        // dz0 = 1
        vec2 dz = vec2(1.0, 0.0);

        bool stripeEnabled = (stripe_s > 0.0) && (stripe_sig > 0.0);
        float stripe_a = 0.0;
        float stripe_t = 0.0;

        out_niter = 0.0;
        out_stripe = 0.0;
        out_dem = 0.0;
        out_normal = vec2(0.0);

        const int MAX_ITER_CAP = 1024;
        int maxIterCapped = (maxIter < MAX_ITER_CAP) ? maxIter : MAX_ITER_CAP;

        for (int i = 0; i < MAX_ITER_CAP; i++) {
          if (i >= maxIterCapped) break;

          // dz = 2*z*dz (use z_n)
          dz = cMul(cMul(vec2(2.0, 0.0), z), dz);

          // z = z^2 + c
          z = cMul(z, z) + c;

          if (stripeEnabled) {
            // stripe_t = (sin(stripe_s * atan2(im, re)) + 1)/2
            stripe_t = (sin(stripe_s * atan(z.y, z.x)) + 1.0) * 0.5;
          }

          float r2 = dot(z, z);

          if (r2 > ESC_RADIUS_2) {
            float modz = sqrt(r2);

            // Smooth iteration count:
            // log_ratio = 2*log(|z|)/log(esc_radius_2)
            float log_ratio = (2.0 * log(modz)) / log(ESC_RADIUS_2);
            log_ratio = max(log_ratio, 1.0e-30);
            float smooth_i = 1.0 - log(log_ratio) / log(2.0);

            float niter = float(i) + smooth_i;

            if (stripeEnabled) {
              // stripe smoothing + linear interpolation, ported from python
              stripe_a = stripe_a * (1.0 + smooth_i * (stripe_sig - 1.0))
                       + stripe_t * smooth_i * (1.0 - stripe_sig);

              float initWeight = pow(stripe_sig, float(i)) * (1.0 + smooth_i * (stripe_sig - 1.0));
              float denom = max(1.0 - initWeight, 1.0e-9);
              stripe_a = stripe_a / denom;
            }

            // Normal vector for lighting: u = z/dz
            vec2 u = cDiv(z, dz);

            // Milton distance estimator:
            // dem = |z|*log(|z|) / |dz| / 2
            float dzAbs = max(length(dz), 1.0e-30);
            float dem = modz * log(modz) / dzAbs / 2.0;

            out_niter = niter;
            out_stripe = stripe_a;
            out_dem = dem;
            out_normal = u;
            return;
          }

          if (stripeEnabled) {
            stripe_a = stripe_a * stripe_sig + stripe_t * (1.0 - stripe_sig);
          }
        }

        // bounded: leave outputs at 0 (black)
      }

      vec3 shadePixel(vec2 fragCoord) {
        float minRes = min(u_resolution.x, u_resolution.y);

        // Map pixel to complex plane like your original: normalized by min dimension,
        // then scaled by BASE_SPAN / zoom (handled via u_zoom)
        vec2 uv = (fragCoord - 0.5 * u_resolution) / minRes;

        float span = ${this.BASE_SPAN.toFixed(6)} / max(u_zoom, 1.0e-6);
        vec2 z0 = uv * span;

        // Frame diagonal in complex-plane units (used to normalize dem like python)
        float spanX = span * (u_resolution.x / minRes);
        float spanY = span * (u_resolution.y / minRes);
        float diag = sqrt(spanX*spanX + spanY*spanY);

        float niter;
        float stripe_a;
        float dem;
        vec2  normal;

        smoothIterJulia(
          z0,
          u_juliaSeed,
          u_maxIter,
          u_stripe_s,
          u_stripe_sig,
          niter,
          stripe_a,
          dem,
          normal
        );

        // Inside set => black (same behavior as mandelbrot.py compute_set)
        if (niter <= 0.0) return vec3(0.0);

        // Port of color_pixel():
        // niter = sqrt(niter) % sqrt(ncycle) / sqrt(ncycle)
        float cycle = sqrt(max(u_ncycle, 1.0e-9));
        float t = fract(sqrt(niter) / cycle);

        // Optional step coloring (quantizes the palette phase like python's col_i update)
        float tColor = t;
        if (u_step_s > 0.0) {
          float stepSize = 1.0 / u_step_s;
          tColor = tColor - mod(tColor, stepSize);
        }

        // Palette phase offsets: python default rgb_thetas=(0.0,0.15,0.25)
        // We add u_hue as a global phase shift, plus a tiny time drift (optional).
        float hueShift = u_hue + 0.02 * u_time; // if you want zero animation, set 0.0 * u_time
        vec3 rgb_thetas = vec3(hueShift, hueShift + 0.15, hueShift + 0.25);

        vec3 base = sinPalette(tColor, rgb_thetas);

        // Saturation control (not in python; keeps your UI meaningful)
        base = mix(vec3(0.5), base, clamp(u_sat, 0.0, 1.0));

        // Brightness with Blinn-Phong
        float bright = blinnPhong(normal);

        // dem normalization by diag + python's log+sigmoid shaping
        float demN = max(dem / max(diag, 1.0e-30), 1.0e-30);
        float demT = -log(demN) / 12.0;
        float demSig = 1.0 / (1.0 + exp(-10.0 * (demT - 0.5)));

        // Shaders: stripes and/or steps (affect brightness like python)
        float nshader = 0.0;
        float shader = 0.0;

        bool stripeEnabled = (u_stripe_s > 0.0) && (u_stripe_sig > 0.0);
        if (stripeEnabled) {
          nshader += 1.0;
          shader += stripe_a;
        }

        if (u_step_s > 0.0) {
          float stepSize = 1.0 / u_step_s;

          // Major step
          float x = mod(t, stepSize) / stepSize;
          float light_step = 6.0 * (1.0 - pow(x, 5.0) - pow(1.0 - x, 100.0)) / 10.0;

          // Minor step
          float stepSize2 = stepSize / 8.0;
          float x2 = mod(t, stepSize2) / stepSize2;
          float light_step2 = 6.0 * (1.0 - pow(x2, 5.0) - pow(1.0 - x2, 30.0)) / 10.0;

          // Overlay merge
          light_step = overlay(light_step2, light_step, 1.0);

          nshader += 1.0;
          shader += light_step;
        }

        if (nshader > 0.0) {
          bright = overlay(bright, shader / nshader, 1.0) * (1.0 - demSig) + demSig * bright;
          bright = clamp(bright, 0.0, 1.0);
        }

        // Apply brightness via overlay mode per channel
        base.r = overlay(base.r, bright, 1.0);
        base.g = overlay(base.g, bright, 1.0);
        base.b = overlay(base.b, bright, 1.0);

        return clamp(base, 0.0, 1.0);
      }

      void main() {
        // No supersampling by default (keep it fast); if you want AA, you can expand this to 4 taps.
        vec3 col = shadePixel(gl_FragCoord.xy);
        gl_FragColor = vec4(col, 1.0);
      }
    `;

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
    this.targetParams = { ...params };
  }

  getCurrentParameters(): VisualParameters {
    return { ...this.currentParams };
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

    this.currentParams.juliaReal += (this.targetParams.juliaReal - this.currentParams.juliaReal) * smoothing;
    this.currentParams.juliaImag += (this.targetParams.juliaImag - this.currentParams.juliaImag) * smoothing;

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
    gl.uniform2f(this.uJuliaSeedLocation!, this.currentParams.juliaReal, this.currentParams.juliaImag);
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

    // Debug: log first frame
    if (this.time === 0) {
      console.log('First render:', {
        seed: [this.currentParams.juliaReal, this.currentParams.juliaImag],
        resolution: [this.canvas.width, this.canvas.height],
        zoom: this.currentParams.zoom,
        maxIter: this.MAX_ITER_DEFAULT,
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
}
