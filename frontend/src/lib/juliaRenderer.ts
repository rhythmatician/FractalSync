/**
 * WebGL-based Julia set renderer for real-time visualization.
 */

import { getGradientByHue, flattenGradient, Gradient, MAX_GRADIENT_STOPS, DEFAULT_HUE } from './toolGradients';

export interface VisualParameters {
  juliaReal: number;
  juliaImag: number;
  colorHue: number;
  colorSat: number;
  colorBright: number;
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
  private time: number = 0;
  private frameCount: number = 0;

  // Uniform locations
  private uJuliaSeedLocation: WebGLUniformLocation | null = null;
  private uZoomLocation: WebGLUniformLocation | null = null;
  private uColorLocation: WebGLUniformLocation | null = null;
  private uTimeLocation: WebGLUniformLocation | null = null;
  private uResolutionLocation: WebGLUniformLocation | null = null;
  private uGradientCountLocation: WebGLUniformLocation | null = null;
  
  // Cached gradient uniform locations for performance
  private gradientUniformLocations: (WebGLUniformLocation | null)[] = [];
  
  private currentGradient: Gradient | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl');
    if (!gl) {
      throw new Error('WebGL not supported');
    }
    this.gl = gl;

    // Initialize default parameters (using shared constant for hue)
    this.currentParams = {
      juliaReal: -0.7269,
      juliaImag: 0.1889,
      colorHue: DEFAULT_HUE,
      colorSat: 0.8,
      colorBright: 0.9,
      zoom: 1.0,
      speed: 0.5
    };
    this.targetParams = { ...this.currentParams };

    // Initialize WebGL
    this.initWebGL();
  }

  private initWebGL(): void {
    const gl = this.gl;

    // Vertex shader (simple full-screen quad)
    const vertexShaderSource = `
      attribute vec2 a_position;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;

    // Fragment shader - Frax-style rendering with 5 stages
    const fragmentShaderSource = `
      precision highp float;
      
      uniform vec2 u_juliaSeed;
      uniform float u_zoom;
      uniform vec3 u_color;  // x=hue, y=intensity, z=contrast
      uniform float u_time;
      uniform vec2 u_resolution;
      uniform int u_gradientCount;
      
      // Gradient stops
      uniform vec4 u_gradient0;
      uniform vec4 u_gradient1;
      uniform vec4 u_gradient2;
      uniform vec4 u_gradient3;
      uniform vec4 u_gradient4;
      uniform vec4 u_gradient5;
      uniform vec4 u_gradient6;
      uniform vec4 u_gradient7;
      
      const int MAX_ITERATIONS = 128;
      const float ESCAPE_RADIUS = 2.0;
      
      // Height mapping parameters (the "art knobs")
      const float HEIGHT_K = 240.0;
      const float HEIGHT_P = 0.9;
      
      // Lighting parameters
      const float SPEC_EXPONENT = 110.0;
      const float FRESNEL_F0 = 0.12;
      const float AMBIENT = 0.15;
      const float DIFFUSE_STRENGTH = 0.6;
      const float SPEC_STRENGTH = 0.8;
      const float REFLECTION_STRENGTH = 0.5;
      
      vec4 getGradientStop(int index) {
        if (index == 0) return u_gradient0;
        if (index == 1) return u_gradient1;
        if (index == 2) return u_gradient2;
        if (index == 3) return u_gradient3;
        if (index == 4) return u_gradient4;
        if (index == 5) return u_gradient5;
        if (index == 6) return u_gradient6;
        if (index == 7) return u_gradient7;
        return vec4(0.0);
      }
      
      vec3 sampleGradient(float t) {
        t = clamp(t, 0.0, 1.0);
        vec4 lowerStop = getGradientStop(0);
        vec4 upperStop = getGradientStop(u_gradientCount - 1);
        
        for (int i = 0; i < 7; i++) {
          if (i >= u_gradientCount - 1) break;
          vec4 stop1 = getGradientStop(i);
          vec4 stop2 = getGradientStop(i + 1);
          if (t >= stop1.w && t <= stop2.w) {
            lowerStop = stop1;
            upperStop = stop2;
            break;
          }
        }
        
        float range = upperStop.w - lowerStop.w;
        float localT = range > 0.0 ? (t - lowerStop.w) / range : 0.0;
        return mix(lowerStop.rgb, upperStop.rgb, localT);
      }
      
      // Complex multiplication
      vec2 cmul(vec2 a, vec2 b) {
        return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
      }
      
      // Stage 1 & 2: Iterate Julia set with derivative tracking
      // Returns: (smoothIter, escaped, distance)
      vec3 juliaIterate(vec2 c, vec2 z0) {
        vec2 z = z0;
        vec2 dz = vec2(0.0, 0.0);  // Derivative z'_0 = 0
        
        float n = 0.0;
        bool escaped = false;
        
        for (int i = 0; i < MAX_ITERATIONS; i++) {
          n = float(i);
          float zMag2 = dot(z, z);
          
          if (zMag2 > ESCAPE_RADIUS * ESCAPE_RADIUS) {
            // Smooth iteration count
            float nu = n + 1.0 - log(log(sqrt(zMag2))) / log(2.0);
            
            // Distance estimator: d ≈ |z|ln|z| / |z'|
            float zMag = sqrt(zMag2);
            float dzMag = length(dz);
            float dist = (dzMag > 0.0) ? (zMag * log(zMag)) / dzMag : 0.0;
            
            return vec3(nu, 1.0, dist);
          }
          
          // Update derivative: z'_{n+1} = 2*z_n*z'_n + 1
          dz = cmul(vec2(2.0, 0.0) * z, dz) + vec2(1.0, 0.0);
          
          // Update z: z_{n+1} = z_n^2 + c
          z = cmul(z, z) + c;
        }
        
        return vec3(float(MAX_ITERATIONS), 0.0, 0.0);
      }
      
      // Stage 3: Map distance to height with nonlinear curve
      float distanceToHeight(float d) {
        float kd = HEIGHT_K * d;
        return 1.0 / (1.0 + pow(kd, HEIGHT_P));
      }
      
      // Sample height at offset (for normal calculation with smoothing)
      float sampleHeight(vec2 uv, vec2 offset, vec2 c) {
        vec3 result = juliaIterate(c, uv + offset);
        if (result.y < 0.5) return 0.0;  // Inside set
        return distanceToHeight(result.z);
      }
      
      // Stage 3d/e: Calculate normals from height field with box blur smoothing
      vec3 calculateNormal(vec2 uv, vec2 c, float pixelSize) {
        // Box blur radius for smoothing
        float blur = pixelSize * 2.0;
        
        // Sample heights in a small neighborhood (3x3 box blur approximation)
        float h = 0.0;
        float hx = 0.0;
        float hy = 0.0;
        
        // Center samples
        for (int dy = -1; dy <= 1; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            vec2 offset = vec2(float(dx), float(dy)) * blur;
            float hSample = sampleHeight(uv, offset, c);
            h += hSample;
            
            // Accumulate for gradients
            if (dx != 0) hx += float(dx) * hSample;
            if (dy != 0) hy += float(dy) * hSample;
          }
        }
        
        // Average
        h /= 9.0;
        hx /= 6.0;  // 6 samples contribute to x gradient
        hy /= 6.0;
        
        // Gradients
        float dhx = hx / blur;
        float dhy = hy / blur;
        
        // Normal: normalize(-∂h/∂x, -∂h/∂y, 1)
        return normalize(vec3(-dhx, -dhy, 1.0));
      }
      
      // Stage 5b: Fresnel (Schlick approximation)
      float fresnel(vec3 normal, vec3 view) {
        float cosTheta = max(dot(normal, view), 0.0);
        return FRESNEL_F0 + (1.0 - FRESNEL_F0) * pow(1.0 - cosTheta, 5.0);
      }
      
      // Stage 5c: Procedural environment map (studio window)
      vec3 sampleEnvironment(vec3 reflectDir) {
        // Warm gradient background
        float y = reflectDir.z * 0.5 + 0.5;
        vec3 bg = mix(vec3(0.3, 0.25, 0.2), vec3(0.6, 0.7, 0.8), y);
        
        // Add two rectangular "window" highlights (softbox reflections)
        vec2 dir2d = reflectDir.xy;
        
        // Window 1 (rotated by time)
        float angle1 = u_time * 0.1;
        vec2 rot1 = vec2(
          dir2d.x * cos(angle1) - dir2d.y * sin(angle1),
          dir2d.x * sin(angle1) + dir2d.y * cos(angle1)
        );
        float window1 = smoothstep(0.3, 0.2, abs(rot1.x)) * smoothstep(0.5, 0.4, abs(rot1.y));
        
        // Window 2 (opposite rotation)
        float angle2 = -u_time * 0.15 + 1.5;
        vec2 rot2 = vec2(
          dir2d.x * cos(angle2) - dir2d.y * sin(angle2),
          dir2d.x * sin(angle2) + dir2d.y * cos(angle2)
        );
        float window2 = smoothstep(0.4, 0.3, abs(rot2.x)) * smoothstep(0.3, 0.2, abs(rot2.y));
        
        return bg + vec3(2.0) * window1 + vec3(1.8, 1.9, 2.0) * window2;
      }
      
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
        uv *= 2.7;
        float pixelSize = 2.7 / min(u_resolution.x, u_resolution.y);
        
        vec2 c = u_juliaSeed;
        
        // Stage 1 & 2: Compute fractal with smooth iteration count
        vec3 fractalData = juliaIterate(c, uv);
        float smoothIter = fractalData.x;
        float escaped = fractalData.y;
        float distance = fractalData.z;
        
        // Stage 2: Normalize smooth iteration to [0,1]
        // Using percentile-like normalization (clamped)
        float t = clamp(smoothIter / float(MAX_ITERATIONS), 0.0, 1.0);
        t = pow(t, 0.7);  // Adjust distribution
        
        // Stage 4: Color from gradient
        float paletteCoord = fract(t * 4.2 * u_color.z);  // Palette cycling
        vec3 baseColor = sampleGradient(paletteCoord);
        
        // Inside set: dark
        if (escaped < 0.5) {
          gl_FragColor = vec4(baseColor * 0.05, 1.0);
          return;
        }
        
        // Stage 3: Calculate height and normal
        float height = distanceToHeight(distance);
        vec3 normal = calculateNormal(uv, c, pixelSize);
        
        // Stage 5: Glossy 3D lighting
        vec3 viewDir = vec3(0.0, 0.0, 1.0);  // Straight down
        
        // Animated light direction
        float lightAngle = u_time * 0.3;
        vec3 lightDir = normalize(vec3(
          cos(lightAngle) * 0.6,
          sin(lightAngle) * 0.4,
          0.8
        ));
        
        // Stage 5a: Diffuse lighting
        float diffuse = max(dot(normal, lightDir), 0.0);
        
        // Stage 5a: Specular (Blinn-Phong)
        vec3 halfVec = normalize(lightDir + viewDir);
        float specular = pow(max(dot(normal, halfVec), 0.0), SPEC_EXPONENT);
        
        // Stage 5b: Fresnel
        float F = fresnel(normal, viewDir);
        
        // Stage 5c: Reflection
        vec3 reflectDir = reflect(-viewDir, normal);
        vec3 envColor = sampleEnvironment(reflectDir);
        
        // Stage 5d: Final composite
        vec3 pigment = baseColor * (AMBIENT + DIFFUSE_STRENGTH * diffuse);
        vec3 gloss = vec3(1.0) * SPEC_STRENGTH * specular;
        vec3 reflection = envColor * REFLECTION_STRENGTH * F;
        
        vec3 finalColor = pigment + gloss + reflection;
        
        // Apply intensity from controls
        finalColor *= (0.5 + u_color.y * 0.5);
        
        // Subtle bloom approximation (brighten bright areas)
        float brightness = dot(finalColor, vec3(0.299, 0.587, 0.114));
        if (brightness > 1.0) {
          finalColor += vec3(brightness - 1.0) * 0.3;
        }
        
        gl_FragColor = vec4(finalColor, 1.0);
      }
    `;

    // Compile shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);

    // Create program
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Failed to link program: ' + gl.getProgramInfoLog(this.program));
    }

    // Get uniform locations
    this.uJuliaSeedLocation = gl.getUniformLocation(this.program, 'u_juliaSeed');
    this.uZoomLocation = gl.getUniformLocation(this.program, 'u_zoom');
    this.uColorLocation = gl.getUniformLocation(this.program, 'u_color');
    this.uTimeLocation = gl.getUniformLocation(this.program, 'u_time');
    this.uResolutionLocation = gl.getUniformLocation(this.program, 'u_resolution');
    this.uGradientCountLocation = gl.getUniformLocation(this.program, 'u_gradientCount');
    
    // Cache gradient uniform locations for performance
    for (let i = 0; i < MAX_GRADIENT_STOPS; i++) {
      this.gradientUniformLocations[i] = gl.getUniformLocation(this.program, `u_gradient${i}`);
    }
    
    // Initialize with the continuous gradient (single gradient for all hue values)
    this.updateGradient(DEFAULT_HUE);

    // Create full-screen quad
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ];
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    // Set up vertex attributes
    const positionLocation = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    // Set viewport
    this.resize();
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error('Failed to compile shader: ' + gl.getShaderInfoLog(shader));
    }

    return shader;
  }

  resize(): void {
    const gl = this.gl;
    const canvas = this.canvas;
    
    // Set canvas size to match display size with device pixel ratio
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      gl.viewport(0, 0, displayWidth, displayHeight);
    }
  }

  updateParameters(params: VisualParameters): void {
    this.targetParams = { ...params };
    
    // Update gradient only on first call (continuous gradient doesn't change)
    if (!this.currentGradient) {
      this.updateGradient(params.colorHue);
    }
  }
  
  getCurrentParameters(): VisualParameters {
    return { ...this.currentParams };
  }
  
  private updateGradient(hue: number): void {
    const gradient = getGradientByHue(hue);
    
    // Only update once (we use a single continuous gradient)
    if (this.currentGradient) {
      return;
    }
    
    this.currentGradient = gradient;
    const gl = this.gl;
    
    // Flatten gradient for WebGL uniform
    const flatData = flattenGradient(gradient);
    
    // Upload to GPU as individual vec4 uniforms (WebGL 1.0 compatible)
    // Note: This changes the active WebGL program state. Callers should be aware
    // that the program will be active after this call (typically called before rendering).
    gl.useProgram(this.program);
    
    // Set gradient count
    gl.uniform1i(this.uGradientCountLocation!, gradient.stops.length);
    
    // Set gradient stops using cached uniform locations (performance optimization)
    for (let i = 0; i < Math.min(gradient.stops.length, MAX_GRADIENT_STOPS); i++) {
      const baseIdx = i * 4;
      const location = this.gradientUniformLocations[i];
      if (location) {
        gl.uniform4f(
          location,
          flatData[baseIdx],     // r
          flatData[baseIdx + 1], // g
          flatData[baseIdx + 2], // b
          flatData[baseIdx + 3]  // position
        );
      }
    }
    
    // Fill remaining slots with black if gradient has fewer stops
    for (let i = gradient.stops.length; i < MAX_GRADIENT_STOPS; i++) {
      const location = this.gradientUniformLocations[i];
      if (location) {
        gl.uniform4f(location, 0.0, 0.0, 0.0, 1.0);
      }
    }
    
    console.log(`🎨 Gradient loaded: ${gradient.name} (continuous smooth transitions)`);
  }

  private interpolateParams(_dt: number): void {
    const smoothing = 0.1; // Smoothing factor
    
    this.currentParams.juliaReal += (this.targetParams.juliaReal - this.currentParams.juliaReal) * smoothing;
    this.currentParams.juliaImag += (this.targetParams.juliaImag - this.currentParams.juliaImag) * smoothing;
    this.currentParams.colorHue += (this.targetParams.colorHue - this.currentParams.colorHue) * smoothing;
    this.currentParams.colorSat += (this.targetParams.colorSat - this.currentParams.colorSat) * smoothing;
    this.currentParams.colorBright += (this.targetParams.colorBright - this.currentParams.colorBright) * smoothing;
    this.currentParams.zoom += (this.targetParams.zoom - this.currentParams.zoom) * smoothing;
    this.currentParams.speed += (this.targetParams.speed - this.currentParams.speed) * smoothing;
  }

  render(): void {
    const gl = this.gl;
    
    // Clear
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Use program
    gl.useProgram(this.program);
    
    // Set uniforms
    gl.uniform2f(this.uJuliaSeedLocation!, this.currentParams.juliaReal, this.currentParams.juliaImag);
    gl.uniform1f(this.uZoomLocation!, this.currentParams.zoom);
    gl.uniform3f(
      this.uColorLocation!,
      this.currentParams.colorHue,
      this.currentParams.colorSat,
      this.currentParams.colorBright
    );
    gl.uniform1f(this.uTimeLocation!, this.time);
    gl.uniform2f(this.uResolutionLocation!, this.canvas.width, this.canvas.height);
    
    // Debug log every 100 frames for deterministic logging
    this.frameCount++;
    if (this.frameCount % 100 === 0) {
      console.log('🎨 Rendering with:', this.currentParams, 'gradient:', this.currentGradient?.name, 'canvas:', this.canvas.width, 'x', this.canvas.height);
    }
    
    // Draw
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  start(): void {
    if (this.animationFrameId !== null) {
      return; // Already running
    }

    const animate = (_timestamp: number) => {
      const dt = 0.016; // ~60 FPS
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
    }
  }

  dispose(): void {
    this.stop();
    const gl = this.gl;
    gl.deleteProgram(this.program);
  }
}
