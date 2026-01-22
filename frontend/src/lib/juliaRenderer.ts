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

    // Fragment shader (Julia set computation with gradient coloring)
    const fragmentShaderSource = `
      precision highp float;
      
      uniform vec2 u_juliaSeed;
      uniform float u_zoom;
      uniform vec3 u_color;  // x=hue, y=intensity, z=contrast
      uniform float u_time;
      uniform vec2 u_resolution;
      uniform int u_gradientCount;
      
      // Gradient stops - using individual uniforms for WebGL 1.0 compatibility
      uniform vec4 u_gradient0;
      uniform vec4 u_gradient1;
      uniform vec4 u_gradient2;
      uniform vec4 u_gradient3;
      uniform vec4 u_gradient4;
      uniform vec4 u_gradient5;
      uniform vec4 u_gradient6;
      uniform vec4 u_gradient7;
      
      const int MAX_ITERATIONS = 100;
      
      // Get gradient stop by index
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
      
      // Sample gradient at position t (0-1)
      vec3 sampleGradient(float t) {
        t = clamp(t, 0.0, 1.0);
        
        // Find the two stops we're between
        // Defaults to first and last stops if no match found (fallback for edge cases)
        vec4 lowerStop = getGradientStop(0);
        vec4 upperStop = getGradientStop(u_gradientCount - 1);
        
        // Loop through stops to find interpolation range
        // Note: hardcoded to MAX_GRADIENT_STOPS-1 for WebGL 1.0 compatibility
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
        
        // Interpolate between stops
        float range = upperStop.w - lowerStop.w;
        float localT = range > 0.0 ? (t - lowerStop.w) / range : 0.0;
        
        vec3 color = mix(lowerStop.rgb, upperStop.rgb, localT);
        return color;
      }
      
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
        // Fixed zoom: always show 2.7 unit range
        uv *= 2.7;
        
        vec2 c = u_juliaSeed;
        vec2 z = uv;
        
        // Orbit trap tracking for organic textures
        float minDist = 1000.0;  // Track closest approach to origin
        vec2 trapPoint = vec2(0.0, 0.0);  // Orbit trap center
        
        int iterations = 0;
        float smoothIter = 0.0;
        
        for (int i = 0; i < MAX_ITERATIONS; i++) {
          float zMagnitude = dot(z, z);
          if (zMagnitude > 4.0) {
            // Smooth coloring: calculate real iteration number using potential function
            // This eliminates banding by using fractional iterations
            smoothIter = float(i) - log2(log2(zMagnitude)) + 4.0;
            break;
          }
          
          // Orbit trap: track minimum distance to trap point
          float dist = length(z - trapPoint);
          minDist = min(minDist, dist);
          
          z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
          );
          iterations = i + 1;
        }
        
        // Normalize smooth iteration count
        if (iterations == MAX_ITERATIONS) {
          smoothIter = float(MAX_ITERATIONS);
        }
        float t = smoothIter / float(MAX_ITERATIONS);
        
        // Apply hue, intensity, and contrast
        float hue = u_color.x;        // colorHue shifts gradient position
        float intensity = u_color.y;  // colorSat controls intensity
        float contrast = u_color.z;   // colorBright controls contrast
        
        // Adjust t based on contrast; add small epsilon to avoid division by zero
        t = pow(t, 1.0 / (contrast + 0.5 + 0.001));
        
        // Blend orbit trap coloring for organic textures
        // Use orbit trap distance to create cloud-like patterns
        float trapInfluence = smoothstep(0.5, 0.0, minDist);
        t = mix(t, minDist * 2.0, trapInfluence * 0.3);
        
        // Shift t by hue for smooth color transitions across the gradient
        // This creates continuous color changes as hue varies
        t = fract(t + hue);
        
        // Sample gradient
        vec3 color = sampleGradient(t);
        
        // Apply intensity
        color = color * (0.5 + intensity * 0.5);
        
        // Lighting effect: simulate 3D depth using derivative-based normal
        // Only calculate lighting for points inside the set or near boundary
        if (iterations > 2 && iterations < MAX_ITERATIONS) {
          // Calculate approximate surface normal from potential gradient
          float h = 0.001;  // Small offset for derivative estimation
          float potentialCenter = log(dot(z, z)) * 0.5;
          
          // Estimate gradient using finite differences (approximates surface normal)
          vec2 dz = vec2(h, 0.0);
          float potentialX = log(dot(z + dz, z + dz)) * 0.5;
          dz = vec2(0.0, h);
          float potentialY = log(dot(z + dz, z + dz)) * 0.5;
          
          vec2 grad = vec2(potentialX - potentialCenter, potentialY - potentialCenter) / h;
          
          // Simple lighting: virtual light from upper-left
          vec3 lightDir = normalize(vec3(-1.0, -1.0, 0.5));
          vec3 normal = normalize(vec3(grad, 1.0));
          float lighting = max(dot(normal, lightDir), 0.0) * 0.5 + 0.5;
          
          // Apply lighting for 3D depth effect
          color *= lighting;
        }
        
        // Add subtle variation based on position for additional depth
        float depthFactor = 1.0 - t * 0.2;
        color *= depthFactor;
        
        gl_FragColor = vec4(color, 1.0);
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
