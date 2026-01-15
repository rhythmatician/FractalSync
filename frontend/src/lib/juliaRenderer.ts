/**
 * WebGL-based Julia set renderer for real-time visualization.
 */

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

  // Uniform locations
  private uJuliaSeedLocation: WebGLUniformLocation | null = null;
  private uZoomLocation: WebGLUniformLocation | null = null;
  private uColorLocation: WebGLUniformLocation | null = null;
  private uTimeLocation: WebGLUniformLocation | null = null;
  private uResolutionLocation: WebGLUniformLocation | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl');
    if (!gl) {
      throw new Error('WebGL not supported');
    }
    this.gl = gl;

    // Initialize default parameters
    this.currentParams = {
      juliaReal: -0.7269,
      juliaImag: 0.1889,
      colorHue: 0.5,
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

    // Fragment shader (Julia set computation)
    const fragmentShaderSource = `
      precision highp float;
      
      uniform vec2 u_juliaSeed;
      uniform float u_zoom;
      uniform vec3 u_color;
      uniform float u_time;
      uniform vec2 u_resolution;
      
      const int MAX_ITERATIONS = 100;
      
      vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
      }
      
      void main() {
        vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);
        // Higher zoom = more zoomed OUT (see more of the fractal)
        uv *= 2.5 / u_zoom;  // Invert zoom so 1.0 = reasonable view
        
        vec2 c = u_juliaSeed;
        vec2 z = uv;
        
        int iterations = 0;
        for (int i = 0; i < MAX_ITERATIONS; i++) {
          if (dot(z, z) > 4.0) break;
          z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            2.0 * z.x * z.y + c.y
          );
          iterations++;
        }
        
        float t = float(iterations) / float(MAX_ITERATIONS);
        
        // Color mapping using HSV
        vec3 hsv = vec3(
          u_color.x + t * 0.3,  // Hue
          u_color.y,              // Saturation
          u_color.z * (1.0 - t * 0.5)  // Brightness
        );
        
        vec3 color = hsv2rgb(hsv);
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
    
    console.log('🔧 Canvas resize:', { displayWidth, displayHeight, clientW: canvas.clientWidth, clientH: canvas.clientHeight });
    
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
      gl.viewport(0, 0, displayWidth, displayHeight);
      console.log('✅ Canvas resized to:', canvas.width, 'x', canvas.height);
    }
  }

  updateParameters(params: VisualParameters): void {
    console.log('📊 Updating renderer with params:', params);
    this.targetParams = { ...params };
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
    
    // Debug log occasionally
    if (Math.random() < 0.01) {
      console.log('🎨 Rendering with:', this.currentParams, 'canvas:', this.canvas.width, 'x', this.canvas.height);
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
