// Render into a separate canvas so the 2D trail and camera overlays can
// keep using the same coordinate system and the CPU renderer can fall back.
export function createMinimapBackground() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
  if (!gl || (gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT)?.precision ?? 0) < 23) return null;

  const program = gl.createProgram();
  if (!program) return null;
  const sources: Array<[number, string]> = [
    [gl.VERTEX_SHADER, `#version 300 es
      void main() {
        vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
      }`],
    [gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      uniform vec2 center;
      uniform float halfExtent;
      uniform float size;
      out vec4 color;
      void main() {
        vec2 c = center + ((gl_FragCoord.xy - 0.5) / (size - 1.0) * 2.0 - 1.0) * halfExtent;
        vec2 z = vec2(0.0);
        int i = 0;
        for (; i < 192; ++i) {
          if (dot(z, z) >= 4.0) break;
          z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        }
        float v = 1.0 - float(i) / 192.0;
        v *= v;
        vec3 rgb = i == 192 ? vec3(8, 14, 36)
          : vec3(25.0 + 230.0*v, 55.0 + 180.0*v, 80.0 + 70.0*(1.0-v) + 105.0*v*v);
        color = vec4(rgb / 255.0, 1.0);
      }`],
  ];
  for (const [type, source] of sources) {
    const shader = gl.createShader(type);
    if (!shader) {
      gl.deleteProgram(program);
      return null;
    }
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    const compiled = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (compiled) gl.attachShader(program, shader);
    gl.deleteShader(shader);
    if (!compiled) {
      gl.deleteProgram(program);
      return null;
    }
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  const centerLocation = gl.getUniformLocation(program, 'center');
  const halfLocation = gl.getUniformLocation(program, 'halfExtent');
  const sizeLocation = gl.getUniformLocation(program, 'size');

  return (ctx: CanvasRenderingContext2D, size: number, center: [number, number], half: number) => {
    // Adjacent samples need room to round independently in float32.
    const largestCoordinate = Math.max(1, Math.abs(center[0]) + half, Math.abs(center[1]) + half);
    if (size < 2 || 2 * half / (size - 1) < largestCoordinate * 2 ** -22 || gl.isContextLost()) return false;
    if (canvas.width !== size) canvas.width = size;
    if (canvas.height !== size) canvas.height = size;
    gl.viewport(0, 0, size, size);
    gl.useProgram(program);
    gl.uniform2f(centerLocation, ...center);
    gl.uniform1f(halfLocation, half);
    gl.uniform1f(sizeLocation, size);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    ctx.drawImage(canvas, 0, 0);
    return true;
  };
}
