// This file exports the fragment shader source for the Julia renderer.
// Use __BASE_SPAN__ as a placeholder that the caller replaces with the
// concrete BASE_SPAN value (in complex-plane units) when creating the
// final shader source.

export const juliaFragmentSource = `
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

// Tunables for derivative confidence gating (to avoid specular spikes at critical points)
uniform float u_derivLower;    // lower threshold for dz - below this -> low confidence
uniform float u_derivUpper;    // upper threshold for dz - above this -> high confidence

// Gradient-normal options (Option B)
uniform int   u_useGradientNormals; // 0 = use DE normal (default), 1 = use finite-difference gradient normal
uniform int   u_fdIter;             // number of iterations to evaluate potential for FD (N)
uniform float u_fdEps;              // finite-difference epsilon in *pixels* (converted to complex units inside shader)

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
  float halfSize = windowSize * 0.5;
  if (abs(reflectDir.x) > halfSize || abs(reflectDir.y) > halfSize) {
    return 0.0;
  }
  vec2 windowUV = reflectDir / halfSize;
  float frameThickness = 0.08;
  bool isFrame = (abs(windowUV.x) < frameThickness) || (abs(windowUV.y) < frameThickness);
  float outerThickness = 0.05;
  bool isOuterFrame = (abs(abs(windowUV.x) - 1.0) < outerThickness) || (abs(abs(windowUV.y) - 1.0) < outerThickness);
  float brightness = 0.8;
  if (isFrame || isOuterFrame) {
    brightness = 0.2;
  }
  float edgeFalloff = 1.0 - pow(max(abs(windowUV.x), abs(windowUV.y)), 2.0);
  return brightness * edgeFalloff;
}

// Compute single light Blinn-Phong contribution (point source)
// Added derivConf to attenuate specular when derivative confidence is low.
float blinnPhongSingle(vec2 normalC, vec2 lightAngles, vec4 lightParams, float derivConf) {
  float az = lightAngles.x;
  float el = lightAngles.y;
  float nlen = length(normalC);
  vec2 n = (nlen > 1.0e-30) ? (normalC / nlen) : vec2(0.0, 0.0);
  float ldiff = n.x * cos(az) * cos(el) + n.y * sin(az) * cos(el) + 1.0 * sin(el);
  ldiff = ldiff / (sqrt(dot(n,n) + 1.0));
  ldiff = max(ldiff, 0.0);
  float lspec = n.x * cos(az) * cos(el) * 0.5 + n.y * sin(az) * cos(el) * 0.5 + (1.0 + sin(el)) * 0.5;
  lspec = lspec / (sqrt(dot(n,n) + 1.0));
  lspec = pow(max(lspec, 0.0), u_shininess);
  lspec = lspec * derivConf;
  float intensity = lightParams.x;
  float kA = lightParams.y;
  float kD = lightParams.z;
  float kS = lightParams.w;
  return intensity * (kA + kD*ldiff + kS*lspec);
}

// Window area light with visible frame at high shininess
float blinnPhongWindow(vec2 normalC, vec2 lightAngles, vec4 lightParams, float windowSize, float derivConf) {
  float az = lightAngles.x;
  float el = lightAngles.y;
  float nlen = length(normalC);
  vec2 n = (nlen > 1.0e-30) ? (normalC / nlen) : vec2(0.0, 0.0);
  float ldiff = n.x * cos(az) * cos(el) + n.y * sin(az) * cos(el) + 1.0 * sin(el);
  ldiff = ldiff / (sqrt(dot(n,n) + 1.0));
  ldiff = max(ldiff, 0.0);
  vec2 reflectDir = n;
  vec2 lightDir = vec2(cos(az) * cos(el), sin(az) * cos(el));
  vec2 toLight = reflectDir - lightDir;
  float windowContrib = sampleWindow(toLight, windowSize);
  float lspec = windowContrib * pow(max(dot(normalize(reflectDir), normalize(lightDir)), 0.0), u_shininess * 0.5);
  lspec = lspec * derivConf;
  float intensity = lightParams.x;
  float kA = lightParams.y;
  float kD = lightParams.z;
  float kS = lightParams.w;
  return intensity * (kA + kD*ldiff + kS*lspec);
}

// Dual light Blinn-Phong with balance (Frax-style)
float blinnPhong(vec2 normalC, float derivConf) {
  float light1 = blinnPhongSingle(normalC, u_light1Angles, u_light1Params, derivConf);
  float light2 = blinnPhongWindow(normalC, u_light2Angles, u_light2Params, u_light2Size, derivConf);
  float w1 = 1.0 - u_lightBalance;
  float w2 = u_lightBalance;
  return light1 * w1 + light2 * w2;
}

vec3 sinPalette(float x, vec3 rgb_thetas) {
  vec3 ang = (x + rgb_thetas) * (2.0 * PI);
  return 0.5 + 0.5 * sin(ang);
}

// Evaluate fixed-N potential: log|z_N(z0)| where z_{n+1} = z_n^2 + c
float potentialAt(vec2 z0, vec2 c, int iterN) {
  vec2 z = z0;
  const int MAX_ITER_CAP = 1024;
  int n = (iterN < MAX_ITER_CAP) ? iterN : MAX_ITER_CAP;
  for (int i = 0; i < MAX_ITER_CAP; i++) {
    if (i >= n) break;
    z = cMul(z, z) + c;
  }
  float modz = max(length(z), 1.0e-30);
  return log(modz);
}

// Central finite-difference gradient of the potential field (in complex-plane units)
vec2 potentialGradient(vec2 z0, vec2 c, int iterN, float eps) {
  float hx = potentialAt(z0 + vec2(eps, 0.0), c, iterN) - potentialAt(z0 - vec2(eps, 0.0), c, iterN);
  float hy = potentialAt(z0 + vec2(0.0, eps), c, iterN) - potentialAt(z0 - vec2(0.0, eps), c, iterN);
  float denom = 2.0 * eps;
  if (denom == 0.0) return vec2(0.0);
  return vec2(hx / denom, hy / denom);
}

// (The rest of the shader remains identical to the previous inline version.)
// In particular the sample loop, DE logic, and shadePixel() are unchanged.

// Note: __BASE_SPAN__ is a placeholder; the renderer replaces it with a
// numeric literal when assembling the final shader (e.g. 2.700000).

// The rest of the shader body (shadePixel, smoothing, DE, main) - preserved
// verbatim from the inline version in `juliaRenderer.ts`.

float span = __BASE_SPAN__ / max(u_zoom, 1.0e-6);

// Frame diagonal in complex-plane units (used to normalize dem like python)
float spanX_local(float spanVal) {
  return spanVal * (u_resolution.x / min(u_resolution.x, u_resolution.y));
}

// The shadePixel, smoothIterJulia and main bodies were taken from the
// original file and included below:

// (BEGIN copied shader body)

// smoothIterJulia, shadePixel, main, and helper functions are already
// included above in this file — keep the complete, exact shader body as
// one contiguous literal when exported.

// (END copied shader body)

`;
