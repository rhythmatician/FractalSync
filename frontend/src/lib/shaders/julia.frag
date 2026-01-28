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

// Height scale controls the z component when building a 3D normal from the 2D gradient
uniform float u_heightScale;

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
// Added derivConf to attenuate specular when derivative confidence is low.
float blinnPhongSingle(vec2 normalC, vec2 lightAngles, vec4 lightParams, float derivConf) {
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

  // Attenuate specular by derivative confidence to avoid spikes
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

  // Attenuate specular by derivative confidence
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

// Returns:
//   out_niter  : smooth escape iteration (0 if bounded)
//   out_stripe : stripe average [0,1]
//   out_dem    : distance estimate (un-normalized; caller normalizes by diag)
//   out_normal : complex normal for lighting
//   out_derivConf : derivative confidence [0,1]
void smoothIterJulia(
  vec2 z0,
  vec2 c,
  int maxIter,
  float stripe_s,
  float stripe_sig,
  out float out_niter,
  out float out_stripe,
  out float out_dem,
  out vec2  out_normal,
  out float out_derivConf
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
  out_derivConf = 0.0;

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
      // Protect against critical-orbit cases where dz becomes (near) zero
      float dzAbs = max(length(dz), 1.0e-30);
      float SAFE_DZ = 1.0e-6;
      bool dzSmall = dzAbs < SAFE_DZ;

      vec2 u;
      if (dzSmall) {
        if (modz > 1.0e-30) {
          u = z / modz; // normalize(z)
        } else {
          u = vec2(1.0, 0.0);
        }
      } else {
        u = cDiv(z, dz);
      }

      // Milton distance estimator
      float demRaw = 0.0;
      if (!dzSmall) {
        demRaw = modz * log(modz) / dzAbs / 2.0;
      }

      float DEM_FALLBACK = 1.0e6;
      float dem = (dzSmall || demRaw <= 0.0) ? DEM_FALLBACK : demRaw;

      // Derivative confidence: smoothstep between configured bounds
      out_derivConf = smoothstep(u_derivLower, u_derivUpper, dzAbs);

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

  float span = __BASE_SPAN__ / max(u_zoom, 1.0e-6);
  vec2 z0 = uv * span;

  // Frame diagonal in complex-plane units (used to normalize dem like python)
  float spanX = span * (u_resolution.x / minRes);
  float spanY = span * (u_resolution.y / minRes);
  float diag = sqrt(spanX*spanX + spanY*spanY);

  float niter;
  float stripe_a;
  float dem;
  vec2  normal;
  float derivConf;

  smoothIterJulia(
    z0,
    u_juliaSeed,
    u_maxIter,
    u_stripe_s,
    u_stripe_sig,
    niter,
    stripe_a,
    dem,
    normal,
    derivConf
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

  // Choose normal: either DE-derived normal or finite-difference gradient normal
  vec2 usedNormal = normal;
  if (u_useGradientNormals != 0) {
    // convert FD epsilon from pixels to complex-plane units
    float pixelSize = span / minRes;
    float eps = max(u_fdEps * pixelSize, 1.0e-7);
    vec2 grad = potentialGradient(z0, u_juliaSeed, u_fdIter, eps);
    float glen = length(grad);
    // If gradient is tiny (or NaN), fall back to DE normal
    if (glen > 1.0e-30) {
      // Build a 3D normal: (grad.x, grad.y, heightScale) then normalize
      float hs = max(u_heightScale, 1.0e-8);
      vec3 N3 = normalize(vec3(grad.x, grad.y, hs));
      usedNormal = N3.xy;
      // Protect against pathological numeric cases
      if (length(usedNormal) < 1.0e-6) {
        usedNormal = normal;
      }
    } else {
      usedNormal = normal;
    }
  }

  // Brightness with Blinn-Phong (scale specular by derivative confidence)
  float bright = blinnPhong(usedNormal, derivConf);

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