/**
 * TOOL-themed gradient system for Julia set visualization.
 * Inspired by the band's aesthetic: dark, intentional, emotionally resonant.
 * 
 * Each gradient is designed to evoke specific moods from TOOL's music:
 * - Introspective depth
 * - Industrial decay
 * - Cosmic transcendence
 * - Raw intensity
 */

// Maximum gradient stops supported by WebGL shader
export const MAX_GRADIENT_STOPS = 8;

export interface GradientStop {
  position: number; // 0.0 to 1.0
  r: number; // 0.0 to 1.0
  g: number; // 0.0 to 1.0
  b: number; // 0.0 to 1.0
}

export interface Gradient {
  name: string;
  description: string;
  stops: GradientStop[];
}

/**
 * Convert hex color to RGB components (0-1 range)
 */
function hex2rgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}

/**
 * Create gradient stops from hex colors
 */
function createGradient(
  name: string,
  description: string,
  colors: string[]
): Gradient {
  const stops: GradientStop[] = colors.map((color, i) => {
    const [r, g, b] = hex2rgb(color);
    return {
      position: i / (colors.length - 1),
      r,
      g,
      b,
    };
  });
  return { name, description, stops };
}

/**
 * TOOL-themed gradient families
 * Each gradient evokes specific moods from the band's aesthetic
 */
export const TOOL_GRADIENTS: Gradient[] = [
  // 1. Flesh Tones & Bronze - Organic, ritualistic, primal
  createGradient(
    "Flesh & Bronze",
    "Organic ritualistic feel - primal connection",
    [
      "#1a1410", // Deep shadow
      "#3d2817", // Dark bronze
      "#6b4423", // Rich bronze
      "#8b6239", // Warm bronze
      "#a67c52", // Light bronze
      "#c4997a", // Flesh tone
      "#d4b5a0", // Pale flesh
    ]
  ),

  // 2. Deep Ocean Blues to Violet - Introspective, watery depth
  createGradient(
    "Ocean Depths",
    "Introspective watery depth - drowning in thought",
    [
      "#050a12", // Near black
      "#0d1a2d", // Deep ocean
      "#1a2f4a", // Dark water
      "#274663", // Mid ocean
      "#335d7c", // Ocean blue
      "#4a6d8a", // Light blue
      "#5d7a9a", // Blue-violet
      "#7086a8", // Pale violet
    ]
  ),

  // 3. Rust & Decay - Industrial decay, mechanical breakdown
  createGradient(
    "Rust & Decay",
    "Industrial decay aesthetic - entropy and breakdown",
    [
      "#1a1412", // Dark decay
      "#2d1f17", // Rust shadow
      "#4a2f1f", // Deep rust
      "#6b422a", // Rust
      "#8d5436", // Orange rust
      "#a96844", // Light rust
      "#c4825a", // Oxidized metal
    ]
  ),

  // 4. Blacks & Charcoal with Crimson - Darkness with intensity bursts
  createGradient(
    "Shadow & Blood",
    "Darkness with crimson bursts - controlled violence",
    [
      "#0a0a0a", // Pure black
      "#1a1a1a", // Charcoal
      "#2a2424", // Dark gray
      "#3d2a2a", // Warm shadow
      "#5a2a2a", // Deep crimson
      "#7a2222", // Blood red
      "#9a3333", // Crimson
      "#b44444", // Bright crimson
    ]
  ),

  // 5. Cosmic Purple to Deep Space - Psychedelic, otherworldly
  createGradient(
    "Cosmic Purple",
    "Psychedelic otherworldly journey - transcendence",
    [
      "#0a0512", // Deep space
      "#150a24", // Void purple
      "#24124a", // Deep purple
      "#331a63", // Royal purple
      "#4a2477", // Purple
      "#5d2f8a", // Bright purple
      "#7a4aa0", // Light purple
      "#8d5fb0", // Lavender purple
    ]
  ),

  // 6. Amber & Gold with Shadow - Fire through smoke
  createGradient(
    "Amber Fire",
    "Fire through smoke - smoldering intensity",
    [
      "#1a1205", // Smoke shadow
      "#2d1f0a", // Dark smoke
      "#4a3312", // Brown smoke
      "#6b4a1a", // Deep amber
      "#8d6324", // Amber
      "#b08030", // Gold amber
      "#d4a040", // Bright gold
      "#f0c060", // Pale gold
    ]
  ),

  // 7. Bone White to Slate Gray - Minimalist, stark contrast
  createGradient(
    "Bone & Stone",
    "Minimalist stark contrast - skeletal structure",
    [
      "#121212", // Near black
      "#242424", // Dark slate
      "#3d3d3d", // Slate
      "#5a5a5a", // Mid gray
      "#7a7a7a", // Light gray
      "#9a9a9a", // Pale gray
      "#c4c4c4", // Bone
      "#e0e0e0", // Bright bone
    ]
  ),
];

/**
 * Get gradient by index (wraps around if out of bounds)
 */
export function getGradient(index: number): Gradient {
  // Handle negative indices and wrap around
  const wrappedIndex = ((index % TOOL_GRADIENTS.length) + TOOL_GRADIENTS.length) % TOOL_GRADIENTS.length;
  return TOOL_GRADIENTS[wrappedIndex];
}

/**
 * Get gradient by hue value (0-1 mapped to gradient families)
 */
export function getGradientByHue(hue: number): Gradient {
  // Map hue 0-1 to gradient indices
  const index = Math.floor(hue * TOOL_GRADIENTS.length);
  return getGradient(index);
}

/**
 * Sample gradient at a specific position (0-1)
 */
export function sampleGradient(gradient: Gradient, position: number): [number, number, number] {
  // Clamp position to [0, 1]
  const t = Math.max(0, Math.min(1, position));
  
  // Find the two stops we're between
  let lowerStop = gradient.stops[0];
  let upperStop = gradient.stops[gradient.stops.length - 1];
  
  for (let i = 0; i < gradient.stops.length - 1; i++) {
    if (t >= gradient.stops[i].position && t <= gradient.stops[i + 1].position) {
      lowerStop = gradient.stops[i];
      upperStop = gradient.stops[i + 1];
      break;
    }
  }
  
  // Interpolate between stops
  const range = upperStop.position - lowerStop.position;
  const localT = range > 0 ? (t - lowerStop.position) / range : 0;
  
  const r = lowerStop.r + (upperStop.r - lowerStop.r) * localT;
  const g = lowerStop.g + (upperStop.g - lowerStop.g) * localT;
  const b = lowerStop.b + (upperStop.b - lowerStop.b) * localT;
  
  return [r, g, b];
}

/**
 * Flatten gradient to array for WebGL uniform (for use in shader)
 * Returns [r1, g1, b1, pos1, r2, g2, b2, pos2, ...]
 */
export function flattenGradient(gradient: Gradient): number[] {
  const result: number[] = [];
  for (const stop of gradient.stops) {
    result.push(stop.r, stop.g, stop.b, stop.position);
  }
  return result;
}
