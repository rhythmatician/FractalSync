/**
 * TOOL-themed gradient system for Julia set visualization.
 * Inspired by the band's aesthetic: dark, intentional, emotionally resonant.
 * 
 * Uses a single continuous gradient that smoothly transitions through
 * badass colors (blacks, blues, and reds) based on hue value.
 */

// Maximum gradient stops supported by WebGL shader
export const MAX_GRADIENT_STOPS = 8;

// Default hue value (mid-range for smooth transitions)
export const DEFAULT_HUE = 0.5;

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
 * @throws Error if hex string is invalid
 */
function hex2rgb(hex: string): [number, number, number] {
  // Validate hex format
  if (!hex || typeof hex !== 'string' || !/^#[0-9A-Fa-f]{6}$/.test(hex)) {
    throw new Error(`Invalid hex color: ${hex}. Expected format: #RRGGBB`);
  }
  
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return [r, g, b];
}

/**
 * Create gradient stops from hex colors
 * @param name - Display name for the gradient
 * @param description - Thematic description
 * @param colors - Array of hex color strings (e.g., ["#000000", "#ffffff"])
 * @returns Gradient object with interpolated stops
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
 * Single continuous gradient for smooth color transitions.
 * Spans full hue range [0, 1] with TOOL-appropriate colors:
 * blacks, deep blues, crimsons, and purples.
 * 
 * This gradient interpolates smoothly based on colorHue parameter,
 * creating natural transitions similar to HSV but confined to badass colors.
 */
export const CONTINUOUS_GRADIENT = createGradient(
  "TOOL Spectrum",
  "Continuous gradient spanning blacks, blues, and reds",
  [
    "#0a0a0a", // Pure black
    "#1a1428", // Dark blue-black
    "#2a2448", // Deep blue
    "#3a1a4a", // Purple-blue
    "#4a1a3a", // Deep purple-red
    "#5a1a2a", // Dark crimson
    "#6a2222", // Blood red
    "#3a1a1a", // Dark shadow (transition back)
  ]
);

/**
 * Get gradient for the current hue value.
 * Returns the continuous gradient that interpolates across the full hue range.
 * 
 * @param _hue - Hue value 0-1 (not used since we have a single continuous gradient)
 * @returns The continuous TOOL gradient
 */
export function getGradientByHue(_hue: number): Gradient {
  return CONTINUOUS_GRADIENT;
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
