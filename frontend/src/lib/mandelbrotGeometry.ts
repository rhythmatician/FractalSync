/**
 * Mandelbrot geometry utilities for orbit synthesis.
 * 
 * AUTO-GENERATED from backend/src/mandelbrot_orbits.py
 * DO NOT EDIT MANUALLY - Run: python backend/generate_frontend_code.py
 */

export class MandelbrotGeometry {
  /**
   * Compute position on lobe boundary at given angle.
   */
  static lobePointAtAngle(
    lobe: number,
    theta: number,
    s: number = 1.0,
    subLobe: number = 0
  ): { real: number; imag: number } {
    if (lobe === 1) {
      // Cardioid parametrization
      const r = 0.25 * (1 - Math.cos(theta));
      const phi = theta;
      return {
        real: s * r * Math.cos(phi),
        imag: s * r * Math.sin(phi)
      };
    } else {
      // Period-n bulb
      const center = this.periodNBulbCenter(lobe, subLobe);
      const radius = this.periodNBulbRadius(lobe, subLobe);
      
      return {
        real: center.real + s * radius * Math.cos(theta),
        imag: center.imag + s * radius * Math.sin(theta)
      };
    }
  }

  /**
   * Compute tangent vector at given angle (for velocity computation).
   */
  static lobeTangentAtAngle(
    lobe: number,
    theta: number,
    s: number = 1.0,
    subLobe: number = 0
  ): { real: number; imag: number } {
    if (lobe === 1) {
      // Cardioid: d/dθ of (r*cos(φ), r*sin(φ))
      const r = 0.25 * (1 - Math.cos(theta));
      const drDtheta = 0.25 * Math.sin(theta);
      
      return {
        real: s * (drDtheta * Math.cos(theta) - r * Math.sin(theta)),
        imag: s * (drDtheta * Math.sin(theta) + r * Math.cos(theta))
      };
    } else {
      // Period-n bulb: circular, so tangent is perpendicular
      const radius = this.periodNBulbRadius(lobe, subLobe);
      return {
        real: -s * radius * Math.sin(theta),
        imag: s * radius * Math.cos(theta)
      };
    }
  }

  /**
   * Get center of period-n bulb.
   */
  static periodNBulbCenter(n: number, k: number = 0): { real: number; imag: number } {
    if (n === 1) {
      return { real: 0, imag: 0 };
    }

    // Hardcoded centers for common periods
    if (n === 2 && k === 0) {
      return { real: -1.0, imag: 0.0 };
    }
    if (n === 3 && k === 0) {
      return { real: -0.125, imag: 0.649519 };
    }
    if (n === 3 && k === 1) {
      return { real: -0.125, imag: -0.649519 };
    }

    // Approximate formula for other periods
    const angle = (2 * Math.PI * k) / n;
    const r = 0.25 * (1 - Math.cos(angle));
    return {
      real: r * Math.cos(angle),
      imag: r * Math.sin(angle)
    };
  }

  /**
   * Get radius of period-n bulb.
   */
  static periodNBulbRadius(n: number, _k: number = 0): number {
    if (n === 1) {
      return 0.25;
    }
    if (n === 2) {
      return 0.25;
    }
    if (n === 3) {
      return 0.0943;
    }
    if (n === 4) {
      return 0.04;
    }
    
    return 0.25 / (n * n);
  }
}
