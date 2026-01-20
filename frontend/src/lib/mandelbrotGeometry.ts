/**
 * Mandelbrot geometry utilities for orbit synthesis.
 * 
 * TypeScript port of backend/src/mandelbrot_orbits.py geometric functions.
 * This ensures the frontend uses the same authoritative geometric calculations.
 */

export class MandelbrotGeometry {
  /**
   * Compute position on lobe boundary at given angle.
   * 
   * @param lobe Period number (1=cardioid, 2=period-2, etc.)
   * @param theta Angle in radians
   * @param s Radius scaling factor (1.0 = boundary)
   * @param subLobe Sub-lobe index
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
      // r = 0.25*(1 - cos(θ)), φ = θ
      // dr/dθ = 0.25*sin(θ)
      const r = 0.25 * (1 - Math.cos(theta));
      const drDtheta = 0.25 * Math.sin(theta);
      
      // d/dθ[r*cos(θ)] = dr/dθ*cos(θ) - r*sin(θ)
      // d/dθ[r*sin(θ)] = dr/dθ*sin(θ) + r*cos(θ)
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
      return { real: 0, imag: 0 }; // Cardioid center
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
      return 0.25; // Cardioid reference
    }
    if (n === 2) {
      return 0.25;
    }
    if (n === 3) {
      return 0.0943; // Hardcoded for period-3
    }
    if (n === 4) {
      return 0.04; // Approximate for period-4
    }
    
    // General approximation
    return 0.25 / (n * n);
  }

  /**
   * Euler's totient function.
   */
  static eulerTotient(n: number): number {
    let result = n;
    let p = 2;
    
    while (p * p <= n) {
      if (n % p === 0) {
        while (n % p === 0) {
          n = Math.floor(n / p);
        }
        result -= Math.floor(result / p);
      }
      p++;
    }
    
    if (n > 1) {
      result -= Math.floor(result / n);
    }
    
    return result;
  }

  /**
   * Get maximum sub-lobe index for a given period.
   */
  static getMaxSubLobeForPeriod(n: number): number {
    if (n === 1) return 0; // Cardioid has no sub-lobes
    return this.eulerTotient(n) - 1;
  }
}
