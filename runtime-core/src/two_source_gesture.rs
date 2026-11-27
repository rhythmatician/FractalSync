//! Experimental two-source gesture in an intrinsic orthonormal frame.
//!
//! This computes an intended displacement, not a physically tracked position.
//! The caller supplies phases; it does not infer musical roles or confidence.

/// The contribution of one selected temporal mode.
#[derive(Clone, Copy, Debug)]
pub struct Source {
    /// Phase in radians, including any deliberately chosen offset.
    pub phase: f64,
    /// Nonnegative contribution strength.
    pub gain: f64,
}

/// A frequency change in the known-phase listening passage.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PhaseSegment {
    pub start_seconds: f64,
    pub frequency_hz: f64,
}

/// The second pulse speeds up, slows down to recover its reference, then locks.
pub const KNOWN_A_HZ: f64 = 1.5;
pub const KNOWN_B_SEGMENTS: [PhaseSegment; 4] = [
    PhaseSegment {
        start_seconds: 0.0,
        frequency_hz: 2.0,
    },
    PhaseSegment {
        start_seconds: 8.0,
        frequency_hz: 2.25,
    },
    PhaseSegment {
        start_seconds: 16.0,
        frequency_hz: 1.75,
    },
    PhaseSegment {
        start_seconds: 24.0,
        frequency_hz: 2.0,
    },
];

/// Phases of the audible sine modulators, referenced to their common start.
pub fn known_phase(seconds: f64) -> Result<(f64, f64), &'static str> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err("passage time must be finite and nonnegative");
    }
    let mut cycles_b = 0.0;
    for (index, segment) in KNOWN_B_SEGMENTS.iter().enumerate() {
        let end = KNOWN_B_SEGMENTS
            .get(index + 1)
            .map_or(seconds, |next| next.start_seconds.min(seconds));
        cycles_b += (end - segment.start_seconds).max(0.0) * segment.frequency_hz;
    }
    let phase = |cycles: f64| std::f64::consts::TAU * cycles - std::f64::consts::FRAC_PI_2;
    let result = (phase(KNOWN_A_HZ * seconds), phase(cycles_b));
    if result.0.is_finite() && result.1.is_finite() {
        Ok(result)
    } else {
        Err("passage phase overflow")
    }
}

/// Comparison with B's departure and recovery occurring eight seconds early.
/// At time zero the phases coincide modulo a full cycle.
pub fn known_phase_misaligned(seconds: f64) -> Result<(f64, f64), &'static str> {
    let (a, _) = known_phase(seconds)?;
    let (_, b) = known_phase(seconds + 8.0)?;
    Ok((a, b))
}

/// The two interpretations used in the initial perceptual comparison.
#[derive(Clone, Copy, Debug)]
pub enum Geometry {
    /// Each mode moves along its own axis.
    Distinct,
    /// Both modes rotate in the same plane and can interfere.
    Rotating,
}

#[derive(Clone, Copy, Debug)]
pub struct Gesture {
    pub geometry: Geometry,
    /// Scale of the combined intrinsic displacement.
    pub excursion: f64,
}

impl Gesture {
    /// Intended displacement in a metric-orthonormal tangent frame.
    pub fn displacement(&self, a: Source, b: Source) -> Result<[f64; 2], &'static str> {
        if !self.excursion.is_finite()
            || self.excursion < 0.0
            || !a.phase.is_finite()
            || !b.phase.is_finite()
            || !a.gain.is_finite()
            || a.gain < 0.0
            || !b.gain.is_finite()
            || b.gain < 0.0
        {
            return Err("gesture inputs must be finite and gains and excursion nonnegative");
        }
        let result = match self.geometry {
            Geometry::Distinct => [a.gain * a.phase.cos(), b.gain * b.phase.cos()],
            Geometry::Rotating => [
                a.gain * a.phase.cos() + b.gain * b.phase.cos(),
                a.gain * a.phase.sin() + b.gain * b.phase.sin(),
            ],
        };
        let result = [result[0] * self.excursion, result[1] * self.excursion];
        if result.iter().all(|x| x.is_finite()) {
            Ok(result)
        } else {
            Err("gesture displacement overflow")
        }
    }

    /// First-order local map using the metric at `anchor`.
    ///
    /// The frame is world-axis aligned by Cholesky decomposition. This is
    /// not an exponential map and makes no claim about large displacements.
    pub fn local_target(
        &self,
        anchor: [f64; 2],
        metric: [[f64; 2]; 2],
        a: Source,
        b: Source,
    ) -> Result<[f64; 2], &'static str> {
        if !anchor.iter().all(|v| v.is_finite())
            || !metric.iter().flatten().all(|v| v.is_finite())
            || metric[0][1] != metric[1][0]
            || metric[0][0] <= 0.0
        {
            return Err("anchor and symmetric positive definite metric required");
        }
        let l00 = metric[0][0].sqrt();
        let l10 = metric[1][0] / l00;
        let remainder = metric[1][1] - l10 * l10;
        if !remainder.is_finite() || remainder <= 0.0 {
            return Err("anchor and symmetric positive definite metric required");
        }
        let [u, v] = self.displacement(a, b)?;
        let y = v / remainder.sqrt();
        let x = (u - l10 * y) / l00;
        let target = [anchor[0] + x, anchor[1] + y];
        if target.iter().all(|v| v.is_finite()) {
            Ok(target)
        } else {
            Err("gesture target overflow")
        }
    }
}
