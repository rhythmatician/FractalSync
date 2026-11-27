use super::{GestureEpochId, GestureSample};
use num_complex::Complex64;

#[derive(Debug)]
pub struct GestureHarmonicMoment {
    pub epoch: GestureEpochId,
    pub support: usize,
    pub cross_moment: Complex64,
    pub audio_energy: f64,
    pub gesture_energy: f64,
    pub coherence: f64,
    /// Available only when every contributing gesture supplies an angle bound.
    pub estimated_cross_moment_angle_error: Option<f64>,
}

fn harmonic(value: Complex64, order: i32) -> Complex64 {
    let radius = value.norm();
    if radius == 0.0 {
        return value;
    }
    Complex64::from_polar(radius, f64::from(order) * value.arg())
}

/// Compare audio and transported motion within one epoch. Positive lag pairs
/// gesture frame i with audio frame i - lag. Weights index gesture frames.
/// Missing motion phases and zero weights remove a pair from all three moments.
pub fn compare_gesture_harmonics(
    audio: &[Complex64],
    gesture: &[GestureSample],
    weights: &[f64],
    orders: [i32; 2],
    lag: isize,
    minimum_support: usize,
) -> Result<Option<GestureHarmonicMoment>, &'static str> {
    if audio.len() != gesture.len()
        || gesture.len() != weights.len()
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || audio.iter().any(|a| !a.re.is_finite() || !a.im.is_finite())
    {
        return Err("invalid harmonic input");
    }
    let Some(first) = gesture.first() else {
        return Ok(None);
    };
    if gesture.iter().any(|sample| sample.id != first.id) {
        return Err("gesture frames must share one epoch and reference");
    }

    let mut total = 0.0;
    let mut support = 0;
    let mut cross = Complex64::default();
    let mut audio_energy = 0.0;
    let mut gesture_energy = 0.0;
    let mut angle_error = Some(0.0);
    for (i, (sample, &weight)) in gesture.iter().zip(weights).enumerate() {
        if weight == 0.0 {
            continue;
        }
        let Some(a) = i
            .checked_sub_signed(lag)
            .and_then(|j| audio.get(j))
            .copied()
        else {
            continue;
        };
        let Some([x, y]) = sample.motion_phasor else {
            continue;
        };
        if !x.is_finite()
            || !y.is_finite()
            || sample
                .angular_error_bound
                .is_some_and(|e| !e.is_finite() || e < 0.0)
        {
            return Err("invalid gesture phasor or angular error");
        }
        let b = Complex64::new(x, y);
        total += weight;
        support += 1;
        cross += weight * harmonic(a, orders[0]) * harmonic(b, orders[1]).conj();
        audio_energy += weight * a.norm_sqr();
        gesture_energy += weight * b.norm_sqr();
        angle_error = angle_error
            .zip(sample.angular_error_bound)
            .map(|(sum, bound)| {
                sum + weight
                    * a.norm()
                    * b.norm()
                    * (f64::from(orders[1].unsigned_abs()) * bound).min(2.0)
            });
    }
    if ![
        total,
        cross.re,
        cross.im,
        audio_energy,
        gesture_energy,
        angle_error.unwrap_or(0.0),
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        return Err("nonfinite harmonic accumulation");
    }
    if support < minimum_support.max(1) || audio_energy <= 0.0 || gesture_energy <= 0.0 {
        return Ok(None);
    }
    let coherence = (cross.norm() / audio_energy.sqrt() / gesture_energy.sqrt()).min(1.0);
    Ok(Some(GestureHarmonicMoment {
        epoch: first.id,
        support,
        cross_moment: cross / total,
        audio_energy: audio_energy / total,
        gesture_energy: gesture_energy / total,
        coherence,
        estimated_cross_moment_angle_error: angle_error.map(|bound| bound / total),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operational_engine::{MechanicalEndpoint, TransportedFrame};

    fn sample(value: Option<[f64; 2]>, epoch: u64) -> GestureSample {
        GestureSample {
            id: GestureEpochId {
                trajectory: 1,
                epoch,
                reference: 2,
            },
            endpoint: MechanicalEndpoint {
                position: [0.0; 2],
                velocity: [0.0; 2],
                mechanical_time: 0.0,
            },
            frame: TransportedFrame {
                axis: [1.0, 0.0],
                companion: [0.0, 1.0],
            },
            speed: value.map_or(0.0, |[x, y]| x.hypot(y)),
            motion_phasor: value,
            angular_error_bound: None,
        }
    }

    #[test]
    fn lag_and_missing_phase_share_one_weighted_support() {
        let audio = [1.0, 2.0, 10.0, 10.0].map(|x| Complex64::new(x, 0.0));
        let frames = [
            sample(Some([99.0, 0.0]), 1),
            sample(Some([2.0, 0.0]), 1),
            sample(None, 1),
            sample(Some([20.0, 0.0]), 1),
        ];
        let weights = [7.0, 1.0, 8.0, 3.0];
        let m = compare_gesture_harmonics(&audio, &frames, &weights, [1, 1], 1, 2)
            .unwrap()
            .unwrap();
        assert_eq!(m.support, 2);
        assert_eq!(m.cross_moment, Complex64::new(150.5, 0.0));
        assert_eq!(m.audio_energy, 75.25);
        assert_eq!(m.gesture_energy, 301.0);
        assert!((m.coherence - 1.0).abs() < 1e-14);
        assert!(m.estimated_cross_moment_angle_error.is_none());
        assert!(
            compare_gesture_harmonics(&audio, &frames, &weights, [1, 1], 1, 3)
                .unwrap()
                .is_none()
        );
        let mut mixed = frames;
        mixed[2].id.epoch += 1;
        assert!(compare_gesture_harmonics(&audio, &mixed, &weights, [1, 1], 1, 2).is_err());
    }

    #[test]
    fn constant_reference_rotation_changes_only_the_complex_orientation() {
        let a = [Complex64::new(1.0, 0.0)];
        let g = [sample(Some([1.0, 0.0]), 1)];
        let mut rotated = g;
        rotated[0].motion_phasor = Some([0.0, -1.0]);
        rotated[0].id.reference += 1;
        let measure = |frames: &[GestureSample]| {
            compare_gesture_harmonics(&a, frames, &[1.0], [1, -2], 0, 1)
                .unwrap()
                .unwrap()
        };
        let m = measure(&g);
        let r = measure(&rotated);
        assert!((r.cross_moment + m.cross_moment).norm() < 1e-14);
        assert_eq!(m.coherence, r.coherence);
    }
}
