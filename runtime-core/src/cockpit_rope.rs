//! Experimental planar tether for the manual cockpit.

#[derive(Clone, Default)]
pub struct CockpitRope {
    arm: [f64; 2],
    ball: [f64; 2],
    velocity: [f64; 2],
    length: f64,
    held: bool,
    paying_out: bool,
    reeling: bool,
    origin: Option<[f64; 2]>,
}

impl CockpitRope {
    pub fn positions(&self) -> [f64; 4] {
        [self.arm[0], self.arm[1], self.ball[0], self.ball[1]]
    }

    /// Points in a flat slice of a rider-centered Poincaré ball,
    /// mapped back to the shared upper half-space.
    pub fn rider_space_points(&self, rider: [f64; 2], z: f64) -> [f64; 6] {
        let inverse = |offset: [f64; 2]| {
            // A factor of two in the inverse map matches the original
            // horizontal scale near the rider when z = rho.
            let radius = 0.09 * offset[0].hypot(offset[1]);
            let scale = 0.09 / (radius / 0.95).max(1.0);
            let bx = scale * offset[0];
            let by = scale * offset[1];
            let bz = -0.023_f64;
            let norm2 = bx * bx + by * by + bz * bz;
            let denom = bx * bx + by * by + (1.0 - bz).powi(2);
            [
                rider[0] + z * 2.0 * bx / denom,
                rider[1] + z * 2.0 * by / denom,
                z * (1.0 - norm2) / denom,
            ]
        };
        let [hx, hy, hz] = inverse(self.arm);
        let [bx, by, bz] = inverse(self.ball);
        [hx, hy, hz, bx, by, bz]
    }

    pub fn step(
        &mut self,
        rider: [f64; 2],
        rho: f64,
        aim: [f64; 2],
        held: bool,
        reel: bool,
        dt: f64,
    ) -> [f64; 2] {
        self.step_analog(
            rider,
            rho,
            aim,
            u8::from(held) as f64,
            u8::from(reel) as f64,
            dt,
        )
    }

    pub fn state(&self) -> [f64; 7] {
        [
            self.arm[0],
            self.arm[1],
            self.ball[0],
            self.ball[1],
            self.velocity[0],
            self.velocity[1],
            self.length,
        ]
    }

    /// Grip resists outward slip. Reel sets the fraction of full retrieval speed.
    pub fn step_analog(
        &mut self,
        rider: [f64; 2],
        rho: f64,
        aim: [f64; 2],
        grip: f64,
        reel_rate: f64,
        dt: f64,
    ) -> [f64; 2] {
        let grip = grip.clamp(0.0, 1.0);
        let reel_rate = reel_rate.clamp(0.0, 1.0);
        if !grip.is_finite()
            || !reel_rate.is_finite()
            || !dt.is_finite()
            || !rider
                .iter()
                .chain(aim.iter())
                .all(|value| value.is_finite())
        {
            return rider;
        }
        let held = grip >= 0.98;
        let reel = reel_rate > 0.0;
        let scale = rho * 0.18;
        if !(scale.is_finite() && scale > 0.0 && dt > 0.0) {
            return rider;
        }
        let old = self.origin.replace(rider);
        if let Some(old) = old {
            for axis in 0..2 {
                self.ball[axis] -= (rider[axis] - old[axis]) / scale;
            }
        } else {
            self.ball = [2.5, 0.0];
            self.length = 2.5;
        }
        let reach = aim[0].hypot(aim[1]).max(1.0);
        let delta = [
            1.4 * aim[0] / reach - self.arm[0],
            1.4 * aim[1] / reach - self.arm[1],
        ];
        let gap = delta[0].hypot(delta[1]);
        let fraction = if gap > 0.0 {
            (6.0 * dt / gap).min(1.0)
        } else {
            0.0
        };
        let distance = (self.ball[0] - self.arm[0]).hypot(self.ball[1] - self.arm[1]);
        if held && !self.held {
            self.length = distance.clamp(0.25, 8.0);
            self.paying_out = false;
        } else if !held && self.held {
            self.paying_out = true;
        }
        if reel || self.reeling {
            self.paying_out = false;
        }
        self.held = held;
        self.reeling = reel;
        let step = dt / 4.0;
        let mut hand_velocity = [0.0; 2];
        for axis in 0..2 {
            hand_velocity[axis] = delta[axis] * fraction / dt
                + old.map_or(0.0, |old| (rider[axis] - old[axis]) / (scale * dt));
        }
        for _ in 0..4 {
            for axis in 0..2 {
                self.arm[axis] += delta[axis] * fraction / 4.0;
                self.velocity[axis] *= (-0.035 * step).exp();
                self.ball[axis] += self.velocity[axis] * step;
            }
            let offset = [self.ball[0] - self.arm[0], self.ball[1] - self.arm[1]];
            let distance = offset[0].hypot(offset[1]);
            let previous_length = self.length;
            if reel {
                self.length = (self.length - 0.8 * reel_rate * step).max(0.25);
            } else if self.paying_out {
                let target = self.length.max(distance).min(8.0);
                self.length = target - grip * (target - self.length);
            }
            if grip > 0.0 && !held && (reel || !self.paying_out) {
                let target = self.length.max(distance).min(8.0);
                self.length += (1.0 - grip) * (1.0 - reel_rate) * (target - self.length);
            }
            if distance > self.length {
                let normal = [offset[0] / distance, offset[1] / distance];
                let radial = (self.velocity[0] - hand_velocity[0]) * normal[0]
                    + (self.velocity[1] - hand_velocity[1]) * normal[1];
                let reel_speed = if reel {
                    (previous_length - self.length) / step
                } else {
                    0.0
                };
                let tension =
                    (300.0 * (distance - self.length) + 24.0 * (radial + reel_speed)).max(0.0);
                for axis in 0..2 {
                    self.velocity[axis] -= tension * normal[axis] * step;
                }
            }
        }
        [
            rider[0] + scale * self.ball[0],
            rider[1] + scale * self.ball[1],
        ]
    }
}
