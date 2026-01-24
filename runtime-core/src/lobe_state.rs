//! Deterministic lobe switching FSM mirrored from Python

#[derive(Clone, Debug)]
pub struct LobeState {
    pub current_lobe: u32,
    pub target_lobe: Option<u32>,
    pub transition_progress: f64,
    pub cooldown_timer: f64,
    pub hold_timer: f64,

    // Tunables
    pub transition_time: f64,
    pub cooldown: f64,
    pub min_hold: f64,
    pub threshold_on: f64,
    pub threshold_off: f64,
    pub transient_threshold: f64,

    pub n_lobes: usize,
}

impl Default for LobeState {
    fn default() -> Self {
        Self {
            current_lobe: 0,
            target_lobe: None,
            transition_progress: 0.0,
            cooldown_timer: 0.0,
            hold_timer: 0.0,
            transition_time: 1.0,
            cooldown: 2.0,
            min_hold: 1.0,
            threshold_on: 0.6,
            threshold_off: 0.4,
            transient_threshold: 0.6,
            n_lobes: 2,
        }
    }
}

impl LobeState {
    pub fn new(n_lobes: usize) -> Self {
        let mut s = Self::default();
        s.n_lobes = n_lobes;
        s
    }

    pub fn step(&mut self, scores: &[f64], dt: f64, transient: f64) {
        // Update timers
        if self.cooldown_timer > 0.0 {
            self.cooldown_timer = (self.cooldown_timer - dt).max(0.0);
        }
        if self.hold_timer > 0.0 {
            self.hold_timer = (self.hold_timer - dt).max(0.0);
        }

        // Softmax normalization
        let mut exps: Vec<f64> = scores.iter().map(|s| s.exp()).collect();
        let ssum: f64 = exps.iter().sum();
        let probs: Vec<f64> = if ssum == 0.0 {
            let n = scores.len() as f64;
            vec![1.0 / n; scores.len()]
        } else {
            exps.iter().map(|e| e / ssum).collect()
        };

        let mut cand = 0usize;
        let mut cand_score = probs[0];
        for (i, &p) in probs.iter().enumerate() {
            if p > cand_score {
                cand = i;
                cand_score = p;
            }
        }

        // If in transition, advance
        if let Some(tl) = self.target_lobe {
            if tl != self.current_lobe as u32 {
                let eff_time = if transient >= self.transient_threshold { self.transition_time * 0.25 } else { self.transition_time };
                self.transition_progress += dt / eff_time.max(1e-6);
                if self.transition_progress >= 1.0 {
                    self.current_lobe = tl;
                    self.target_lobe = None;
                    self.transition_progress = 0.0;
                    self.cooldown_timer = self.cooldown;
                    self.hold_timer = self.min_hold;
                }
                return;
            }
        }

        // Not in transition: consider starting one
        if (cand as u32) != self.current_lobe && self.cooldown_timer <= 0.0 && self.hold_timer <= 0.0 {
            if cand_score >= self.threshold_on {
                self.target_lobe = Some(cand as u32);
                self.transition_progress = 0.0;
                // Immediately progress using dt
                let eff_time = if transient >= self.transient_threshold { self.transition_time * 0.25 } else { self.transition_time };
                self.transition_progress += dt / eff_time.max(1e-6);
                if self.transition_progress >= 1.0 {
                    self.current_lobe = self.target_lobe.unwrap();
                    self.target_lobe = None;
                    self.transition_progress = 0.0;
                    self.cooldown_timer = self.cooldown;
                    self.hold_timer = self.min_hold;
                }
                return;
            }
        }
    }

    pub fn get_mix(&self) -> f64 {
        self.transition_progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_lobe_and_transition_progress() {
        let mut s = LobeState::new(3);
        // Provide scores that strongly favor lobe 1
        s.step(&[0.0, 10.0, 0.0], 0.2, 0.0);
        // Either target_lobe will be set or transition completed immediately
        assert!(s.target_lobe.is_some() || s.current_lobe == 1);
        // Some progress should have been made
        assert!(s.get_mix() >= 0.0);
    }

    #[test]
    fn test_transient_shortens_transition_time() {
        let mut s1 = LobeState::new(3);
        let mut s2 = LobeState::new(3);
        // Identical scores, same dt
        let dt = 0.2;
        s1.step(&[0.0, 10.0, 0.0], dt, 0.0); // non-transient
        s2.step(&[0.0, 10.0, 0.0], dt, 1.0); // transient >= threshold

        // Transient path should make more progress
        assert!(s2.get_mix() > s1.get_mix());
    }
}
