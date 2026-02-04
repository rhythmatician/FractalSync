use runtime_core::clock::{SlowClock, SlowState};

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn slow_clock_time_advances() {
    let mut clock = SlowClock::new();
    let hop = vec![0.0_f32; 1024];
    let mut state = clock.process_hop(&hop);
    let dt = 1024.0 / 48_000.0;
    assert!(approx_eq(state.beat.t_sec, dt, 1e-6));

    for _ in 0..9 {
        state = clock.process_hop(&hop);
    }

    assert!(approx_eq(state.beat.t_sec, dt * 10.0, 1e-6));
}

#[test]
fn slow_clock_phase_wraps() {
    let mut clock = SlowClock::new();
    let hop = vec![0.0_f32; 1024];
    let dt = 1024.0 / 48_000.0;
    let spb = 0.5;

    let mut state = clock.process_hop(&hop);
    let mut expected_phase = (dt / spb) % 1.0;
    assert!(approx_eq(state.beat.phase, expected_phase, 1e-6));
    assert!(state.beat.phase >= 0.0 && state.beat.phase < 1.0);

    for _ in 0..9 {
        state = clock.process_hop(&hop);
    }

    expected_phase = (dt * 10.0 / spb) % 1.0;
    assert!(approx_eq(state.beat.phase, expected_phase, 1e-6));
    assert!(state.beat.phase >= 0.0 && state.beat.phase < 1.0);
}

#[test]
fn slow_state_roundtrip_json() {
    let mut clock = SlowClock::new();
    let hop = vec![0.0_f32; 1024];
    let state = clock.process_hop(&hop);
    let json = serde_json::to_string(&state).expect("serialize");
    let decoded: SlowState = serde_json::from_str(&json).expect("deserialize");
    assert!(approx_eq(state.beat.t_sec, decoded.beat.t_sec, 1e-6));
    assert_eq!(state.features.len(), decoded.features.len());
}
