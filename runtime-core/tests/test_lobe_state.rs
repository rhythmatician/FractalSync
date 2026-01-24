use runtime_core::lobe_state::LobeState;

#[test]
fn test_basic_switch_and_cooldown() {
    let mut ls = LobeState::new(3);
    let scores = vec![0.1, 2.0, 0.0];
    ls.step(&scores, 0.1, 0.0);
    assert_eq!(ls.target_lobe, Some(1));
    // Finish transition by stepping enough time
    ls.step(&scores, ls.transition_time, 0.0);
    assert_eq!(ls.current_lobe, 1);
    // Attempt quick switch back should be blocked by cooldown
    let scores2 = vec![5.0, 0.1, 0.0];
    ls.step(&scores2, 0.1, 0.0);
    assert_eq!(ls.target_lobe, None);
}

#[test]
fn test_transient_fast_switch() {
    let mut ls = LobeState::new(2);
    ls.transition_time = 1.0;
    ls.transient_threshold = 0.5;
    let scores = vec![0.0, 3.0];
    ls.step(&scores, 0.1, 1.0);
    assert_eq!(ls.target_lobe, Some(1));
    // High transient reduces effective time -> finish faster
    ls.step(&scores, 0.25, 1.0);
    assert_eq!(ls.current_lobe, 1);
}