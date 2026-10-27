//! Opt-in diagnostic timings. No certificate or acceptance decisions depend on them.
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

const NAMES: [&str; 7] = [
    "solve", "phase", "certify", "edge", "join", "geometry", "final",
];
type Totals = [(u64, Duration); 7];
thread_local! {
    static TOTALS: RefCell<Option<Totals>> = const { RefCell::new(None) };
}

pub(super) struct Scope(Option<Option<Totals>>);
impl Scope {
    pub(super) fn begin() -> Self {
        if std::env::var_os("GEOMETRY145_COVER_PROFILE").is_none() {
            return Self(None);
        }
        Self(Some(
            TOTALS.with(|v| v.replace(Some([(0, Duration::ZERO); 7]))),
        ))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            if let Some(totals) = TOTALS.with(|v| v.replace(previous)) {
                for (name, (calls, elapsed)) in NAMES.into_iter().zip(totals) {
                    eprintln!(
                        "[COVER_PHASE145] name={name} calls={calls} elapsed_s={}",
                        elapsed.as_secs_f64()
                    );
                }
            }
        }
    }
}

pub(super) fn timed<T>(stage: usize, operation: impl FnOnce() -> T) -> T {
    if !TOTALS.with(|v| v.borrow().is_some()) {
        return operation();
    }
    let start = Instant::now();
    let result = operation();
    TOTALS.with(|v| {
        if let Some(totals) = v.borrow_mut().as_mut() {
            totals[stage].0 += 1;
            totals[stage].1 += start.elapsed();
        }
    });
    result
}
