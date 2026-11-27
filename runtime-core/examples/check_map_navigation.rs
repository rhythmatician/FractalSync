//! Compare target scans on the Debug Cockpit's four common-start crossing replays.
use runtime_core::controller::{OrbitController, HOP_LENGTH, SAMPLE_RATE};
use runtime_core::controls::MotionControls;
use runtime_core::map_navigation::{evaluate_map_navigation, MapRoute};
use serde_json::json;

const THROTTLES: [f64; 4] = [0.05, 0.4, 0.8, 1.0];
const DT: f64 = HOP_LENGTH as f64 / SAMPLE_RATE as f64;

fn drive(throttle: f64, brake: f64, grip: f64) -> MotionControls {
    MotionControls {
        direction: [1.0, 0.0],
        throttle,
        brake,
        grip,
        impulse: 0.0,
    }
}

fn replay(commit: f64) -> (Vec<[f64; 2]>, Vec<u8>) {
    let mut controller = OrbitController::default();
    let mut positions = Vec::with_capacity(1221);
    let mut valid = Vec::with_capacity(1221);
    let mut record = |controls| {
        let c = controller.step_with_controls(DT, &controls);
        positions.push([c.re, c.im]);
        valid.push(u8::from(controller.manifold_error.is_none()));
    };
    // The same action sequence as baselineVariants in the Debug Cockpit.
    // Start at the first recorded state, as the browser does.
    for _ in 0..120 {
        record(drive(0.25, 0.0, 0.0));
    }
    for _ in 0..400 {
        record(drive(commit, 0.0, 0.0));
        record(drive(0.4, 0.6, 1.0));
    }
    for _ in 0..300 {
        record(drive(0.4, 0.6, 1.0));
    }
    (positions, valid)
}

fn main() -> Result<(), String> {
    let replays = THROTTLES.map(replay);
    let candidates = replays.each_ref().map(|(positions, valid_steps)| MapRoute {
        positions,
        valid_steps,
    });
    let start = candidates[0].positions[0];
    let config = OrbitController::default().manifold_config;
    let target = [0.26, 0.0];
    let radius = 0.005;
    let ranking_frame = 760;
    let report =
        evaluate_map_navigation(start, target, radius, ranking_frame, &config, &candidates)?;
    let last_positions = replays.each_ref().map(|(path, _)| path.last());
    let ranking_positions = replays.each_ref().map(|(path, _)| path[ranking_frame]);
    println!(
        "{}",
        json!({
            "start": start,
            "targetCenter": target,
            "targetRadius": radius,
            "commitThrottles": THROTTLES,
            "rankingPositions": ranking_positions,
            "lastPositions": last_positions,
            "report": report,
        })
    );
    assert_eq!(report.baseline.arrivals, report.bounded.arrivals);
    assert_eq!(
        report.baseline.target_queries,
        report.bounded.target_queries
    );
    assert_eq!(report.baseline.first_arrival_route, Some(3));
    assert_eq!(report.bounded.first_arrival_route, Some(3));
    Ok(())
}
