//! Offline public-query derivative diagnostic for issue #145.
//! Run with --check to turn the known inconsistency into a failing command.

use num_complex::Complex64;
use runtime_core::geometry_provider::{
    query_geometry, GeometryJet, GeometryValidity, GEOMETRY_PROVIDER_VERSION,
};
use serde::Serialize;

#[derive(Serialize)]
struct AxisProbe {
    validities: [GeometryValidity; 2],
    resolved_scales: [f64; 2],
    distance_derivative: f64,
    gradient_derivative: [f64; 2],
}

#[derive(Serialize)]
struct Probe {
    case: &'static str,
    c: [f64; 2],
    step: f64,
    center: GeometryJet,
    axes: [AxisProbe; 2],
}

#[derive(Serialize)]
struct Report {
    diagnostic_only: bool,
    provider_version: &'static str,
    epsilon: f64,
    method: &'static str,
    rows: Vec<Probe>,
}

fn axis_probe(c: Complex64, step: f64, axis: usize) -> Result<AxisProbe, String> {
    let delta = if axis == 0 {
        Complex64::new(step, 0.0)
    } else {
        Complex64::new(0.0, step)
    };
    let minus = query_geometry(c - delta, 1e-4)?;
    let plus = query_geometry(c + delta, 1e-4)?;
    Ok(AxisProbe {
        validities: [minus.validity, plus.validity],
        resolved_scales: [minus.resolved_scale, plus.resolved_scale],
        distance_derivative: (plus.d - minus.d) / (2.0 * step),
        gradient_derivative: [
            (plus.grad_d[0] - minus.grad_d[0]) / (2.0 * step),
            (plus.grad_d[1] - minus.grad_d[1]) / (2.0 * step),
        ],
    })
}

fn consistent(probe: &Probe) -> bool {
    probe.center.validity == GeometryValidity::Regular
        && probe.axes.iter().enumerate().all(|(axis, values)| {
            values
                .validities
                .iter()
                .all(|v| *v == GeometryValidity::Regular)
                && values
                    .resolved_scales
                    .iter()
                    .all(|&h| h == probe.center.resolved_scale)
                && (values.distance_derivative - probe.center.grad_d[axis]).abs() < 1e-4
                && (0..2).all(|component| {
                    let expected = probe.center.hessian_d[component][axis];
                    (values.gradient_derivative[component] - expected).abs()
                        < 1e-4 * expected.abs().max(1.0)
                })
        })
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut selected_case = None;
    let mut check = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--check" => check = true,
            "--case" => {
                index += 1;
                selected_case = Some(args.get(index).ok_or("--case requires a name")?.as_str());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        index += 1;
    }
    let mut rows = Vec::new();
    for (case, c) in [
        (
            "inward_before",
            Complex64::new(0.39494073887023606, 0.04996931486326729),
        ),
        (
            "inward_after",
            Complex64::new(0.3948230501337527, 0.0499674412699672),
        ),
        (
            "near_shore_before",
            Complex64::new(0.30007240651412576, 0.05348910097156354),
        ),
        (
            "near_shore_after",
            Complex64::new(0.3000728655316197, 0.05350153585945869),
        ),
        ("regular_tile_boundary", Complex64::new(-0.5, 0.1)),
        ("near_shore_tile_boundary", Complex64::new(0.390625, 0.05)),
    ] {
        if selected_case.is_some_and(|selected| selected != case) {
            continue;
        }
        for step in [1e-6, 1e-7] {
            rows.push(Probe {
                case,
                c: [c.re, c.im],
                step,
                center: query_geometry(c, 1e-4)?,
                axes: [axis_probe(c, step, 0)?, axis_probe(c, step, 1)?],
            });
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "unknown case: {}",
            selected_case.unwrap_or_default()
        ));
    }
    let failures = rows.iter().filter(|probe| !consistent(probe)).count();
    let report = Report {
        diagnostic_only: true,
        provider_version: GEOMETRY_PROVIDER_VERSION,
        epsilon: 1e-4,
        method:
            "Centered differences of public query_geometry D and grad_D; never used by Physics.",
        rows,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    if check && failures > 0 {
        return Err(format!(
            "{failures} derivative probes disagree; see JSON report"
        ));
    }
    Ok(())
}
