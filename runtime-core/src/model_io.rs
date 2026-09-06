//! Canonical model-output schemas and post-activation decoders.
//!
//! Training may reproduce the activation descriptors with differentiable tensor
//! operations. Runtime consumers decode the resulting post-activation values here.

use serde::{Deserialize, Serialize};

pub const MODEL_IO_VERSION: &str = "model-io/1";
pub const LEGACY_AUDIO_FEATURES_PER_FRAME: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Identity,
    Sigmoid,
    Tanh,
    ScaledSigmoid,
    ScaledSoftplusClamped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputField {
    pub name: String,
    pub group: String,
    pub activation: Activation,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub scale: f64,
    pub offset: f64,
}

fn field(name: impl Into<String>, activation: Activation, min: f64, max: f64) -> OutputField {
    OutputField {
        name: name.into(),
        group: String::new(),
        activation,
        min: Some(min),
        max: Some(max),
        scale: 1.0,
        offset: 0.0,
    }
}

pub fn orbit_control_schema(k_bands: usize) -> Vec<OutputField> {
    let mut fields = vec![
        OutputField {
            name: "s_target".into(),
            group: "orbit".into(),
            activation: Activation::ScaledSigmoid,
            min: Some(0.2),
            max: Some(3.0),
            scale: 2.8,
            offset: 0.2,
        },
        OutputField {
            name: "alpha".into(),
            group: "orbit".into(),
            activation: Activation::ScaledSigmoid,
            min: Some(0.05),
            max: Some(0.95),
            scale: 0.9,
            offset: 0.05,
        },
        OutputField {
            name: "omega_scale".into(),
            group: "orbit".into(),
            activation: Activation::ScaledSoftplusClamped,
            min: Some(0.1),
            max: Some(5.0),
            scale: 0.5,
            offset: 0.1,
        },
    ];
    fields.extend(
        (0..k_bands)
            .map(|index| field(format!("band_gate_{index}"), Activation::Sigmoid, 0.0, 1.0)),
    );
    for (index, output) in fields.iter_mut().enumerate() {
        output.group = if index < 3 { "orbit" } else { "band_gates" }.to_string();
    }
    fields
}

pub fn controls_v2_schema() -> Vec<OutputField> {
    let ranges = crate::controls::ControlsV2::parameter_ranges();
    crate::controls::ControlsV2::model_output_order()
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let activation = if (2..=5).contains(&index) {
                Activation::Sigmoid
            } else {
                Activation::Tanh
            };
            let [min, max] = ranges[name];
            let mut output = field(name, activation, min, max);
            output.group = if index < 6 { "motion" } else { "view" }.to_string();
            output
        })
        .collect()
}

pub fn legacy_visual_schema() -> Vec<OutputField> {
    [
        "julia_real",
        "julia_imag",
        "color_hue",
        "color_sat",
        "color_bright",
        "zoom",
        "speed",
    ]
    .into_iter()
    .map(|name| OutputField {
        name: name.into(),
        group: "visual".to_string(),
        activation: Activation::Identity,
        min: None,
        max: None,
        scale: 1.0,
        offset: 0.0,
    })
    .collect()
}

/// Historical presentation ranges written into legacy ONNX metadata.
/// These describe declared consumer-facing ranges, not raw-logit limits.
pub fn legacy_visual_export_ranges() -> Vec<OutputField> {
    [
        ("julia_real", -2.0, 2.0),
        ("julia_imag", -2.0, 2.0),
        ("color_hue", 0.0, 1.0),
        ("color_sat", 0.0, 1.0),
        ("color_bright", 0.0, 1.0),
        ("zoom", 0.1, 10.0),
        ("speed", 0.0, 1.0),
    ]
    .into_iter()
    .map(|(name, min, max)| {
        let mut output = field(name, Activation::Identity, min, max);
        output.group = "legacy_export_metadata".to_string();
        output
    })
    .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrbitControlOutput {
    pub s_target: f64,
    pub alpha: f64,
    pub omega_scale: f64,
    pub band_gates: Vec<f64>,
}

pub fn decode_orbit_control(values: &[f64], k_bands: usize) -> Result<OrbitControlOutput, String> {
    let expected = 3 + k_bands;
    if values.len() != expected {
        return Err(format!(
            "orbit_control expects {expected} outputs, got {}",
            values.len()
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("orbit_control outputs must be finite".to_string());
    }
    Ok(OrbitControlOutput {
        s_target: values[0],
        alpha: values[1],
        omega_scale: values[2],
        band_gates: values[3..].to_vec(),
    })
}

pub fn decode_controls_v2(values: &[f64]) -> Result<crate::controls::ControlsV2, String> {
    if !values.iter().all(|value| value.is_finite()) {
        return Err("controls/2 outputs must be finite".to_string());
    }
    crate::controls::ControlsV2::from_model_output(values)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioFeatureAverages {
    pub rms: f64,
    pub onset: f64,
}

pub fn audio_feature_averages(
    features: &[f64],
    features_per_frame: usize,
) -> Result<AudioFeatureAverages, String> {
    if features_per_frame <= 4 {
        return Err("audio feature layout must include RMS at 2 and onset at 4".to_string());
    }
    if features.len() < features_per_frame {
        return Err(format!(
            "feature history needs at least one complete frame of {features_per_frame} values"
        ));
    }
    if !features.iter().all(|value| value.is_finite()) {
        return Err("audio feature history must be finite".to_string());
    }
    let frames = features.len() / features_per_frame;
    let (rms, onset) = features
        .chunks_exact(features_per_frame)
        .fold((0.0, 0.0), |(rms, onset), frame| {
            (rms + frame[2], onset + frame[4])
        });
    Ok(AudioFeatureAverages {
        rms: rms / frames as f64,
        onset: onset / frames as f64,
    })
}

pub fn legacy_audio_feature_averages(features: &[f64]) -> Result<AudioFeatureAverages, String> {
    audio_feature_averages(features, LEGACY_AUDIO_FEATURES_PER_FRAME)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JuliaSeed {
    pub real: f64,
    pub imag: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualParameters {
    pub julia_seed: JuliaSeed,
    pub color_hue: f64,
    pub color_sat: f64,
    pub color_bright: f64,
    pub zoom: f64,
    pub speed: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LegacyOrbitDriveInputs {
    pub transient: f64,
    pub energy: f64,
    pub thrust: f64,
}

pub fn legacy_orbit_drive_inputs(
    audio: AudioFeatureAverages,
) -> Result<LegacyOrbitDriveInputs, String> {
    if !audio.rms.is_finite() || !audio.onset.is_finite() {
        return Err("audio feature averages must be finite".to_string());
    }
    let energy = 1.0 / (1.0 + (-audio.rms).exp());
    Ok(LegacyOrbitDriveInputs {
        transient: audio.onset.clamp(0.0, 1.0),
        energy,
        thrust: energy * 0.06,
    })
}

pub fn controls_v2_visual_parameters(
    c: [f64; 2],
    controls: &crate::controls::ControlsV2,
    presentation: Option<[f64; 4]>,
) -> Result<VisualParameters, String> {
    if !c.iter().all(|value| value.is_finite())
        || presentation.is_some_and(|values| !values.iter().all(|value| value.is_finite()))
    {
        return Err("controls/2 visual inputs must be finite".to_string());
    }
    let [color_hue, color_sat, color_bright, zoom] = presentation.unwrap_or([0.0, 0.6, 0.6, 2.5]);
    let direction = controls.motion.direction;
    Ok(VisualParameters {
        julia_seed: JuliaSeed {
            real: c[0],
            imag: c[1],
        },
        color_hue: color_hue % 1.0,
        color_sat,
        color_bright,
        zoom,
        speed: direction[0].hypot(direction[1]) * controls.motion.throttle,
    })
}

pub fn decode_legacy_visual(
    values: &[f64],
    audio: Option<AudioFeatureAverages>,
) -> Result<VisualParameters, String> {
    if values.len() != 7 {
        return Err(format!(
            "legacy visual model expects 7 outputs, got {}",
            values.len()
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err("legacy visual outputs must be finite".to_string());
    }
    let (hue, saturation, brightness) = match audio {
        Some(features) if features.rms.is_finite() && features.onset.is_finite() => (
            (values[2] + features.rms * 2.0) % 1.0,
            (0.7 + features.onset * 0.3).clamp(0.5, 1.0),
            (0.6 + features.rms * 0.3).clamp(0.5, 0.9),
        ),
        Some(_) => return Err("audio feature averages must be finite".to_string()),
        None => (
            values[2] % 1.0,
            (values[3] * 0.8 + 0.5).clamp(0.5, 1.0),
            (values[4] * 0.5 + 0.5).clamp(0.6, 0.9),
        ),
    };
    Ok(VisualParameters {
        julia_seed: JuliaSeed {
            real: (values[0] * 0.6) % 1.4 - 0.7,
            imag: (values[1] * 0.6) % 1.4 - 0.7,
        },
        color_hue: hue,
        color_sat: saturation,
        color_bright: brightness,
        zoom: (values[5] * 2.0 + 1.5).clamp(1.5, 4.0),
        speed: values[6].clamp(0.3, 0.7),
    })
}

pub fn orbit_visual_parameters(
    c: [f64; 2],
    controls: &OrbitControlOutput,
    audio: AudioFeatureAverages,
) -> Result<VisualParameters, String> {
    if !c.iter().all(|value| value.is_finite())
        || !audio.rms.is_finite()
        || !audio.onset.is_finite()
    {
        return Err("orbit visual inputs must be finite".to_string());
    }
    Ok(VisualParameters {
        julia_seed: JuliaSeed {
            real: c[0],
            imag: c[1],
        },
        color_hue: (audio.rms * 2.0) % 1.0,
        color_sat: (0.7 + audio.onset * 0.3).clamp(0.5, 1.0),
        color_bright: (0.6 + audio.rms * 0.3).clamp(0.5, 0.9),
        zoom: 2.5,
        speed: (controls.omega_scale / 5.0).clamp(0.3, 0.7),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOutputKind {
    ControlsV2,
    OrbitControl,
    LegacyVisual,
}

pub fn model_output_kind(
    model_type: Option<&str>,
    controls_version: Option<&str>,
) -> ModelOutputKind {
    if controls_version == Some(crate::controls::CONTROLS_VERSION)
        || model_type == Some("controls_v2")
    {
        ModelOutputKind::ControlsV2
    } else if model_type == Some("orbit_control") {
        ModelOutputKind::OrbitControl
    } else {
        ModelOutputKind::LegacyVisual
    }
}
