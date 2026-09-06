use runtime_core::model_io::{
    audio_feature_averages, controls_v2_schema, controls_v2_visual_parameters, decode_controls_v2,
    decode_legacy_visual, decode_orbit_control, legacy_orbit_drive_inputs,
    legacy_visual_export_ranges, legacy_visual_schema, orbit_control_schema, Activation,
    AudioFeatureAverages,
};

#[test]
fn orbit_schema_describes_training_head_exactly() {
    let schema = orbit_control_schema(2);
    assert_eq!(schema.len(), 5);
    assert_eq!(schema[0].name, "s_target");
    assert_eq!(schema[0].activation, Activation::ScaledSigmoid);
    assert_eq!((schema[0].min, schema[0].max), (Some(0.2), Some(3.0)));
    assert_eq!(schema[1].name, "alpha");
    assert_eq!((schema[1].min, schema[1].max), (Some(0.05), Some(0.95)));
    assert_eq!(schema[2].activation, Activation::ScaledSoftplusClamped);
    assert_eq!((schema[2].scale, schema[2].offset), (0.5, 0.1));
    assert_eq!(schema[3].name, "band_gate_0");
}

#[test]
fn remaining_browser_model_interpretations_are_canonical() {
    let audio = AudioFeatureAverages {
        rms: 0.0,
        onset: 1.4,
    };
    let drive = legacy_orbit_drive_inputs(audio).unwrap();
    assert_eq!(drive.transient, 1.0);
    assert_eq!(drive.energy, 0.5);
    assert_eq!(drive.thrust, 0.03);

    let controls = decode_controls_v2(&[
        1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])
    .unwrap();
    let fallback = controls_v2_visual_parameters([0.2, -0.3], &controls, None).unwrap();
    assert_eq!(fallback.color_hue, 0.0);
    assert_eq!(fallback.color_sat, 0.6);
    assert_eq!(fallback.color_bright, 0.6);
    assert_eq!(fallback.zoom, 2.5);
    assert_eq!(fallback.speed, 0.5);
}

#[test]
fn controls_v2_schema_owns_head_activations() {
    let schema = controls_v2_schema();
    assert_eq!(schema.len(), 13);
    assert_eq!(schema[0].activation, Activation::Tanh);
    assert_eq!(schema[2].activation, Activation::Sigmoid);
    assert_eq!(schema[6].activation, Activation::Tanh);
    assert!(decode_controls_v2(&[f64::NAN; 13]).is_err());
}

#[test]
fn audio_averages_use_canonical_rms_and_onset_slots() {
    let averages = audio_feature_averages(
        &[9.0, 8.0, 0.2, 7.0, 0.4, 6.0, 5.0, 4.0, 0.6, 3.0, 0.8, 2.0],
        6,
    )
    .unwrap();
    assert!((averages.rms - 0.4).abs() < 1e-12);
    assert!((averages.onset - 0.6).abs() < 1e-12);
    let with_partial_tail =
        audio_feature_averages(&[0.0, 0.0, 0.2, 0.0, 0.4, 0.0, 99.0], 6).unwrap();
    assert_eq!(
        with_partial_tail,
        AudioFeatureAverages {
            rms: 0.2,
            onset: 0.4
        }
    );
}

#[test]
fn orbit_decoder_preserves_post_activation_values_and_band_order() {
    let decoded = decode_orbit_control(&[1.25, 0.3, 2.4, 0.1, 0.9], 2).unwrap();
    assert_eq!(decoded.s_target, 1.25);
    assert_eq!(decoded.alpha, 0.3);
    assert_eq!(decoded.omega_scale, 2.4);
    assert_eq!(decoded.band_gates, vec![0.1, 0.9]);
    let external = decode_orbit_control(&[3.2, 1.1, 5.5, -0.1, 1.1], 2).unwrap();
    assert_eq!(external.s_target, 3.2);
    assert_eq!(external.band_gates, vec![-0.1, 1.1]);
    assert!(decode_orbit_control(&[1.0, 0.5], 2).is_err());
}

#[test]
fn legacy_visual_decoder_matches_browser_post_processing() {
    let schema = legacy_visual_schema();
    assert_eq!(
        schema
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "julia_real",
            "julia_imag",
            "color_hue",
            "color_sat",
            "color_bright",
            "zoom",
            "speed"
        ]
    );
    let decoded = decode_legacy_visual(
        &[0.5, -0.5, 0.2, 0.4, 0.8, 0.6, 0.9],
        Some(AudioFeatureAverages {
            rms: 0.1,
            onset: 0.5,
        }),
    )
    .unwrap();
    assert!((decoded.julia_seed.real + 0.4).abs() < 1e-12);
    assert!((decoded.julia_seed.imag + 1.0).abs() < 1e-12);
    assert!((decoded.color_hue - 0.4).abs() < 1e-12);
    assert!((decoded.color_sat - 0.85).abs() < 1e-12);
    assert!((decoded.color_bright - 0.63).abs() < 1e-12);
    assert!((decoded.zoom - 2.7).abs() < 1e-12);
    assert_eq!(decoded.speed, 0.7);
}

#[test]
fn legacy_export_ranges_preserve_metadata_without_bounding_decoder() {
    let ranges = legacy_visual_export_ranges();
    assert_eq!(ranges[0].name, "julia_real");
    assert_eq!((ranges[0].min, ranges[0].max), (Some(-2.0), Some(2.0)));
    assert_eq!(ranges[4].name, "color_bright");
    assert_eq!((ranges[5].min, ranges[5].max), (Some(0.1), Some(10.0)));
    assert!(
        (decode_legacy_visual(&[9.0; 7], None)
            .unwrap()
            .julia_seed
            .real
            - 0.5)
            .abs()
            < 1e-12
    );
}
