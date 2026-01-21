use runtime_core::features::FeatureExtractor;

#[test]
fn test_extract_windowed_features_dimensions() {
    let extractor = FeatureExtractor::new(22050, 512, 2048, false, false);
    
    // Test various lengths
    let test_cases = [
        (22050, "1 second"),
        (110250, "5 seconds"),
        (220500, "10 seconds"),
    ];

    for (audio_len, description) in test_cases {
        let audio: Vec<f32> = (0..audio_len).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let features = extractor.extract_windowed_features(&audio, 10);
        
        // Should have 60-dim features (6 features * 10 frames)
        assert!(
            !features.is_empty(),
            "Expected non-empty features for {}",
            description
        );
        assert_eq!(
            features[0].len(),
            60,
            "Expected 60-dim features for {}, got {}",
            description,
            features[0].len()
        );
    }
}

#[test]
fn test_extract_windowed_features_consistency() {
    let extractor = FeatureExtractor::new(22050, 512, 2048, false, false);
    let audio: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin()).collect();
    
    let features1 = extractor.extract_windowed_features(&audio, 10);
    let features2 = extractor.extract_windowed_features(&audio, 10);
    
    assert_eq!(features1.len(), features2.len());
    for (f1, f2) in features1.iter().zip(features2.iter()) {
        for (v1, v2) in f1.iter().zip(f2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "Features should be deterministic");
        }
    }
}

#[test]
fn test_delta_features_enabled() {
    let extractor_no_delta = FeatureExtractor::new(22050, 512, 2048, false, false);
    let extractor_with_delta = FeatureExtractor::new(22050, 512, 2048, true, false);
    
    assert_eq!(extractor_no_delta.num_features_per_frame(), 6);
    assert_eq!(extractor_with_delta.num_features_per_frame(), 12); // 6 base + 6 delta
    
    let audio: Vec<f32> = (0..22050).map(|i| (i as f32 * 0.001).sin()).collect();
    
    let features_no_delta = extractor_no_delta.extract_windowed_features(&audio, 10);
    let features_with_delta = extractor_with_delta.extract_windowed_features(&audio, 10);
    
    assert_eq!(features_no_delta[0].len(), 60); // 6 * 10
    assert_eq!(features_with_delta[0].len(), 120); // 12 * 10
}

#[test]
fn test_delta_delta_features_enabled() {
    let extractor = FeatureExtractor::new(22050, 512, 2048, true, true);
    
    assert_eq!(extractor.num_features_per_frame(), 18); // 6 base + 6 delta + 6 delta-delta
    
    let audio: Vec<f32> = (0..22050).map(|i| (i as f32 * 0.001).sin()).collect();
    let features = extractor.extract_windowed_features(&audio, 10);
    
    assert_eq!(features[0].len(), 180); // 18 * 10
}

#[test]
fn test_feature_extraction_with_silence() {
    let extractor = FeatureExtractor::default();
    let silence: Vec<f32> = vec![0.0; 48000];
    
    let features = extractor.extract_windowed_features(&silence, 10);
    
    assert!(!features.is_empty());
    // All features should be finite (not NaN or inf)
    for window in features {
        for &value in &window {
            assert!(value.is_finite(), "Feature values must be finite, got {}", value);
        }
    }
}

#[test]
fn test_feature_extraction_with_noise() {
    let extractor = FeatureExtractor::default();
    let noise: Vec<f32> = (0..48000).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
    
    let features = extractor.extract_windowed_features(&noise, 10);
    
    assert!(!features.is_empty());
    for window in features {
        for &value in &window {
            assert!(value.is_finite());
        }
    }
}

#[test]
fn test_windowing_with_different_frame_counts() {
    let extractor = FeatureExtractor::default();
    let audio: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.001).sin()).collect();
    
    for window_frames in [1, 5, 10, 20, 50] {
        let features = extractor.extract_windowed_features(&audio, window_frames);
        
        assert!(!features.is_empty());
        assert_eq!(
            features[0].len(),
            extractor.num_features_per_frame() * window_frames
        );
    }
}

#[test]
fn test_feature_values_in_reasonable_range() {
    let extractor = FeatureExtractor::default();
    let audio: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
    
    let features = extractor.extract_windowed_features(&audio, 10);
    
    // Features should generally be in a reasonable range (not wildly large)
    for window in features {
        for &value in &window {
            assert!(
                value.abs() < 1000.0,
                "Feature value {} seems unreasonably large",
                value
            );
        }
    }
}
