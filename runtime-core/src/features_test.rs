#[cfg(test)]
mod tests {
    use crate::features::FeatureExtractor;

    #[test]
    fn test_empty_audio() {
        let extractor = FeatureExtractor::new(48000, 1024, 4096, false, false);
        let audio: Vec<f32> = vec![];
        let result = extractor.extract_windowed_features(&audio, 2);
        println!("Empty audio result: {} windows", result.len());
    }

    #[test]
    fn test_single_sample() {
        let extractor = FeatureExtractor::new(48000, 1024, 4096, false, false);
        let audio: Vec<f32> = vec![0.5];
        let result = extractor.extract_windowed_features(&audio, 2);
        println!("Single sample result: {} windows", result.len());
    }

    #[test]
    fn test_5000_samples() {
        let extractor = FeatureExtractor::new(48000, 1024, 4096, false, false);
        let mut audio: Vec<f32> = Vec::with_capacity(5000);
        for i in 0..5000 {
            audio.push((i as f32 / 5000.0).sin() * 0.3);
        }
        let result = extractor.extract_windowed_features(&audio, 3);
        println!("5000 samples result: {} windows", result.len());
        assert!(result.len() > 0);
    }
}
