#[cfg(test)]
mod tests {
    use crate::features::FeatureExtractor;
    use std::env;
    use std::fs;
    use std::path::Path;

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
    
    #[test]
    fn test_parity_extract() {
        // This test is called by test_feature_parity.py to extract features
        // from audio and save them to JSON for comparison
        
        let audio_path = env::var("PARITY_TEST_AUDIO_PATH")
            .unwrap_or_else(|_| "../backend/data/cache/parity_test_audio.npy".to_string());
        
        // Read numpy array (simple .npy format parser for f32)
        let audio = read_npy_f32(&audio_path).expect("Failed to read audio file");
        
        log::debug!("Loaded {} audio samples from {}", audio.len(), audio_path);
        
        // Extract features with same params as Python test
        let extractor = FeatureExtractor::new(48000, 1024, 4096, false, false);
        let features = extractor.extract_windowed_features(&audio, 10);
        
        log::debug!("Extracted {} windows with {} features each", 
                  features.len(), 
                  if features.is_empty() { 0 } else { features[0].len() });
        
        // Write to JSON
        let json = serde_json::to_string_pretty(&features).expect("Failed to serialize");
        let output_path = Path::new("../backend/data/cache/parity_test_features.json");
        fs::write(output_path, json).expect("Failed to write output");
        
        log::debug!("Wrote features to {}", output_path.display());
    }
    
    // Simple .npy reader for f32 arrays (header + data)
    fn read_npy_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        
        // .npy format: magic (6 bytes) + version (2) + header_len (2) + header + data
        if &bytes[0..6] != b"\x93NUMPY" {
            return Err("Not a valid .npy file".into());
        }
        
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data_start = 10 + header_len;
        
        // Convert bytes to f32
        let float_data = &bytes[data_start..];
        let mut audio = Vec::with_capacity(float_data.len() / 4);
        
        for chunk in float_data.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            audio.push(value);
        }
        
        Ok(audio)
    }
}
