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
