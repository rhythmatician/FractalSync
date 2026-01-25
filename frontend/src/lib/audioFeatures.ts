/**
 * Browser-based audio feature extraction using Web Audio API.
 * Matches backend feature extraction pipeline.
 */

export interface AudioFeatures {
  spectralCentroid: number;  // Brightness/timbre
  spectralFlux: number;      // Transients
  rmsEnergy: number;         // Loudness
  zeroCrossingRate: number;  // Noisiness/distortion
  onsets: number;            // Hits/transients
  spectralRolloff: number;   // Tone vs noise
}

export class AudioFeatureExtractor {
  // private audioContext: AudioContext;
  private analyser: AnalyserNode;
  // private dataArray: Float32Array;
  private previousSpectrum: Float32Array | null = null;
  private sampleRate: number;
  private fftSize: number = 2048;
  // private hopLength: number = 512;
  
  // Sliding window buffer for maintaining history of features
  private featureBuffer: AudioFeatures[] = [];
  // private windowSize: number = 10;
  private frameCount: number = 0;

  constructor(audioContext: AudioContext, analyser: AnalyserNode, fftSize?: number) {
    // this.audioContext = audioContext;
    this.analyser = analyser;
    this.sampleRate = audioContext.sampleRate;

    // If an explicit fftSize is provided use it; otherwise prefer the analyser's current fftSize
    if (typeof fftSize === 'number') {
      this.fftSize = fftSize;
    } else if (typeof (analyser as any).fftSize === 'number') {
      this.fftSize = (analyser as any).fftSize as number;
    }

    analyser.fftSize = this.fftSize;
    analyser.smoothingTimeConstant = 0.8;
  }

  /**
   * Extract all features from current audio buffer.
   */
  extractFeatures(): AudioFeatures {
    // Get frequency data
    const frequencyData = new Float32Array(this.analyser.frequencyBinCount);
    this.analyser.getFloatFrequencyData(frequencyData);
    
    // Convert to magnitude (frequencyData is in dB)
    const magnitude = frequencyData.map(db => Math.pow(10, db / 20));
    
    // Get time domain data for ZCR and RMS
    const timeData = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(timeData);
    
    const flux = this.computeSpectralFlux(magnitude);

    return {
      spectralCentroid: this.computeSpectralCentroid(magnitude),
      spectralFlux: flux,
      rmsEnergy: this.computeRMSEnergy(timeData),
      zeroCrossingRate: this.computeZeroCrossingRate(timeData),
      onsets: flux, // use the already-computed flux as onset proxy (matches Rust behavior)
      spectralRolloff: this.computeSpectralRolloff(magnitude)
    };
  }

  /**
   * Extract windowed features matching backend sliding window format.
   * Maintains a circular buffer of the last `windowSize` feature frames.
   * Returns flattened: [frame_t-9_f0, ..., frame_t-9_f5, frame_t-8_f0, ..., frame_t_f5]
   */
  extractWindowedFeatures(windowSize: number = 10): number[] {
    // Extract current frame features
    const currentFeatures = this.extractFeatures();
    
    // Add to buffer (maintain sliding window)
    this.featureBuffer.push(currentFeatures);
    if (this.featureBuffer.length > windowSize) {
      this.featureBuffer.shift(); // Remove oldest frame
    }
    
    this.frameCount++;
    
    // Flatten buffer into model input format
    const featureVector: number[] = [];
    
    // During startup, pad with first frame if not enough history
    const startIdx = Math.max(0, this.featureBuffer.length - windowSize);
    const padCount = Math.max(0, windowSize - this.featureBuffer.length);
    
    // Build an array of frames for the window (including padding by repeating the first frame)
    const frames: AudioFeatures[] = [];
    if (padCount > 0 && this.featureBuffer.length > 0) {
      const firstFrame = this.featureBuffer[0];
      for (let p = 0; p < padCount; p++) frames.push(firstFrame);
    }
    for (let i = startIdx; i < this.featureBuffer.length; i++) frames.push(this.featureBuffer[i]);

    // Normalize spectralFlux, rmsEnergy, and onsets across the window (min-max) to match Rust
    function normaliseInPlace(arr: number[]) {
      if (arr.length === 0) return;
      let min = Infinity, max = -Infinity;
      for (const v of arr) {
        if (!Number.isFinite(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min;
      if (range > 0) {
        for (let i = 0; i < arr.length; i++) arr[i] = (arr[i] - min) / range;
      }
    }

    const fluxes: number[] = frames.map(f => f.spectralFlux);
    const rmses: number[] = frames.map(f => f.rmsEnergy);
    const onsets: number[] = frames.map(f => f.onsets);
    normaliseInPlace(fluxes);
    normaliseInPlace(rmses);
    normaliseInPlace(onsets);

    // Reconstruct flattened feature vector using normalized flux/rms/onset series
    for (let fi = 0; fi < frames.length; fi++) {
      const fr = frames[fi];
      featureVector.push(
        fr.spectralCentroid,
        fluxes[fi],
        rmses[fi],
        fr.zeroCrossingRate,
        onsets[fi],
        fr.spectralRolloff
      );
    }    
    // Ensure output is always windowSize * 6 (60 for default)
    while (featureVector.length < windowSize * 6) {
      // Shouldn't happen, but failsafe
      featureVector.push(0);
    }
    
    return featureVector;
  }
  
  /**
   * Get frame count since extractor started (for debugging/monitoring).
   */
  getFrameCount(): number {
    return this.frameCount;
  }
  
  /**
   * Get current buffer size (for debugging/monitoring).
   */
  getBufferSize(): number {
    return this.featureBuffer.length;
  }

  private computeSpectralCentroid(magnitude: Float32Array): number {
    const nyquist = this.sampleRate / 2;
    const binWidth = nyquist / magnitude.length;
    
    let weightedSum = 0;
    let magnitudeSum = 0;
    
    for (let i = 0; i < magnitude.length; i++) {
      const freq = i * binWidth;
      weightedSum += freq * magnitude[i];
      magnitudeSum += magnitude[i];
    }
    
    const centroid = magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
    return centroid / nyquist; // Normalize to [0, 1]
  }

  private computeSpectralFlux(magnitude: Float32Array): number {
    if (this.previousSpectrum === null) {
      this.previousSpectrum = new Float32Array(magnitude.length);
      magnitude.forEach((val, i) => {
        this.previousSpectrum![i] = val;
      });
      return 0; // first frame flux = 0 (no previous frame)
    }

    let flux = 0;
    for (let i = 0; i < magnitude.length; i++) {
      const diff = magnitude[i] - this.previousSpectrum[i];
      // squared L2 difference (match Rust implementation)
      flux += diff * diff;
    }

    // Update previous spectrum
    magnitude.forEach((val, i) => {
      this.previousSpectrum![i] = val;
    });

    // Return raw flux (post-loop normalization will be applied across frames)
    return flux;
  }

  private computeRMSEnergy(timeData: Float32Array): number {
    let sumSquares = 0;
    for (let i = 0; i < timeData.length; i++) {
      sumSquares += timeData[i] * timeData[i];
    }

    const rms = Math.sqrt(sumSquares / timeData.length);
    return rms; // raw RMS; normalise across frames later
  }

  private computeZeroCrossingRate(timeData: Float32Array): number {
    let crossings = 0;
    for (let i = 1; i < timeData.length; i++) {
      if ((timeData[i] >= 0) !== (timeData[i - 1] >= 0)) {
        crossings++;
      }
    }
    
    return crossings / timeData.length;
  }


  private computeSpectralRolloff(magnitude: Float32Array, rolloffPercent: number = 0.85): number {
    const nyquist = this.sampleRate / 2;
    const binWidth = nyquist / magnitude.length;

    // Compute total energy as sum of magnitudes (match Rust's use of magnitude)
    let totalEnergy = 0;
    for (let i = 0; i < magnitude.length; i++) {
      totalEnergy += magnitude[i];
    }

    // Find frequency below which rolloffPercent of energy is contained
    let cumulativeEnergy = 0;
    for (let i = 0; i < magnitude.length; i++) {
      cumulativeEnergy += magnitude[i];
      if (cumulativeEnergy >= rolloffPercent * totalEnergy) {
        const rolloffFreq = i * binWidth;
        return rolloffFreq / nyquist; // Normalize to [0, 1]
      }
    }

    return 1.0;
  }
}
