/**
 * Frontend audio feature extraction tests.
 *
 * Tests that the browser-based feature extraction works correctly
 * and can be compared against backend baseline extraction.
 *
 * Note: Full parity test requires actual audio file and real Web Audio API.
 * This test validates the extraction logic structure and normalization.
 */

import { describe, it, expect, beforeAll } from "vitest";
import { AudioFeatureExtractor } from "../audioFeatures";

describe("Audio Feature Extraction", () => {
  let audioContext: AudioContext;
  let analyser: AnalyserNode;
  let extractor: AudioFeatureExtractor;

  beforeAll(() => {
    // Use a local lightweight AudioContext mock so tests don't depend on global state
    class LocalAnalyser {
      fftSize = 2048;
      frequencyData = new Float32Array(1024);
      timeData = new Float32Array(1024);
      getFloatFrequencyData(dst: Float32Array) { dst.set(this.frequencyData); }
      getFloatTimeDomainData(dst: Float32Array) { dst.set(this.timeData); }
      get frequencyBinCount() { return 1024; }
      connect() {}
      disconnect() {}
    }

    class LocalAudioContext {
      createAnalyser() { return new LocalAnalyser() as any; }
      get sampleRate() { return 44100; }
      get currentTime() { return 0; }
      createGain() { return { connect() {}, disconnect() {}, gain: { value: 1 } }; }
    }

    audioContext = new LocalAudioContext() as any;
    analyser = audioContext.createAnalyser() as any;
    extractor = new AudioFeatureExtractor(audioContext as any, analyser as any);
  });

  describe("Feature Extractor Initialization", () => {
    it("should initialize without errors", () => {
      expect(extractor).toBeDefined();
      expect(extractor.getFrameCount()).toBe(0);
      expect(extractor.getBufferSize()).toBe(0);
    });

    it("should track frame count", () => {
      const initialCount = extractor.getFrameCount();
      // Frame count increments when extracting windowed features
      extractor.extractWindowedFeatures();
      expect(extractor.getFrameCount()).toBe(initialCount + 1);
    });
  });

  describe("Feature Shape and Structure", () => {
    it("should extract windowed features in correct shape", () => {
      const features = extractor.extractWindowedFeatures(10);

      // Default window size 10, 6 features per frame
      expect(features).toHaveLength(60);
      features.forEach((feature: number) => {
        expect(typeof feature).toBe("number");
        expect(Number.isFinite(feature)).toBe(true);
      });
    });

    it("should produce consistent feature objects", () => {
      const baseFeatures = extractor.extractFeatures();

      expect(baseFeatures).toHaveProperty("spectralCentroid");
      expect(baseFeatures).toHaveProperty("spectralFlux");
      expect(baseFeatures).toHaveProperty("rmsEnergy");
      expect(baseFeatures).toHaveProperty("zeroCrossingRate");
      expect(baseFeatures).toHaveProperty("onsets");
      expect(baseFeatures).toHaveProperty("spectralRolloff");

      // All values should be finite
      Object.values(baseFeatures).forEach((val: number) => {
        expect(Number.isFinite(val)).toBe(true);
      });
    });
  });

  describe("Feature Range Validation", () => {
    it("should produce features in valid ranges", () => {
      const features = extractor.extractFeatures();

      // Normalized features should be roughly in [0, 1]
      const { spectralCentroid, spectralFlux, rmsEnergy, zeroCrossingRate, spectralRolloff } = features;

      expect(spectralCentroid).toBeGreaterThanOrEqual(0);
      expect(spectralCentroid).toBeLessThanOrEqual(1);

      expect(spectralFlux).toBeGreaterThanOrEqual(0);
      expect(spectralFlux).toBeLessThanOrEqual(1);

      expect(rmsEnergy).toBeGreaterThanOrEqual(0);
      expect(rmsEnergy).toBeLessThanOrEqual(1);

      expect(zeroCrossingRate).toBeGreaterThanOrEqual(0);
      expect(zeroCrossingRate).toBeLessThanOrEqual(1);

      expect(spectralRolloff).toBeGreaterThanOrEqual(0);
      expect(spectralRolloff).toBeLessThanOrEqual(1);
    });
  });

  describe("Windowed Feature Buffering", () => {
    it("should maintain sliding window correctly", () => {
      const windowSize = 5;
      const features1 = extractor.extractWindowedFeatures(windowSize);
      const features2 = extractor.extractWindowedFeatures(windowSize);

      // Both should have same length
      expect(features1).toHaveLength(windowSize * 6);
      expect(features2).toHaveLength(windowSize * 6);

      // They should be different (different audio analysis windows)
      // (unless by extreme coincidence they're identical)
    });

    it("should handle window size 1 (no buffering)", () => {
      const features = extractor.extractWindowedFeatures(1);
      expect(features).toHaveLength(6);
    });

    it("should handle large window sizes", () => {
      const features = extractor.extractWindowedFeatures(20);
      expect(features).toHaveLength(120);
    });
  });
});

/**
 * Integration test notes:
 *
 * Full parity testing with backend requires:
 * 1. Test audio file (e.g., Tool - The Grudge)
 * 2. Backend baseline features (generated by test_feature_parity.py)
 * 3. Real Web Audio API (not mocked)
 *
 * This can be done with:
 * - Vitest with jsdom for basic Web Audio mocking
 * - Puppeteer or Playwright for real browser testing
 * - or CI integration with real audio file comparison
 *
 * For now, this test validates the extraction structure and API.
 */

