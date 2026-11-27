import { sampleTerrainPatch, type TerrainPatch } from './debugCockpit';
import { initOrbitSynth } from './orbitSynthesizer';

const ready = initOrbitSynth();

self.onmessage = async ({ data }: MessageEvent<Parameters<typeof sampleTerrainPatch>>) => {
  try {
    await ready;
    const patch: TerrainPatch = sampleTerrainPatch(...data);
    self.postMessage({ patch });
  } catch (error) {
    self.postMessage({ error: String(error) });
  }
};
