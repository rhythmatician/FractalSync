import { renderMinimapPixels, type MinimapView } from './cockpitMinimap';

self.onmessage = ({ data }: MessageEvent<MinimapView>) => {
  const pixels = renderMinimapPixels(data);
  self.postMessage({ view: data, pixels }, { transfer: [pixels.buffer] });
};
