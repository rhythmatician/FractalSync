import ReactDOM from 'react-dom/client';
import * as ort from 'onnxruntime-web';
import App from './App';

// Force canonical WASM runtime configuration early (single-thread, non-SIMD) so the correct
// artifact (`public/ort-wasm.wasm`) is selected and initialized deterministically.
ort.env.wasm.wasmPaths = '/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;

ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
