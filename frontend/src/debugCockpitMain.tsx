import ReactDOM from 'react-dom/client';
import { DebugCockpit } from './components/DebugCockpit';

// Standalone debug cockpit entry (issue #111 Phase A). No ONNX runtime, no
// audio: this page replays hand-authored Controls v2 trajectories through
// the wasm destination physics and renders the third-person manifold view.
ReactDOM.createRoot(document.getElementById('root')!).render(<DebugCockpit />);
