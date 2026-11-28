import ReactDOM from 'react-dom/client';
import { DebugCockpit } from './components/DebugCockpit';

// Standalone debug cockpit entry (issue #111 Phase A). No ONNX runtime, no
// audio: this page replays hand-authored Controls v2 trajectories through
// the wasm destination physics and renders the third-person manifold view.
const bowlMode = new URLSearchParams(window.location.search).get('mode') === 'bowl';
ReactDOM.createRoot(document.getElementById('root')!).render(
    bowlMode ? (
        <main style={{ background: '#121619', color: '#e9e7de', minHeight: '100vh', fontFamily: 'system-ui' }}>
            <nav style={{ padding: '12px 24px' }}>
                <strong>Debug Cockpit · Exterior bowl</strong>
                {' · '}
                <a style={{ color: '#9ed0ff' }} href="/debugCockpit.html">Standard cockpit</a>
            </nav>
            <iframe
                title="Exterior bowl controls and trajectory"
                src="/cockpit-bowl/"
                style={{ display: 'block', width: '100%', height: 'calc(100vh - 48px)', border: 0 }}
            />
        </main>
    ) : <DebugCockpit />,
);
