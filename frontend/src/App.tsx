import { useState } from 'react';
import { Visualizer } from './components/Visualizer';
import { TrainingPanel } from './components/TrainingPanel';
import './App.css';

function App() {
  const [view, setView] = useState<'visualizer' | 'training'>('visualizer');

  return (
    <div style={{ width: '100%', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <nav style={{ background: '#2a2a2a', padding: '10px', display: 'flex', gap: '10px' }}>
        <button
          onClick={() => setView('visualizer')}
          style={{
            padding: '10px 20px',
            background: view === 'visualizer' ? '#44ff44' : '#444',
            color: view === 'visualizer' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Visualizer
        </button>
        <button
          onClick={() => setView('training')}
          style={{
            padding: '10px 20px',
            background: view === 'training' ? '#44ff44' : '#444',
            color: view === 'training' ? '#000' : '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer'
          }}
        >
          Training
        </button>
      </nav>
      {view === 'visualizer' ? <Visualizer /> : <TrainingPanel />}
    </div>
  );
}

export default App;
