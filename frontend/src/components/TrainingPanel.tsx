/**
 * Training panel component for managing model training.
 */

import { useState, useEffect } from 'react';

interface TrainingStatus {
  status: string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  loss_history: Array<{
    epoch: number;
    loss: number;
    timbre_color_loss: number;
    transient_impact_loss: number;
    silence_stillness_loss: number;
    distortion_roughness_loss: number;
    smoothness_loss: number;
  }>;
  error: string | null;
}

interface TrainingRequest {
  data_dir: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  window_frames: number;
  input_dim: number;
  // GPU rendering optimizations (commit 75c1a43)
  no_gpu_rendering?: boolean;
  julia_resolution?: number;
  julia_max_iter?: number;
  num_workers?: number;
}

export function TrainingPanel() {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [audioFiles, setAudioFiles] = useState<File[]>([]);
  const [trainingConfig, setTrainingConfig] = useState<TrainingRequest>({
    data_dir: 'data/audio',
    epochs: 100,
    batch_size: 32,
    learning_rate: 1e-4,
    window_frames: 10,
    input_dim: 60,
    // GPU rendering optimizations (commit 75c1a43) - enabled by default
    no_gpu_rendering: false,
    julia_resolution: 64,
    julia_max_iter: 50,
    num_workers: 4,
  });

  useEffect(() => {
    // Poll training status if training is active
    let intervalId: number | null = null;
    if (isTraining) {
      intervalId = window.setInterval(async () => {
        try {
          const response = await fetch('/api/train/status');
          const status = await response.json() as TrainingStatus;
          setTrainingStatus(status);
          
          if (status.status === 'completed' || status.status === 'error') {
            setIsTraining(false);
            if (intervalId) clearInterval(intervalId);
          }
        } catch (error) {
          console.error('Failed to fetch training status:', error);
        }
      }, 1000); // Poll every second
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isTraining]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setAudioFiles(files);

    // Upload files to server
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('/api/audio/upload', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      console.log('Files uploaded:', result);
    } catch (error) {
      console.error('Failed to upload files:', error);
    }
  };

  const startTraining = async () => {
    try {
      const response = await fetch('/api/train/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(trainingConfig)
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      setIsTraining(true);
      setTrainingStatus(null);
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const downloadModel = async () => {
    try {
      const response = await fetch('/api/model/latest');
      if (!response.ok) {
        throw new Error('Failed to download model');
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model.onnx';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download model:', error);
    }
  };

  return (
    <div style={{ padding: '20px', background: '#1a1a1a', color: '#fff', minHeight: '100vh' }}>
      <h2>Training Panel</h2>

      <div style={{ marginBottom: '20px' }}>
        <h3>Upload Audio Files</h3>
        <input
          type="file"
          multiple
          accept="audio/*"
          onChange={handleFileUpload}
          style={{ marginBottom: '10px' }}
        />
        {audioFiles.length > 0 && (
          <div>
            <p>Uploaded {audioFiles.length} file(s):</p>
            <ul>
              {audioFiles.map((file, idx) => (
                <li key={idx}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>Training Configuration</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <div>
            <label>Epochs:</label>
            <input
              type="number"
              value={trainingConfig.epochs}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
          </div>
          <div>
            <label>Batch Size:</label>
            <input
              type="number"
              value={trainingConfig.batch_size}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
          </div>
          <div>
            <label>Learning Rate:</label>
            <input
              type="number"
              step="0.0001"
              value={trainingConfig.learning_rate}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
          </div>
          <div>
            <label>Window Frames:</label>
            <input
              type="number"
              value={trainingConfig.window_frames}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, window_frames: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
          </div>
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3>
          GPU Rendering Optimizations 
          <span style={{ fontSize: '14px', color: '#888', marginLeft: '10px' }}>(commit 75c1a43)</span>
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
          <div style={{ gridColumn: '1 / -1' }}>
            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={!trainingConfig.no_gpu_rendering}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, no_gpu_rendering: !e.target.checked })}
                style={{ marginRight: '8px' }}
              />
              Enable GPU-accelerated Julia rendering
            </label>
          </div>
          <div>
            <label>Julia Resolution:</label>
            <input
              type="number"
              value={trainingConfig.julia_resolution}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, julia_resolution: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
            <small style={{ color: '#888' }}>Default: 64 (original: 128)</small>
          </div>
          <div>
            <label>Julia Max Iterations:</label>
            <input
              type="number"
              value={trainingConfig.julia_max_iter}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, julia_max_iter: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
            <small style={{ color: '#888' }}>Default: 50 (original: 100)</small>
          </div>
          <div>
            <label>DataLoader Workers:</label>
            <input
              type="number"
              value={trainingConfig.num_workers}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, num_workers: parseInt(e.target.value) })}
              style={{ width: '100%', padding: '5px' }}
            />
            <small style={{ color: '#888' }}>Default: 4 (original: 0)</small>
          </div>
        </div>
        <div style={{ padding: '10px', background: '#2a2a2a', borderRadius: '5px' }}>
          <p style={{ margin: 0, fontSize: '14px', color: '#aaa' }}>
            <strong>Expected speedup:</strong> 3-5x faster training with GPU rendering + parallel data loading.
            Disable optimizations to match pre-75c1a43 behavior (slower but higher quality).
          </p>
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <button
          onClick={startTraining}
          disabled={isTraining || audioFiles.length === 0}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            background: isTraining ? '#666' : '#44ff44',
            color: '#000',
            border: 'none',
            borderRadius: '5px',
            cursor: isTraining || audioFiles.length === 0 ? 'not-allowed' : 'pointer'
          }}
        >
          {isTraining ? 'Training...' : 'Start Training'}
        </button>
        <button
          onClick={downloadModel}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            background: '#4444ff',
            color: '#fff',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            marginLeft: '10px'
          }}
        >
          Download Latest Model
        </button>
      </div>

      {trainingStatus && (
        <div style={{ marginTop: '20px' }}>
          <h3>Training Status</h3>
          <div>
            <p>Status: <strong>{trainingStatus.status}</strong></p>
            <p>Progress: {(trainingStatus.progress * 100).toFixed(1)}%</p>
            <p>Epoch: {trainingStatus.current_epoch} / {trainingStatus.total_epochs}</p>
            
            {trainingStatus.status === 'training' && (
              <div style={{ width: '100%', background: '#333', borderRadius: '5px', overflow: 'hidden', marginTop: '10px' }}>
                <div
                  style={{
                    width: `${trainingStatus.progress * 100}%`,
                    background: '#44ff44',
                    height: '20px',
                    transition: 'width 0.3s'
                  }}
                />
              </div>
            )}

            {trainingStatus.error && (
              <div style={{ color: '#ff4444', marginTop: '10px' }}>
                Error: {trainingStatus.error}
              </div>
            )}

            {trainingStatus.loss_history.length > 0 && (
              <div style={{ marginTop: '20px' }}>
                <h4>Loss History</h4>
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ background: '#333' }}>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Epoch</th>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Total Loss</th>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Timbre-Color</th>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Transient-Impact</th>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Silence-Stillness</th>
                        <th style={{ padding: '5px', border: '1px solid #555' }}>Distortion-Roughness</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingStatus.loss_history.slice(-20).map((entry, idx) => (
                        <tr key={idx}>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.epoch}</td>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.loss.toFixed(4)}</td>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.timbre_color_loss.toFixed(4)}</td>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.transient_impact_loss.toFixed(4)}</td>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.silence_stillness_loss.toFixed(4)}</td>
                          <td style={{ padding: '5px', border: '1px solid #555' }}>{entry.distortion_roughness_loss.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
