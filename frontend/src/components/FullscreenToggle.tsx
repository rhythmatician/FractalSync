import { useState, useEffect } from 'react';

interface FullscreenToggleProps {
  targetId?: string;
}

export function FullscreenToggle({ targetId = 'visualizerCanvas' }: FullscreenToggleProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = () => {
    const element = document.getElementById(targetId);
    if (element) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        element.requestFullscreen();
      }
    }
  };

  return (
    <button
      onClick={toggleFullscreen}
      style={{
        padding: '8px 16px',
        background: isFullscreen ? '#44ff44' : '#444',
        color: isFullscreen ? '#000' : '#fff',
        border: '1px solid #666',
        borderRadius: '4px',
        cursor: 'pointer',
        fontFamily: 'monospace',
        fontSize: '14px',
        transition: 'all 0.2s'
      }}
      title={isFullscreen ? 'Exit Fullscreen (Esc)' : 'Enter Fullscreen'}
    >
      {isFullscreen ? '⛶ Exit Fullscreen' : '⛶ Fullscreen'}
    </button>
  );
}