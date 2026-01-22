import { useState, useEffect } from 'react';

interface FullscreenToggleProps {
  targetId?: string;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export function FullscreenToggle({ targetId = 'visualizerCanvas', position = 'top-right' }: FullscreenToggleProps) {
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

  const positionStyles = {
    'top-left': { top: '10px', left: '10px' },
    'top-right': { top: '10px', right: '10px' },
    'bottom-left': { bottom: '10px', left: '10px' },
    'bottom-right': { bottom: '10px', right: '10px' }
  };

  return (
    <button
      onClick={toggleFullscreen}
      style={{
        position: 'absolute',
        ...positionStyles[position],
        padding: '8px 12px',
        background: '#444',
        color: '#fff',
        border: '1px solid #666',
        borderRadius: '4px',
        cursor: 'pointer',
        fontFamily: 'monospace',
        fontSize: '14px',
        transition: 'all 0.2s',
        opacity: 0.7,
        zIndex: 10
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = '#66ff66';
        e.currentTarget.style.color = '#000';
        e.currentTarget.style.opacity = '1';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = '#444';
        e.currentTarget.style.color = '#fff';
        e.currentTarget.style.opacity = '0.7';
      }}
      title={isFullscreen ? 'Exit Fullscreen (Esc)' : 'Enter Fullscreen'}
    >
      ⛶
    </button>
  );
}