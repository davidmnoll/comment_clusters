// frontend/src/components/Visualization.tsx
import { useState } from 'react';
import { ZoomIn, ZoomOut } from 'lucide-react';
import { useVisualization } from '../hooks/useComment';

interface VisualizationProps {
  path: string;
}

export function Visualization({ path }: VisualizationProps) {
  const [scale, setScale] = useState(1);
  const imageUrl = useVisualization(path);

  return (
    <div className="relative">
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => setScale(s => Math.min(s * 1.2, 3))}
          className="p-2 rounded bg-white shadow"
        >
          <ZoomIn />
        </button>
        <button
          onClick={() => setScale(s => Math.max(s / 1.2, 0.5))}
          className="p-2 rounded bg-white shadow"
        >
          <ZoomOut />
        </button>
      </div>
      <div 
        className="overflow-auto border rounded"
        style={{ maxHeight: 'calc(100vh - 200px)' }}
      >
        <img 
          src={imageUrl} 
          alt="Comment visualization"
          style={{ transform: `scale(${scale})`, transformOrigin: '0 0' }}
          className="transition-transform"
        />
      </div>
    </div>
  );
}