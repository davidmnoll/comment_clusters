// frontend/src/components/Navigation.tsx
import { useState } from 'react';
import { ChevronUp, Home, RefreshCw } from 'lucide-react';

interface NavigationProps {
  path: string;
  onNavigate: (path: string) => void;
}

export function Navigation({ path, onNavigate }: NavigationProps) {
  const segments = path.split('-');

  return (
    <div className="flex flex-row gap-2">
  
        {segments.length > 1 && (<button 
          onClick={() => onNavigate(segments.slice(0, -1).join('-') || '0')}
          className="p-3 rounded-sm bg-blue-950 hover:bg-blue-900 text-white"
          key={'-1'}
        >
          <ChevronUp  size={"1rem"} />
        </button>)}
        {segments.map((segment, i) => (
            <button
              onClick={() => onNavigate(segments.slice(0, i + 1).join('-'))}
              className="hover:bg-slate-800 p-3 rounded-sm"
              key={i}
            >
              {i === 0 ? <Home size={"1rem"} /> : `Child ${segment}`}
            </button>
        ))}
    </div>
  );
}