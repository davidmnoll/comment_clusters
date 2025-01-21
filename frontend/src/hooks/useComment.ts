import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { CommentNode } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export function useComment(path: string) {
  return useQuery({
    queryKey: ['comment', path],
    queryFn: async () => {
      const url = path.length > 0 
        ? `${API_BASE}/${path}` 
        : `${API_BASE}/`;
      
      const { data } = await axios.get<CommentNode>(url);
      return data;
    }
  });
}

export function useVisualization(path: string) {
  return `${API_BASE}/comment/${path}/visualization`;
}