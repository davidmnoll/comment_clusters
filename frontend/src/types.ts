// First, let's define the base types to make the code more readable
export type CommentNode = {
  id: string;
  text: string;
  route: string;
  vector: number[];
  kmeans_clusters: CommentNode[];
  louvain_clusters: CommentNode[];
  raw_comments: CommentNode[];
  is_cluster: boolean;
  confidence: number;
  parent_id: string;
  timestamp: string;
  author: string;
  similarity_matrix: number[][];
  config: {
    min_cluster_size: number;
    confidence_threshold: number;
    louvain_resolution: number;
    similarity_threshold: number;
  
  }
};
