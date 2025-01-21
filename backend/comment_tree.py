import json
import os
from re import sub
from typing import List, Dict, Any, Optional, Union, Literal, TypedDict
import numpy as np
import networkx as nx
import aiohttp
import asyncio
from dataclasses import dataclass, field, asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from community import community_louvain
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import hashlib
import graphviz
from datetime import datetime, timedelta
from cachetools import LRUCache

os.environ["OMP_NUM_THREADS"] = "1"


# EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

TOKENIZER = AutoTokenizer.from_pretrained("microsoft/phi-2")
LM_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
LM_MODEL.eval()

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# GEN_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/phi-2")  
# GEN_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
# GEN_MODEL.eval()


class CommentTree:
    def __init__(self, filename: str = 'comment_tree.json'):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache = LRUCache(maxsize=1000)
        self.root = None
        self.cluster_version = 0
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = filename


        
    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def fetch_comment(self, comment_id: str) -> Dict:
        """Fetch comment data with caching"""
        cache_key = f"comment_{comment_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        session = await self.get_session()
        async with session.get(f"{HN_BASE_URL}/item/{comment_id}.json") as response:
            if response.status == 200:
                data = await response.json()
                print(f"Fetched comment {comment_id}", data)
                self._cache[cache_key] = data
                return data
            raise ValueError(f"Failed to fetch comment {comment_id}")

    async def get_comment_by_path(self, path_string: str) -> 'CommentNode':
        if not self.root:
            raise ValueError("Root comment not loaded")        
        """Navigate to a specific comment using recursive path traversal"""
        if not path_string:
            if not self.root:
                await self.load_top_story()
            return self.root
        else:
            path_parts = path_string.split('-')
            return await self.root.navigate_path(path_parts)
        
     

    async def load_top_story(self) -> None:

        """Load the top story if needed"""
        session = await self.get_session()
        async with session.get(f"{HN_BASE_URL}/topstories.json") as response:

            if response.status == 200:
                stories = await response.json()
                if stories:
                    story_id = stories[0]
                    story_data = await self.fetch_comment(str(story_id))
                    self.root = CommentNode(
                        id=str(story_data['id']),
                        text=story_data.get('title', ''),
                        is_cluster=False,
                        tree=self,
                        vector=EMBEDDING_MODEL.encode([story_data.get('title', '')])[0],
                        route=[],
                        confidence=1.0
                    )
                    await self.root.load_raw_comments()

    def save(self, filename: str | None = None) -> bool:
        filename = filename or self.filename
        """Save complete tree state including clusters"""
        try:
            tree_data = {
                'version': self.cluster_version,
                'tree': self.root.to_dict() if self.root else {},
                'config': {}  # Add any config parameters here
            }
            
            full_path = os.path.join(self.data_dir, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w') as f:
                json.dump(tree_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving tree: {str(e)}")
            return False

    @classmethod
    async def loadFileOrFetch(cls, filename: str = 'comment_tree.json') -> 'CommentTree':
        """Load complete tree state including clusters"""

        try:
            full_path = os.path.join(os.path.dirname(__file__), 'data', filename)
            
            with open(full_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Invalid data format")

            tree = cls()
            tree.root = await CommentNode.from_dict(tree, data.get('tree', {}))
            if not tree.root:
                raise ValueError("Root comment not loaded2")
            if len(tree.root.raw_comments) < tree.root.min_cluster_size:
                await tree.root.load_raw_comments()
            tree.cluster_version = data.get('version', 0)
            return tree
        except FileNotFoundError:
            tree = cls()
            await tree.load_top_story()
            if not tree.root:
                raise ValueError("Root comment not loaded3")
            
            return tree


    @classmethod
    async def load(cls, filename: str) -> Optional['CommentTree']:
        """Load complete tree state including clusters"""
        try:
            full_path = os.path.join(os.path.dirname(__file__), 'data', filename)
            
            with open(full_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None

            tree = cls()
            tree.root = await CommentNode.from_dict(tree, data.get('tree', {}))
            if not tree.root:
                raise ValueError("Root comment not loaded")
            if len(tree.root.raw_comments) < tree.root.min_cluster_size:
                await tree.root.load_raw_comments()
            tree.cluster_version = data.get('version', 0)
            return tree
        except FileNotFoundError:
            return cls()
        

class CommentNode:
    def __init__(self, tree: CommentTree, *, 
                id: str,
                text: str,
                is_cluster: bool = False,
                vector: np.ndarray,
                confidence: float,
                route: List[str],
                min_cluster_size: int = 5,
                children: List["CommentNode"] = [],
                similarity_matrix: Optional[np.ndarray] = None,
                confidence_threshold: float = 0.7,
                similarity_threshold: float = 0.5,
                louvain_resolution: float = 1.0,
                parent_id: Optional[str] = None,
                timestamp: Optional[datetime] = None,
                author: Optional[str] = None,
                
                ):
        self.tree = tree
        self.id = id
        self.text = text
        self.is_cluster = is_cluster
        self.route = route

        self.confidence = confidence
        self.parent_id = parent_id
        self.timestamp = timestamp
        self.author = author
        
        self.raw_comments: List[CommentNode] = children
        self.kmeans_clusters: List[CommentNode] = []
        self.louvain_clusters: List[CommentNode] = []


        self.similarity_matrix: Optional[np.ndarray] = similarity_matrix
        # Clustering parameters
        self.min_cluster_size = min_cluster_size
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.louvain_resolution = louvain_resolution
        
        # Initialize models lazily
        self._embedding_model = EMBEDDING_MODEL
        self._tokenizer = TOKENIZER
        self._lm_model = LM_MODEL
        # self._lm_model.eval()

        self.vector: np.ndarray = vector

    

    async def navigate_path(self, path_parts: List[str]) -> 'CommentNode':
        await self.load_raw_comments()
        if len(path_parts) == 0:
            return self
        else:             
            head = path_parts[0]
            child = None
            if head.startswith('k'):
                cluster_id = int(head[1:])
                child = self.kmeans_clusters[cluster_id] 
            elif head.startswith('l'): 
                cluster_id = int(head[1:])
                child = self.louvain_clusters[cluster_id]
            else:
                if len(self.raw_comments) == 0:
                    raise ValueError(f"Comment {self.id} has no children")
                child = self.raw_comments[int(head)]

            childNode = await CommentNode.from_dict(self.tree, child.to_dict())
                  
     
            return await childNode.navigate_path(path_parts[1:])
                    
  
   

    def _process_kmeans_clusters(self, embeddings: np.ndarray) -> List["CommentNode"]:
        """Process KMeans clusters with updated typing"""
        print(f"Processing KMeans clusters for {self.id}")
        n_clusters = max(2, min(len(self.raw_comments) // 3, 5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        distances = euclidean_distances(embeddings, kmeans.cluster_centers_)
        confidences = 1 - (distances / np.max(distances))
        cluster_infos = []
        for cluster_idx in range(n_clusters):
            cluster_mask = clusters == cluster_idx
            cluster_nodes = [self.raw_comments[i] for i in range(len(self.raw_comments)) if cluster_mask[i]]
            cluster_confidences = confidences[cluster_mask][:, cluster_idx]
            
            summary = self.generate_cluster_summary(
                cluster_nodes, 
                kmeans.cluster_centers_[cluster_idx]
            )
            
            cluster = CommentNode(
                id=str(cluster_idx),
                confidence=float(np.mean(cluster_confidences)),
                vector=kmeans.cluster_centers_[cluster_idx],
                text=summary,
                children=cluster_nodes,
                tree=self.tree,
                route=self.route + [str(cluster_idx)],
                is_cluster=True
            )
            
            # Update member assignments
            for node, confidence in zip(cluster_nodes, cluster_confidences):
                if confidence >= self.confidence_threshold:
                    cluster.raw_comments.append(node)
            
            if len(cluster.raw_comments) > 0:
                cluster_infos.append(cluster)
        print(f"Processed {len(cluster_infos)} KMeans clusters")                
        return cluster_infos

    def _process_louvain_clusters(self, embeddings: np.ndarray) -> List["CommentNode"]:
        """Process Louvain clusters with updated typing"""
        print(f"Processing Louvain clusters for {self.id}")
        similarities = cosine_similarity(embeddings)
        G = nx.Graph()
        
        # Build graph
        for i in range(len(self.raw_comments)):
            G.add_node(i)
            for j in range(i + 1, len(self.raw_comments)):
                if similarities[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarities[i, j])
        
        communities = community_louvain.best_partition(G, resolution=self.louvain_resolution)
        cluster_infos = []
        
        for index, community_id in enumerate(set(communities.values())):
            community_indices = [i for i, c in communities.items() if c == community_id]
            community_nodes = [self.raw_comments[i] for i in community_indices]
            community_center = np.mean(embeddings[community_indices], axis=0)
            community_similarities = similarities[community_indices][:, community_indices]
            
            summary = self.generate_cluster_summary(
                community_nodes,
                community_center
            )
            
            cluster = CommentNode(
                id=community_id,
                confidence=0.0,  # Updated after member assignment
                vector=community_center,
                text=summary,
                children=community_nodes,
                tree=self.tree,
                similarity_matrix=community_similarities,
                route=self.route + [str(index)],
                is_cluster=True
            )
            
            confidences = [
                np.mean([similarities[idx, j] for j in community_indices])
                for idx in community_indices
            ]
            
            # Update member assignments
            for node, confidence in zip(community_nodes, confidences):
                if confidence >= self.confidence_threshold:
                    cluster.raw_comments.append(node)
                    node.louvain_assignment = cluster
            
            if len(cluster.raw_comments) > 0:
                cluster.confidence = float(np.mean(confidences))
                cluster_infos.append(cluster)
        print(f"Processed {len(cluster_infos)} Louvain clusters")
        return cluster_infos



    async def load_raw_comments(self):
        """Lazy cluster processing"""

        if not self.raw_comments:
            data = await self.tree.fetch_comment(self.id)
            self.raw_comments = [
                CommentNode(
                    id=str(child_id),
                    text=(await self.tree.fetch_comment(str(child_id))).get('text', ''),
                    route=self.route + [str(index)],
                    confidence=1.0,
                    tree=self.tree,
                    is_cluster=False,
                    vector=self._embedding_model.encode([data.get('text', '')])[0],
                )
                for index, child_id in enumerate(data.get('kids', []))
            ]
            print(f"Fetched {len(self.raw_comments)} children for {self.id}")

          
        if len(self.raw_comments) < self.min_cluster_size:
            return


        if self.is_cluster:
            if len(self.raw_comments) == 0:
                raise ValueError("Cluster has no children")

        # Get embeddings for all children
        embeddings = np.array([child.vector for child in self.raw_comments])
        
        # Process both cluster types
        if len(self.raw_comments) >= self.min_cluster_size:
          self.kmeans_clusters = self._process_kmeans_clusters(embeddings)
          self.louvain_clusters = self._process_louvain_clusters(embeddings)
        self.tree.save()



    
    def generate_cluster_summary(self, nodes: List['CommentNode'], cluster_center: np.ndarray) -> str:
        """Generate a summary for a cluster of comments"""
        if not nodes:
            return "Empty cluster"
        
        sorted_nodes = sorted(nodes, key=lambda x: x.confidence, reverse=True)
        top_samples = [node.text for node in sorted_nodes[:5]]
            
        prompt = f"Please provide a short summary of these related comments:\n" + "\n".join(f"- {text}" for text in top_samples)
        
        # Yes you still need the tokenizer - it converts text to the numbers the model needs
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._lm_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


    def visualize_clusters(self, output_path: str, cluster_type: str = 'kmeans',
                         confidence_threshold: float = 0.7) -> Optional[str]:
        clusters = self.kmeans_clusters if cluster_type == 'kmeans' else self.louvain_clusters
        if not clusters:
            return None
            
        dot = graphviz.Digraph(comment='Comment Clusters')
        dot.attr(rankdir='TB')
        dot.node('Root', self.text[:30] + '...' if len(self.text) > 30 else self.text, shape='circle')
        
        for i, cluster in enumerate(clusters):
            subgraph = dot.subgraph(name=f'cluster_{i}')
            if subgraph:
              with subgraph as c:
                c.attr(label=cluster.text)
                for member in cluster.raw_comments:
                    member_id = member.id
                    node_id = hashlib.md5(member_id.encode()).hexdigest()[:8]
                    node_color = '#90EE90' if cluster.confidence >= confidence_threshold else '#FFB6C1'
                    c.node(node_id, f"{member_id}\n({cluster.confidence:.2f})",
                          style='filled', fillcolor=node_color)
              dot.edge('Root', f'cluster_{i}')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dot.render(output_path, format='png', cleanup=True)
        return f"{output_path}.png"




    def to_dict(self) -> Dict:
        """Convert node to dictionary format for saving"""
        return {
            'id': self.id,
            'text': self.text,
            'vector': self.vector.tolist(),
            'route': self.route,
            'raw_comments': [c.to_dict() for c in self.raw_comments],
            'kmeans_clusters': [c.to_dict() for c in self.kmeans_clusters],
            'louvain_clusters': [c.to_dict() for c in self.louvain_clusters],
            'is_cluster': self.is_cluster,
            'confidence': self.confidence,
            'similarity_matrix': self.similarity_matrix.tolist() if self.similarity_matrix is not None else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'author': self.author,
            'parent_id': self.parent_id,

            'config': {
                'min_cluster_size': self.min_cluster_size,
                'confidence_threshold': self.confidence_threshold,
                'similarity_threshold': self.similarity_threshold,
                'louvain_resolution': self.louvain_resolution
            }
        }

    @classmethod
    async def from_dict(cls, tree: CommentTree, data: Dict) -> 'CommentNode':
        """Create node from dictionary format"""
        if not isinstance(data, dict):
            return None

        config = data.get('config', {})
        node = cls(
            tree,
            id=str(data.get('id', id(data))),
            text=data['text'],
            vector=np.array(data.get('vector', [])),
            route=data.get('route', []),
            confidence=data.get('confidence', 0.0),
            min_cluster_size=config.get('min_cluster_size', 5),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            similarity_threshold=config.get('similarity_threshold', 0.5),
            louvain_resolution=config.get('louvain_resolution', 1.0),
            is_cluster=data.get('is_cluster', False),
            similarity_matrix=np.array(data.get('similarity_matrix', [])),
            author=data.get('author'),
            timestamp=datetime.fromisoformat(data.get('timestamp' , datetime.now())) if data.get('timestamp') else None,
            parent_id=data.get('parent_id')

        )
        
        # Load cluster data
        node.raw_comments = [await CommentNode.from_dict(tree, c) for c in data.get('raw_comments', [])]
        node.kmeans_clusters = [await CommentNode.from_dict(tree, c) for c in data.get('kmeans_clusters', [])]
        node.louvain_clusters = [await CommentNode.from_dict(tree, c) for c in data.get('louvain_clusters', [])]
        
        return node

if __name__ == "__main__":
    async def main():
        tree = CommentTree()
        await tree.load_top_story()
        if tree.root:
            tree.save()
            
    asyncio.run(main())