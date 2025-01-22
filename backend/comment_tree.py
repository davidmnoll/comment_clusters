import json
import os
from typing import List, Dict, Any, Optional, Union, Literal, TypedDict
import numpy as np
import networkx as nx
import aiohttp
import asyncio
from dataclasses import dataclass, field, asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from community import community_louvain
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import hashlib
import graphviz
from datetime import datetime, timedelta
from cachetools import LRUCache
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["OMP_NUM_THREADS"] = "1"



HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"

# GEN_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/phi-2")  
# GEN_MODEL = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
# GEN_MODEL.eval()

default_filepath = os.path.join(os.path.dirname(__file__), 'data', 'comment_tree.json')
default_vizdir = os.path.join(os.path.dirname(__file__), 'data', 'visualizations')

class CommentTree:
    def __init__(self, filepath: str = default_filepath, skipload: bool = False, visualization_dir: str = default_vizdir):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache = LRUCache(maxsize=1000)
        self.cluster_version = 0
        self.filepath = filepath
        self.vizpath = visualization_dir
        self.table = {}
        self.embedding_model = None
        self.tokenizer = None
        self.lm_model = None
        if not skipload:
            print("Loading tree...")
            self.load()
        else: 
            print("Skipping load")
        


        
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
                # print(f"Fetched comment {comment_id}", data.get('text', '')[:30])
                self._cache[cache_key] = data
                return data
            raise ValueError(f"Failed to fetch comment {comment_id}")

    async def get_comment_by_path(self, path_string: str) -> 'CommentNode':
        return self.table[path_string]
        
     

    async def load_story(self, story_id: str | None = None ) -> None:
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        if story_id:
            print(f"Loading story {story_id}")
        else:
            print("Loading top story")
            session = await self.get_session()
            async with session.get(f"{HN_BASE_URL}/topstories.json") as response:

                if response.status == 200:
                    stories = await response.json()
                    if stories:
                        story_id = stories[0]
            print(f"Top story ID: {story_id}")
            if not story_id:
                raise ValueError("Failed to fetch top story")
        story_data = await self.fetch_comment(str(story_id))
        self.table[''] = CommentNode(
            id=str(story_data['id']),
            text=story_data.get('title', ''),
            is_cluster=False,
            tree=self,
            vector=self.embedding_model.encode([story_data.get('title', '')])[0],
            route="",
            confidence=1.0
        )

    def save(self, file_path: str | None = None) -> bool:
        full_path = file_path or self.filepath
        """Save complete tree state including clusters"""
        tree_data = {
            'version': self.cluster_version,
            'table': { route: x.to_dict() for route, x in  self.table.items() } ,
            'config': {
                
            }  # Add any config parameters here
        }
                    
        with open(full_path, 'w') as f:
            json.dump(tree_data, f, indent=2)
        return True


    async def fetch_comment_tree(self, story_id: str | None = None):
        await self.load_story(story_id=story_id)
        await self.table[''].fetch_comments(force=True)
    
    async def generate_visualizations(self, output_dir: str = default_vizdir):
        # self.table[''].generate_visualizations(force=True)
        await self.table[''].generate(force_reload=False, force_fetch=False, force_cluster=False, force_visualize=False, visualize=True)

    async def generate(self, 
        story: str | None = None, 
        reload_all: bool = False,
        reload_comments: bool = False, 
        reload_clusters: bool = False, 
        reload_visualizations: bool = False,
        depth: int = 5
        ):
        print("Generating all", f"story: {story}")

        self.init_models()
        if self.table.get('') is None:
            await self.load_story(story)
        await self.table[''].generate(force_reload=reload_all, force_fetch=reload_comments, force_cluster=reload_clusters, force_visualize=reload_visualizations, visualize=False, depth=depth)
        print("Finished generating all")


    async def generate_clusters(self):
        print("Regenerating clusters")
        self.init_models()
        self.load()
        self.table[''].generate_clusters(force=True)
        self.save()
        print("Finished regenerating clusters")

    def init_models(self):
        if self.embedding_model is None:
            print("Initializing embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Test the embedding model with two very different texts
            test_text1 = "This is a test sentence."
            test_text2 = "This is a completely different sentence about cats."
            emb1 = self.embedding_model.encode([test_text1])[0]
            emb2 = self.embedding_model.encode([test_text2])[0]
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"Test embedding shapes: {emb1.shape}, {emb2.shape}")
            print(f"Test cosine similarity: {cos_sim}")

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

        if self.lm_model is None:
            self.lm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
            self.lm_model.eval()


    def load(self) -> Optional['CommentNode']:
        """Load complete tree state including clusters"""
        # if file doesn't exist, return
        if not os.path.exists(self.filepath):
            print(f"File {self.filepath} not found")
            return None
        else:
            print(f"Loading tree from {self.filepath}")
            with open(self.filepath, 'r') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Invalid data format in file")

            self.cluster_version = data.get('version', 0)
            self.table = {
                route: CommentNode.from_dict(self, x)
                for route, x in data.get('table', {}).items()
            }

            print(f"Loaded tree with {len(self.table)} nodes")
            return self.table['']
        

class CommentNode:
    def __init__(self, tree: CommentTree, *, 
                id: str,
                text: str,
                is_cluster: bool = False,
                vector: np.ndarray,
                confidence: float,
                route: str,
                min_cluster_size: int = 5,
                children: List["CommentNode"] = [],
                similarity_matrix: Optional[np.ndarray] = None,
                confidence_threshold: float = 0.7,
                similarity_threshold: float = 0.5,
                louvain_resolution: float = 1.0,
                parent_id: Optional[str] = None,
                fetched: bool = False,
                clustered: bool = False,
                visualized: bool = False,
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
        
        self.fetched = False
        self.clustered = False
        self.visualized = False
        self.visualization_file: Optional[str] = None
        self.vector: np.ndarray = vector

  
   
    def _process_kmeans_clusters(self, children: List["CommentNode"]) -> List["CommentNode"]:
        """Process KMeans clusters with adjusted tolerance"""
        print(f"Processing KMeans clusters for {self.id}, - {len(children)} children")



        # Check each child's vector
        for i, child in enumerate(children):
            if len(child.vector) == 0:
                print(f"Warning: Child {child.id} has empty vector")
            elif i > 0:  # Compare with previous vector
                diff = np.max(np.abs(children[i-1].vector - child.vector))
                if diff < 1e-6:
                    print(f"Warning: Child {child.id} has nearly identical vector to {children[i-1].id}")
                    print(f"Text for {child.id}: {child.text[:50]}")
                    print(f"Text for {children[i-1].id}: {children[i-1].text[:50]}")


        # Get embeddings for all children
        embeddings = np.array([child.vector for child in children])


        # Add diagnostic prints
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding variance: {np.var(embeddings)}")
        
        # Check if embeddings are too similar
        similarities = cosine_similarity(embeddings)
        avg_similarity = np.mean(similarities)
        print(f"Average cosine similarity between vectors: {avg_similarity}")

        # Normalize embeddings to unit length
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Calculate number of clusters - more conservative now
        # n_clusters = min(
        #     max(2, len(children) // 5),  # One cluster per 5 comments
        #     5  # Maximum of 5 clusters
        # )
        n_clusters = self.min_cluster_size
        
        # Use KMeans with adjusted parameters
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,  # More initialization attempts
            tol=1e-3,   # More relaxed tolerance
            max_iter=500  # More iterations allowed
        )
        
        # clusters = kmeans.fit_predict(normalized_embeddings)
        clusters = kmeans.fit_predict(embeddings)
        
        # Get distances to cluster centers
        # distances = euclidean_distances(normalized_embeddings, kmeans.cluster_centers_)
        distances = euclidean_distances(embeddings, kmeans.cluster_centers_)
        confidences = 1 - (distances / np.max(distances))
        
        cluster_infos = []
        for cluster_idx in range(n_clusters):
            cluster_mask = clusters == cluster_idx
            if not any(cluster_mask):  # Skip empty clusters
                continue
                
            cluster_nodes = [children[i] for i in range(len(children)) if cluster_mask[i]]
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
                route=self.route + "-" + str(cluster_idx) if self.route else str(cluster_idx),
                is_cluster=True
            )
            
            # Update member assignments with relaxed confidence threshold
            for node, confidence in zip(cluster_nodes, cluster_confidences):
                if confidence >= self.confidence_threshold * 0.8:  # Slightly more lenient
                    cluster.raw_comments.append(node)
            
            if len(cluster.raw_comments) > 0:
                cluster_infos.append(cluster)
                
        print(f"Created {len(cluster_infos)} KMeans clusters")
        return cluster_infos



    def _process_louvain_clusters(self, children: List["CommentNode"]) -> List["CommentNode"]:
        """Process Louvain clusters with updated typing"""
        print(f"Processing Louvain clusters for {self.id}, - {len(children)} children")
        # Get embeddings for all children
        embeddings = np.array([child.vector for child in children])

        similarities = cosine_similarity(embeddings)
        G = nx.Graph()
        
        # Build graph
        for i in range(len(children)):
            G.add_node(i)
            for j in range(i + 1, len(children)):
                if similarities[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarities[i, j])
        
        communities = community_louvain.best_partition(G, resolution=self.louvain_resolution)
        cluster_infos = []
        
        for index, community_id in enumerate(set(communities.values())):
            community_indices = [i for i, c in communities.items() if c == community_id]
            community_nodes = [children[i] for i in community_indices]
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
                route=self.route + "-" + str(index) if self.route else str(index),
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



    
    def generate_cluster_summary(self, nodes: List['CommentNode'], cluster_center: np.ndarray) -> str:
        """Generate a summary for a cluster of comments"""
        if not self.tree.tokenizer:
            raise ValueError("Tokenizer not initialized")
        if not self.tree.lm_model:
            raise ValueError("LM model not initialized")

        if not nodes:
            return "Empty cluster"
        
        sorted_nodes = sorted(nodes, key=lambda x: x.confidence, reverse=True)
        top_samples = [node.text for node in sorted_nodes[:5]]
            
        prompt = f"Please provide a short summary of these related comments:\n" + "\n".join(f"- {text}" for text in top_samples)
        
        # Tokenize the input
        inputs = self.tree.tokenizer(prompt, return_tensors="pt")
        
        # Get the length of input tokens
        input_length = inputs.input_ids.shape[1]
        
        # Generate with the same parameters
        with torch.no_grad():
            outputs = self.tree.lm_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        # Decode only the new tokens by slicing from input_length onwards
        new_tokens = outputs[0][input_length:]
        summary = self.tree.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Generated summary for cluster: {summary}")
        return summary.strip()


    

    async def fetch_comments(self, force: bool = False):
        """Lazy cluster processing"""
        if not self.tree.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        if self.is_cluster:
            return

        if self.fetched and not force:
            return 
        else:
            data = await self.tree.fetch_comment(self.id)

            raw_comments = []
            for index, child_id in enumerate(data.get('kids', [])):
                child_data = await self.tree.fetch_comment(str(child_id))
                text = child_data.get('text', '')
                if not text or text == '[dead]':
                    continue
                # print(f"Fetched comment {child_id}", data)
                # print(f"\nCreating embedding for comment {child_id}")
                # print(f"Text: {text[:100]}")  # Print first 100 chars
                vector = self.tree.embedding_model.encode([text])[0]
                # print(f"Vector first 5 values: {vector[:5]}")
                new_comment = CommentNode(
                    id=str(child_id),
                    text=text,
                    route=self.route + "-" + str(index) if self.route else str(index),
                    confidence=1.0,
                    tree=self.tree,
                    is_cluster=False,
                    vector=vector,
                )
                raw_comments.append(new_comment)

            self.raw_comments = raw_comments
            print(f"Fetched {len(self.raw_comments)} children for {self.id}")
            self.fetched = True
            self.tree.save()
        





    async def generate(self, force_reload: bool = False, force_fetch: bool = False, force_cluster: bool = False, force_visualize: bool = False, visualize: bool = False, depth: int = 5):
        if depth == 0:
            return
        print(f"Generating for {self.id}")
        if not self.visualization_file or force_visualize:
            self.visualized = False
            self.visualization_file = None
        if not len(self.kmeans_clusters) or len(self.louvain_clusters) or force_cluster:
            self.clustered = False
            self.kmeans_clusters = []
            self.louvain_clusters = []
        if not self.fetched or force_fetch:
            self.fetched = False
            self.raw_comments = []
        if not self.tree.table.get(self.route) or force_reload:
            self.tree.table[self.route] = self
        if not self.fetched:
            await self.fetch_comments()
        if not self.clustered:
            self.generate_clusters()
        if not self.visualized and visualize:
            self.generate_visualizations()
        print(f"Finished generating for {self.id}")

        for child in self.raw_comments + self.kmeans_clusters + self.louvain_clusters:
            await child.generate(force_reload=force_reload, force_fetch=force_fetch, force_cluster=force_cluster, force_visualize=force_visualize, visualize=visualize, depth=depth - 1)

    
          
    def generate_clusters(self, force: bool = False):
        print(f"Generating clusters for {self.id}")
        if self.clustered and not force:
            return
        if self.is_cluster:
            if len(self.raw_comments) == 0:
                raise ValueError("Cluster has no children")

        if self.is_cluster:
            children = [c for child in self.raw_comments for c in child.raw_comments]
        else: 
            children = self.raw_comments

        # Process both cluster types
        if len(children) >= self.min_cluster_size:
          self.kmeans_clusters = self._process_kmeans_clusters(children)
          self.louvain_clusters = self._process_louvain_clusters(children)

        self.tree.table[self.route] = self
        self.clustered = True
        self.tree.save()
        print(f"Finished generating clusters for {self.id}")
        # for child in self.raw_comments + self.kmeans_clusters + self.louvain_clusters:
        #     child.generate_clusters()

  
    def generate_visualizations(self, force: bool = False):
        print(f"Generating visualizations for {self.id}")
        if self.visualized and not force:
            return
            
        # Collect all nodes for visualization
        if self.is_cluster:
            nodes = [c for child in self.raw_comments for c in child.raw_comments]
        else:
            nodes = self.raw_comments
            
        if len(nodes) < 2:  # Need at least 2 points for t-SNE
            print(f"Not enough nodes for visualization: {len(nodes)}")
            return
            
        # Get embeddings and create t-SNE projection
        embeddings = np.array([node.vector for node in nodes])
        tsne = TSNE(random_state=42, n_iter=1000)
        tsne_results = tsne.fit_transform(embeddings)
        
        # Create DataFrames for both clustering methods
        kmeans_df = pl.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Cluster_Type': ['KMeans'] * len(nodes)
        })
        
        louvain_df = pl.DataFrame({
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Cluster_Type': ['Louvain'] * len(nodes)
        })
        
        # Set up warm colors for KMeans
        kmeans_colors = sns.color_palette("Reds", n_colors=len(self.kmeans_clusters))
        # Set up cool colors for Louvain
        louvain_colors = sns.color_palette("Blues", n_colors=len(self.louvain_clusters))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot KMeans clusters
        for idx, cluster in enumerate(self.kmeans_clusters):
            cluster_points = [nodes.index(node) for node in cluster.raw_comments]
            if cluster_points:
                sns.scatterplot(
                    data=kmeans_df.filter(pl.Series(range(len(nodes))).is_in(cluster_points)),
                    x='TSNE1',
                    y='TSNE2',
                    color=kmeans_colors[idx],
                    label=f'KMeans {idx}',
                    alpha=0.6,
                    s=100
                )
        
        # Plot Louvain clusters
        for idx, cluster in enumerate(self.louvain_clusters):
            cluster_points = [nodes.index(node) for node in cluster.raw_comments]
            if cluster_points:
                sns.scatterplot(
                    data=louvain_df.filter(pl.Series(range(len(nodes))).is_in(cluster_points)),
                    x='TSNE1',
                    y='TSNE2',
                    color=louvain_colors[idx],
                    label=f'Louvain {idx}',
                    alpha=0.6,
                    s=100,
                    marker='s'  # Square markers for Louvain clusters
                )
        
        # Customize the plot
        plt.title(f'Comment Clusters Visualization - {self.id}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Adjust legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        # Save the visualization
        viz_path = os.path.join(self.tree.vizpath, f'clusters_{self.id}.png')
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.visualization_file = viz_path
        self.visualized = True
        self.tree.save()
        print(f"Finished generating visualizations for {self.id}")
        

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
            'fetched': self.fetched,
            'clustered': self.clustered,
            'visualized': self.visualized,
            'config': {
                'min_cluster_size': self.min_cluster_size,
                'confidence_threshold': self.confidence_threshold,
                'similarity_threshold': self.similarity_threshold,
                'louvain_resolution': self.louvain_resolution
            }
        }

    @classmethod
    def from_dict(cls, tree: CommentTree, data: Dict) -> 'CommentNode':
        """Create node from dictionary format"""
        if not isinstance(data, dict):
            return None
        # print(f"Loading node {data.get('id', id(data))} - {data.get('text', '')[:30]}")
        # print(data)

        vector = np.array(data.get('vector', []))
        if not len(vector) > 0:  # Only print if vector exists
            raise ValueError("Vector not found")
        else: 
            # print(f"Vector for node {data.get('id')}: shape={vector.shape}, first 3 values={vector[:3]}")
            pass

        config = data.get('config', {})
        node = cls(
            tree,
            id=str(data.get('id', id(data))),
            text=data['text'],
            vector=np.array(data.get('vector', [])),
            route=data.get('route', ""),
            confidence=data.get('confidence', 0.0),
            min_cluster_size=config.get('min_cluster_size', 5),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            similarity_threshold=config.get('similarity_threshold', 0.5),
            louvain_resolution=config.get('louvain_resolution', 1.0),
            is_cluster=data.get('is_cluster', False),
            similarity_matrix=np.array(data.get('similarity_matrix', [])),
            author=data.get('author'),
            fetched=data.get('fetched', False),
            timestamp=datetime.fromisoformat(data.get('timestamp' , datetime.now())) if data.get('timestamp') else None,
            parent_id=data.get('parent_id'),
            clustered=data.get('clustered', False),
            visualized=data.get('visualized', False)
        )


        # Load cluster data
        node.raw_comments = [CommentNode.from_dict(tree, c) for c in data.get('raw_comments', [])]
        node.kmeans_clusters = [CommentNode.from_dict(tree, c) for c in data.get('kmeans_clusters', [])]
        node.louvain_clusters = [CommentNode.from_dict(tree, c) for c in data.get('louvain_clusters', [])]
        
        return node

