import numpy as np
import pandas as pd
import torch
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from dash.exceptions import PreventUpdate
import base64
from io import BytesIO
from PIL import Image
import umap
import os
import json
import networkx.algorithms.community.louvain as community_louvain
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from sklearn.decomposition import PCA
from collections import defaultdict, Counter

class PrototypeVisualizer:
    def __init__(self, model, dataset=None, num_examples=5, num_features=128, prototypes=None, 
                 reducer=None, prototype_graph=None, device='cuda:0'):
        """
        Initialize the enhanced prototype visualizer for PIPNet
        
        Parameters:
        -----------
        model : PIPNet model
            The model containing prototypes to visualize
        dataset : torch Dataset
            Dataset used for finding prototype examples
        num_examples : int
            Number of example patches to show per prototype
        num_features : int
            Number of features in the prototype vectors
        prototypes : torch.Tensor
            The prototype representations, if already computed
        reducer : umap.UMAP
            UMAP reducer for dimensionality reduction
        prototype_graph : networkx.Graph
            Graph of prototype relationships
        device : str
            Device to use for computation
        """
        self.model = model
        self.dataset = dataset
        self.num_examples = num_examples
        self.num_features = num_features
        self.device = device
        
        # Get model information
        if model is not None:
            self.num_prototypes = model.module._num_prototypes
            self.num_classes = model.module._classification.weight.shape[0]
            
            # Extract prototype representations if not provided
            if prototypes is None:
                self.prototype_representations = model.module._add_on[0].weight.data.squeeze(2).squeeze(2).cpu().numpy()
            else:
                self.prototype_representations = prototypes.cpu().numpy() if isinstance(prototypes, torch.Tensor) else prototypes
                
            # Extract class weights
            self.class_weights = model.module._classification.weight.detach().cpu().numpy()
        else:
            # Default values if model not provided
            self.num_prototypes = prototypes.shape[0] if prototypes is not None else 50
            self.num_classes = 10  # Default
            self.prototype_representations = prototypes.cpu().numpy() if isinstance(prototypes, torch.Tensor) else prototypes
            self.class_weights = np.random.random((self.num_classes, self.num_prototypes)) * 2 - 1
            self.class_weights[self.class_weights < 0.7] = 0
        
        # Set up UMAP reducer
        self.reducer = reducer if reducer is not None else umap.UMAP(n_components=2, random_state=42)
        
        # Compute the embedding if not already done
        self.embedding = self._compute_embedding()
        
        # Use provided graph or create a new one
        self.prototype_graph = prototype_graph if prototype_graph is not None else self._create_prototype_graph()
        
        # For storing circuit analysis results
        self.prototype_circuits = {}
        self.polysemantic_prototypes = set()
        self.virtual_prototypes = {}
        
        # For storing prototype examples from real data
        self.top_activating_patches = None
        
        # Find maximally activating patches
        if dataset is not None and model is not None:
            self.find_maximally_activating_patches()
        else:
            # Generate synthetic examples if no dataset provided
            self.prototype_examples = self._generate_synthetic_examples()
        
        # Identify prototype communities
        self.prototype_communities = self._find_prototype_communities()
        
        # Compute local PCA embeddings
        self.local_pca_embeddings = {}
        
        # Initialize the app
        self.app = self._create_app()
    
    def _compute_embedding(self):
        """Compute 2D embedding of prototype representations using UMAP"""
        if self.reducer is None:
            # Create a new reducer if one wasn't provided
            self.reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = self.reducer.fit_transform(self.prototype_representations)
        else:
            # Use the provided reducer
            try:
                # Try transforming with existing reducer
                embedding = self.reducer.transform(self.prototype_representations)
            except:
                # If that fails, fit and transform
                embedding = self.reducer.fit_transform(self.prototype_representations)
        
        return embedding
    
    def _find_prototype_neighbors(self, similarity_threshold=0.7):
        """Find neighboring prototypes based on similarity"""
        # Compute pairwise similarities between prototypes
        sim_matrix = cosine_similarity(self.prototype_representations)
        np.fill_diagonal(sim_matrix, 0)  # Remove self-similarity
        
        # For each prototype, get the indices of its neighbors
        neighbors = {}
        for i in range(self.num_prototypes):
            neighbors[i] = list(np.where(sim_matrix[i] > similarity_threshold)[0])
        
        return neighbors
    
    def _create_prototype_graph(self):
        """Create a network graph of prototype relationships"""
        # If we already have a graph, use it
        if hasattr(self, 'prototype_graph') and self.prototype_graph is not None:
            return self.prototype_graph
            
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.num_prototypes):
            # Find the primary class for this prototype (for coloring)
            if hasattr(self, 'class_weights'):
                primary_class = np.argmax(self.class_weights[:, i])
            else:
                primary_class = 0
            G.add_node(i, primary_class=primary_class)
        
        # Find prototype neighbors if not already done
        if not hasattr(self, 'prototype_neighbors'):
            self.prototype_neighbors = self._find_prototype_neighbors()
        
        # Add edges based on prototype neighborhoods
        for proto, neighbors in self.prototype_neighbors.items():
            for neighbor in neighbors:
                G.add_edge(proto, neighbor)
        
        return G
    
    def _find_prototype_communities(self):
        """
        Find communities of related prototypes using the Louvain algorithm
        
        Returns:
        --------
        dict : Mapping from prototype index to community ID
        """
        if self.prototype_graph is None or self.prototype_graph.number_of_nodes() == 0:
            return {}
        
        # Apply Louvain community detection
        try:
            communities = community_louvain.best_partition(self.prototype_graph)
            return communities
        except:
            # Fallback to a simpler connected components approach
            communities = {}
            for i, component in enumerate(nx.connected_components(self.prototype_graph)):
                for node in component:
                    communities[node] = i
            return communities
    
    def find_shared_prototype_sets(self, min_set_size=2, min_frequency=2):
        """
        Find sets of prototypes that frequently co-activate together
        
        Parameters:
        -----------
        min_set_size : int
            Minimum number of prototypes in a set
        min_frequency : int
            Minimum number of times a set must appear to be included
        
        Returns:
        --------
        list : List of (prototype_set, frequency) tuples
        """
        if self.top_activating_patches is None:
            print("No activation data available for finding shared prototype sets")
            return []
        
        # Collect sets of co-activating prototypes
        coactivation_sets = []
        
        for batch_idx, batch_data in enumerate(self.top_activating_patches.values()):
            for position, activations in batch_data['positions'].items():
                # Get prototypes that activate above threshold (e.g., 0.5)
                active_protos = [proto_idx for proto_idx, score in activations.items() if score > 0.5]
                
                # Only consider sets of at least min_set_size
                if len(active_protos) >= min_set_size:
                    coactivation_sets.append(frozenset(active_protos))
        
        # Count frequencies of each set
        set_counter = Counter(coactivation_sets)
        
        # Filter by minimum frequency
        frequent_sets = [(set(s), count) for s, count in set_counter.items() if count >= min_frequency]
        
        # Sort by frequency (descending)
        frequent_sets.sort(key=lambda x: x[1], reverse=True)
        
        return frequent_sets
    
    def compute_local_pca_embedding(self, prototype_indices, n_components=2):
        """
        Compute local PCA embedding for a set of prototypes
        
        Parameters:
        -----------
        prototype_indices : list
            Indices of prototypes to include in the embedding
        n_components : int
            Number of PCA components to use
        
        Returns:
        --------
        dict : PCA embedding results
        """
        if len(prototype_indices) < 2:
            return None
        
        # Get the prototype representations
        proto_vectors = self.prototype_representations[prototype_indices]
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(prototype_indices)))
        embedding = pca.fit_transform(proto_vectors)
        
        # Collect feature vectors if we have activation data
        feature_vectors = []
        patch_indices = []
        
        if self.top_activating_patches is not None:
            for proto_idx in prototype_indices:
                if proto_idx in self.top_activating_patches:
                    for i, patch in enumerate(self.top_activating_patches[proto_idx]['patches']):
                        if 'feature_vector' in patch:
                            feature_vectors.append(patch['feature_vector'])
                            patch_indices.append((proto_idx, i))
        
        # Project feature vectors onto PCA space if available
        if feature_vectors:
            try:
                feature_embedding = pca.transform(np.array(feature_vectors))
            except Exception as e:
                print(f"Error transforming feature vectors: {e}")
                feature_embedding = None
        else:
            feature_embedding = None
        
        result = {
            'prototype_indices': prototype_indices,
            'pca': pca,
            'prototype_embedding': embedding,
            'feature_embedding': feature_embedding,
            'patch_indices': patch_indices,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
        
        # Cache the result
        key = frozenset(prototype_indices)
        self.local_pca_embeddings[key] = result
        
        return result
    
    def find_maximally_activating_patches(self, top_k=10, batch_size=16, max_batches=50):
        """
        Find the image patches that maximally activate each prototype
        
        Parameters:
        -----------
        top_k : int
            Number of top activating patches to keep for each prototype
        batch_size : int
            Batch size for processing dataset
        max_batches : int
            Maximum number of batches to process
        """
        if self.dataset is None or self.model is None:
            print("No dataset or model provided for finding activating patches")
            return
        
        print(f"Finding top {top_k} activating patches for each prototype...")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )
        
        # Initialize storage for top activating patches
        self.top_activating_patches = {}
        for i in range(self.num_prototypes):
            self.top_activating_patches[i] = {
                'patches': [],  # Will store the actual patches
                'scores': [],   # Activation scores
                'positions': {} # Spatial positions with activations
            }
        
        # Process batches
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            inputs = inputs.to(self.device)
            
            # Forward pass to get prototype activations
            with torch.no_grad():
                proto_features, _, _, features = self.model(inputs, features_save=True)
            
            # Process each image in the batch
            for img_idx in range(inputs.size(0)):
                img = inputs[img_idx]
                proto_acts = proto_features[img_idx]
                feat = features[img_idx]
                
                # For each spatial position
                for h in range(proto_acts.size(1)):
                    for w in range(proto_acts.size(2)):
                        position_key = (h, w)
                        activations = proto_acts[:, h, w].cpu().numpy()
                        
                        # Record activations at this position
                        if position_key not in self.top_activating_patches[0]['positions']:
                            for proto_idx in range(self.num_prototypes):
                                if position_key not in self.top_activating_patches[proto_idx]['positions']:
                                    self.top_activating_patches[proto_idx]['positions'][position_key] = {}
                        
                        # Record all significant activations
                        for proto_idx in range(self.num_prototypes):
                            if activations[proto_idx] > 0.1:  # Only record significant activations
                                self.top_activating_patches[proto_idx]['positions'][position_key][proto_idx] = activations[proto_idx]
                        
                        # Find which prototypes are activated at this position
                        top_protos = np.argsort(activations)[::-1][:5]  # Top 5 prototypes
                        
                        # Get the feature vector at this position
                        feature_vector = feat[:, h, w].cpu().numpy()
                        
                        # For each activated prototype, record the patch
                        for proto_idx in top_protos:
                            activation = activations[proto_idx]
                            
                            # Skip if activation is too low
                            if activation < 0.1:
                                continue
                                
                            # Extract patch centered at this position
                            # We need to determine patch size based on the receptive field
                            # For simplicity, we'll use a fixed size of 32x32
                            patch_size = min(32, img.size(1), img.size(2))
                            half_size = patch_size // 2
                            
                            # Get patch coordinates
                            # Map from feature map coordinates to image coordinates
                            scale_factor = img.size(2) / proto_acts.size(2)
                            img_h = int((h + 0.5) * scale_factor)
                            img_w = int((w + 0.5) * scale_factor)
                            
                            # Extract patch
                            h_start = max(0, img_h - half_size)
                            h_end = min(img.size(1), img_h + half_size)
                            w_start = max(0, img_w - half_size)
                            w_end = min(img.size(2), img_w + half_size)
                            
                            patch = img[:, h_start:h_end, w_start:w_end].cpu()
                            
                            # Add to top activating patches for this prototype
                            if len(self.top_activating_patches[proto_idx]['scores']) < top_k or activation > min(self.top_activating_patches[proto_idx]['scores']):
                                patch_info = {
                                    'patch': patch,
                                    'score': activation,
                                    'position': (h, w),
                                    'feature_vector': feature_vector,
                                    'img_idx': img_idx,
                                    'batch_idx': batch_idx
                                }
                                
                                self.top_activating_patches[proto_idx]['patches'].append(patch_info)
                                self.top_activating_patches[proto_idx]['scores'].append(activation)
                                
                                # Keep only top_k patches
                                if len(self.top_activating_patches[proto_idx]['patches']) > top_k:
                                    # Find index of lowest score
                                    min_idx = np.argmin(self.top_activating_patches[proto_idx]['scores'])
                                    # Remove patch and score with lowest value
                                    del self.top_activating_patches[proto_idx]['patches'][min_idx]
                                    del self.top_activating_patches[proto_idx]['scores'][min_idx]
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{max_batches} batches")
        
        # Create visual examples from the patches
        self.prototype_examples = []
        for proto_idx in range(self.num_prototypes):
            examples = []
            
            if proto_idx in self.top_activating_patches and self.top_activating_patches[proto_idx]['patches']:
                # Sort patches by activation score
                sorted_patches = sorted(
                    self.top_activating_patches[proto_idx]['patches'], 
                    key=lambda x: x['score'], 
                    reverse=True
                )
                
                # Take top examples
                for patch_info in sorted_patches[:self.num_examples]:
                    patch = patch_info['patch']
                    
                    # Convert tensor to numpy array
                    patch_np = patch.permute(1, 2, 0).numpy()
                    
                    # Normalize to [0, 255] for visualization
                    patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8)
                    patch_np = (patch_np * 255).astype(np.uint8)
                    
                    examples.append(patch_np)
            
            # If we don't have enough examples, pad with synthetic ones
            while len(examples) < self.num_examples:
                # Create a synthetic example
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                color = [(proto_idx * 50) % 256, ((proto_idx+1) * 70) % 256, ((proto_idx+2) * 90) % 256]
                
                # Pattern type based on prototype index
                pattern_type = proto_idx % 3
                if pattern_type == 0:
                    # Circle
                    center = np.array([32, 32])
                    for y in range(64):
                        for x in range(64):
                            dist = np.sqrt(np.sum((np.array([x, y]) - center)**2))
                            if dist < 20:
                                img[y, x] = color
                elif pattern_type == 1:
                    # Square
                    size = 15
                    start = 32 - size
                    end = 32 + size
                    img[start:end, start:end] = color
                else:
                    # Lines
                    thickness = 2
                    for y in range(64):
                        for t in range(thickness):
                            x = (y + proto_idx*5) % 64
                            if x + t < 64:
                                img[y, x+t] = color
                
                examples.append(img)
            
            self.prototype_examples.append(examples)
        
        print(f"Found top activating patches for {len(self.top_activating_patches)} prototypes")
    
    def analyze_prototype_circuits(self, n_ref_samples=100, batch_size=16, use_activations=True):
        """
        Analyze the circuits for prototypes to identify polysemantic units
        
        Parameters:
        -----------
        n_ref_samples : int
            Number of reference samples to use for circuit analysis
        batch_size : int
            Batch size for processing dataset
        use_activations : bool
            Whether to use activation patterns instead of gradients for circuit analysis
            (activations are faster but may be less accurate for circuit identification)
            
        Returns:
        --------
        dict : Results of circuit analysis
        """
        if self.dataset is None:
            print("No dataset provided for circuit analysis")
            return
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )
        
        # Collect activation data for prototypes
        prototype_activations = {i: [] for i in range(self.num_prototypes)}
        attributions = {i: [] for i in range(self.num_prototypes)}
        
        # Use activation data from top activating patches if available
        if hasattr(self, 'top_activating_patches') and self.top_activating_patches:
            for proto_idx in range(self.num_prototypes):
                if proto_idx in self.top_activating_patches:
                    for patch_info in self.top_activating_patches[proto_idx]['patches']:
                        if 'feature_vector' in patch_info:
                            attributions[proto_idx].append(patch_info['feature_vector'])
                            
                            prototype_activations[proto_idx].append({
                                'activation': patch_info['score'],
                                'feature_vector': patch_info['feature_vector'],
                                'input_idx': patch_info.get('img_idx', 0),
                                'position': patch_info.get('position', (0, 0))
                            })
        else:
            # Function to get intermediate layer activations via hooks
            activation_dict = {}
            
            def get_activation(name):
                def hook(module, input, output):
                    activation_dict[name] = output.detach()
                return hook
            
            # Register hooks for the backbone features
            handles = []
            backbone_layer = self.model.module.feature_net
            handles.append(backbone_layer.register_forward_hook(get_activation('backbone')))
            
            sample_count = 0
            print(f"Collecting data for {n_ref_samples} samples...")
            
            try:
                for batch_idx, (inputs, _) in enumerate(dataloader):
                    if sample_count >= n_ref_samples:
                        break
                    
                    inputs = inputs.to(self.device)
                    
                    # Forward pass
                    with torch.no_grad():
                        # Get prototype activations and features
                        proto_features, _, _, features = self.model(inputs, features_save=True)
                    
                    # Process each sample in the batch
                    for sample_idx in range(inputs.size(0)):
                        if sample_count >= n_ref_samples:
                            break
                        
                        # Get the prototype activations for this sample
                        sample_proto = proto_features[sample_idx]
                        
                        # For each spatial position, record which prototypes are activated
                        for h in range(sample_proto.size(1)):
                            for w in range(sample_proto.size(2)):
                                activations = sample_proto[:, h, w].cpu().numpy()
                                
                                # Get the feature vector at this position
                                feature_vector = features[sample_idx, :, h, w].cpu().numpy()
                                
                                # For each prototype, record activation and feature vector
                                for proto_idx in range(self.num_prototypes):
                                    activation = activations[proto_idx]
                                    
                                    # Only store if activation is significant
                                    if activation > 0.1:
                                        prototype_activations[proto_idx].append({
                                            'activation': activation,
                                            'feature_vector': feature_vector,
                                            'input_idx': sample_count,
                                            'position': (h, w)
                                        })
                                        
                                        # Store attribution (feature vector)
                                        attributions[proto_idx].append(feature_vector)
                        
                        sample_count += 1
                    
                    # Status update
                    if batch_idx % 5 == 0:
                        print(f"Processed {sample_count} samples")
            finally:
                # Remove hooks
                for handle in handles:
                    handle.remove()
            
            print(f"Collected data for {sample_count} samples")
        
        # Analyze the attribution patterns to find polysemantic prototypes
        self._analyze_circuits(prototype_activations, attributions)
        
        return {
            'prototype_circuits': self.prototype_circuits,
            'polysemantic_prototypes': self.polysemantic_prototypes,
            'virtual_prototypes': self.virtual_prototypes
        }
    
    def _analyze_circuits(self, prototype_activations, attributions):
        """
        Analyze the circuits for prototypes to identify polysemantic units
        
        Parameters:
        -----------
        prototype_activations : dict
            Activations for each prototype
        attributions : dict
            Attributions for each prototype
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        print("Analyzing prototype circuits...")
        
        for proto_idx in prototype_activations.keys():
            # Skip prototypes with insufficient data
            if len(attributions[proto_idx]) < 5:
                continue
                
            try:
                # Get attributions for this prototype
                proto_attributions = attributions[proto_idx]
                
                # Form attribution matrix
                attribution_matrix = np.vstack(proto_attributions)
                
                # Check if the prototype is polysemantic using k-means clustering
                # as described in the PURE paper
                n_clusters = min(2, len(attribution_matrix))
                
                # Use k-means to find clusters in the attribution space
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(attribution_matrix)
                
                # Calculate silhouette score to determine if clustering is meaningful
                sil_score = 0
                if len(np.unique(cluster_labels)) > 1:
                    sil_score = silhouette_score(attribution_matrix, cluster_labels)
                
                # Store circuit information
                self.prototype_circuits[proto_idx] = {
                    'attribution_matrix': attribution_matrix,
                    'cluster_labels': cluster_labels,
                    'silhouette_score': sil_score,
                    'cluster_centers': kmeans.cluster_centers_,
                    'activations': [item['activation'] for item in prototype_activations[proto_idx]]
                }
                
                # Determine if prototype is polysemantic
                polysemantic_threshold = 0.1
                is_polysemantic = sil_score > polysemantic_threshold
                
                if is_polysemantic:
                    print(f"Prototype {proto_idx} is polysemantic (silhouette score: {sil_score:.3f})")
                    self.polysemantic_prototypes.add(proto_idx)
                    
                    # Create virtual prototypes for each cluster
                    self.virtual_prototypes[proto_idx] = {}
                    
                    for cluster_idx in range(n_clusters):
                        # Get samples in this cluster
                        cluster_mask = (cluster_labels == cluster_idx)
                        cluster_attributions = attribution_matrix[cluster_mask]
                        
                        # Count samples in cluster
                        cluster_size = cluster_attributions.shape[0]
                        
                        # Calculate the centroid of the cluster
                        centroid = kmeans.cluster_centers_[cluster_idx]
                        
                        # Find the most representative sample (closest to centroid)
                        distances = np.linalg.norm(cluster_attributions - centroid, axis=1)
                        representative_idx = np.argmin(distances)
                        
                        # Store virtual prototype information
                        self.virtual_prototypes[proto_idx][cluster_idx] = {
                            'centroid': centroid,
                            'size': cluster_size,
                            'attribution_indices': np.where(cluster_mask)[0],
                            'representative_idx': representative_idx,
                            'sample_indices': [prototype_activations[proto_idx][i]['input_idx'] 
                                              for i in range(len(prototype_activations[proto_idx]))
                                              if i in np.where(cluster_mask)[0]]
                        }
            
            except Exception as e:
                print(f"Error analyzing prototype {proto_idx}: {e}")
    
    def _generate_synthetic_examples(self):
        """Generate synthetic example patches for each prototype"""
        examples = []
        for i in range(self.num_prototypes):
            # Create examples per prototype
            prototype_examples = []
            
            # Color derived from prototype index for differentiation
            color = [(i * 50) % 256, ((i+1) * 70) % 256, ((i+2) * 90) % 256]
            
            # Create visual pattern based on prototype index
            for j in range(self.num_examples):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                
                pattern_type = i % 3
                if pattern_type == 0:
                    # Draw a circle
                    center = np.array([32, 32])
                    for y in range(64):
                        for x in range(64):
                            dist = np.sqrt(np.sum((np.array([x, y]) - center)**2))
                            if dist < 20 + j*2:
                                img[y, x] = color
                elif pattern_type == 1:
                    # Draw a square
                    size = 15 + j*3
                    start = 32 - size
                    end = 32 + size
                    img[start:end, start:end] = color
                else:
                    # Draw diagonal lines
                    thickness = 2 + j
                    for y in range(64):
                        for t in range(thickness):
                            x = (y + i*5 + j*10) % 64
                            if x + t < 64:
                                img[y, x+t] = color
                
                prototype_examples.append(img)
            
            examples.append(prototype_examples)
        
        return examples
    
    def _compute_circuit_similarity(self, proto1, proto2):
        """Compute similarity between circuits of two prototypes"""
        if proto1 not in self.prototype_circuits or proto2 not in self.prototype_circuits:
            # If circuit info isn't available, fall back to representation similarity
            return cosine_similarity(
                [self.prototype_representations[proto1]], 
                [self.prototype_representations[proto2]]
            )[0][0]
        
        # Use the cluster centers for similarity computation
        centers1 = self.prototype_circuits[proto1]['cluster_centers']
        centers2 = self.prototype_circuits[proto2]['cluster_centers']
        
        # Calculate pairwise similarities between all centers
        similarities = []
        for c1 in centers1:
            for c2 in centers2:
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
                similarities.append(sim)
        
        # Return maximum similarity
        return max(similarities) if similarities else 0.0
    
    def _create_app(self):
        """Create the Dash application for visualization"""
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Define layout
        app.layout = html.Div([
            html.H1("PIPNet Prototype Visualization Tool"),
            
            # Main tabs
            dcc.Tabs([
                # Global visualization tab
                dcc.Tab(label="Prototype Manifold", children=[
                    html.Div([
                        html.Div([
                            html.H3("Global Prototype Manifold"),
                            dcc.Graph(
                                id='prototype-manifold',
                                figure=self._create_manifold_figure(),
                                style={'height': '600px'}
                            ),
                            html.Div([
                                html.Button('Reset View', id='reset-button', n_clicks=0),
                                html.Button('Apply Changes', id='apply-button', n_clicks=0, 
                                          style={'margin-left': '10px', 'background-color': '#4CAF50', 'color': 'white'})
                            ], style={'margin-top': '10px'}),
                            html.Div(id='drag-status')
                        ], style={'width': '55%', 'display': 'inline-block', 'vertical-align': 'top'}),
                        
                        html.Div([
                            html.H3("Prototype Details"),
                            html.Div([
                                html.H4("Selected Prototype: ", id='selected-prototype-title'),
                                html.Div([
                                    html.Div([
                                        html.H5("Example Patches"),
                                        html.Div(id='prototype-examples', style={'display': 'flex', 'flex-wrap': 'wrap'})
                                    ], style={'margin-bottom': '20px'}),
                                    
                                    html.Div([
                                        html.H5("Class Weights"),
                                        dcc.Graph(id='class-weights-chart', style={'height': '200px'})
                                    ], style={'margin-bottom': '20px'}),
                                    
                                    html.Div([
                                        html.H5("Circuit Information"),
                                        dcc.Graph(id='circuit-info-chart', style={'height': '200px'})
                                    ]),
                                    
                                    html.H5("Prototype Relationships"),
                                    dcc.Graph(id='related-prototypes', style={'height': '200px'}),
                                    
                                    html.Div([
                                        html.H5("Polysemantic Analysis"),
                                        html.Div(id='polysemantic-analysis')
                                    ], style={'margin-top': '20px'})
                                ], id='prototype-details-container', style={'display': 'none'})
                            ])
                        ], style={'width': '42%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3%'})
                    ]),
                ]),
                
                # Local manifold analysis tab
                dcc.Tab(label="Local Manifold Analysis", children=[
                    html.Div([
                        html.H3("Local Manifold Analysis"),
                        html.Div([
                            html.Label("Select Prototype Set:"),
                            dcc.Dropdown(
                                id='prototype-set-dropdown',
                                options=[
                                    {'label': f'Community {i} (Size: {len([n for n, c in self.prototype_communities.items() if c == i])})', 
                                     'value': i} 
                                    for i in set(self.prototype_communities.values())
                                ] + [
                                    {'label': 'Custom Selection', 'value': 'custom'}
                                ],
                                value=next(iter(set(self.prototype_communities.values()))) if self.prototype_communities else 'custom'
                            ),
                            html.Div(id='custom-prototype-selection', style={'display': 'none'}, children=[
                                html.Label("Enter prototype indices (comma-separated):"),
                                dcc.Input(id='custom-prototypes-input', type='text', value=''),
                                html.Button('Apply', id='apply-custom-prototypes', n_clicks=0)
                            ]),
                            html.Label("PCA Components:"),
                            dcc.Slider(id='pca-components-slider', min=2, max=5, step=1, value=2,
                                     marks={i: str(i) for i in range(2, 6)}),
                        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
                        
                        html.Div([
                            html.H4("Local PCA Embedding"),
                            dcc.Graph(id='local-pca-embedding', style={'height': '500px'}),
                            html.Div(id='pca-stats')
                        ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'})
                    ])
                ]),
                
                # Shared prototype sets tab
                dcc.Tab(label="Shared Prototype Sets", children=[
                    html.Div([
                        html.H3("Shared Prototype Sets Analysis"),
                        html.Div([
                            html.Label("Minimum Set Size:"),
                            dcc.Slider(id='min-set-size-slider', min=2, max=5, step=1, value=2,
                                     marks={i: str(i) for i in range(2, 6)}),
                            html.Label("Minimum Frequency:"),
                            dcc.Slider(id='min-frequency-slider', min=2, max=10, step=1, value=3,
                                     marks={i: str(i) for i in range(2, 11, 2)}),
                            html.Button('Find Shared Sets', id='find-shared-sets-button', n_clicks=0,
                                       style={'margin-top': '20px'})
                        ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
                        
                        html.Div([
                            html.H4("Shared Prototype Sets"),
                            html.Div(id='shared-sets-list')
                        ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'})
                    ])
                ])
            ]),
            
            # Store component to save the current state of the prototype positions
            dcc.Store(id='prototype-positions', data=self.embedding.tolist()),
            
            # Store component for the currently selected prototype
            dcc.Store(id='selected-prototype', data=None),
            
            # Store component for the current local PCA embeddings
            dcc.Store(id='current-pca-embedding', data=None),
            
            # Store component for UMAP-selected prototypes
            dcc.Store(id='umap-selected-prototypes-store', data=[])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('prototype-details-container', 'style'),
             Output('selected-prototype-title', 'children'),
             Output('prototype-examples', 'children'),
             Output('class-weights-chart', 'figure'),
             Output('circuit-info-chart', 'figure'),
             Output('related-prototypes', 'figure'),
             Output('polysemantic-analysis', 'children'),
             Output('selected-prototype', 'data')],
            [Input('prototype-manifold', 'clickData')],
            [State('prototype-positions', 'data')]
        )
        def display_prototype_details(clickData, positions):
            if not clickData:
                return {'display': 'none'}, "Selected Prototype: None", [], {}, {}, {}, [], None
            
            # Get the index of the clicked prototype
            point_index = clickData['points'][0]['customdata'][0]
            
            # Update selected prototype title
            title = f"Selected Prototype: {point_index}"
            
            # Create example patches display
            examples = []
            for i, example_img in enumerate(self.prototype_examples[point_index]):
                # Convert numpy array to base64 encoded image
                img = Image.fromarray(example_img)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Add activation score if available
                score_text = ""
                if point_index in self.top_activating_patches and i < len(self.top_activating_patches[point_index]['scores']):
                    score = self.top_activating_patches[point_index]['scores'][i]
                    score_text = f"Score: {score:.2f}"
                
                example_div = html.Div([
                    html.Img(src=f'data:image/png;base64,{encoded_image}', 
                             style={'width': '80px', 'height': '80px', 'margin': '5px', 'border': '1px solid #ddd'}),
                    html.Div(score_text, style={'text-align': 'center', 'font-size': '10px'})
                ], style={'display': 'inline-block'})
                
                examples.append(example_div)
            
            # Create class weights chart
            weights = self.class_weights[:, point_index] if self.class_weights.shape[0] == self.num_classes else self.class_weights[point_index]
            weights_fig = go.Figure(data=[
                go.Bar(
                    x=[f'Class {i}' for i in range(len(weights))],
                    y=weights,
                    marker_color='royalblue'
                )
            ])
            weights_fig.update_layout(
                title="Weights to Classes",
                xaxis_title="Class",
                yaxis_title="Weight",
                margin=dict(l=40, r=20, t=40, b=30),
                height=200
            )
            
            # Create circuit information chart
            if point_index in self.prototype_circuits:
                circuit_info = self.prototype_circuits[point_index]
                
                # Get the number of samples in each cluster
                n_clusters = len(np.unique(circuit_info['cluster_labels']))
                cluster_counts = [np.sum(circuit_info['cluster_labels'] == i) for i in range(n_clusters)]
                
                circuit_fig = go.Figure(data=[
                    go.Bar(
                        x=[f'Cluster {i}' for i in range(n_clusters)],
                        y=cluster_counts,
                        marker_color=['forestgreen' if i == 0 else 'coral' for i in range(n_clusters)]
                    )
                ])
                circuit_fig.update_layout(
                    title=f"Circuit Clusters (Silhouette Score: {circuit_info['silhouette_score']:.3f})",
                    xaxis_title="Cluster",
                    yaxis_title="Number of Samples",
                    margin=dict(l=40, r=20, t=40, b=30),
                    height=200
                )
            else:
                circuit_fig = go.Figure()
                circuit_fig.update_layout(
                    title="No circuit information available",
                    margin=dict(l=40, r=20, t=40, b=30),
                    height=200
                )
            
            # Create related prototypes visualization
            if hasattr(self, 'prototype_neighbors'):
                related_prototypes = [n for n, nbrs in self.prototype_neighbors.items() 
                                     if point_index in nbrs or n == point_index]
            else:
                # Find related prototypes based on cosine similarity
                related_prototypes = []
                for i in range(self.num_prototypes):
                    if i != point_index:
                        sim = cosine_similarity([self.prototype_representations[point_index]], 
                                               [self.prototype_representations[i]])[0][0]
                        if sim > 0.7:  # Threshold for considering prototypes related
                            related_prototypes.append(i)
                related_prototypes.append(point_index)
            
            # Prepare the data for the network graph
            if len(related_prototypes) > 1:
                # Create a subgraph with the selected prototype and its neighbors
                H = nx.Graph()
                
                # Add nodes
                for node in related_prototypes:
                    H.add_node(node, pos=(positions[node][0], positions[node][1]))
                
                # Add edges
                for i in related_prototypes:
                    for j in related_prototypes:
                        if i < j and (j in self.prototype_neighbors.get(i, []) or i in self.prototype_neighbors.get(j, [])):
                            H.add_edge(i, j)
                
                # Position nodes using the embedding
                pos = {node: positions[node] for node in related_prototypes}
                
                # Create edge traces
                edge_x = []
                edge_y = []
                for edge in H.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                # Create node traces
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                
                for node in H.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(f'Prototype {node}')
                    if node == point_index:
                        node_colors.append('red')  # Selected prototype
                    else:
                        # Calculate circuit similarity
                        sim = self._compute_circuit_similarity(point_index, node)
                        # Color based on similarity (green for high similarity)
                        node_colors.append(f'rgba(0, 255, 0, {sim})')
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        color=node_colors,
                        size=15,
                        line=dict(width=2, color='DarkSlateGrey')
                    )
                )
                
                related_fig = go.Figure(data=[edge_trace, node_trace],
                                 layout=go.Layout(
                                     title=f'Related Prototypes (green intensity = similarity)',
                                     showlegend=False,
                                     margin=dict(l=40, r=20, t=40, b=30),
                                     height=200,
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                 ))
            else:
                related_fig = go.Figure()
                related_fig.update_layout(
                    title="No related prototypes found",
                    margin=dict(l=40, r=20, t=40, b=30),
                    height=200
                )
            
            # Create polysemantic analysis information
            if point_index in self.polysemantic_prototypes:
                polysemantic_info = [
                    html.Div([
                        html.H6(f"This prototype is polysemantic (Silhouette Score: {self.prototype_circuits[point_index]['silhouette_score']:.3f})"),
                        html.P("This prototype appears to represent multiple distinct concepts. You can split it into separate prototypes.")
                    ], style={'color': 'coral', 'font-weight': 'bold'})
                ]
                
                # Add virtual prototype information if available
                if point_index in self.virtual_prototypes:
                    for cluster_idx, virtual_proto in self.virtual_prototypes[point_index].items():
                        polysemantic_info.append(
                            html.Div([
                                html.H6(f"Virtual Prototype {point_index}-{cluster_idx}"),
                                html.P(f"Cluster Size: {virtual_proto['size']} samples")
                            ])
                        )
                        
                        # Add option to promote this virtual prototype to a real one
                        polysemantic_info.append(
                            html.Button(
                                f"Promote Virtual Prototype {point_index}-{cluster_idx}",
                                id=f'promote-button-{point_index}-{cluster_idx}',
                                style={'margin-top': '5px', 'margin-bottom': '15px'}
                            )
                        )
            else:
                if point_index in self.prototype_circuits:
                    polysemantic_info = [
                        html.Div([
                            html.H6(f"This prototype is monosemantic (Silhouette Score: {self.prototype_circuits[point_index]['silhouette_score']:.3f})"),
                            html.P("This prototype appears to represent a single concept.")
                        ], style={'color': 'forestgreen'})
                    ]
                else:
                    polysemantic_info = [
                        html.P("No circuit analysis available for this prototype.")
                    ]
            
            return {'display': 'block'}, title, examples, weights_fig, circuit_fig, related_fig, polysemantic_info, point_index
        
        @app.callback(
            [Output('prototype-manifold', 'figure'),
             Output('umap-selected-prototypes-store', 'data')],
            [Input('reset-button', 'n_clicks'),
             Input('prototype-manifold', 'selectedData')],
            [State('prototype-positions', 'data'),
             State('selected-prototype', 'data'),
             State('umap-selected-prototypes-store', 'data')]
        )
        def update_manifold(reset_clicks, selectedData, positions, selected_prototype, selected_prototypes):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'reset-button':
                # Reset to original embedding and clear selections
                return self._create_manifold_figure(), []
            
            elif trigger_id == 'prototype-manifold' and selectedData:
                # Extract selected prototype indices from the selection
                selected_indices = []
                for point in selectedData['points']:
                    if 'customdata' in point and len(point['customdata']) > 0:
                        selected_indices.append(point['customdata'][0])
                
                # User has selected points - store selection and update view
                return self._create_manifold_figure(positions, None, selected_prototype, selected_indices), selected_indices
            
            raise PreventUpdate
        
        @app.callback(
            Output('prototype-positions', 'data'),
            [Input('apply-button', 'n_clicks'),
             Input('prototype-manifold', 'selectedData')],
            [State('prototype-positions', 'data')]
        )
        def update_positions(apply_clicks, selectedData, positions):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'prototype-manifold' and selectedData:
                # User has dragged points - update the stored positions
                selected_indices = [point['customdata'][0] for point in selectedData['points']]
                
                for i, idx in enumerate(selected_indices):
                    pos_x = selectedData['points'][i]['x']
                    pos_y = selectedData['points'][i]['y']
                    positions[idx] = [pos_x, pos_y]
                
                return positions
            
            return positions
        
        @app.callback(
            Output('custom-prototype-selection', 'style'),
            [Input('prototype-set-dropdown', 'value')]
        )
        def toggle_custom_prototype_input(value):
            if value == 'custom':
                return {'display': 'block', 'margin-top': '10px', 'margin-bottom': '10px'}
            return {'display': 'none'}
        
        @app.callback(
            [Output('umap-selection-info', 'style'),
             Output('custom-prototype-selection', 'style'),
             Output('umap-selected-prototypes', 'children')],
            [Input('prototype-set-dropdown', 'value')],
            [State('umap-selected-prototypes-store', 'data')]
        )
        def toggle_prototype_input_panels(value, selected_prototypes):
            # Show/hide appropriate input panels based on dropdown selection
            umap_style = {'display': 'block', 'margin-top': '10px', 'margin-bottom': '10px'} if value == 'umap_selection' else {'display': 'none'}
            custom_style = {'display': 'block', 'margin-top': '10px', 'margin-bottom': '10px'} if value == 'custom' else {'display': 'none'}
            
            # Format the currently selected prototypes from UMAP
            if not selected_prototypes or len(selected_prototypes) == 0:
                selected_text = "No prototypes selected in UMAP view."
            else:
                selected_text = f"{len(selected_prototypes)} prototypes selected: " + ", ".join([str(p) for p in sorted(selected_prototypes)])
            
            return umap_style, custom_style, selected_text
            
        @app.callback(
            [Output('local-pca-embedding', 'figure'),
             Output('pca-stats', 'children'),
             Output('current-pca-embedding', 'data')],
            [Input('prototype-set-dropdown', 'value'),
             Input('pca-components-slider', 'value'),
             Input('apply-custom-prototypes', 'n_clicks')],
            [State('custom-prototypes-input', 'value'),
             State('current-pca-embedding', 'data'),
             State('umap-selected-prototypes-store', 'data')]
        )
        def update_local_pca_embedding(community_id, n_components, apply_custom_clicks, custom_protos_input, current_embedding, umap_selected):
            ctx = callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            prototype_indices = []
            
            if community_id == 'custom':
                # Parse custom prototype indices
                if custom_protos_input:
                    try:
                        prototype_indices = [int(idx.strip()) for idx in custom_protos_input.split(',') if idx.strip()]
                    except:
                        prototype_indices = []
            elif community_id == 'umap_selection':
                # Use prototypes selected in UMAP view
                prototype_indices = umap_selected if umap_selected else []
            else:
                # Get prototypes in this community
                prototype_indices = [proto_idx for proto_idx, comm_id in self.prototype_communities.items() 
                                   if comm_id == community_id]
            
            # If no valid prototype indices, return empty figure
            if not prototype_indices:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No prototypes selected",
                    xaxis_title="Component 1",
                    yaxis_title="Component 2"
                )
                return empty_fig, "No prototypes selected for PCA analysis", None
            
            # Compute local PCA embedding
            pca_result = self.compute_local_pca_embedding(prototype_indices, n_components)
            
            if pca_result is None:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="Insufficient data for PCA",
                    xaxis_title="Component 1",
                    yaxis_title="Component 2"
                )
                return empty_fig, "Need at least 2 prototypes for PCA analysis", None
            
            # Create visualization
            fig = go.Figure()
            
            # Add prototype points
            fig.add_trace(go.Scatter(
                x=pca_result['prototype_embedding'][:, 0],
                y=pca_result['prototype_embedding'][:, 1] if n_components > 1 else np.zeros(len(prototype_indices)),
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=[f'rgba({(i*30)%256}, {(i*70)%256}, {(i*110)%256}, 0.7)' for i in prototype_indices],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=[f'P{i}' for i in prototype_indices],
                textposition='top center',
                name='Prototypes'
            ))
            
            # Add feature points if available
            if pca_result['feature_embedding'] is not None and pca_result['patch_indices']:
                # Create hover texts with images when possible
                hover_texts = []
                
                for i, (proto_idx, patch_idx) in enumerate(pca_result['patch_indices']):
                    # Create hover text with prototype info
                    hover_text = f'Prototype: {proto_idx}<br>Feature Vector {patch_idx}'
                    
                    # Try to add image if available
                    if hasattr(self, 'top_activating_patches') and self.top_activating_patches and \
                       proto_idx in self.top_activating_patches and \
                       'patches' in self.top_activating_patches[proto_idx] and \
                       patch_idx < len(self.top_activating_patches[proto_idx]['patches']) and \
                       'patch' in self.top_activating_patches[proto_idx]['patches'][patch_idx]:
                        
                        # Get the patch and convert to base64 for display
                        try:
                            patch = self.top_activating_patches[proto_idx]['patches'][patch_idx]['patch']
                            
                            # Convert tensor to PIL Image
                            patch_np = patch.permute(1, 2, 0).numpy()
                            patch_np = (patch_np - patch_np.min()) / (patch_np.max() - patch_np.min() + 1e-8)
                            patch_np = (patch_np * 255).astype(np.uint8)
                            
                            img = Image.fromarray(patch_np)
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            
                            # Add image to hover text
                            hover_text += f'<br><img src="data:image/png;base64,{img_str}" width="100">'
                            hover_text += f'<br>Score: {self.top_activating_patches[proto_idx]["scores"][patch_idx]:.3f}'
                        except Exception as e:
                            print(f"Error creating hover image: {e}")
                    
                    hover_texts.append(hover_text)
                
                fig.add_trace(go.Scatter(
                    x=pca_result['feature_embedding'][:, 0],
                    y=pca_result['feature_embedding'][:, 1] if n_components > 1 else np.zeros(len(pca_result['feature_embedding'])),
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=['rgba(100, 100, 100, 0.5)' if i % 2 == 0 else 'rgba(150, 150, 150, 0.5)' 
                               for i in range(len(pca_result['feature_embedding']))],
                        symbol='circle'
                    ),
                    hoverinfo='text',
                    hovertext=hover_texts,
                    name='Feature Vectors'
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Local PCA Embedding for {len(prototype_indices)} Prototypes",
                xaxis_title="Component 1",
                yaxis_title="Component 2" if n_components > 1 else "",
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Create stats display
            stats = [
                html.H5("PCA Analysis"),
                html.P(f"Number of prototypes: {len(prototype_indices)}"),
                html.P(f"Prototypes included: {', '.join([str(i) for i in prototype_indices])}")
            ]
            
            if 'explained_variance_ratio' in pca_result:
                var_ratio = pca_result['explained_variance_ratio']
                stats.append(html.P(f"Explained variance: {', '.join([f'{v:.2%}' for v in var_ratio])}"))
                stats.append(html.P(f"Cumulative variance: {np.sum(var_ratio):.2%}"))
            
            return fig, stats, prototype_indices
        
        @app.callback(
            Output('shared-sets-list', 'children'),
            [Input('find-shared-sets-button', 'n_clicks')],
            [State('min-set-size-slider', 'value'),
             State('min-frequency-slider', 'value')]
        )
        def update_shared_sets(n_clicks, min_set_size, min_frequency):
            if n_clicks == 0:
                return html.P("Click 'Find Shared Sets' to analyze prototype co-activations")
                
            # Find shared prototype sets
            shared_sets = self.find_shared_prototype_sets(min_set_size, min_frequency)
            
            if not shared_sets:
                return html.P("No shared prototype sets found with the current parameters")
            
            # Create display
            set_items = []
            for proto_set, frequency in shared_sets[:20]:  # Limit to top 20
                set_items.append(html.Div([
                    html.H5(f"Prototype Set (Frequency: {frequency})"),
                    html.P(f"Prototypes: {', '.join([str(i) for i in sorted(proto_set)])}"),
                    html.Button(
                        "Analyze This Set", 
                        id=f'analyze-set-{"-".join([str(i) for i in sorted(proto_set)])}',
                        n_clicks=0,
                        style={'margin-bottom': '15px'}
                    )
                ]))
            
            return set_items
        
        return app
    
    def _create_manifold_figure(self, positions=None, selectedData=None, selected_prototype=None, selected_prototypes=None):
        """Create the figure for the prototype manifold visualization"""
        if positions is None:
            # Use the original embedding
            positions = self.embedding.tolist()
        
        # Prepare data for the figure
        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]
        
        # Color points based on their primary class
        if hasattr(self, 'class_weights'):
            if self.class_weights.shape[0] == self.num_classes:
                # Class weights shape is (num_classes, num_prototypes)
                primary_classes = [np.argmax(self.class_weights[:, i]) for i in range(self.num_prototypes)]
            else:
                # Class weights shape is (num_prototypes, num_classes)
                primary_classes = [np.argmax(self.class_weights[i]) for i in range(self.num_prototypes)]
        else:
            # Default to 0 if no class weights available
            primary_classes = [0 for _ in range(self.num_prototypes)]
        
        # Initialize selected_prototypes if it's None
        if selected_prototypes is None:
            selected_prototypes = []
        
        # Add the single selected prototype to the list if it exists
        if selected_prototype is not None and selected_prototype not in selected_prototypes:
            selected_prototypes.append(selected_prototype)
        
        # Determine sizes: make selected prototypes larger
        sizes = [25 if i in selected_prototypes else 15 for i in range(self.num_prototypes)]
        
        # Determine color: highlight selected and polysemantic prototypes
        colors = []
        for i in range(self.num_prototypes):
            if i in selected_prototypes:
                colors.append('red')  # Selected prototype
            elif i in self.polysemantic_prototypes:
                colors.append('coral')  # Polysemantic
            else:
                c = primary_classes[i]
                colors.append(f'rgba({(c*30)%256}, {(c*70)%256}, {(c*110)%256}, 0.7)')
        
        fig = go.Figure()
        
        # Add scatter plot for prototypes
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[str(i) for i in range(self.num_prototypes)],
            textposition='top center',
            hovertext=[f'Prototype {i}<br>Primary Class: {primary_classes[i]}<br>Polysemantic: {i in self.polysemantic_prototypes}' 
                     for i in range(self.num_prototypes)],
            customdata=[[i] for i in range(self.num_prototypes)],
            hoverinfo='text'
        ))
        
        # Add edges between related prototypes (if graph is available)
        if hasattr(self, 'prototype_graph') and self.prototype_graph is not None:
            edge_x = []
            edge_y = []
            
            for u, v in self.prototype_graph.edges():
                x0, y0 = positions[u]
                x1, y1 = positions[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none'
            ))
        
        # Update layout
        fig.update_layout(
            title='Prototype Manifold (Colored by Primary Class, Polysemantic in Coral)',
            hovermode='closest',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            dragmode='lasso',  # Allow selecting multiple points
            height=600,
            clickmode='event+select'  # Enable both clicking and selection
        )
        
        # Add annotation for instructions
        fig.add_annotation(
            text="Select prototypes (click+drag) to add them to PCA analysis",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    def run_server(self, debug=True, port=8050):
        """Run the Dash server"""
        self.app.run_server(debug=debug, port=port)

def create_graph_from_pairs(pairs):
    """
    Create a networkx graph from a list of prototype pairs
    
    Parameters:
    -----------
    pairs : list of tuples
        List of prototype pairs (i, j) that are connected
        
    Returns:
    --------
    networkx.Graph
    """
    G = nx.Graph()
    
    # Add all unique nodes first
    unique_nodes = set()
    for i, j in pairs:
        unique_nodes.add(i)
        unique_nodes.add(j)
    
    for node in unique_nodes:
        G.add_node(node)
    
    # Add edges
    for i, j in pairs:
        if G.has_edge(i, j):
            # Increment edge weight if it already exists
            G[i][j]['weight'] += 1
        else:
            # Create new edge with weight 1
            G.add_edge(i, j, weight=1)
    
    return G

# Add a function to make it easier to run the visualization from your existing code
def run_prototype_visualization(model, dataset=None, reducer=None, prototype_graph=None, port=8050):
    """
    Run the prototype visualization tool
    
    Parameters:
    -----------
    model : PIPNet model
        The model to visualize
    dataset : torch Dataset
        Dataset used for analyzing prototypes
    reducer : umap.UMAP
        UMAP reducer for dimensionality reduction
    prototype_graph : networkx.Graph
        Graph of prototype relationships
    port : int
        Port for the web server
        
    Returns:
    --------
    PrototypeVisualizer
    """
    visualizer = PrototypeVisualizer(
        model=model,
        dataset=dataset,
        reducer=reducer,
        prototype_graph=prototype_graph
    )
    
    print(f"Starting visualization server on port {port}...")
    visualizer.app.run(debug=True, port=port)
    
    return visualizer