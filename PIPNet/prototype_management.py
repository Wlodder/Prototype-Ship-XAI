import torch
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy
import matplotlib.pyplot as plt

class PrototypeManager:
    """
    A utility class for managing PIPNet prototypes, including:
    - Splitting polysemantic prototypes using PURE
    - Merging similar prototypes based on various criteria
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the prototype manager.
        
        Args:
            model: PIPNet model
            device: Computing device
        """
        self.model = model
        self.device = device
        self.num_prototypes = model.module._num_prototypes
        self.num_classes = model.module._num_classes
        
    def compute_prototype_attributions(self, x, prototype_idx):
        """
        Compute attributions (circuit) for a specific prototype using Gradient × Activation.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype to compute attributions for
            
        Returns:
            Tensor of attributions for lower-level features
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass to get feature representations
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass
        _, pooled, _, features = self.model(x, features_save=True)
        
        # Get the specific prototype activation
        target_activation = pooled[:, prototype_idx]
        
        # Compute gradients with respect to features
        gradients = torch.autograd.grad(target_activation.sum(), features, 
                                        retain_graph=True, allow_unused=True)[0]
        
        # Gradient × Activation attribution
        attributions = gradients * features
        
        # Sum over spatial dimensions to get attribution per channel
        attributions = attributions.sum(dim=(2, 3))
        
        return attributions
    
    def find_top_activating_samples(self, dataloader, prototype_idx, num_samples=100):
        """
        Find the top activating samples for a prototype.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype 
            num_samples: Number of top samples to return
            
        Returns:
            Tuple of (top samples, top activations)
        """
        self.model.eval()
        activations = []
        images = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different dataloader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass
                _, pooled, _ = self.model(inputs, inference=False)
                
                # Get activations for the target prototype
                batch_activations = pooled[:, prototype_idx].cpu()
                
                for i, activation in enumerate(batch_activations):
                    activations.append(activation.item())
                    images.append(inputs[i].cpu())
        
        # Sort by activation and get top samples
        sorted_indices = np.argsort(activations)[::-1]  # Descending order
        top_indices = sorted_indices[:num_samples]
        
        top_samples = torch.stack([images[i] for i in top_indices])
        top_activations = [activations[i] for i in top_indices]
        
        return top_samples, top_activations
    
    def compute_circuits(self, top_samples, prototype_idx):
        """
        Compute circuits for each top activating sample.
        
        Args:
            top_samples: Tensor of top activating samples
            prototype_idx: Index of the prototype
            
        Returns:
            Tensor of circuit attributions for each sample
        """
        circuits = []
        
        for sample in top_samples:
            sample = sample.unsqueeze(0)  # Add batch dimension
            attribution = self.compute_prototype_attributions(sample, prototype_idx)
            circuits.append(attribution.cpu().detach())
        
        # Stack all circuits
        all_circuits = torch.stack(circuits)
        
        return all_circuits
    
    def split_polysemantic_prototype(self, dataloader, prototype_idx, n_clusters=None, 
                                    visualize=True, adaptive=True, max_clusters=5):
        """
        Split a potentially polysemantic prototype into multiple pure features.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to analyze
            n_clusters: Number of clusters to create (if None, determined automatically)
            visualize: Whether to visualize the results
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Dictionary containing split results including centroids and clusters
        """
        print(f"Analyzing prototype {prototype_idx}...")
        
        # Find top activating samples
        top_samples, top_activations = self.find_top_activating_samples(dataloader, prototype_idx)
        print(f"Found {len(top_samples)} top activating samples.")
        
        # Compute circuits
        circuits = self.compute_circuits(top_samples, prototype_idx)
        
        # Determine the optimal number of clusters if adaptive
        if adaptive and n_clusters is None:
            n_clusters = self._determine_optimal_clusters(circuits, max_clusters)
            print(f"Automatically determined {n_clusters} clusters.")
        elif n_clusters is None:
            n_clusters = 2  # Default to 2 clusters
            
        # Skip splitting if only one cluster is needed
        if n_clusters <= 1:
            print(f"Prototype {prototype_idx} appears to be monosemantic. No split needed.")
            return {
                'prototype_idx': prototype_idx,
                'is_polysemantic': False,
                'n_clusters': 1,
                'cluster_labels': np.zeros(len(top_samples)),
                'centroids': circuits.mean(dim=0).unsqueeze(0),
                'circuits': circuits,
                'top_samples': top_samples,
                'top_activations': top_activations
            }
            
        # Cluster circuits
        cluster_labels, centroids = self._cluster_circuits(circuits, n_clusters)
        
        # Calculate silhouette score to determine if actually polysemantic
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(flat_circuits, cluster_labels)
            is_polysemantic = sil_score > 0.1  # Threshold for considering it polysemantic
        else:
            sil_score = 0
            is_polysemantic = False
            
        print(f"Silhouette score: {sil_score:.3f}, Polysemantic: {is_polysemantic}")
        
        # Visualize if requested
        if visualize:
            self._visualize_clusters(top_samples, cluster_labels, prototype_idx)
            
        # Return the split results
        return {
            'prototype_idx': prototype_idx,
            'is_polysemantic': is_polysemantic,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'cluster_labels': cluster_labels,
            'centroids': centroids,
            'circuits': circuits,
            'top_samples': top_samples,
            'top_activations': top_activations
        }
    
    def split_multiple_prototypes(self, dataloader, prototype_indices, n_clusters=None, 
                                 visualize=True, adaptive=True, max_clusters=5):
        """
        Split multiple prototypes and return their results.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of indices of prototypes to analyze
            n_clusters: Number of clusters per prototype
            visualize: Whether to visualize the results
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Dictionary mapping prototype indices to their split results
        """
        results = {}
        
        for proto_idx in prototype_indices:
            split_result = self.split_polysemantic_prototype(
                dataloader, proto_idx, n_clusters, visualize, adaptive, max_clusters
            )
            results[proto_idx] = split_result
            
        return results
    
    def expand_model_with_split_prototypes(self, split_results, scaling=0.5):
        """
        Expand the model with new prototypes from splitting.
        
        Args:
            split_results: Dictionary of split results from split_multiple_prototypes
            scaling: Scaling factor for the new prototypes (controls how much they deviate from original)
            
        Returns:
            Updated model with expanded prototypes
        """
        from util.pure_expander import expand_pipnet_with_pure_centroids
        
        # Create a copy of the original model
        expanded_model, prototype_mapping = expand_pipnet_with_pure_centroids(
            self.model.module, split_results, device=self.device, scaling=scaling
        )
        
        # Replace the model module with the expanded version
        self.model.module = expanded_model.to(self.device)
        
        # Update internal state
        self.num_prototypes = expanded_model._num_prototypes
        
        return self.model, prototype_mapping
        
    def _cluster_circuits(self, circuits, n_clusters):
        """
        Cluster circuits to identify different semantics.
        
        Args:
            circuits: Tensor of circuit attributions
            n_clusters: Number of clusters (virtual neurons) to create
            
        Returns:
            Tuple of (cluster labels, centroids)
        """
        # Reshape for clustering
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(flat_circuits)
        
        # Get centroids
        centroids = kmeans.cluster_centers_
        centroids = torch.tensor(centroids).reshape(n_clusters, *circuits.shape[1:])
        
        return cluster_labels, centroids
    
    def _determine_optimal_clusters(self, circuits, max_clusters=5, random_state=42):
        """
        Determine the optimal number of clusters for prototype disentanglement.
        
        Args:
            circuits: Tensor of circuit attributions
            max_clusters: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
        # Reshape circuits for clustering
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Must have at least max_clusters+1 samples
        n_samples = flat_circuits.shape[0]
        max_possible = min(max_clusters, n_samples - 1)
        
        if max_possible <= 1:
            return 1
        
        # Store scores for different numbers of clusters
        silhouette_scores = []
        ch_scores = []
        
        # Evaluate different numbers of clusters
        for n_clusters in range(2, max_possible + 1):  # Start from 2 clusters
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(flat_circuits)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters exist
                s_score = silhouette_score(flat_circuits, cluster_labels)
                silhouette_scores.append(s_score)
                
                # Calculate Calinski-Harabasz score (variance ratio)
                ch_score = calinski_harabasz_score(flat_circuits, cluster_labels)
                ch_scores.append(ch_score)
            else:
                silhouette_scores.append(-1)
                ch_scores.append(0)
        
        # Normalize scores
        if silhouette_scores:
            norm_silhouette = np.array(silhouette_scores) / max(max(silhouette_scores), 1e-10)
            norm_ch = np.array(ch_scores) / max(max(ch_scores), 1e-10)
            
            # Combined score (weighted average)
            combined_scores = 0.7 * norm_silhouette + 0.3 * norm_ch
            
            # Get optimal number of clusters (add 2 because we started from 2)
            optimal_clusters = np.argmax(combined_scores) + 2
            
            # If best silhouette score is too low, don't split 
            if max(silhouette_scores) < 0.1:
                optimal_clusters = 1
        else:
            optimal_clusters = 1  # Default if we couldn't compute scores
        
        return optimal_clusters
    
    def _visualize_clusters(self, samples, cluster_labels, prototype_idx):
        """
        Visualize the clusters of a prototype.
        
        Args:
            samples: Tensor of samples
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
        """
        n_clusters = len(np.unique(cluster_labels))
        
        # Create a figure
        fig, axs = plt.subplots(n_clusters, 1, figsize=(12, 3*n_clusters))
        
        # Make axs a list if there's only one cluster
        if n_clusters == 1:
            axs = [axs]
        
        # Visualize samples for each cluster
        for i in range(n_clusters):
            cluster_samples = samples[cluster_labels == i]
            
            # Calculate grid dimensions
            n_samples = min(10, len(cluster_samples))
            if n_samples == 0:
                continue
                
            grid_cols = min(5, n_samples)
            grid_rows = (n_samples + grid_cols - 1) // grid_cols
            
            # Create subplot grid
            for j in range(min(n_samples, 10)):
                ax = plt.subplot(grid_rows, grid_cols, j + 1)
                
                # Denormalize and convert tensor to image
                img = cluster_samples[j].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                ax.imshow(img)
                ax.axis('off')
            
            axs[i].set_title(f"Cluster {i+1} samples (virtual prototype {prototype_idx}.{i+1})")
        
        plt.tight_layout()
        plt.show()
    
    def compute_prototype_similarities(self, similarity_type='feature'):
        """
        Compute similarity matrix between all prototypes.
        
        Args:
            similarity_type: Type of similarity to compute ('feature', 'weight', or 'combined')
            
        Returns:
            Numpy array of similarity scores between prototypes
        """
        if similarity_type not in ['feature', 'weight', 'combined']:
            raise ValueError("similarity_type must be 'feature', 'weight' or 'combined'")
            
        # Get prototype feature vectors (from add-on layer weights)
        proto_features = self.model.module._add_on[0].weight.data.squeeze(2).squeeze(2)
        num_prototypes = proto_features.shape[0]
        
        # Compute feature similarity
        feature_sim = torch.zeros((num_prototypes, num_prototypes), device=self.device)
        for i in range(num_prototypes):
            for j in range(i, num_prototypes):
                # Compute cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    proto_features[i].unsqueeze(0), 
                    proto_features[j].unsqueeze(0),
                    dim=1
                ).item()
                
                feature_sim[i, j] = sim
                feature_sim[j, i] = sim  # Symmetric
                
        # Get classification weights
        class_weights = self.model.module._classification.weight.data
        
        # Compute weight similarity
        weight_sim = torch.zeros((num_prototypes, num_prototypes), device=self.device)
        for i in range(num_prototypes):
            for j in range(i, num_prototypes):
                # Compute cosine similarity between weight vectors
                sim = torch.nn.functional.cosine_similarity(
                    class_weights[:, i].unsqueeze(0),
                    class_weights[:, j].unsqueeze(0),
                    dim=1
                ).item()
                
                weight_sim[i, j] = sim
                weight_sim[j, i] = sim  # Symmetric
                
        # Return the requested similarity
        if similarity_type == 'feature':
            return feature_sim.cpu().numpy()
        elif similarity_type == 'weight':
            return weight_sim.cpu().numpy()
        else:  # combined
            # Weighted average of both similarities
            return (0.5 * feature_sim + 0.5 * weight_sim).cpu().numpy()
    
    def identify_merge_candidates(self, similarity_threshold=0.85, 
                                 similarity_type='combined',
                                 min_weight=0.01,
                                 max_candidates=20):
        """
        Identify candidate pairs of prototypes for merging.
        
        Args:
            similarity_threshold: Threshold for considering prototypes similar enough to merge
            similarity_type: Type of similarity to use ('feature', 'weight', or 'combined')
            min_weight: Minimum weight in classification layer to consider a prototype relevant
            max_candidates: Maximum number of candidate pairs to return
            
        Returns:
            List of (prototype1, prototype2, similarity) tuples sorted by similarity
        """
        # Get similarities
        similarities = self.compute_prototype_similarities(similarity_type)
        
        # Get classification weights
        class_weights = self.model.module._classification.weight.data
        
        # Identify which prototypes are actually used (have sufficient weight)
        used_prototypes = []
        for p in range(self.num_prototypes):
            if torch.max(class_weights[:, p]) > min_weight:
                used_prototypes.append(p)
        
        print(f"Found {len(used_prototypes)} prototypes with weight > {min_weight}")
        
        # Find pairs of prototypes with high similarity
        candidates = []
        for i in range(len(used_prototypes)):
            for j in range(i+1, len(used_prototypes)):
                p1, p2 = used_prototypes[i], used_prototypes[j]
                sim = similarities[p1, p2]
                
                if sim > similarity_threshold:
                    candidates.append((p1, p2, sim))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Limit the number of candidates
        candidates = candidates[:max_candidates]
        
        print(f"Found {len(candidates)} candidate pairs for merging with similarity > {similarity_threshold}")
        return candidates
    
    def merge_prototypes(self, prototype_pairs, merge_strategy='weighted_average'):
        """
        Merge pairs of prototypes.
        
        Args:
            prototype_pairs: List of (prototype1, prototype2) tuples to merge
            merge_strategy: Strategy for merging ('weighted_average' or 'max')
            
        Returns:
            Updated model
        """
        # Create a copy of the model to modify
        new_model = deepcopy(self.model)
        
        # Get add-on layer weights and classification weights
        proto_weights = new_model.module._add_on[0].weight.data
        class_weights = new_model.module._classification.weight.data
        
        # Check for conflicts in merge pairs
        processed_prototypes = set()
        valid_pairs = []
        
        for p1, p2 in prototype_pairs:
            if p1 in processed_prototypes or p2 in processed_prototypes:
                continue  # Skip if either prototype has already been merged
            
            valid_pairs.append((p1, p2))
            processed_prototypes.add(p1)
            processed_prototypes.add(p2)
        
        # Process each pair
        for p1, p2 in valid_pairs:
            print(f"Merging prototypes {p1} and {p2}")
            
            # Merge prototype features
            if merge_strategy == 'weighted_average':
                # Weighted by their maximum class weights
                w1 = torch.max(class_weights[:, p1]).item()
                w2 = torch.max(class_weights[:, p2]).item()
                total_weight = w1 + w2
                
                # Compute weighted average
                merged_features = ((w1 * proto_weights[p1] + w2 * proto_weights[p2]) / total_weight)
                
            elif merge_strategy == 'max':
                # Take the prototype with higher maximum class weight
                if torch.max(class_weights[:, p1]) >= torch.max(class_weights[:, p2]):
                    merged_features = proto_weights[p1].clone()
                else:
                    merged_features = proto_weights[p2].clone()
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
            # Update the first prototype with merged features
            proto_weights[p1] = merged_features
            
            # Sum the classification weights
            class_weights[:, p1] += class_weights[:, p2]
            
            # Zero out the second prototype (effectively removing it)
            class_weights[:, p2] = 0.0
        
        # Update the model
        self.model = new_model
        
        print(f"Merged {len(valid_pairs)} prototype pairs")
        return self.model
    
    def visualize_prototype_activations(self, dataloader, prototype_idx, n_samples=5):
        """
        Visualize what activates a specific prototype.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to visualize
            n_samples: Number of samples to show
        """
        # Get top activating samples
        top_samples, top_activations = self.find_top_activating_samples(
            dataloader, prototype_idx, n_samples
        )
        
        # Create a figure
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        
        # Show each sample
        for i in range(n_samples):
            # Denormalize and convert tensor to image
            img = top_samples[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            axes[i].imshow(img)
            axes[i].set_title(f"Activation: {top_activations[i]:.3f}")
            axes[i].axis('off')
        
        plt.suptitle(f"Top activations for prototype {prototype_idx}")
        plt.tight_layout()
        plt.show()
    
    def compare_prototypes(self, prototype_indices, dataloader, n_samples=3):
        """
        Compare activations of multiple prototypes to help decide if they should be merged.
        
        Args:
            prototype_indices: List of prototype indices to compare
            dataloader: DataLoader containing the dataset
            n_samples: Number of samples to show per prototype
        """
        # Get top activating samples for each prototype
        all_samples = []
        all_activations = []
        
        for idx in prototype_indices:
            samples, activations = self.find_top_activating_samples(
                dataloader, idx, n_samples
            )
            all_samples.append(samples)
            all_activations.append(activations)
        
        # Get classification weights for these prototypes
        class_weights = self.model.module._classification.weight.data.cpu().numpy()
        max_classes = np.argmax(class_weights, axis=0)
        
        # Create a figure
        fig, axes = plt.subplots(
            len(prototype_indices), 
            n_samples, 
            figsize=(n_samples * 3, len(prototype_indices) * 3)
        )
        
        # Show each prototype's samples
        for i, idx in enumerate(prototype_indices):
            # Get max class for this prototype
            max_class_idx = max_classes[idx]
            max_weight = class_weights[max_class_idx, idx]
            
            for j in range(n_samples):
                # Denormalize and convert tensor to image
                img = all_samples[i][j].permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Act: {all_activations[i][j]:.3f}")
                axes[i, j].axis('off')
            
            # Add a label with prototype info
            axes[i, 0].set_ylabel(f"Proto {idx}\nClass {max_class_idx}\nWeight {max_weight:.3f}")
        
        # Calculate feature similarity
        proto_features = self.model.module._add_on[0].weight.data
        feature_similarity = torch.nn.functional.cosine_similarity(
            proto_features[prototype_indices[0]].flatten().unsqueeze(0),
            proto_features[prototype_indices[1]].flatten().unsqueeze(0)
        ).item()
        
        # Calculate weight similarity
        weight_similarity = torch.nn.functional.cosine_similarity(
            self.model.module._classification.weight[:, prototype_indices[0]].unsqueeze(0),
            self.model.module._classification.weight[:, prototype_indices[1]].unsqueeze(0)
        ).item()
        
        plt.suptitle(f"Comparing prototypes: Feature similarity={feature_similarity:.3f}, Weight similarity={weight_similarity:.3f}")
        plt.tight_layout()
        plt.show()