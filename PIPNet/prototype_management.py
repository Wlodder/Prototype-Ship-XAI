import torch
import umap
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from typing import List, Dict, Tuple, Optional, Union, Any
from copy import deepcopy
from util.attribution_analyzer import MultiLayerAttributionAnalyzer
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
            for batch in tqdm(dataloader):
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
                    if activation > 0.5:
                        activations.append(activation.item())
                        images.append(inputs[i].cpu())
        
        # Sort by activation and get top samples
        sorted_indices = np.argsort(activations)[::-1]  # Descending order
        # top_indices = sorted_indices[:num_samples]
        top_indices = sorted_indices[:min(num_samples, len(activations))]
        
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

    def visualize_prototype_patches(self, samples, cluster_labels, prototype_idx):
        """
        Visualize the image patches that activate a prototype, grouped by cluster.

        This function displays both the original images (with highlighted regions) and
        the extracted patches that activate the prototype, organized by cluster.
        This helps understand exactly what visual features each prototype cluster
        is detecting and in what context they appear.

        Args:
            samples: Tensor of samples that activate the prototype
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
        """
        n_clusters = len(np.unique(cluster_labels))

        # Create a figure for showing original images and their corresponding patches
        # Each row represents a different cluster, with pairs of images (original + patch)
        fig, axs = plt.subplots(n_clusters, 6, figsize=(15, 3*n_clusters))

        # Make axs a 2D array even if there's only one cluster
        if n_clusters == 1:
            axs = np.array([axs])

        # Process each cluster separately
        for i in range(n_clusters):
            # Get samples belonging to this cluster
            cluster_samples = samples[cluster_labels == i]
            
            # Skip empty clusters (shouldn't happen, but just in case)
            if len(cluster_samples) == 0:
                continue
                
            # Find the most strongly activating samples for this cluster
            with torch.no_grad():
                # We'll calculate activations and locate the patches
                activations = []
                patch_locations = []
                
                for sample in cluster_samples[:5]:  # Process up to 5 samples
                    # Forward pass to get activations
                    sample = sample.to(self.device).unsqueeze(0)
                    _, pooled, _, features = self.model(sample, features_save=True)
                    proto_features = self.model.module._add_on(features)
                    
                    # Record activation strength for this prototype
                    activation = pooled[0, prototype_idx].item()
                    activations.append(activation)
                    
                    # Find where in the feature map this prototype activates most strongly
                    proto_map = proto_features[0, prototype_idx]
                    h_idx, w_idx = np.unravel_index(torch.argmax(proto_map).cpu(), proto_map.shape)
                    
                    # Map feature map coordinates to image space
                    img_size = sample.shape[2]  # Assuming square image
                    feature_size = proto_map.shape[0]
                    scale = img_size / feature_size
                    
                    # Calculate patch size - we want a reasonably sized crop
                    patch_size = max(32, int(scale))
                    
                    # Calculate patch center coordinates
                    center_h = int((h_idx.item() + 0.5) * scale)
                    center_w = int((w_idx.item() + 0.5) * scale)
                    
                    # Calculate patch boundaries
                    h_min = max(0, center_h - patch_size//2)
                    h_max = min(img_size, center_h + patch_size//2)
                    w_min = max(0, center_w - patch_size//2)
                    w_max = min(img_size, center_w + patch_size//2)
                    
                    patch_locations.append((h_min, h_max, w_min, w_max))
                
                # Sort samples by activation strength and take top 3
                sorted_indices = np.argsort(activations)[::-1][:3]
                
            # Display each of the top 3 samples along with their patches
            for j, idx in enumerate(sorted_indices):
                if j >= 3:  # Only show top 3
                    break
                    
                # Display the original image
                img = cluster_samples[idx].cpu()
                img = img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                axs[i, j*2].imshow(img)
                axs[i, j*2].set_title(f"Act: {activations[idx]:.3f}")
                axs[i, j*2].axis('off')
                
                # Draw rectangle around the activating region
                h_min, h_max, w_min, w_max = patch_locations[idx]
                rect = plt.Rectangle((w_min, h_min), w_max-w_min, h_max-h_min, 
                                    linewidth=2, edgecolor='r', facecolor='none')
                axs[i, j*2].add_patch(rect)
                
                # Display the extracted patch
                patch = cluster_samples[idx][:, h_min:h_max, w_min:w_max].cpu()
                patch = patch.permute(1, 2, 0).numpy()
                patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                
                axs[i, j*2+1].imshow(patch)
                axs[i, j*2+1].set_title("Patch")
                axs[i, j*2+1].axis('off')
            
            # Add row label to identify the cluster
            axs[i, 0].set_ylabel(f"Cluster {i+1}\n{len(cluster_samples)} samples")

        plt.suptitle(f"Activating Samples and Patches for Prototype {prototype_idx}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show() 


    def visualize_circuit_clusters_umap(self, circuits, cluster_labels, prototype_idx):
        """
        Visualize UMAP embedding of circuit attributions colored by cluster.
        
        This function creates a 2D UMAP projection of the high-dimensional circuit 
        attributions, allowing you to visualize how well-separated the clusters are.
        Points are colored according to their cluster assignment, making it easy to
        see if the prototype is truly polysemantic.
        
        Args:
            circuits: Tensor of circuit attributions (activation patterns)
            cluster_labels: Numpy array of cluster labels from KMeans
            prototype_idx: Index of the prototype being analyzed
        """
        # Reshape circuits for UMAP processing - flatten all dimensions except batch
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Standardize the data to have zero mean and unit variance
        # This is important for UMAP to work well with neural network activations
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(flat_circuits)
        
        # Apply UMAP for dimensionality reduction
        # Adjust n_neighbors based on dataset size to avoid errors with small datasets
        umap_reducer = umap.UMAP(
            n_components=2,  # We want a 2D visualization
            n_neighbors=min(15, len(flat_circuits)-1),  # Adapt to dataset size
            min_dist=0.1,    # Controls how tightly points cluster together
            random_state=42  # For reproducibility
        )
        embedding = umap_reducer.fit_transform(scaled_data)
        
        # Create the visualization plot
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        # Plot each cluster with a different color
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=f'Cluster {i+1}',
                alpha=0.7,
                s=100
            )
        
        plt.title(f"UMAP Embedding of Circuit Attributions for Prototype {prototype_idx}")
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        # Add density contours to highlight cluster boundaries
        # This helps visualize the distribution density within each cluster
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            if np.sum(mask) > 5:  # Need enough points for density estimation
                try:
                    from scipy.stats import gaussian_kde
                    x = embedding[mask, 0]
                    y = embedding[mask, 1]
                    xy = np.vstack([x, y])
                    kde = gaussian_kde(xy)
                    
                    # Create a grid and compute density
                    x_grid = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 50)
                    y_grid = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 50)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = kde(positions).reshape(X.shape)
                    
                    # Plot contours to show density regions
                    plt.contour(X, Y, Z, colors=[colors[i]], alpha=0.5, linewidths=2)
                except:
                    # Skip contour if density estimation fails (can happen with very small clusters)
                    pass
        
        plt.tight_layout()
        plt.show()

    def split_polysemantic_prototype_multilayer(self, dataloader, prototype_idx, 
                                            layer_indices=[-1, -5, -10], 
                                            n_clusters=None, visualize=True, 
                                            adaptive=True, max_clusters=5,
                                            clustering_method='hdbscan'):
        """
        Enhanced version of split_polysemantic_prototype that uses attributions 
        from multiple network layers.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to analyze
            layer_indices: List of layer indices to analyze (negative indices supported)
            n_clusters: Number of clusters to create (if None, determined automatically)
            visualize: Whether to visualize the results
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            clustering_method: Clustering method to use
            
        Returns:
            Dictionary containing split results
        """
        # Initialize the multi-layer analyzer
        analyzer = MultiLayerAttributionAnalyzer(self.model, device=self.device)
        
        # Perform the multi-layer analysis
        results = analyzer.analyze_prototype(
            dataloader=dataloader,
            prototype_idx=prototype_idx,
            layer_indices=layer_indices,
            n_clusters=n_clusters,
            adaptive=adaptive,
            max_clusters=max_clusters,
            clustering_method=clustering_method,
            visualize=visualize
        )
        
        # Extract the relevant information for prototype splitting
        return {
            'prototype_idx': prototype_idx,
            'is_polysemantic': results['is_polysemantic'],
            'n_clusters': results['n_clusters'],
            'silhouette_score': results['silhouette_score'],
            'cluster_labels': results['cluster_labels'],
            'centroids': results['centroids'],
            'circuits': results['circuits'],
            'top_samples': results['top_samples'],
            'top_activations': results['top_activations'],
            'layer_indices': results['layer_indices']
        }

    def split_polysemantic_prototype(self, dataloader, prototype_idx, n_clusters=None, 
                                    visualize=True, adaptive=True, max_clusters=5):
        """
        Split a potentially polysemantic prototype into multiple pure features.
        
        This function analyzes a prototype to determine if it represents multiple
        distinct concepts (polysemantic) or a single concept (monosemantic). If it's
        polysemantic, the function identifies the different concepts through clustering
        and visualizes them.
        
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
        
        # Step 1: Find top activating samples for this prototype
        top_samples, top_activations = self.find_top_activating_samples(dataloader, prototype_idx)
        print(f"Found {len(top_samples)} top activating samples.")
        
        # Step 2: Compute circuit representations for each sample
        circuits = self.compute_circuits(top_samples, prototype_idx)
        
        # Step 3: Determine the optimal number of clusters if needed
        if adaptive and n_clusters is None:
            # Let the algorithm determine the right number of clusters
            n_clusters = self._determine_optimal_clusters(circuits, max_clusters)
            print(f"Automatically determined {n_clusters} clusters.")
        elif n_clusters is None:
            # Use default if not specified
            n_clusters = 2
            
        # If only one cluster is needed, the prototype is likely monosemantic
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
                
        # Step 4: Cluster circuits to identify different concepts
        cluster_labels, centroids = self._cluster_circuits(circuits, n_clusters)
        
        # Step 5: Calculate silhouette score to evaluate cluster quality
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(flat_circuits, cluster_labels)
            # A silhouette score > 0.1 indicates decent clustering structure
            is_polysemantic = sil_score > 0.1
        else:
            sil_score = 0
            is_polysemantic = False
                
        print(f"Silhouette score: {sil_score:.3f}, Polysemantic: {is_polysemantic}")
        
        # Step 6: Visualize the results if requested
        if visualize:
            print("Visualizing UMAP embedding of circuit clusters...")
            self.visualize_circuit_clusters_umap(circuits, cluster_labels, prototype_idx)
            
            print("Visualizing prototype clusters and patches...")
            self._visualize_clusters(top_samples, cluster_labels, prototype_idx)
            self.visualize_prototype_patches(top_samples, cluster_labels, prototype_idx)
                
        # Return all the analysis results
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
                                 visualize=True, adaptive=True, max_clusters=5,
                                 algorithm='kmeans'):
        """
        Split multiple prototypes and return their results.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of indices of prototypes to analyze
            n_clusters: Number of clusters per prototype
            visualize: Whether to visualize the results
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            algorithm: Clustering algorithm to use ('kmeans', 'hdbscan', 'gmm', or 'spectral')
            
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

    def split_multiple_prototypes_multi_depth(self, dataloader, prototype_indices, n_clusters=None, 
                                 visualize=True, adaptive=True, max_clusters=5,
                                 algorithm='kmeans'):
        """
        Split multiple prototypes and return their results.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of indices of prototypes to analyze
            n_clusters: Number of clusters per prototype
            visualize: Whether to visualize the results
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            algorithm: Clustering algorithm to use ('kmeans', 'hdbscan', 'gmm', or 'spectral')
            
        Returns:
            Dictionary mapping prototype indices to their split results
        """
        analyzer = MultiLayerAttributionAnalyzer(self.model, device=self.device)
        results = {}

        for proto_idx in prototype_indices:
            split_result = analyzer.analyze_prototype(
                dataloader=dataloader,
                prototype_idx=proto_idx,  # Choose any prototype to analyze
                layer_indices=[-2, -3, -4],  # Analyze at different depths
                adaptive=True,
                clustering_method='hdbscan',
                visualize=True,
            )
            results[proto_idx] = split_result

        html = analyzer.create_activation_patch_visualization(results, prototype_indices)

        return results
    
    
    def project_to_prototype_manifold(self, new_prototypes, reference_dataloader=None, steps=10):
        """
        Project new prototype vectors onto the manifold of valid prototypes.
        
        This uses a PCA-based approach to ensure new prototypes remain in the valid subspace.
        
        Args:
            new_prototypes: Tensor of new prototype vectors to project
            reference_dataloader: DataLoader for computing the manifold
            steps: Number of steps for gradual projection
            
        Returns:
            Projected prototype vectors
        """
        # Get existing prototypes for reference
        existing_prototypes = self.model.module._add_on[0].weight.data.clone()
        
        # Flatten for easier processing
        flat_existing = existing_prototypes.view(existing_prototypes.size(0), -1)
        flat_new = new_prototypes.view(new_prototypes.size(0), -1)
        
        # Compute PCA on existing prototypes to define the manifold
        from sklearn.decomposition import PCA
        
        # Center the data
        mean_prototype = torch.mean(flat_existing, dim=0, keepdim=True)
        centered_existing = flat_existing - mean_prototype
        
        # Compute PCA basis
        pca = PCA(n_components=min(flat_existing.size(0)-1, flat_existing.size(1)))
        pca.fit(centered_existing.cpu().numpy())
        
        # Optional: augment manifold with actual feature vectors from data
        if reference_dataloader is not None:
            print("Computing prototype manifold from reference data...")
            feature_vectors = []
            
            # Collect feature vectors from reference data
            with torch.no_grad():
                for batch in reference_dataloader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                    
                    # Get features at the level where prototypes operate
                    features = self.model.module._net(inputs)
                    
                    # Sample random locations from feature map
                    b, c, h, w = features.shape
                    for i in range(b):
                        for _ in range(10):  # Sample 10 locations per image
                            h_idx = np.random.randint(0, h)
                            w_idx = np.random.randint(0, w)
                            feature_vectors.append(features[i, :, h_idx, w_idx].cpu().numpy())
                            
                    if len(feature_vectors) > 500:  # Limit to 500 samples
                        break
            
            # Recompute PCA with both existing prototypes and feature vectors
            if len(feature_vectors) > 0:
                combined_data = np.vstack([centered_existing.cpu().numpy(), np.array(feature_vectors)])
                pca.fit(combined_data)
        
        # Project new prototypes to the manifold in steps (gradual projection)
        projected_new = flat_new.clone()
        for step in range(steps):
            # Interpolate between original and fully projected
            alpha = (step + 1) / steps
            
            # Center
            projected_new = projected_new.to(self.device)
            centered_new = projected_new - mean_prototype
            
            # Project to PCA space and back
            projected = pca.inverse_transform(pca.transform(centered_new.cpu().numpy()))
            projected = torch.tensor(projected, device=self.device)
            
            # Add mean back
            projected = projected + mean_prototype
            
            # Interpolate between current and projected
            projected = projected.to(self.device)
            flat_new = flat_new.to(self.device)
            projected_new = (1 - alpha) * flat_new + alpha * projected
        
        # Reshape back to original prototype shape
        projected_new = projected_new.view(new_prototypes.size())
        
        return projected_new

    def adaptive_prototype_expansion(self, split_results, base_step_size=0.5, 
                                   manifold_projection=True, reference_dataloader=None):
        """
        Expand model with new prototypes using adaptive step sizing and manifold projection.
        
        Args:
            split_results: Dictionary of split results from PURE
            base_step_size: Initial step size for prototype expansion
            manifold_projection: Whether to project new prototypes to the prototype manifold
            reference_dataloader: DataLoader for computing the prototype manifold
            
        Returns:
            Updated model and prototype mapping
        """
        # Create a copy of the original model to avoid modifying the original
        from copy import deepcopy
        import torch.nn as nn
        
        expanded_model = deepcopy(self.model.module)
        
        # Extract model parameters
        original_num_prototypes = expanded_model._num_prototypes
        num_classes = expanded_model._num_classes
        
        # Count new prototypes needed
        total_new_clusters = 0
        for proto_idx, result in split_results.items():
            n_clusters = len(np.unique(result['cluster_labels']))
            if isinstance(proto_idx, list):
                total_new_clusters += n_clusters - len(proto_idx)
            else:
                total_new_clusters += (n_clusters - 1)  # Subtract 1 as first cluster uses original prototype
        
        num_new_prototypes = original_num_prototypes + total_new_clusters
        print(f"Expanding from {original_num_prototypes} to {num_new_prototypes} prototypes")
        
        # Create prototype mapping
        prototype_mapping = {}
        next_idx = original_num_prototypes
        
        # Get prototype feature weights
        proto_weights = expanded_model._add_on[0].weight.data
        class_weights = expanded_model._classification.weight.data
        
        # Create expanded layers
        new_proto_layer = nn.Conv2d(
            in_channels=expanded_model._add_on[0].in_channels,
            out_channels=num_new_prototypes,
            kernel_size=expanded_model._add_on[0].kernel_size,
            stride=expanded_model._add_on[0].stride,
            padding=expanded_model._add_on[0].padding,
            bias=(expanded_model._add_on[0].bias is not None)
        )
        
        new_classification = nn.Linear(
            in_features=num_new_prototypes,
            out_features=num_classes,
            bias=(expanded_model._classification.bias is not None)
        )
        
        # Copy original weights
        with torch.no_grad():
            new_proto_layer.weight.data[:original_num_prototypes] = proto_weights
            if expanded_model._add_on[0].bias is not None:
                new_proto_layer.bias.data[:original_num_prototypes] = expanded_model._add_on[0].bias.data
                
            new_classification.weight.data[:, :original_num_prototypes] = class_weights
            if expanded_model._classification.bias is not None:
                new_classification.bias.data = expanded_model._classification.bias.data
        
        # Process each prototype
        for proto_idx, result in split_results.items():
            cluster_labels = result['cluster_labels']
            centroids = result['centroids']
            unique_clusters = np.unique(cluster_labels)
            
            # Process as single or list of prototype indices
            proto_indices = [proto_idx] if not isinstance(proto_idx, list) else proto_idx
            
            # Create mapping dict
            for p_idx in proto_indices:
                prototype_mapping[p_idx] = []
            
            # First cluster: modify original prototype
            if not isinstance(proto_idx, list):
                # Identify the optimal step size for this prototype
                first_centroid = centroids[0]
                original_proto = proto_weights[proto_idx].clone()
                
                # Compute prototype activation statistics to determine step size
                importance = torch.max(class_weights[:, proto_idx]).item()
                
                # Adaptive step size based on prototype importance
                # More important prototypes get smaller steps to prevent disruption
                adaptive_step = base_step_size / (1 + importance)
                print(f"Prototype {proto_idx} importance: {importance:.4f}, adaptive step: {adaptive_step:.4f}")
                
                # Compute the direction vector from original to centroid
                if isinstance(first_centroid, torch.Tensor):
                    centroid_tensor = first_centroid.view_as(original_proto)
                else:
                    centroid_tensor = torch.tensor(first_centroid, device=self.device).view_as(original_proto)
                    
                centroid_tensor = centroid_tensor.to(self.device) 
                step_vector = centroid_tensor - original_proto
                
                # Apply the step
                updated_proto = original_proto + adaptive_step * step_vector
                
                # Update the prototype
                new_proto_layer.weight.data[proto_idx] = updated_proto
                
                # Add to mapping
                for p_idx in proto_indices:
                    prototype_mapping[p_idx].append(proto_idx)
            
            # Process remaining clusters (create new prototypes)
            start_cluster = 1 if not isinstance(proto_idx, list) else 0
            for i, cluster_idx in enumerate(unique_clusters[start_cluster:], start=start_cluster):
                # Get centroid for this cluster
                centroid = centroids[i]
                
                # Pick a reference prototype (first one if list, or the one specified)
                ref_proto_idx = proto_indices[0] if isinstance(proto_idx, list) else proto_idx
                
                # Get the original prototype
                original_proto = proto_weights[ref_proto_idx].clone()
                
                # Create the new prototype by stepping from original toward centroid
                if isinstance(centroid, torch.Tensor):
                    centroid_tensor = centroid.view_as(original_proto)
                else:
                    centroid_tensor = torch.tensor(centroid, device=self.device).view_as(original_proto)
                    
                # Compute step vector
                centroid_tensor = centroid_tensor.to(self.device) 
                step_vector = centroid_tensor - original_proto
                
                # Use adaptive step size as before
                importance = torch.max(class_weights[:, ref_proto_idx]).item()
                adaptive_step = base_step_size / (1 + importance * 0.5)  # Slightly different formula for new prototypes
                
                # Apply the step
                new_proto = original_proto + adaptive_step * step_vector
                
                # Set the new prototype weights
                new_proto_layer.weight.data[next_idx] = new_proto
                
                # Set classification weights for the new prototype
                # Copy from original prototype
                new_classification.weight.data[:, next_idx] = class_weights[:, ref_proto_idx].clone()
                
                # Add to mapping
                for p_idx in proto_indices:
                    prototype_mapping[p_idx].append(next_idx)
                    
                # Increment index
                next_idx += 1
        
        # Project new prototypes to the manifold if requested
        if manifold_projection and next_idx > original_num_prototypes:
            # Project all prototypes to ensure they're on the manifold
            new_protos = new_proto_layer.weight.data[original_num_prototypes:next_idx]
            if len(new_protos) > 0:
                projected_protos = self.project_to_prototype_manifold(
                    new_protos, reference_dataloader
                )
                new_proto_layer.weight.data[original_num_prototypes:next_idx] = projected_protos
        
        # Update the model
        # Replace the prototype layer
        expanded_model._add_on[0] = new_proto_layer
        expanded_model._classification = new_classification
        expanded_model._num_prototypes = num_new_prototypes
        
        # Convert to same device/structure as original
        expanded_model = expanded_model.to(self.device)
        
        # Replace the model module
        self.model.module = expanded_model
        
        # Update internal state
        self.num_prototypes = expanded_model._num_prototypes
        
        return self.model, prototype_mapping


    def finetune_split_prototypes(self, split_results, expanded_model, mapping, 
                                learning_rate=0.001, epochs=5):
        """Fine-tune split prototypes with supervision from cluster assignments"""
        # Set up optimizer for prototype weights only
        proto_params = []
        for proto_idx, new_indices in mapping.items():
            # Add parameters for original and new prototypes
            for idx in new_indices:
                proto_params.append({
                    'params': expanded_model.module._add_on[0].weight[idx], 
                    'lr': learning_rate
                })
        
        optimizer = torch.optim.Adam(proto_params)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Process each split prototype
            for proto_idx, new_indices in mapping.items():
                result = split_results[proto_idx]
                samples = result['top_samples'].to(self.device)
                cluster_labels = result['cluster_labels']
                
                # Forward pass
                proto_features, pooled, _ = expanded_model(samples)
                
                # Compute specialization loss
                loss = 0
                for cluster_idx, proto_idx in enumerate(new_indices):
                    # Get samples from this cluster
                    cluster_mask = (cluster_labels == cluster_idx)
                    
                    if not np.any(cluster_mask):
                        continue
                    
                    # Positive examples: Encourage high activation
                    pos_activations = pooled[cluster_mask, proto_idx]
                    pos_loss = -torch.mean(pos_loss)
                    
                    # Negative examples: Encourage low activation
                    neg_activations = pooled[~cluster_mask, proto_idx]
                    neg_loss = torch.mean(F.relu(neg_activations - 0.1))
                    
                    # Specialization loss
                    loss += pos_loss + neg_loss
                
                # Backprop and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return expanded_model


    def split_prototype_with_centroids(self, split_results, proto_idx, step_size=0.5):
        """Split prototype by directly applying centroid steps to weights"""
        # Create a copy of the model
        model = deepcopy(self.model.module)
        result = split_results[proto_idx]
        
        # Get centroids and determine number of new prototypes needed
        n_clusters = len(np.unique(result['cluster_labels']))
        n_new_prototypes = n_clusters - 1  # First cluster uses original prototype
        
        # Choose the most informative layer for centroids
        layer_idx = -1  # Default to add-on layer
        for idx in sorted(result['centroids'].keys(), reverse=True):
            layer_idx = idx
            break
        
        # Get original prototype weights
        original_weights = model._add_on[0].weight.data[proto_idx].clone()
        original_shape = original_weights.shape
        
        # 1. Expand the model's capacity
        old_num_protos = model._num_prototypes
        new_num_protos = old_num_protos + n_new_prototypes
        
        # Create expanded add-on layer
        old_addon = model._add_on[0]
        new_addon = nn.Conv2d(
            in_channels=old_addon.in_channels,
            out_channels=new_num_protos,
            kernel_size=old_addon.kernel_size,
            stride=old_addon.stride,
            padding=old_addon.padding,
            bias=(old_addon.bias is not None)
        )
        
        # Copy original weights
        with torch.no_grad():
            new_addon.weight.data[:old_num_protos] = old_addon.weight.data
            if old_addon.bias is not None:
                new_addon.bias.data[:old_num_protos] = old_addon.bias.data
        
        # Create expanded classification layer
        old_class = model._classification
        new_class = nn.Linear(
            in_features=new_num_protos,
            out_features=model._num_classes,
            bias=(old_class.bias is not None)
        )
        
        # Copy original weights
        with torch.no_grad():
            new_class.weight.data[:, :old_num_protos] = old_class.weight.data
            if old_class.bias is not None:
                new_class.bias.data = old_class.bias.data
        
        # 2. Add the new prototypes with centroids
        next_idx = old_num_protos
        mapping = {proto_idx: [proto_idx]}
        
        with torch.no_grad():
            # First cluster: modify original prototype
            centroid = result['centroids'][layer_idx][0].to(self.device)
            reshaped_centroid = centroid.view(-1)[:original_shape[0]].view(original_shape)
            new_addon.weight.data[proto_idx] = original_weights + step_size * reshaped_centroid
            
            # Additional clusters: create new prototypes
            for cluster_idx in range(1, n_clusters):
                centroid = result['centroids'][layer_idx][cluster_idx].to(self.device)
                reshaped_centroid = centroid.view(-1)[:original_shape[0]].view(original_shape)
                
                # Create new prototype
                new_addon.weight.data[next_idx] = original_weights + step_size * reshaped_centroid
                
                # Copy classification weights
                new_class.weight.data[:, next_idx] = old_class.weight.data[:, proto_idx].clone()
                
                # Add to mapping
                mapping[proto_idx].append(next_idx)
                next_idx += 1
        
        # Update model
        model._add_on[0] = new_addon
        model._classification = new_class
        model._num_prototypes = new_num_protos
        
        # Wrap model and return
        expanded_model = nn.DataParallel(model.to(self.device))
        self.model = expanded_model
        return expanded_model, mapping

    def expand_model_with_split_prototypes(self, split_results, scaling=0.5, 
                                         use_adaptive_expansion=True,
                                         manifold_projection=True,
                                         reference_dataloader=None):
        """
        Expand the model with new prototypes from splitting.
        
        Args:
            split_results: Dictionary of split results from split_multiple_prototypes
            scaling: Scaling factor for the new prototypes (controls how much they deviate from original)
            use_adaptive_expansion: Whether to use the advanced adaptive expansion method
            manifold_projection: Whether to project new prototypes to the prototype manifold
            reference_dataloader: DataLoader for computing the prototype manifold
            
        Returns:
            Updated model with expanded prototypes
        """
        if use_adaptive_expansion:
            return self.adaptive_prototype_expansion(
                split_results, 
                base_step_size=scaling,
                manifold_projection=manifold_projection,
                reference_dataloader=reference_dataloader
            )
        else:
            # Fall back to the original expansion method
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
        
    def _cluster_circuits(self, circuits, n_clusters, algorithm='kmeans'):
        """
        Cluster circuits using various algorithms to identify different semantics.
        
        Args:
            circuits: Tensor of circuit attributions
            n_clusters: Number of clusters (virtual neurons) to create
            algorithm: Clustering algorithm to use ('kmeans', 'hdbscan', 'gmm', or 'spectral')
            
        Returns:
            Tuple of (cluster labels, centroids)
        """
        # Reshape for clustering
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        if algorithm == 'kmeans':
            # Standard K-means clustering
            from sklearn.cluster import KMeans
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clustering.fit_predict(flat_circuits)
            centroids = clustering.cluster_centers_
            
        elif algorithm == 'hdbscan':
            # HDBSCAN for density-based clustering (automatically determines clusters)
            try:
                import hdbscan
            except ImportError:
                print("HDBSCAN not installed. Run 'pip install hdbscan' to use this algorithm.")
                print("Falling back to K-means clustering.")
                return self._cluster_circuits(circuits, n_clusters, algorithm='kmeans')
                
            clustering = hdbscan.HDBSCAN(min_cluster_size=max(5, len(flat_circuits)//10),
                                         min_samples=3,
                                         prediction_data=True)
            cluster_labels = clustering.fit_predict(flat_circuits)
            
            # Handle noise points (labeled as -1)
            if -1 in cluster_labels:
                # Assign noise points to nearest cluster
                noise_indices = np.where(cluster_labels == -1)[0]
                for idx in noise_indices:
                    distances = np.linalg.norm(flat_circuits[idx] - flat_circuits, axis=1)
                    # Find nearest non-noise point
                    valid_indices = np.where(cluster_labels != -1)[0]
                    if len(valid_indices) > 0:
                        nearest_idx = valid_indices[np.argmin(distances[valid_indices])]
                        cluster_labels[idx] = cluster_labels[nearest_idx]
                    else:
                        # If all points are noise, create a single cluster
                        cluster_labels[idx] = 0
            
            # Relabel clusters to be 0-indexed consecutive integers
            unique_clusters = np.unique(cluster_labels)
            mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
            cluster_labels = np.array([mapping[label] for label in cluster_labels])
            
            # Compute centroids for each cluster
            n_clusters = len(np.unique(cluster_labels))
            centroids = np.zeros((n_clusters, flat_circuits.shape[1]))
            for i in range(n_clusters):
                cluster_points = flat_circuits[cluster_labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty clusters (shouldn't happen with proper relabeling)
                    centroids[i] = np.zeros(flat_circuits.shape[1])
                
        elif algorithm == 'gmm':
            # Gaussian Mixture Model clustering
            from sklearn.mixture import GaussianMixture
            clustering = GaussianMixture(n_components=n_clusters, random_state=42, n_init=5)
            cluster_labels = clustering.fit_predict(flat_circuits)
            centroids = clustering.means_
            
        elif algorithm == 'spectral':
            # Spectral clustering for complex geometries
            from sklearn.cluster import SpectralClustering
            clustering = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                          affinity='nearest_neighbors', n_neighbors=10)
            cluster_labels = clustering.fit_predict(flat_circuits)
            
            # Compute centroids for each cluster
            centroids = np.zeros((n_clusters, flat_circuits.shape[1]))
            for i in range(n_clusters):
                cluster_points = flat_circuits[cluster_labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    centroids[i] = np.zeros(flat_circuits.shape[1])
                
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        
        # Convert centroids to tensor and reshape
        centroids = torch.tensor(centroids).reshape(n_clusters, *circuits.shape[1:])
        
        return cluster_labels, centroids

    def visualize_multi_layer_umap_interactive(self, circuits, cluster_labels, prototype_idx, 
                                         samples, layer_indices=None, layer_names=None):
        """
        Create interactive UMAP visualizations with patch preview on hover.
        
        Args:
            circuits: Dictionary of circuit attributions
            cluster_labels: Cluster assignment for each sample
            prototype_idx: Index of the prototype being analyzed
            samples: Original input samples for generating previews
            layer_indices: Specific layer indices to visualize (defaults to all)
            layer_names: Optional dictionary of {layer_idx: display_name} for labels
            
        Returns:
            Dictionary of Plotly figures for each layer
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import umap
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import base64
        from io import BytesIO
        from PIL import Image
        import torch
        
        # Determine which layers to visualize
        if layer_indices is None:
            layer_indices = sorted(circuits.keys())
        else:
            layer_indices = [idx for idx in layer_indices if idx in circuits]
        
        if not layer_indices:
            print("No valid layers to visualize")
            return {}
        
        # Use default names if not provided
        if layer_names is None:
            if hasattr(self, 'layer_names'):
                layer_names = {idx: name for idx, name in enumerate(self.layer_names) 
                            if idx in layer_indices}
            else:
                layer_names = {idx: f"Layer {idx}" for idx in layer_indices}
        
        # Define a function to convert tensor to base64 image for hover
        def tensor_to_base64_img(tensor):
            # Denormalize and convert to PIL image
            img = tensor.cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Convert to base64
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        # Get number of clusters for color mapping
        unique_clusters = np.unique(cluster_labels)
        
        # Create color mapping
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))]
        
        # Process each layer and create figures
        figures = {}
        
        for layer_idx in layer_indices:
            # Get layer attributions
            layer_circuits = circuits[layer_idx]
            
            # Reshape for UMAP
            flat_circuits = layer_circuits.reshape(layer_circuits.shape[0], -1).numpy()
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(flat_circuits)
            
            # Apply UMAP dimensionality reduction
            reducer = umap.UMAP(
                n_components=2,
                min_dist=0.1,
                n_neighbors=min(15, len(flat_circuits)-1),
                random_state=42
            )
            
            # Compute embedding
            embedding = reducer.fit_transform(scaled_data)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Process each cluster
            for j, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_indices = np.where(mask)[0]
                
                # Pre-compute hover images
                hover_images = [tensor_to_base64_img(samples[idx]) for idx in cluster_indices]
                
                # Add scatter trace for this cluster
                fig.add_trace(go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=colors[j],
                        size=12,
                        opacity=0.7
                    ),
                    name=f'Cluster {j+1}',
                    customdata=list(zip(cluster_indices, hover_images)),
                    hovertemplate="""
                    <b>Sample Index: %{customdata[0]}</b><br>
                    <img src='%{customdata[1]}' width=150><br>
                    <extra></extra>
                    """
                ))
            
            # Update layout
            layer_name = layer_names.get(layer_idx, f"Layer {layer_idx}")
            fig.update_layout(
                title=f"{layer_name} Attributions for Prototype {prototype_idx}",
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                legend_title="Clusters",
                height=600,
                width=800
            )
            
            figures[layer_idx] = fig
        
        return figures 

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
    
    def visualize_prototypes_after_modification(self, dataloader, prototype_indices, 
                                        operation_name="Split", n_samples=10, 
                                        output_dir=None, max_prototypes=50):
        """
        Create comprehensive visualizations for prototypes after splitting or merging.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of original prototype indices involved in the operation
            operation_name: Name of the operation performed ("Split" or "Merge")
            n_samples: Number of high-activating samples to show per prototype
            output_dir: Directory to save visualizations (if None, just display)
            max_prototypes: Maximum number of prototypes to visualize
        """
        import os
        
        # Get class weights for coloring by most influential class
        class_weights = self.model.module._classification.weight.data.cpu().numpy()
        max_class_indices = np.argmax(class_weights, axis=0)
        max_class_weights = np.max(class_weights, axis=0)
        
        # Get the prototype indices to visualize
        if operation_name == "Split":
            # For split, we need the original prototypes and their "offspring"
            all_prototype_indices = []
            for proto_idx in prototype_indices:
                all_prototype_indices.append(proto_idx)
                
                # Find child prototypes (those with indices > original_num_prototypes)
                original_num_prototypes = self.model.module._num_prototypes - \
                                        (self.model.module._add_on[0].weight.data.shape[0] - 
                                         self.num_prototypes)
                
                # Look for weights coming from this prototype to find children
                for i in range(original_num_prototypes, self.model.module._num_prototypes):
                    # Check if it's strongly connected to the original prototype's class
                    original_class = max_class_indices[proto_idx]
                    new_class = max_class_indices[i]
                    
                    # If they serve the same class and have similar magnitude
                    if original_class == new_class and max_class_weights[i] > 0.5 * max_class_weights[proto_idx]:
                        all_prototype_indices.append(i)
                        
        elif operation_name == "Merge":
            # For merge, we include both merged prototypes and the resulting prototype
            all_prototype_indices = prototype_indices
        else:
            all_prototype_indices = prototype_indices
            
        # Limit the number of prototypes to visualize
        if len(all_prototype_indices) > max_prototypes:
            print(f"Limiting visualization to {max_prototypes} prototypes")
            all_prototype_indices = all_prototype_indices[:max_prototypes]
            
        # Get top activating samples for each prototype
        all_visualizations = []
        
        print(f"Creating visualizations for {len(all_prototype_indices)} prototypes...")
        for proto_idx in all_prototype_indices:
            # Find top activating samples
            top_samples, top_activations = self.find_top_activating_samples(
                dataloader, proto_idx, n_samples
            )
            
            # Skip if no samples found
            if len(top_samples) == 0:
                continue
                
            # Get the class information
            class_idx = max_class_indices[proto_idx]
            class_weight = max_class_weights[proto_idx]
            
            # Create figure
            fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
            
            # Handle case with single sample
            if n_samples == 1:
                axes = [axes]
            
            # Show each sample
            for i, (sample, activation) in enumerate(zip(top_samples, top_activations)):
                if i >= len(axes):
                    break
                    
                # Convert tensor to image
                img = sample.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                # Display
                axes[i].imshow(img)
                axes[i].set_title(f"Act: {activation:.3f}")
                axes[i].axis('off')
            
            # Set title with prototype and class info
            plt.suptitle(f"Prototype {proto_idx} (Class {class_idx}, Weight: {class_weight:.3f})")
            
            # Save or display
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"prototype_{proto_idx}_activations.png"))
                plt.close()
            else:
                all_visualizations.append(fig)
        
        # Display all figures if not saving
        if not output_dir and all_visualizations:
            for fig in all_visualizations:
                plt.figure(fig.number)
                plt.show()
                
        return all_visualizations
                
    def visualize_prototype_heatmaps(self, dataloader, prototype_indices, n_samples=3, 
                                   output_dir=None, max_prototypes=20):
        """
        Create heatmap visualizations for prototypes showing where they activate on images.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of prototype indices to visualize
            n_samples: Number of example images to show per prototype
            output_dir: Directory to save visualizations (if None, just display)
            max_prototypes: Maximum number of prototypes to visualize
        """
        import os
        import torch.nn.functional as F
        
        # Limit the number of prototypes to visualize
        if len(prototype_indices) > max_prototypes:
            print(f"Limiting visualization to {max_prototypes} prototypes")
            prototype_indices = prototype_indices[:max_prototypes]
        
        # Get class weights for coloring by most influential class
        class_weights = self.model.module._classification.weight.data.cpu().numpy()
        max_class_indices = np.argmax(class_weights, axis=0)
        
        # Create output directory if saving
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        all_visualizations = []
        
        # Process each prototype
        for proto_idx in prototype_indices:
            # Find top activating samples
            top_samples, _ = self.find_top_activating_samples(
                dataloader, proto_idx, n_samples
            )
            
            # Skip if no samples found
            if len(top_samples) == 0:
                continue
            
            # Get the class information
            class_idx = max_class_indices[proto_idx]
            
            # Create figure
            fig, axes = plt.subplots(n_samples, 2, figsize=(10, n_samples * 4))
            
            # Handle case with single sample
            if n_samples == 1:
                axes = [axes]
            
            # For each sample, show original and heatmap
            for i, sample in enumerate(top_samples[:n_samples]):
                if i >= len(axes):
                    break
                
                # Show original image
                img = sample.permute(1, 2, 0).cpu().numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[i, 0].imshow(img)
                axes[i, 0].axis('off')
                axes[i, 0].set_title(f"Original Image")
                
                # Get activation heatmap
                sample_batch = sample.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    proto_features, _, _ = self.model(sample_batch, inference=True)
                    
                # Get activation map for this prototype
                activation_map = proto_features[0, proto_idx].cpu()
                
                # Upsample to match image size
                upsampled = F.interpolate(
                    activation_map.unsqueeze(0).unsqueeze(0),
                    size=sample.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                # Normalize for visualization
                heatmap = upsampled.numpy()
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Create colored heatmap
                from matplotlib import cm
                colored_heatmap = cm.jet(heatmap)
                
                # Overlay heatmap on image
                overlay = 0.7 * img + 0.3 * colored_heatmap[:, :, :3]
                overlay = np.clip(overlay, 0, 1)
                
                # Show overlay
                axes[i, 1].imshow(overlay)
                axes[i, 1].axis('off')
                axes[i, 1].set_title(f"Activation Heatmap")
            
            # Set overall title
            plt.suptitle(f"Prototype {proto_idx} (Class {class_idx}) Activation Heatmaps")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save or display
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"prototype_{proto_idx}_heatmaps.png"))
                plt.close()
            else:
                all_visualizations.append(fig)
        
        # Display all figures if not saving
        if not output_dir and all_visualizations:
            for fig in all_visualizations:
                plt.figure(fig.number)
                plt.show()
                
        return all_visualizations
    
    def create_prototype_gallery(self, dataloader, prototype_indices=None, 
                               n_samples=5, n_cols=5, max_prototypes=100,
                               output_dir=None, sort_by_weight=True):
        """
        Create a comprehensive gallery of prototypes showing their activations.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_indices: List of prototype indices to visualize (if None, use all)
            n_samples: Number of example images to show per prototype
            n_cols: Number of columns in the gallery grid
            max_prototypes: Maximum number of prototypes to visualize
            output_dir: Directory to save visualizations (if None, just display)
            sort_by_weight: Whether to sort prototypes by their maximum weight
            
        Returns:
            List of figure objects if not saving to disk
        """
        import os
        import math
        
        # Get class weights for coloring by most influential class
        class_weights = self.model.module._classification.weight.data.cpu().numpy()
        max_class_indices = np.argmax(class_weights, axis=0)
        max_class_weights = np.max(class_weights, axis=0)
        
        # If no indices provided, use all prototypes with significant weights
        if prototype_indices is None:
            significant_weight = 0.01  # Threshold for considering a prototype used
            prototype_indices = [p for p in range(self.num_prototypes) 
                               if max_class_weights[p] > significant_weight]
        
        # Sort by weight if requested
        if sort_by_weight:
            prototype_indices = sorted(prototype_indices, 
                                      key=lambda p: max_class_weights[p], 
                                      reverse=True)
        
        # Limit the number of prototypes to visualize
        if len(prototype_indices) > max_prototypes:
            print(f"Limiting gallery to {max_prototypes} prototypes (out of {len(prototype_indices)})")
            prototype_indices = prototype_indices[:max_prototypes]
        
        # Create output directory if saving
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Calculate layout
        n_rows = math.ceil(len(prototype_indices) / n_cols)
        
        # Create a large figure
        fig_height = 3 * n_rows
        fig_width = 4 * n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Make sure axes is 2D even with single row/column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Process each prototype
        print(f"Creating gallery for {len(prototype_indices)} prototypes...")
        for i, proto_idx in enumerate(prototype_indices):
            row = i // n_cols
            col = i % n_cols
            
            # Skip if beyond array bounds (shouldn't happen)
            if row >= axes.shape[0] or col >= axes.shape[1]:
                continue
                
            ax = axes[row, col]
            
            # Find top activating samples
            top_samples, top_activations = self.find_top_activating_samples(
                dataloader, proto_idx, num_samples=1  # Just get the top one for the gallery
            )
            
            # Skip if no samples found
            if len(top_samples) == 0:
                ax.axis('off')
                continue
            
            # Get class and weight info
            class_idx = max_class_indices[proto_idx]
            weight = max_class_weights[proto_idx]
            
            # Display top sample
            img = top_samples[0].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            
            # Add border color based on class
            class_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            color = class_colors[class_idx % 20]
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(5)
            
            # Set title with prototype info
            ax.set_title(f"P{proto_idx} (C{class_idx}: {weight:.2f})")
            ax.axis('off')
        
        # Turn off unused axes
        for i in range(len(prototype_indices), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')
        
        # Add title and adjust layout
        plt.suptitle(f"Prototype Gallery: Top Activating Examples", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save or display
        if output_dir:
            plt.savefig(os.path.join(output_dir, "prototype_gallery.png"), dpi=300)
            print(f"Gallery saved to {os.path.join(output_dir, 'prototype_gallery.png')}")
            plt.close()
            return None
        else:
            return [fig]
    
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