from torch import nn
import torch
import numpy as np

class MultiLayerAttributionAnalyzer:
    """
    A class for analyzing and clustering circuit attributions across multiple layers
    of a deep neural network.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize the analyzer.
        
        Args:
            model: The PIPNet model to analyze
            device: Computing device ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.layer_names = []
        self.layer_modules = []
        
        # Extract and index all relevant modules
        self._index_network_layers()
    
    def _index_network_layers(self):
        """
        Create an index of network layers for attribution analysis.
        """
        # Get all modules in the feature network
        feature_net = self.model.module._net
        
        # For ResNets and ConvNeXt, we want key layers where semantic transformations happen
        
        # Handle different network architectures
        if hasattr(feature_net, 'layer1'):  # ResNet architecture
            # For ResNet, index the output of each residual block
            self.layer_names = ['input']
            self.layer_modules = [feature_net.conv1]  # Input convolution
            
            # Add stem layers
            if hasattr(feature_net, 'bn1'):
                self.layer_names.append('stem_bn')
                self.layer_modules.append(feature_net.bn1)
            
            if hasattr(feature_net, 'relu'):
                self.layer_names.append('stem_relu')
                self.layer_modules.append(feature_net.relu)
                
            if hasattr(feature_net, 'maxpool'):
                self.layer_names.append('stem_pool')
                self.layer_modules.append(feature_net.maxpool)
            
            # Add each layer group
            for group_idx in range(1, 5):  # ResNets typically have 4 layer groups
                layer_attr = f'layer{group_idx}'
                if hasattr(feature_net, layer_attr):
                    layer_group = getattr(feature_net, layer_attr)
                    
                    # Add each block in the layer group
                    for block_idx, block in enumerate(layer_group):
                        self.layer_names.append(f'layer{group_idx}.{block_idx}')
                        self.layer_modules.append(block)
        
        elif hasattr(feature_net, 'features'):  # ConvNeXt or similar
            # For ConvNeXt, index each stage
            self.layer_names = []
            self.layer_modules = []
            
            # Add each feature layer
            for idx, module in enumerate(feature_net.features):
                self.layer_names.append(f'features.{idx}')
                self.layer_modules.append(module)
        
        else:
            # Generic approach for other architectures
            self.layer_names = []
            self.layer_modules = []
            
            # Recursively index all convolutional and normalization layers
            def index_module(module, path=''):
                for name, child in module.named_children():
                    current_path = f"{path}.{name}" if path else name
                    
                    # Only index layers that transform features
                    if isinstance(child, (nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm, 
                                         nn.ReLU, nn.GELU, nn.MaxPool2d)):
                        self.layer_names.append(current_path)
                        self.layer_modules.append(child)
                    
                    # Recurse into child modules
                    index_module(child, current_path)
            
            index_module(feature_net)
        
        # Add the add-on layer that produces prototype features
        self.layer_names.append('add_on')
        self.layer_modules.append(self.model.module._add_on)
        
        print(f"Indexed {len(self.layer_names)} layers for attribution analysis")
    
    def compute_layer_attributions(self, x, prototype_idx, layer_indices=None):
        """
        Compute attributions for a specific prototype across multiple network layers.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype to compute attributions for
            layer_indices: List of layer indices to analyze. If None, use all layers.
                           Negative indices count from the end, like Python lists.
                           
        Returns:
            Dictionary of {layer_index: attribution_tensor} for each analyzed layer
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Determine which layers to analyze
        if layer_indices is None:
            # Use all layers
            layer_indices = list(range(len(self.layer_modules)))
        else:
            # Convert negative indices to positive
            layer_indices = [idx if idx >= 0 else len(self.layer_modules) + idx 
                           for idx in layer_indices]
        
        # Ensure all indices are valid
        layer_indices = [idx for idx in layer_indices 
                       if 0 <= idx < len(self.layer_modules)]
        
        # Forward pass to get feature representations
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # We need to collect activations from specific layers
        activations = {}
        
        # Register hooks for the layers we want to analyze
        handles = []
        
        def get_activation(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
        
        # Register hooks for the selected layers
        for idx in layer_indices:
            layer_name = self.layer_names[idx]
            module = self.layer_modules[idx]
            handles.append(module.register_forward_hook(
                get_activation(layer_name)
            ))
        
        # Forward pass through the model
        with torch.no_grad():
            _, pooled, _ = self.model(x, inference=False)
        
        # Get the prototype activation
        target_activation = pooled[0, prototype_idx]
        
        # Compute attributions for each layer
        attributions = {}
        
        for idx in layer_indices:
            layer_name = self.layer_names[idx]
            
            if layer_name in activations:
                # Get the activation tensor
                layer_activation = activations[layer_name]
                
                # Compute gradients
                gradients = torch.autograd.grad(
                    target_activation, 
                    layer_activation,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if gradients is not None:
                    # Compute Gradient × Activation attribution
                    layer_attr = gradients * layer_activation
                    
                    # For spatial layers, sum over spatial dimensions
                    if len(layer_attr.shape) > 2:
                        # Sum over spatial dimensions (H, W)
                        layer_attr = layer_attr.sum(dim=(2, 3))
                    
                    attributions[idx] = layer_attr
        
        # Clean up hooks
        for handle in handles:
            handle.remove()
        
        return attributions
    
    def compute_multi_layer_circuits(self, samples, prototype_idx, layer_indices=None):
        """
        Compute circuit attributions across multiple layers for a set of samples.
        
        Args:
            samples: Tensor of input samples [N, C, H, W]
            prototype_idx: Index of the prototype to analyze
            layer_indices: List of layer indices to analyze
            
        Returns:
            Dictionary of {layer_index: circuit_tensor} for each analyzed layer
        """
        circuits = {}
        
        for i, sample in enumerate(samples):
            # Add batch dimension
            sample_batch = sample.unsqueeze(0)
            
            # Compute attributions for this sample
            sample_attributions = self.compute_layer_attributions(
                sample_batch, prototype_idx, layer_indices
            )
            
            # Add to circuits dictionary
            for layer_idx, attribution in sample_attributions.items():
                if layer_idx not in circuits:
                    circuits[layer_idx] = []
                
                circuits[layer_idx].append(attribution.detach().cpu())
            
            # Log progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples")
        
        # Stack attributions for each layer
        for layer_idx in circuits:
            circuits[layer_idx] = torch.stack(circuits[layer_idx])
        
        return circuits
    
    def cluster_multi_layer_circuits(self, circuits, n_clusters=2, method='kmeans', 
                                   layer_weights=None):
        """
        Cluster circuit attributions using information from multiple layers.
        
        Args:
            circuits: Dictionary of {layer_index: circuit_tensor} from compute_multi_layer_circuits
            n_clusters: Number of clusters to create
            method: Clustering algorithm to use
            layer_weights: Optional dictionary of {layer_index: weight} for weighting layers
                         If None, all layers are weighted equally
                         
        Returns:
            Tuple of (cluster_labels, centroids_dict)
        """
        # If no weights provided, weight all layers equally
        if layer_weights is None:
            layer_weights = {layer_idx: 1.0 for layer_idx in circuits.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(layer_weights.values())
        layer_weights = {idx: w/total_weight for idx, w in layer_weights.items()}
        
        # Combine attributions from all layers
        n_samples = next(iter(circuits.values())).shape[0]
        combined_features = []
        
        for i in range(n_samples):
            # Collect attributions for this sample from all layers
            sample_features = []
            
            for layer_idx, weight in layer_weights.items():
                if layer_idx in circuits:
                    # Get attribution for this layer and sample
                    layer_attr = circuits[layer_idx][i]
                    
                    # Flatten and normalize the attribution
                    flat_attr = layer_attr.flatten()
                    norm = torch.norm(flat_attr)
                    if norm > 0:
                        flat_attr = flat_attr / norm
                    
                    # Weight by layer importance
                    flat_attr = flat_attr * weight
                    
                    sample_features.append(flat_attr)
            
            # Concatenate all layer features
            if sample_features:
                combined_attr = torch.cat(sample_features)
                combined_features.append(combined_attr)
        
        # Stack all combined features
        combined_features = torch.stack(combined_features)
        combined_features_np = combined_features.numpy()
        
        # Apply clustering using specified method
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_features_np)
            
            # Compute centroids for each layer
            centroids = {}
            for layer_idx in circuits:
                layer_circuits = circuits[layer_idx].numpy()
                layer_centroids = np.zeros((n_clusters,) + layer_circuits.shape[1:])
                
                for c in range(n_clusters):
                    cluster_mask = cluster_labels == c
                    if np.any(cluster_mask):
                        layer_centroids[c] = np.mean(layer_circuits[cluster_mask], axis=0)
                
                centroids[layer_idx] = torch.tensor(layer_centroids)
        
        elif method == 'hdbscan':
            import hdbscan
            
            # HDBSCAN for density-based clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, n_samples // 10),
                min_samples=5,
                metric='euclidean',
                prediction_data=True
            )
            
            cluster_labels = clusterer.fit_predict(combined_features_np)
            
            # Handle noise points (labeled as -1)
            noise_mask = cluster_labels == -1
            if np.any(noise_mask):
                # Assign noise points to nearest cluster
                for i in np.where(noise_mask)[0]:
                    if np.any(~noise_mask):
                        dists = np.linalg.norm(
                            combined_features_np[i] - combined_features_np[~noise_mask], 
                            axis=1
                        )
                        nearest_cluster = cluster_labels[~noise_mask][np.argmin(dists)]
                        cluster_labels[i] = nearest_cluster
                    else:
                        # All points are noise, make a single cluster
                        cluster_labels[i] = 0
            
            # Compute centroids for each layer
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            
            centroids = {}
            for layer_idx in circuits:
                layer_circuits = circuits[layer_idx].numpy()
                layer_centroids = np.zeros((n_clusters,) + layer_circuits.shape[1:])
                
                for i, c in enumerate(unique_clusters):
                    cluster_mask = cluster_labels == c
                    if np.any(cluster_mask):
                        layer_centroids[i] = np.mean(layer_circuits[cluster_mask], axis=0)
                
                centroids[layer_idx] = torch.tensor(layer_centroids)
        
        elif method == 'spectral':
            from sklearn.cluster import SpectralClustering
            
            # Spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                assign_labels='kmeans',
                random_state=42,
                affinity='nearest_neighbors',
                n_neighbors=min(10, n_samples-1)
            )
            
            cluster_labels = clustering.fit_predict(combined_features_np)
            
            # Compute centroids for each layer
            centroids = {}
            for layer_idx in circuits:
                layer_circuits = circuits[layer_idx].numpy()
                layer_centroids = np.zeros((n_clusters,) + layer_circuits.shape[1:])
                
                for c in range(n_clusters):
                    cluster_mask = cluster_labels == c
                    if np.any(cluster_mask):
                        layer_centroids[c] = np.mean(layer_circuits[cluster_mask], axis=0)
                
                centroids[layer_idx] = torch.tensor(layer_centroids)
        
        elif method == 'bgmm':
            from sklearn.mixture import BayesianGaussianMixture
            
            # Bayesian Gaussian Mixture Model
            bgmm = BayesianGaussianMixture(
                n_components=n_clusters,
                random_state=42,
                max_iter=300,
                n_init=3
            )
            
            cluster_labels = bgmm.fit_predict(combined_features_np)
            
            # Compute centroids for each layer
            centroids = {}
            for layer_idx in circuits:
                layer_circuits = circuits[layer_idx].numpy()
                layer_centroids = np.zeros((n_clusters,) + layer_circuits.shape[1:])
                
                for c in range(n_clusters):
                    cluster_mask = cluster_labels == c
                    if np.any(cluster_mask):
                        layer_centroids[c] = np.mean(layer_circuits[cluster_mask], axis=0)
                
                centroids[layer_idx] = torch.tensor(layer_centroids)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return cluster_labels, centroids
    
    def visualize_multi_layer_umap(self, circuits, cluster_labels, prototype_idx, 
                                 layer_indices=None, layer_names=None):
        """
        Create UMAP visualizations for attributions at each analyzed layer.
        
        Args:
            circuits: Dictionary of circuit attributions
            cluster_labels: Cluster assignment for each sample
            prototype_idx: Index of the prototype being analyzed
            layer_indices: Specific layer indices to visualize (defaults to all)
            layer_names: Optional dictionary of {layer_idx: display_name} for labels
        """
        import umap
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Determine which layers to visualize
        if layer_indices is None:
            layer_indices = sorted(circuits.keys())
        else:
            # Filter to only include layers that exist in circuits
            layer_indices = [idx for idx in layer_indices if idx in circuits]
        
        if not layer_indices:
            print("No valid layers to visualize")
            return
        
        # Use default names if not provided
        if layer_names is None:
            if hasattr(self, 'layer_names'):
                layer_names = {idx: name for idx, name in enumerate(self.layer_names) 
                             if idx in layer_indices}
            else:
                layer_names = {idx: f"Layer {idx}" for idx in layer_indices}
        
        # Calculate grid dimensions based on number of layers
        n_layers = len(layer_indices)
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Get number of clusters for color mapping
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        # Process each layer
        for i, layer_idx in enumerate(layer_indices):
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
            
            # Create subplot
            row, col = i // n_cols, i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # Plot each cluster
            for j, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[j]],
                    label=f'Cluster {j+1}',
                    alpha=0.7,
                    s=80
                )
            
            # Set plot title and labels
            layer_name = layer_names.get(layer_idx, f"Layer {layer_idx}")
            ax.set_title(f"{layer_name} Attributions", fontsize=12)
            ax.grid(alpha=0.3)
            
            # Add legend to first plot only
            if i == 0:
                ax.legend()
        
        plt.suptitle(f"Multi-Layer Attribution Analysis for Prototype {prototype_idx}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
        
        return fig
    
    def visualize_multi_layer_attributions(self, circuits, cluster_labels, prototype_idx,
                                         samples, layer_indices=None, max_examples=3):
        """
        Visualize the attributions across different layers for samples in each cluster.
        
        Args:
            circuits: Dictionary of circuit attributions
            cluster_labels: Cluster assignments
            prototype_idx: Index of the prototype
            samples: Original input samples
            layer_indices: Specific layer indices to visualize (defaults to all)
            max_examples: Maximum number of examples per cluster
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Determine which layers to visualize
        if layer_indices is None:
            layer_indices = sorted(circuits.keys())
        else:
            # Filter to only include layers that exist in circuits
            layer_indices = [idx for idx in layer_indices if idx in circuits]
        
        if not layer_indices:
            print("No valid layers to visualize")
            return
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # For each cluster, visualize examples
        for cluster_idx, cluster in enumerate(unique_clusters):
            # Find samples in this cluster
            cluster_mask = cluster_labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Select examples (either all or up to max_examples)
            n_examples = min(max_examples, len(cluster_indices))
            example_indices = cluster_indices[:n_examples]
            
            # Create a figure for this cluster
            fig = plt.figure(figsize=(5*len(layer_indices), 4*n_examples))
            gs = GridSpec(n_examples, len(layer_indices) + 1)  # +1 for the original image
            
            # For each example
            for i, example_idx in enumerate(example_indices):
                # Show the original image
                ax_img = fig.add_subplot(gs[i, 0])
                img = samples[example_idx].cpu()
                img = img.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax_img.imshow(img)
                ax_img.set_title(f"Sample {example_idx}")
                ax_img.axis('off')
                
                # For each layer, visualize attribution
                for j, layer_idx in enumerate(layer_indices):
                    if layer_idx in circuits:
                        ax_attr = fig.add_subplot(gs[i, j+1])
                        
                        # Get attribution for this sample and layer
                        attribution = circuits[layer_idx][example_idx]
                        
                        # Determine how to visualize based on attribution shape
                        if len(attribution.shape) == 1:
                            # 1D attribution (e.g., channel attributions)
                            values = attribution.cpu().numpy()
                            ax_attr.bar(range(len(values)), values)
                            ax_attr.set_title(f"Layer {layer_idx} Attribution")
                        elif len(attribution.shape) == 2:
                            # 2D attribution (e.g., spatial heatmap)
                            values = attribution.cpu().numpy()
                            im = ax_attr.imshow(values, cmap='viridis')
                            plt.colorbar(im, ax=ax_attr)
                            ax_attr.set_title(f"Layer {layer_idx} Attribution")
            
            plt.suptitle(f"Cluster {cluster_idx+1} Examples for Prototype {prototype_idx}", fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.show()
    
    def analyze_prototype(self, dataloader, prototype_idx, layer_indices=None, 
                        n_clusters=None, adaptive=True, max_clusters=5,
                        clustering_method='kmeans', visualize=True,
                        max_samples=100, layer_weights=None):
        """
        Comprehensive analysis of a prototype using multi-layer attributions.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to analyze
            layer_indices: List of layer indices to analyze (negative indices supported)
            n_clusters: Number of clusters to create (if None, determined automatically)
            adaptive: Whether to adaptively determine the number of clusters
            max_clusters: Maximum number of clusters to consider
            clustering_method: Clustering method to use
            visualize: Whether to visualize the results
            max_samples: Maximum number of samples to analyze
            layer_weights: Optional dictionary of weights for each layer
            
        Returns:
            Dictionary of analysis results
        """
        # Convert negative indices to positive if needed
        if layer_indices is not None:
            # Convert negative indices to positive
            n_layers = len(self.layer_modules)
            layer_indices = [idx if idx >= 0 else n_layers + idx for idx in layer_indices]
            # Filter out invalid indices
            layer_indices = [idx for idx in layer_indices if 0 <= idx < n_layers]
        
        print(f"Analyzing prototype {prototype_idx} across layers: {layer_indices}")
        
        # Find top activating samples
        from util.pure import PURE
        pure_analyzer = PURE(self.model, device=self.device)
        top_samples, top_activations = pure_analyzer.find_top_activating_samples(
            dataloader, prototype_idx)        
        print(f"Found {len(top_samples)} top activating samples")
        
        # Compute multi-layer attributions
        circuits = self.compute_multi_layer_circuits(top_samples, prototype_idx, layer_indices)
        
        # Determine number of clusters if needed
        if adaptive and n_clusters is None:
            # Use the most informative layer for cluster determination
            # (typically the deepest layer or add-on layer)
            if -1 in circuits:  # Add-on layer
                informative_layer = -1
            else:
                # Use the deepest available layer
                informative_layer = max(circuits.keys())
            
            # Implement a method to determine optimal clusters
            # For example, silhouette analysis or elbow method
            from sklearn.metrics import silhouette_score
            
            # Get attributions for the informative layer
            layer_circuits = circuits[informative_layer]
            flat_circuits = layer_circuits.reshape(layer_circuits.shape[0], -1).numpy()
            
            # Try different numbers of clusters
            best_score = -1
            best_n_clusters = 1
            
            for k in range(2, min(max_clusters + 1, len(flat_circuits))):
                # Skip if too few samples
                if len(flat_circuits) <= k:
                    continue
                    
                # Cluster with k-means
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(flat_circuits)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                    score = silhouette_score(flat_circuits, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = k
            
            n_clusters = best_n_clusters
            print(f"Automatically determined {n_clusters} clusters (score: {best_score:.3f})")
        elif n_clusters is None:
            n_clusters = 2  # Default
        
        # Cluster the multi-layer circuits
        cluster_labels, centroids = self.cluster_multi_layer_circuits(
            circuits, n_clusters, method=clustering_method, layer_weights=layer_weights
        )
        
        # Evaluate clustering quality
        from sklearn.metrics import silhouette_score
        
        # Combine attributions from all layers for evaluation
        combined_circuits = []
        for i in range(len(top_samples)):
            sample_attrs = []
            for layer_idx in circuits:
                attr = circuits[layer_idx][i].flatten()
                sample_attrs.append(attr)
            combined_attr = torch.cat(sample_attrs)
            combined_circuits.append(combined_attr)
        
        combined_circuits = torch.stack(combined_circuits).numpy()
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(combined_circuits, cluster_labels)
            is_polysemantic = sil_score > 0.1  # Threshold for polysemanticity
        else:
            sil_score = 0
            is_polysemantic = False
            
        print(f"Silhouette score: {sil_score:.3f}, Polysemantic: {is_polysemantic}")
        
        # Visualize if requested
        if visualize:
            print("Visualizing multi-layer analysis results...")
            
            # Visualize UMAP projections
            self.visualize_multi_layer_umap(circuits, cluster_labels, prototype_idx)
            
            # Visualize attributions
            self.visualize_multi_layer_attributions(
                circuits, cluster_labels, prototype_idx, top_samples
            )
        
        # Return comprehensive results
        return {
            'prototype_idx': prototype_idx,
            'layer_indices': layer_indices,
            'n_clusters': len(np.unique(cluster_labels)),
            'cluster_labels': cluster_labels,
            'silhouette_score': sil_score,
            'is_polysemantic': is_polysemantic,
            'circuits': circuits,
            'centroids': centroids,
            'top_samples': top_samples,
            'top_activations': top_activations
        }