from torch import nn
from util.pure import PURE
import torch
import matplotlib.pyplot as plt
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
        
        print(f"Indexed {len(self.layer_names)} layers for attribution analysis: {self.layer_names}")
    
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
        # with torch.no_grad():
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
        else:
            layer_weights = layer_weights
        print(layer_weights)
        
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
            overall_centroids = torch.from_numpy(kmeans.cluster_centers_)
            
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
                min_cluster_size=max(6, n_samples // 10),
                min_samples=5,
                metric='euclidean',
                prediction_data=True,
                leaf_size=20
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
        elif method == 'hdbscan_kmeans':
            # Implemented hybrid approach here
            import hdbscan
            
            # Run HDBSCAN first to get initial clusters
            clustering = hdbscan.HDBSCAN(min_cluster_size=max(5, n_samples//10),
                                        min_samples=3,
                                        prediction_data=True)
            hdbscan_labels = clustering.fit_predict(combined_features_np)
            
            # Handle noise points
            if -1 in hdbscan_labels:
                # Assign noise points to nearest cluster
                noise_indices = np.where(hdbscan_labels == -1)[0]
                valid_indices = np.where(hdbscan_labels != -1)[0]
                
                if len(valid_indices) > 0:
                    # Assign to nearest non-noise cluster
                    for idx in noise_indices:
                        distances = np.linalg.norm(combined_features_np[idx] - combined_features_np[valid_indices], axis=1)
                        nearest_idx = valid_indices[np.argmin(distances)]
                        hdbscan_labels[idx] = hdbscan_labels[nearest_idx]
                else:
                    # If all points are noise, create a single cluster
                    hdbscan_labels = np.zeros_like(hdbscan_labels)
            
            # Get number of clusters and calculate centroids
            unique_clusters = np.unique(hdbscan_labels)
            n_clusters = len(unique_clusters)
            
            # Now use K-means with the determined number of clusters
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(combined_features_np)
            centroids = kmeans.cluster_centers_
            overall_centroids = torch.from_numpy(kmeans.cluster_centers_)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return cluster_labels, centroids, overall_centroids
    
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

    def visualize_multi_layer_umap_with_gallery(self, circuits, cluster_labels, prototype_idx, 
                                            samples, layer_indices=None, layer_names=None,
                                            max_display_images=12):
        """
        Create interactive UMAP visualizations with synchronized image gallery display.
        When points are selected in the scatter plot, corresponding images appear in the gallery.
        
        Args:
            circuits: Dictionary of circuit attributions
            cluster_labels: Cluster assignment for each sample
            prototype_idx: Index of the prototype being analyzed
            samples: Original input samples for generating previews
            layer_indices: Specific layer indices to visualize (defaults to all)
            layer_names: Optional dictionary of {layer_idx: display_name} for labels
            max_display_images: Maximum number of images to display in the gallery
            
        Returns:
            Dictionary of HTML files with interactive visualizations
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import umap
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        import base64
        from io import BytesIO
        from PIL import Image
        import os
        import math
        
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
        
        # Define a function to convert tensor to base64 image for embedding
        def tensor_to_base64_img(tensor, size=150):
            # Denormalize and convert to PIL image
            img = tensor.cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Resize if needed
            if size:
                pil_img = pil_img.resize((size, size), Image.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=80)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        # Get number of clusters for color mapping
        unique_clusters = np.unique(cluster_labels)
        
        # Create color mapping
        colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))]
        
        results = {}
        
        # For each layer, create a complete dashboard
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
            
            # Create a UMAP scatter plot figure
            fig_scatter = go.Figure()
            
            # Process each cluster
            sample_indices_by_cluster = {}
            images_by_cluster = {}
            
            for j, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_indices = np.where(mask)[0]
                sample_indices_by_cluster[j] = cluster_indices.tolist()
                
                # Store images for this cluster (for gallery display)
                images_by_cluster[j] = [tensor_to_base64_img(samples[idx]) for idx in cluster_indices]
                
                # Add scatter trace
                fig_scatter.add_trace(go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=colors[j],
                        size=10,
                        opacity=0.8
                    ),
                    name=f'Cluster {j+1}',
                    customdata=list(zip(cluster_indices, [j] * len(cluster_indices))),
                    hovertemplate="Sample: %{customdata[0]}<br>Cluster: %{customdata[1] + 1}<extra></extra>"
                ))
            
            # Update layout
            layer_name = layer_names.get(layer_idx, f"Layer {layer_idx}")
            fig_scatter.update_layout(
                title=f"{layer_name} Attributions for Prototype {prototype_idx}",
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                legend_title="Clusters",
                height=500,
                width=800,
                hovermode='closest',
                clickmode='event+select'
            )
            
            # Generate HTML for the dashboard with embedded JavaScript for interactivity
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Interactive UMAP Visualization - Layer {layer_idx}</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ display: flex; flex-direction: column; }}
                    .plot-container {{ width: 800px; }}
                    .gallery-container {{ 
                        margin-top: 20px;
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                    }}
                    .cluster-selector {{
                        margin: 20px 0;
                        display: flex;
                        gap: 10px;
                    }}
                    .cluster-btn {{
                        padding: 8px 15px;
                        cursor: pointer;
                        border: none;
                        border-radius: 4px;
                    }}
                    .gallery-image {{
                        width: 150px;
                        height: 150px;
                        object-fit: cover;
                        border-radius: 4px;
                        transition: transform 0.2s;
                    }}
                    .gallery-image:hover {{
                        transform: scale(1.05);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    }}
                    .image-container {{
                        position: relative;
                        display: inline-block;
                    }}
                    .image-label {{
                        position: absolute;
                        bottom: 0;
                        left: 0;
                        background: rgba(0,0,0,0.7);
                        color: white;
                        padding: 2px 6px;
                        font-size: 12px;
                        border-radius: 0 0 4px 4px;
                    }}
                    h2 {{ margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>Interactive Visualization for Prototype {prototype_idx} - {layer_name}</h1>
                <div class="container">
                    <div class="plot-container" id="scatter-plot"></div>
                    
                    <div class="cluster-selector">
                        <span><strong>Show cluster: </strong></span>
            """
            
            # Add cluster selector buttons
            for j, cluster in enumerate(unique_clusters):
                html += f"""
                <button class="cluster-btn" 
                        style="background-color: {colors[j]}; color: white;" 
                        onclick="showClusterImages({j})">
                    Cluster {j+1} ({len(sample_indices_by_cluster[j])} samples)
                </button>
                """
            
            html += """
                    </div>
                    
                    <h2>Image Gallery</h2>
                    <div id="gallery-info">Select points in the plot or click a cluster button to view images.</div>
                    <div class="gallery-container" id="image-gallery"></div>
                </div>
                
                <script>
                    // Store all image data
                    const clusterData = {
            """
            
            # Embed image data as JavaScript variables
            for j in range(len(unique_clusters)):
                html += f"""
                        {j}: {{
                            indices: {sample_indices_by_cluster[j]},
                            images: {images_by_cluster[j]},
                            color: "{colors[j]}"
                        }},
                """
            
            html += """
                    };
                    
                    // Create the scatter plot
                    const scatterData = 
            """
            
            # Embed the Plotly figure data
            html += fig_scatter.to_json()
            
            html += """
                    ;
                    
                    Plotly.newPlot('scatter-plot', scatterData.data, scatterData.layout);
                    
                    // Function to display images for a specific cluster
                    function showClusterImages(clusterIdx) {
                        const gallery = document.getElementById('image-gallery');
                        const info = document.getElementById('gallery-info');
                        const cluster = clusterData[clusterIdx];
                        
                        // Update info text
                        info.innerHTML = `Showing ${Math.min(cluster.images.length, 12)} of ${cluster.images.length} images from Cluster ${clusterIdx + 1}`;
                        
                        // Clear gallery
                        gallery.innerHTML = '';
                        
                        // Add images (limit to max_display_images)
                        const maxImages = Math.min(cluster.images.length, 12);
                        for (let i = 0; i < maxImages; i++) {
                            const div = document.createElement('div');
                            div.className = 'image-container';
                            
                            const img = document.createElement('img');
                            img.src = cluster.images[i];
                            img.className = 'gallery-image';
                            img.style.border = `3px solid ${cluster.color}`;
                            
                            const label = document.createElement('div');
                            label.className = 'image-label';
                            label.textContent = `Sample ${cluster.indices[i]}`;
                            
                            div.appendChild(img);
                            div.appendChild(label);
                            gallery.appendChild(div);
                        }
                    }
                    
                    // Handle selection events from the plot
                    document.getElementById('scatter-plot').on('plotly_selected', function(eventData) {
                        if (!eventData || !eventData.points || eventData.points.length === 0) {
                            return;
                        }
                        
                        const gallery = document.getElementById('image-gallery');
                        const info = document.getElementById('gallery-info');
                        
                        // Clear gallery
                        gallery.innerHTML = '';
                        
                        // Collect selected points
                        const selectedIndices = eventData.points.map(pt => pt.customdata[0]);
                        const numToShow = Math.min(selectedIndices.length, 12);
                        
                        // Update info text
                        info.innerHTML = `Showing ${numToShow} of ${selectedIndices.length} selected samples`;
                        
                        // Display images for selected points
                        for (let i = 0; i < numToShow; i++) {
                            const sampleIdx = selectedIndices[i];
                            let clusterIdx, imageIdx;
                            
                            // Find which cluster this sample belongs to
                            for (const [cIdx, cluster] of Object.entries(clusterData)) {
                                const localIdx = cluster.indices.indexOf(sampleIdx);
                                if (localIdx !== -1) {
                                    clusterIdx = parseInt(cIdx);
                                    imageIdx = localIdx;
                                    break;
                                }
                            }
                            
                            if (clusterIdx === undefined) continue;
                            
                            const div = document.createElement('div');
                            div.className = 'image-container';
                            
                            const img = document.createElement('img');
                            img.src = clusterData[clusterIdx].images[imageIdx];
                            img.className = 'gallery-image';
                            img.style.border = `3px solid ${clusterData[clusterIdx].color}`;
                            
                            const label = document.createElement('div');
                            label.className = 'image-label';
                            label.textContent = `Sample ${sampleIdx}`;
                            
                            div.appendChild(img);
                            div.appendChild(label);
                            gallery.appendChild(div);
                        }
                    });
                    
                    // Show the first cluster by default
                    showClusterImages(0);
                </script>
            </body>
            </html>
            """
            
            # Store the HTML for this layer
            results[layer_idx] = html
        
        # Save HTML files to disk
        output_dir = f"prototype_{prototype_idx}_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_idx, html_content in results.items():
            layer_name = layer_names.get(layer_idx, f"layer_{layer_idx}")
            file_path = os.path.join(output_dir, f"{layer_name}_visualization.html")
            
            with open(file_path, "w") as f:
                f.write(html_content)
            
            print(f"Saved visualization for layer {layer_idx} to {file_path}")
        
        return results


    def visualize_top_activating_patches(self, prototype_idx, cluster_labels, samples, 
                                    activations, proto_features=None, n_patches=6,
                                    save_dir=None):
        """
        Visualize the top activating patches for each cluster of a prototype.
        
        Args:
            prototype_idx: Index of the prototype to visualize
            cluster_labels: Cluster assignments for samples
            samples: Tensor of input samples that activate the prototype
            activations: Activation values for each sample
            proto_features: Optional prototype feature maps to locate activating regions
            n_patches: Number of top patches to show per cluster
            save_dir: Directory to save visualizations (if None, just display)
            
        Returns:
            List of matplotlib figures, one per cluster
        """
        import matplotlib.pyplot as plt
        import torch
        import numpy as np
        import os
        from matplotlib.patches import Rectangle
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # List to store figures
        figures = []
        
        # For each cluster
        for cluster_idx, cluster in enumerate(unique_clusters):
            # Find samples in this cluster
            cluster_mask = cluster_labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            cluster_activations = [activations[i] for i in cluster_indices]
            
            # Sort by activation strength (descending)
            sorted_indices = np.argsort(cluster_activations)[::-1]
            top_indices = [cluster_indices[i] for i in sorted_indices[:n_patches]]
            top_activations = [cluster_activations[i] for i in sorted_indices[:n_patches]]
            
            # Skip if no samples
            if not top_indices:
                continue
                
            # Create figure - showing pairs of (original image, activation patch)
            n_cols = 2  # original + patch
            n_rows = min(n_patches, len(top_indices))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
            
            # Make axes 2D if only one row
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Process top samples
            for i, (idx, activation) in enumerate(zip(top_indices, top_activations)):
                if i >= n_rows:
                    break
                    
                # Get the sample
                sample = samples[idx]
                
                # Show original image
                img = sample.cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Sample {idx}\nActivation: {activation:.3f}")
                axes[i, 0].axis('off')
                
                # Find activation region if proto_features provided
                if proto_features is not None:
                    # Get prototype activation map for this sample
                    activation_map = proto_features[idx, prototype_idx]
                    
                    # Find coordinates of maximum activation
                    if len(activation_map.shape) == 2:  # Spatial map
                        h_idx, w_idx = np.unravel_index(
                            torch.argmax(activation_map).cpu(), 
                            activation_map.shape
                        )
                        
                        # Calculate receptive field in input image
                        # This is approximate and depends on model architecture
                        input_size = sample.shape[1]  # Assuming square input
                        feature_size = activation_map.shape[0]
                        scale = input_size / feature_size
                        
                        # Calculate patch size (relative to activation map)
                        patch_size = max(1, min(feature_size // 4, 3))
                        
                        # Calculate patch boundaries in input space
                        h_min = max(0, int((h_idx - patch_size/2) * scale))
                        h_max = min(input_size, int((h_idx + patch_size/2) * scale))
                        w_min = max(0, int((w_idx - patch_size/2) * scale))
                        w_max = min(input_size, int((w_idx + patch_size/2) * scale))
                        
                        # Draw rectangle on original image
                        rect = Rectangle((w_min, h_min), w_max-w_min, h_max-h_min,
                                    linewidth=2, edgecolor='r', facecolor='none')
                        axes[i, 0].add_patch(rect)
                        
                        # Extract and display the patch
                        patch = sample[:, h_min:h_max, w_min:w_max].cpu()
                        patch = patch.permute(1, 2, 0).numpy()
                        patch = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                        axes[i, 1].imshow(patch)
                        axes[i, 1].set_title("Activation Patch")
                        axes[i, 1].axis('off')
                    else:
                        # If not spatial, just show the full image again
                        axes[i, 1].imshow(img)
                        axes[i, 1].set_title("No spatial information")
                        axes[i, 1].axis('off')
                else:
                    # Without proto_features, use a simple center crop
                    h, w = img.shape[:2]
                    crop_size = min(h, w) // 3
                    h_mid, w_mid = h // 2, w // 2
                    h_min, h_max = max(0, h_mid - crop_size//2), min(h, h_mid + crop_size//2)
                    w_min, w_max = max(0, w_mid - crop_size//2), min(w, w_mid + crop_size//2)
                    
                    # Draw rectangle on original
                    rect = Rectangle((w_min, h_min), w_max-w_min, h_max-h_min,
                                linewidth=2, edgecolor='r', facecolor='none')
                    axes[i, 0].add_patch(rect)
                    
                    # Show cropped patch
                    patch = img[h_min:h_max, w_min:w_max]
                    axes[i, 1].imshow(patch)
                    axes[i, 1].set_title("Center Crop (approximate)")
                    axes[i, 1].axis('off')
            
            plt.suptitle(f"Prototype {prototype_idx} - Cluster {cluster_idx+1} Top Activating Patches", fontsize=16)
            plt.tight_layout()
            
            # Save or display
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"prototype_{prototype_idx}_cluster_{cluster_idx+1}_patches.png"), 
                        dpi=150, bbox_inches='tight')
                plt.close()
            else:
                figures.append(fig)
                plt.show()
        
        return figures

    def visualize_all_prototypes_patches(self, pure_results, dataloader, 
                                        save_dir=None, n_patches=4):
        """
        Visualize highly activating patches for all prototypes that have been analyzed with PURE.
        
        Args:
            pure_results: Dictionary of results from analyzing prototypes with PURE
            dataloader: DataLoader to get additional samples if needed
            save_dir: Directory to save visualizations
            n_patches: Number of patches to show per cluster
            
        Returns:
            Dictionary mapping prototype indices to visualization figures
        """
        import os
        from tqdm import tqdm
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Process each prototype in the results
        all_visualizations = {}
        
        for proto_idx, result in tqdm(pure_results.items(), desc="Visualizing prototypes"):
            # Extract needed data
            samples = result['top_samples']
            cluster_labels = result['cluster_labels']
            activations = result['top_activations']
            
            # Get prototype feature maps if available
            proto_features = None
            if 'proto_features' in result:
                proto_features = result['proto_features']
            
            # Create subdirectory for this prototype
            proto_save_dir = None
            if save_dir:
                proto_save_dir = os.path.join(save_dir, f"prototype_{proto_idx}")
                os.makedirs(proto_save_dir, exist_ok=True)
            
            # Visualize patches
            figures = self.visualize_top_activating_patches(
                proto_idx, 
                cluster_labels, 
                samples, 
                activations, 
                proto_features,
                n_patches=n_patches,
                save_dir=proto_save_dir
            )
            
            all_visualizations[proto_idx] = figures
            
        return all_visualizations

    def extract_activation_patches(self, prototype_idx, sample_tensor, 
                                proto_features=None, patch_size=64):
        """
        Extract the actual patch that maximally activates a given prototype.
        
        Args:
            prototype_idx: Index of the prototype
            sample_tensor: Input sample tensor
            proto_features: Prototype feature map (if available)
            patch_size: Size of the patch to extract (in pixels)
            
        Returns:
            Tuple of (patch tensor, (x, y, width, height))
        """
        import torch
        import numpy as np
        
        # Get sample dimensions
        _, h, w = sample_tensor.shape
        
        # If we have prototype features, use them to find activation region
        if proto_features is not None:
            # Get activation map
            activation_map = proto_features[0, prototype_idx]
            
            # Find coordinates of maximum activation
            if len(activation_map.shape) == 2:  # Spatial map
                max_h, max_w = np.unravel_index(
                    torch.argmax(activation_map).cpu(), 
                    activation_map.shape
                )
                
                # Convert to input image coordinates
                feature_size = activation_map.shape[0]
                scale = h / feature_size  # Assuming square input
                
                # Center of activation
                center_h = int(max_h * scale)
                center_w = int(max_w * scale)
            else:
                # If not spatial, use center of image
                center_h, center_w = h // 2, w // 2
        else:
            # Without features, use center of image
            center_h, center_w = h // 2, w // 2
        
        # Calculate patch boundaries
        half_size = patch_size // 2
        h_min = max(0, center_h - half_size)
        h_max = min(h, center_h + half_size)
        w_min = max(0, center_w - half_size)
        w_max = min(w, center_w + half_size)
        
        # Extract patch
        patch = sample_tensor[:, h_min:h_max, w_min:w_max]
        
        # Return patch and coordinates
        return patch, (w_min, h_min, w_max - w_min, h_max - h_min)

    def create_activation_patch_visualization(self, pure_results, prototype_indices=None):
        """Create HTML visualization showing highly activating patches with rectangles highlighting activated regions."""
        import json
        import base64
        from io import BytesIO
        from PIL import Image, ImageDraw
        import torch
        import os
        import numpy as np
        
        # Helper function for tensor to base64 image conversion WITH highlighted region
        def tensor_to_base64(tensor, coordinates=None):
            try:
                img = tensor.cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = (img * 255).astype('uint8')
                pil_img = Image.fromarray(img)
                
                # Draw rectangle if coordinates provided
                if coordinates:
                    draw = ImageDraw.Draw(pil_img)
                    x, y, w, h = coordinates
                    # Draw red rectangle with 3px width
                    draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return img_str
            except Exception as e:
                print(f"Error converting image: {e}")
                return ""
        
        def find_activation_patch(sample, proto_idx):
            """Find patch that maximally activates prototype or prototype group"""
            sample_batch = sample.unsqueeze(0).to(self.device)
            
            # Handle different types of prototype references
            if isinstance(proto_idx, list):
                # Direct list of prototype indices
                proto_indices = proto_idx
                
                with torch.no_grad():
                    proto_features, pooled, _ = self.model(sample_batch, inference=True)
                    activations = [pooled[0, idx].item() for idx in proto_indices]
                    strongest_idx = proto_indices[np.argmax(activations)]
                    max_val = max(activations)
                    activation_map = proto_features[0, strongest_idx]
            elif isinstance(proto_idx, str) and "_" in proto_idx:
                # String representation of prototype group (e.g. "3_22")
                proto_indices = [int(idx) for idx in proto_idx.split("_")]
                
                with torch.no_grad():
                    proto_features, pooled, _ = self.model(sample_batch, inference=True)
                    activations = [pooled[0, idx].item() for idx in proto_indices]
                    strongest_idx = proto_indices[np.argmax(activations)]
                    max_val = max(activations)
                    activation_map = proto_features[0, strongest_idx]
            else:
                # Single prototype case
                proto_num = int(proto_idx) if isinstance(proto_idx, str) else proto_idx
                with torch.no_grad():
                    proto_features, pooled, _ = self.model(sample_batch, inference=True)
                    max_val = pooled[0, proto_num].item()
                    activation_map = proto_features[0, proto_num]
                

            
            # Find coordinates of maximum activation
            if proto_features.ndim == 4:
                max_pos = torch.where(activation_map == torch.max(activation_map))
                
                if len(max_pos[0]) > 0:
                    max_h, max_w = max_pos[0][0].item(), max_pos[1][0].item()
                    
                    # Convert feature map coordinates to image coordinates
                    img_h, img_w = sample.shape[1:]
                    feat_h, feat_w = activation_map.shape
                    scale_h, scale_w = img_h / feat_h, img_w / feat_w
                    
                    # Use fixed 32px patch size
                    patch_size = 32
                    
                    # Calculate patch center in image coordinates
                    center_h = int(max_h * scale_h)
                    center_w = int(max_w * scale_w)
                    
                    # Calculate patch boundaries
                    h_min = max(0, center_h - patch_size // 2)
                    h_max = min(img_h, center_h + patch_size // 2)
                    w_min = max(0, center_w - patch_size // 2)
                    w_max = min(img_w, center_w + patch_size // 2)
                    
                    # Extract patch
                    patch = sample[:, h_min:h_max, w_min:w_max]
                    return patch, (w_min, h_min, w_max-w_min, h_max-h_min), float(max_val)
            
            # Fallback to center crop
            img_h, img_w = sample.shape[1:]
            patch_size = min(img_h, img_w) // 3
            center_h, center_w = img_h // 2, img_w // 2
            h_min = max(0, center_h - patch_size // 2)
            h_max = min(img_h, center_h + patch_size // 2)
            w_min = max(0, center_w - patch_size // 2)
            w_max = min(img_w, center_w + patch_size // 2)
            
            patch = sample[:, h_min:h_max, w_min:w_max]
            return patch, (w_min, h_min, w_max-w_min, h_max-h_min), float(max_val)
        
        # Normalize prototype_indices to handle groups
        if prototype_indices is None:
            prototype_indices = list(pure_results.keys())
        
        # Process prototypes
        output_data = {}
        print("Processing prototype data...")
        print(pure_results.keys())
        for proto_idx in prototype_indices:
            proto_key = proto_idx
            
            if isinstance(proto_idx, list):
                proto_key = "_".join(map(str, proto_idx))

            if isinstance(proto_idx, list):
                proto_idx = "_".join(map(str, proto_idx))
            
            if proto_key not in pure_results: #and proto_idx not in pure_results:
                print(f"Cannot find prototype {proto_key}")
                continue
            
            # Try to get with both key formats
            result = pure_results.get(proto_key, pure_results.get(proto_idx, None))
            if result is None:
                continue
                
            samples = result.get('samples', result.get('top_samples'))
            cluster_labels = result['cluster_labels'] 
            activations = result.get('top_activations', [1.0] * len(samples))
            
            # Generate group name for prototype group
            proto_name = proto_key if isinstance(proto_key, str) else str(proto_key)
            
            proto_data = {"clusters": {}}
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue
                    
                print('Getting highest activations ')
                # Take top 8 samples by activation
                cluster_acts = [activations[i] for i in indices]
                sorted_idx = np.argsort(cluster_acts)[::-1]
                top_indices = [indices[i] for i in sorted_idx[:min(10, len(sorted_idx))]]
                
                cluster_samples = []
                for idx in top_indices:
                    # Get sample and find activation
                    sample = samples[idx]
                    print(f'finding top activating patches and samples for {idx}')
                    patch, coords, act_val = find_activation_patch(sample, proto_idx)
                    
                    # Create images WITH highlighted regions
                    full_image = tensor_to_base64(sample, coords)
                    patch_image = tensor_to_base64(patch)
                    
                    sample_data = {
                        "id": int(idx),
                        "activation": float(activations[idx]) if idx < len(activations) else 0.0,
                        "act_value": float(act_val),
                        "image": full_image,
                        "patch": patch_image,
                        "coords": coords
                    }
                    cluster_samples.append(sample_data)
                
                proto_data["clusters"][int(cluster_id)] = cluster_samples
            
            output_data[proto_name] = proto_data
        
        # Create JS and HTML for visualization
        js_file = "prototype_data.js"
        with open(js_file, "w") as f:
            f.write(f"const prototypeData = {json.dumps(output_data)};")
        
        # HTML file content remains the same
        html_file = "prototype_visualization.html"
        html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prototype Activation Patches</title>
        <script src="prototype_data.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .controls { margin: 20px 0; }
            select, button { padding: 8px 12px; margin-right: 10px; }
            .cluster-btn { 
                margin-right: 5px; 
                border: none; 
                border-radius: 4px; 
                padding: 8px 15px; 
                cursor: pointer; 
                font-weight: bold;
            }
            .active-cluster { box-shadow: 0 0 0 3px rgba(0,0,0,0.3); }
            .gallery { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); 
                gap: 20px; 
                margin-top: 20px;
            }
            .card { 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                overflow: hidden; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover { 
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .card-header { 
                padding: 10px 15px; 
                background: #f5f5f5; 
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #ddd;
            }
            .card-body { padding: 15px; }
            img { 
                width: 100%; 
                border-radius: 4px; 
                display: block;
            }
            .patch-img { 
                border: 3px solid red; 
                margin-bottom: 12px;
            }
            .section { margin-bottom: 15px; }
            h2, h3, h4 { margin-top: 0; color: #333; }
            .proto-info {
                background: #f0f4f8;
                padding: 10px 15px;
                border-radius: 6px;
                margin: 10px 0 20px 0;
            }
            .activation-value {
                font-weight: bold;
                color: #d04040;
            }
        </style>
    </head>
    <body>
        <h1>Prototype Activation Patches</h1>
        
        <div class="proto-info" id="proto-info">
            Select a prototype to begin exploring activation patches
        </div>
        
        <div class="controls">
            <label for="prototype-select">Select Prototype:</label>
            <select id="prototype-select">
                <option value="">-- Select Prototype --</option>
            </select>
            
            <div id="cluster-buttons" style="display: inline-block; margin-left: 20px;"></div>
        </div>
        
        <div id="gallery" class="gallery"></div>
        
        <script>
            // Initialize UI
            const protoSelect = document.getElementById('prototype-select');
            const clusterButtons = document.getElementById('cluster-buttons');
            const gallery = document.getElementById('gallery');
            const protoInfo = document.getElementById('proto-info');
            
            let currentCluster = null;
            
            // Populate prototype dropdown with proper naming
            for (const protoId in prototypeData) {
                const option = document.createElement('option');
                option.value = protoId;
                
                // Display prototype name (handle groups differently)
                if (protoId.includes('_')) {
                    option.textContent = `Prototype Group [${protoId}]`;
                } else {
                    option.textContent = `Prototype ${protoId}`;
                }
                protoSelect.appendChild(option);
            }
            
            // Handle prototype selection
            protoSelect.addEventListener('change', function() {
                const protoId = this.value;
                if (!protoId) {
                    clusterButtons.innerHTML = '';
                    gallery.innerHTML = '';
                    protoInfo.textContent = 'Select a prototype to begin exploring activation patches';
                    return;
                }
                
                showPrototype(protoId);
            });
            
            function showPrototype(protoId) {
                const protoData = prototypeData[protoId];
                currentCluster = null;
                
                // Create cluster buttons
                clusterButtons.innerHTML = '';
                const clusters = Object.keys(protoData.clusters);
                
                // Update prototype info
                const totalClusters = clusters.length;
                let protoDisplayName = protoId.includes('_') ? 
                    `Prototype Group [${protoId}]` : `Prototype ${protoId}`;
                protoInfo.innerHTML = `<strong>${protoDisplayName}</strong> - ${totalClusters} clusters detected`;
                
                clusters.forEach((clusterId, i) => {
                    const clusterSamples = protoData.clusters[clusterId];
                    const btn = document.createElement('button');
                    btn.textContent = `Cluster ${parseInt(clusterId) + 1} (${clusterSamples.length})`;
                    btn.className = 'cluster-btn';
                    btn.dataset.clusterId = clusterId;
                    btn.style.backgroundColor = getColor(i);
                    btn.style.color = 'white';
                    
                    btn.onclick = () => {
                        document.querySelectorAll('.cluster-btn').forEach(b => 
                            b.classList.remove('active-cluster'));
                        btn.classList.add('active-cluster');
                        showCluster(protoId, clusterId);
                    };
                    
                    clusterButtons.appendChild(btn);
                });
                
                // Show first cluster by default
                if (clusters.length > 0) {
                    const firstButton = document.querySelector('.cluster-btn');
                    if (firstButton) {
                        firstButton.classList.add('active-cluster');
                        showCluster(protoId, clusters[0]);
                    }
                }
            }
            
            function showCluster(protoId, clusterId) {
                currentCluster = clusterId;
                const samples = prototypeData[protoId].clusters[clusterId];
                
                // Update gallery
                gallery.innerHTML = '';
                
                samples.forEach(sample => {
                    const card = document.createElement('div');
                    card.className = 'card';
                    
                    // Create header
                    const header = document.createElement('div');
                    header.className = 'card-header';
                    header.innerHTML = `
                        <div>Sample ${sample.id}</div>
                        <div class="activation-value">Act: ${sample.activation.toFixed(3)}</div>
                    `;
                    
                    // Create body
                    const body = document.createElement('div');
                    body.className = 'card-body';
                    
                    // Add full image with highlighted region
                    const fullSection = document.createElement('div');
                    fullSection.className = 'section';
                    
                    const fullLabel = document.createElement('h4');
                    fullLabel.textContent = 'Activated Region';
                    
                    const fullImg = document.createElement('img');
                    fullImg.src = `data:image/jpeg;base64,${sample.image}`;
                    fullImg.alt = 'Full Image with Highlighted Region';
                    
                    fullSection.appendChild(fullLabel);
                    fullSection.appendChild(fullImg);
                    
                    // Add patch image section
                    const patchSection = document.createElement('div');
                    patchSection.className = 'section';
                    
                    const patchLabel = document.createElement('h4');
                    patchLabel.textContent = 'Activation Patch';
                    
                    const patchImg = document.createElement('img');
                    patchImg.src = `data:image/jpeg;base64,${sample.patch}`;
                    patchImg.className = 'patch-img';
                    patchImg.alt = 'Activation Patch';
                    
                    patchSection.appendChild(patchLabel);
                    patchSection.appendChild(patchImg);
                    
                    // Assemble card
                    body.appendChild(fullSection);
                    body.appendChild(patchSection);
                    
                    card.appendChild(header);
                    card.appendChild(body);
                    gallery.appendChild(card);
                });
            }
            
            function getColor(index) {
                const colors = [
                    '#4285F4', '#EA4335', '#FBBC05', '#34A853', 
                    '#FF6D01', '#46BDC6', '#9C27B0', '#795548',
                    '#5c6bc0', '#26a69a', '#ec407a', '#ab47bc',
                    '#42a5f5', '#66bb6a', '#ffca28', '#8d6e63'
                ];
                return colors[index % colors.length];
            }
        </script>
    </body>
    </html>
        """
        
        with open(html_file, "w") as f:
            f.write(html)
        
        print(f"Visualization saved to {html_file}")
        print(f"Data saved to {js_file}")
        
        return html_file

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
        pure_analyzer = PURE(self.model, device=self.device, num_ref_samples=max_samples)
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
            # is_polysemantic = sil_score > 0.1  # Threshold for polysemanticity
            is_polysemantic = True
        else:
            sil_score = 0
            is_polysemantic = False
            
        print(f"Silhouette score: {sil_score:.3f}, Polysemantic: {is_polysemantic}")
        
        # Visualize if requested
        if visualize:
            print("Visualizing multi-layer analysis results...")
            
            # Visualize UMAP projections
            # self.visualize_multi_layer_umap(circuits, cluster_labels, prototype_idx)
            
            figures = self.visualize_multi_layer_umap_with_gallery(circuits, cluster_labels, prototype_idx, top_samples)
            # Visualize attributions
            # self.visualize_multi_layer_attributions(
            #     circuits, cluster_labels, prototype_idx, top_samples
            # )
        
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

    def analyze_related_prototypes(self, dataloader, prototype_groups, layer_indices=[-1, -5, -10], 
                            n_clusters=None, adaptive=True, max_clusters=5,
                            clustering_method='hdbscan', visualize=True, max_samples=100,
                            layer_weights=None, output_path='.'):
        """
        Analyze potentially related prototypes by jointly clustering their circuits.
        
        Args:
            dataloader: DataLoader containing dataset samples
            prototype_groups: List where each item is either a single prototype index 
                            or a list of related prototype indices
            layer_indices: List of network layers to analyze
            n_clusters: Fixed number of clusters (if None, determine adaptively)
            adaptive: Whether to adaptively determine cluster count
            max_clusters: Maximum clusters to consider if adaptive
            clustering_method: Clustering algorithm to use
            visualize: Whether to visualize results
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary of analysis results in a format compatible with create_activation_patch_visualization
        """
        print(f"Analyzing {len(prototype_groups)} prototype groups")
        
        # Collect top activating samples for all prototypes
        all_samples = []
        all_activations = []
        all_activated_samples = []
        sample_prototype_map = []  # Track which prototype each sample belongs to
        
        for group_idx, proto_group in enumerate(prototype_groups):
            # Generate group name for prototype group (e.g., "3_22" for [3, 22])
            if isinstance(proto_group, list):
                group_name = "_".join(map(str, proto_group))
            else:
                group_name = str(proto_group)
                proto_group = [proto_group]  # Normalize for consistent handling

            for proto_idx in proto_group:
                print(f"Finding top activating samples for {proto_idx} in {group_name}")
                # Find top activating samples
                pure_analyzer = PURE(self.model, device=self.device, num_ref_samples=max_samples)
                samples, activations, activated_samples = pure_analyzer.find_top_activating_samples(dataloader, proto_idx, True, 0.3)
                
                # Store samples and track which prototype they belong to
                all_samples.append(samples)
                all_activations.extend(activations)
                all_activated_samples.append(activated_samples)
                sample_prototype_map.extend([(group_name, proto_idx)] * len(samples))

        
        # Combine all samples
        if not all_samples:
            return {"error": "No samples found for prototypes"}
            

        combined_act_samples = torch.cat(all_activated_samples)
        combined_samples = torch.cat(all_samples)
        print(f"Collected {len(combined_samples)} samples from all prototypes")
        
        # Compute multi-layer attributions for all samples
        combined_circuits = {}
        
        
        for i, sample in enumerate(combined_samples):
            # Add batch dimension
            sample_batch = sample.unsqueeze(0)
            
            # Get prototype from sample mapping
            group_idx, proto_idx = sample_prototype_map[i]
            
            # Compute attributions
            sample_attributions = self.compute_layer_attributions(
                sample_batch, proto_idx, layer_indices
            )
            
            # Add to circuits dictionary
            for layer_idx, attribution in sample_attributions.items():
                if layer_idx not in combined_circuits:
                    combined_circuits[layer_idx] = []
                
                combined_circuits[layer_idx].append(attribution.detach().cpu())
        
        # Stack attributions for each layer
        print("Finished computing attributions")
        for layer_idx in combined_circuits:
            combined_circuits[layer_idx] = torch.stack(combined_circuits[layer_idx])
        
        # Determine number of clusters if adaptive
        if adaptive and n_clusters is None:
            # Use silhouette score approach
            from sklearn.metrics import silhouette_score
            
            # Choose an informative layer
            # informative_layer = combined_circuits.keys()
            informative_layer = 5
            layer_circuits = combined_circuits[informative_layer]
            flat_circuits = layer_circuits.reshape(layer_circuits.shape[0], -1).numpy()
            
            best_score = -1
            best_n_clusters = 2  # Default
            
            for k in range(2, min(max_clusters + 1, len(flat_circuits))):
                # Skip if too few samples
                if len(flat_circuits) <= k:
                    continue
                    
                # Try clustering with k clusters
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(flat_circuits)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(flat_circuits, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = k
            
            n_clusters = best_n_clusters
            print(f"Determined optimal number of clusters: {n_clusters} with score {best_score:.3f}")
        
        # Cluster the combined circuits
        cluster_labels, centroids, overall_centroids = self.cluster_multi_layer_circuits(
            combined_circuits, n_clusters, method=clustering_method, layer_weights=layer_weights
        )
        
        # Analyze how samples from each prototype map to clusters
        prototype_cluster_map = {}
        
        for i, (group_idx, proto_idx) in enumerate(sample_prototype_map):
            cluster = cluster_labels[i]
            
            if proto_idx not in prototype_cluster_map:
                prototype_cluster_map[proto_idx] = {}
            
            if cluster not in prototype_cluster_map[proto_idx]:
                prototype_cluster_map[proto_idx][cluster] = 0
                
            prototype_cluster_map[proto_idx][cluster] += 1
        
        # Summarize cluster distributions for each prototype
        prototype_cluster_distributions = {}
        for proto_idx, clusters in prototype_cluster_map.items():
            total = sum(clusters.values())
            distribution = {k: v/total for k, v in clusters.items()}
            prototype_cluster_distributions[proto_idx] = distribution
        
        # Organize data by prototype for create_activation_patch_visualization compatibility
        # For each prototype, we need to construct a result with:
        # - samples: the images that activate this prototype
        # - cluster_labels: the cluster assignments for these samples
        # - top_activations: activation values for these samples
        
        reorganized_results = {}
        
        # Get unique prototype indices
        unique_prototypes = set([p for _, p in sample_prototype_map])
        
        
        
        # Add group-to-cluster naming mappings
        group_to_named_clusters = {}
        for group_name in set(item[0] for item in sample_prototype_map):
            group_to_named_clusters[group_name] = {}
            for cluster_idx in range(n_clusters):
                named_cluster = f"{group_name}.{cluster_idx+1}"
                group_to_named_clusters[group_name][cluster_idx] = named_cluster


        # Also include the original combined results for reference
        combined_results = {
            "cluster_labels": cluster_labels,
            "centroids": centroids,
            "sample_prototype_map": sample_prototype_map,
            "group_to_named_clusters": group_to_named_clusters,
            "n_clusters": n_clusters,
            "samples": combined_samples,
            "all_activations": all_activations,
            "circuits": combined_circuits,
            "overall_centroids": overall_centroids
        }
        
        # Visualize if requested
        if visualize:
            print("Visualizing prototype clusters...")
            path = '/media/wlodder/Data/XAI/proto_results/convnext_tiny_26_3_30_200_8_8_256/trial_5_2/15_6_main/log/visualised_prototypes_represent'
            figures = self.visualize_multi_layer_umap_with_gallery_2(combined_circuits, cluster_labels, prototype_groups,
                                                                      combined_act_samples, prototype_patches_dir=path,
                                                                      output_path=output_path)

            # create_interactive_umap_server(circuits=combined_circuits, cluster_labels=cluster_labels, samples=combined_act_samples, prototype_idx=prototype_groups)
            # html_file = self.create_activation_patch_visualization(reorganized_results)
            # combined_results["visualization_file"] = html_file
        
        return combined_results

    def visualize_multi_layer_umap_with_gallery_2(self, circuits, cluster_labels, prototype_idx, 
                                                samples, layer_indices=None, layer_names=None,
                                                max_display_images=12, prototype_patches_dir=None,
                                                additional_samples=None, output_path='.'):
            """
            Create interactive UMAP visualizations with synchronized image gallery display and prototype patch explorer.
            When points are selected in the scatter plot, corresponding images appear in the gallery.
            Users can also browse and select prototype patches in a side panel.
            
            Args:
                circuits: Dictionary of circuit attributions
                cluster_labels: Cluster assignment for each sample
                prototype_idx: Index of the prototype being analyzed
                samples: Original input samples for generating previews
                layer_indices: Specific layer indices to visualize (defaults to all)
                layer_names: Optional dictionary of {layer_idx: display_name} for labels
                max_display_images: Maximum number of images to display in the gallery
                prototype_patches_dir: Base directory containing folders of prototype patches
                                    Format: prototype_patches_dir/prototype_{idx}/patch_{n}.jpg
                additional_samples: Optional dictionary of additional samples to allow selection
                                Format: {sample_id: tensor_image}
                
            Returns:
                Dictionary of HTML files with interactive visualizations
            """
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import umap
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            import base64
            from io import BytesIO
            from PIL import Image
            import os
            import math
            import glob
            import matplotlib.pyplot as plt
            
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
            
            # Define a function to convert tensor to base64 image for embedding
            def tensor_to_base64_img(tensor, size=150):
                # Denormalize and convert to PIL image
                img = tensor.cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                
                # Resize if needed
                if size:
                    pil_img = pil_img.resize((size, size), Image.LANCZOS)
                
                # Convert to base64
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG", quality=80)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            
            # Function to convert file path to base64 image
            def file_to_base64_img(file_path, size=150):
                try:
                    pil_img = Image.open(file_path)
                    
                    # Resize if needed
                    if size:
                        pil_img = pil_img.resize((size, size), Image.LANCZOS)
                    
                    # Convert to base64
                    buffered = BytesIO()
                    pil_img.save(buffered, format="JPEG", quality=80)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    return f"data:image/jpeg;base64,{img_str}"
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
                    return None
            
                        # Get prototype patches if directory is provided
            prototype_patches = []
            prototype_folders = []
            all_prototype_patches = {}  # Dictionary to store patches for multiple prototypes
            
            if prototype_patches_dir:
                # Get all prototype folders
                prototype_folders = sorted(glob.glob(os.path.join(prototype_patches_dir, "prototype_*")))
                
                # Get info about available prototypes for the selector
                available_prototypes = []
                for folder in prototype_folders:
                    proto_id = os.path.basename(folder).replace("prototype_", "")
                    try:
                        proto_id = int(proto_id)
                        available_prototypes.append(proto_id)
                    except ValueError:
                        continue
                
                # Process patches for each prototype
                # Limit to current prototype + a few others to avoid making the HTML too large
                prototypes_to_include = [prototype_idx]  # Always include the current prototype
                
                # Process each prototype's patches
                for proto_id in prototypes_to_include:
                    proto_folder = os.path.join(prototype_patches_dir, f"prototype_{proto_id}")
                    if not os.path.exists(proto_folder):
                        continue
                    
                    # Get all image files for this prototype
                    patch_files = sorted(glob.glob(os.path.join(proto_folder, "*.jpg")) + 
                                    glob.glob(os.path.join(proto_folder, "*.png")) +
                                    glob.glob(os.path.join(proto_folder, "*.jpeg")))
                    
                    proto_patches = []
                    # Convert to base64 for embedding
                    for file_path in patch_files:
                        base64_img = file_to_base64_img(file_path)
                        if base64_img:
                            patch_name = os.path.basename(file_path)
                            patch_data = {
                                "name": patch_name,
                                "path": file_path,
                                "image": base64_img
                            }
                            
                            # Add to the prototype-specific list
                            proto_patches.append(patch_data)
                            
                            # Also add to the current prototype's list if it matches
                            if proto_id == prototype_idx:
                                prototype_patches.append(patch_data)
                    
                    # Store the patches for this prototype
                    all_prototype_patches[proto_id] = proto_patches
            
            # Process additional samples if provided
            additional_images = {}
            if additional_samples:
                for sample_id, tensor in additional_samples.items():
                    additional_images[sample_id] = tensor_to_base64_img(tensor)
            
            # Get number of clusters for color mapping
            unique_clusters = np.unique(cluster_labels)
            
            # Create color mapping
            colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                    for r, g, b, _ in plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))]
            
            results = {}
            
            # For each layer, create a complete dashboard
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
                
                # Create a UMAP scatter plot figure
                fig_scatter = go.Figure()
                
                # Process each cluster
                sample_indices_by_cluster = {}
                images_by_cluster = {}
                
                for j, cluster in enumerate(unique_clusters):
                    mask = cluster_labels == cluster
                    cluster_indices = np.where(mask)[0]
                    sample_indices_by_cluster[j] = cluster_indices.tolist()
                    
                    # Store images for this cluster (for gallery display)
                    images_by_cluster[j] = [tensor_to_base64_img(samples[idx]) for idx in cluster_indices]
                    
                    # Add scatter trace
                    fig_scatter.add_trace(go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode='markers',
                        marker=dict(
                            color=colors[j],
                            size=10,
                            opacity=0.8
                        ),
                        name=f'Cluster {j+1}',
                        customdata=list(zip(cluster_indices, [j] * len(cluster_indices))),
                        hovertemplate="Sample: %{customdata[0]}<br>Cluster: %{customdata[1] + 1}<extra></extra>"
                    ))
                
                # Update layout
                layer_name = layer_names.get(layer_idx, f"Layer {layer_idx}")
                fig_scatter.update_layout(
                    title=f"{layer_name} Attributions for Prototype {prototype_idx}",
                    xaxis_title="UMAP Dimension 1",
                    yaxis_title="UMAP Dimension 2",
                    legend_title="Clusters",
                    height=500,
                    width=800,
                    hovermode='closest',
                    clickmode='event+select'
                )
                
                # Generate HTML for the dashboard with embedded JavaScript for interactivity
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Interactive UMAP Visualization - Prototype {prototype_idx}, Layer {layer_idx}</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .main-container {{ 
                            display: flex; 
                            flex-direction: row;
                            gap: 20px;
                        }}
                        .left-panel {{ 
                            flex: 7;
                            display: flex;
                            flex-direction: column;
                        }}
                        .right-panel {{ 
                            flex: 3;
                            display: flex;
                            flex-direction: column;
                            border-left: 1px solid #ddd;
                            padding-left: 20px;
                        }}
                        .plot-container {{ width: 100%; }}
                        .gallery-container {{ 
                            margin-top: 20px;
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                        }}
                        .cluster-selector {{
                            margin: 20px 0;
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                        }}
                        .cluster-btn {{
                            padding: 8px 15px;
                            cursor: pointer;
                            border: none;
                            border-radius: 4px;
                        }}
                        .gallery-image {{
                            width: 150px;
                            height: 150px;
                            object-fit: cover;
                            border-radius: 4px;
                            transition: transform 0.2s;
                        }}
                        .gallery-image:hover {{
                            transform: scale(1.05);
                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                        }}
                        .image-container {{
                            position: relative;
                            display: inline-block;
                        }}
                        .image-label {{
                            position: absolute;
                            bottom: 0;
                            left: 0;
                            background: rgba(0,0,0,0.7);
                            color: white;
                            padding: 2px 6px;
                            font-size: 12px;
                            border-radius: 0 0 4px 4px;
                        }}
                        .prototype-selector {{
                            margin: 20px 0;
                        }}
                        .prototype-patches {{
                            margin-top: 10px;
                            display: flex;
                            flex-wrap: wrap;
                            gap: 10px;
                            max-height: 500px;
                            overflow-y: auto;
                        }}
                        .load-more {{
                            margin-top: 15px;
                            padding: 10px;
                            background-color: #4CAF50;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                        }}
                        .load-more:hover {{
                            background-color: #45a049;
                        }}
                        .additional-selector {{
                            margin-top: 20px;
                        }}
                        select {{
                            padding: 8px;
                            border-radius: 4px;
                            border: 1px solid #ddd;
                            margin-right: 10px;
                        }}
                        .selected {{
                            border: 3px solid #ff9800 !important;
                        }}
                        h2, h3 {{ margin-bottom: 10px; }}
                        .control-row {{
                            display: flex;
                            align-items: center;
                            margin-bottom: 15px;
                        }}
                        button {{
                            padding: 8px 15px;
                            cursor: pointer;
                            border: none;
                            border-radius: 4px;
                            background-color: #2196F3;
                            color: white;
                        }}
                        button:hover {{
                            background-color: #0b7dda;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Interactive Visualization for Prototype {prototype_idx} - {layer_name}</h1>
                    <div class="main-container">
                        <div class="left-panel">
                            <div class="plot-container" id="scatter-plot"></div>
                            
                            <div class="cluster-selector">
                                <span><strong>Show cluster: </strong></span>
                """
                
                # Add cluster selector buttons
                for j, cluster in enumerate(unique_clusters):
                    html += f"""
                    <button class="cluster-btn" 
                            style="background-color: {colors[j]}; color: white;" 
                            onclick="showClusterImages({j})">
                        Cluster {j+1} ({len(sample_indices_by_cluster[j])} samples)
                    </button>
                    """
                
                html += """
                            </div>
                            
                            <div class="control-row">
                                <h3>Image Display: </h3>
                                <select id="display-count" onchange="updateMaxImages()">
                                    <option value="12">Show 12 images</option>
                                    <option value="24">Show 24 images</option>
                                    <option value="48">Show 48 images</option>
                                    <option value="96">Show 96 images</option>
                                    <option value="200">Show 200 images</option>
                                </select>
                            </div>
                            
                            <h2>Image Gallery</h2>
                            <div id="gallery-info">Select points in the plot or click a cluster button to view images.</div>
                            <div class="gallery-container" id="image-gallery"></div>
                            <button class="load-more" id="load-more-btn" onclick="loadMoreImages()" style="display:none;">
                                Load More Images
                            </button>
                        </div>
                        
                        <div class="right-panel">
                """
                
                # Prototype Selector Section
                if prototype_folders:
                    available_prototypes = []
                    for folder in prototype_folders:
                        proto_id = os.path.basename(folder).replace("prototype_", "")
                        try:
                            proto_id = int(proto_id)
                            available_prototypes.append(proto_id)
                        except ValueError:
                            continue
                    
                    if available_prototypes:
                        html += """
                            <h2>Prototype Explorer</h2>
                            <div class="prototype-selector">
                                <div class="control-row">
                                    <select id="prototype-select">
                        """
                        
                        for proto_id in sorted(available_prototypes):
                            selected = "selected" if proto_id == prototype_idx else ""
                            html += f"""
                                        <option value="{proto_id}" {selected}>Prototype {proto_id}</option>
                            """
                        
                        html += """
                                    </select>
                                    <button onclick="loadPrototypeView()">View Prototype</button>
                                </div>
                            </div>
                        """
                
                # Prototype Patches Section
                html += """
                            <h2 id="prototype-title">Prototype {prototype_idx} Patches</h2>
                """
                
                html += """
                        <div class="prototype-patches" id="prototype-patches">
                """
                
                if prototype_patches:
                    # Add each patch
                    for i, patch in enumerate(prototype_patches):
                        html += f"""
                        <div class="image-container">
                            <img src="{patch['image']}" class="gallery-image" 
                                onclick="selectPatch(this, {i})" alt="{patch['name']}"/>
                            <div class="image-label">{patch['name']}</div>
                        </div>
                        """
                else:
                    html += """
                        <p>No prototype patches available for this prototype.</p>
                    """
                    
                html += """
                        </div>
                """
                    
                # Close the right panel and main container divs
                html += """
                        </div>
                    </div>
                    
                    <script>
                        // Store all image data
                        const clusterData = {
                """
                
                # Embed image data as JavaScript variables
                for j in range(len(unique_clusters)):
                    html += f"""
                            {j}: {{
                                indices: {sample_indices_by_cluster[j]},
                                images: {images_by_cluster[j]},
                                color: "{colors[j]}"
                            }},
                    """
                
                # Add additional samples if available
                if additional_images:
                    html += """
                        additional: {
                    """
                    for sample_id, img_data in additional_images.items():
                        html += f"""
                            "{sample_id}": "{img_data}",
                        """
                    html += """
                        },
                    """
                    
                # Add prototype patch data
                html += """
                        };
                        
                        // Store prototype patch data
                        const prototypePatches = [
                """
                
                for patch in prototype_patches:
                    html += f"""
                        {{
                            name: "{patch['name']}",
                            image: "{patch['image']}"
                        }},
                    """
                    
                html += """
                        ];
                        
                        // Variables to track current state
                        let currentCluster = null;
                        let currentSelection = [];
                        let currentMaxImages = 12;
                        let displayedCount = 0;
                        
                        // Create the scatter plot
                        const scatterData = 
                """
                
                # Embed the Plotly figure data
                html += fig_scatter.to_json()
                
                html += """
                        ;
                        
                        Plotly.newPlot('scatter-plot', scatterData.data, scatterData.layout);
                        
                        // All prototype patches data - will be populated during initialization
                        const allPrototypePatches = {};
                        
                        // Function to display a different prototype without navigation
                        function loadPrototypeView() {
                            const prototypeSelect = document.getElementById('prototype-select');
                            const selectedPrototype = prototypeSelect.value;
                            const prototypePatchesElement = document.getElementById('prototype-patches');
                            const prototypeTitle = document.getElementById('prototype-title');
                            
                            // Update the title
                            if (prototypeTitle) {
                                prototypeTitle.textContent = `Prototype ${selectedPrototype} Patches`;
                            }
                            
                            // Clear current patches
                            prototypePatchesElement.innerHTML = '';
                            
                            // Check if we have the data for this prototype
                            if (allPrototypePatches[selectedPrototype]) {
                                // Display patches for the selected prototype
                                const patches = allPrototypePatches[selectedPrototype];
                                
                                if (patches.length === 0) {
                                    prototypePatchesElement.innerHTML = '<p>No patches available for this prototype</p>';
                                    return;
                                }
                                
                                // Add each patch to the display
                                patches.forEach((patch, i) => {
                                    const div = document.createElement('div');
                                    div.className = 'image-container';
                                    
                                    const img = document.createElement('img');
                                    img.src = patch.image;
                                    img.className = 'gallery-image';
                                    img.onclick = function() { selectPatch(this, i); };
                                    img.alt = patch.name;
                                    
                                    const label = document.createElement('div');
                                    label.className = 'image-label';
                                    label.textContent = patch.name;
                                    
                                    div.appendChild(img);
                                    div.appendChild(label);
                                    prototypePatchesElement.appendChild(div);
                                });
                            } else {
                                // We don't have this prototype's data
                                prototypePatchesElement.innerHTML = 
                                    '<p>Data for this prototype is not preloaded. You can still view it by navigating to its dedicated page.</p>' +
                                    '<button onclick="navigateToPrototype()">Go to Prototype ' + selectedPrototype + '</button>';
                            }
                        }
                        
                        // Fallback function to navigate to a different prototype if needed
                        function navigateToPrototype() {
                            const prototypeSelect = document.getElementById('prototype-select');
                            const selectedPrototype = prototypeSelect.value;
                            
                            // Get current URL path and file name
                            const currentPath = window.location.pathname;
                            const currentDir = currentPath.substring(0, currentPath.lastIndexOf('/'));
                            const baseDir = currentDir.substring(0, currentDir.lastIndexOf('/'));
                            
                            // Construct new path to the selected prototype's visualization
                            const layerName = "${layer_name}";
                            const newPath = `${baseDir}/prototype_${selectedPrototype}_visualizations/${layerName}_visualization.html`;
                            
                            // Navigate to the new URL
                            window.location.href = newPath;
                        }
                        
                        // Function to update max images to display
                        function updateMaxImages() {
                            const selectElement = document.getElementById('display-count');
                            currentMaxImages = parseInt(selectElement.value);
                            
                            // Re-display current view with new limit
                            if (currentCluster !== null) {
                                showClusterImages(currentCluster, 0, true);
                            } else if (currentSelection.length > 0) {
                                showSelectedImages(currentSelection, 0, true);
                            }
                        }
                        
                        // Function to select a prototype patch
                        function selectPatch(element, index) {
                            // Toggle selection visual state
                            const allPatches = document.querySelectorAll('#prototype-patches .gallery-image');
                            allPatches.forEach(img => img.classList.remove('selected'));
                            element.classList.add('selected');
                            
                            // You could add more functionality here to show this patch in detail
                            console.log(`Selected patch ${index}: ${prototypePatches[index].name}`);
                            
                            // Display the selected patch in a larger view or with more information
                            // This could be expanded to show more details about the patch
                        }
                        
                        // Function to display images for a specific cluster
                        function showClusterImages(clusterIdx, startIndex = 0, reset = false) {
                            currentCluster = clusterIdx;
                            currentSelection = [];
                            
                            const gallery = document.getElementById('image-gallery');
                            const info = document.getElementById('gallery-info');
                            const loadMoreBtn = document.getElementById('load-more-btn');
                            const cluster = clusterData[clusterIdx];
                            
                            // Clear gallery if starting from beginning or resetting
                            if (startIndex === 0 || reset) {
                                gallery.innerHTML = '';
                                displayedCount = 0;
                            }
                            
                            // Calculate how many to show
                            const remaining = cluster.images.length - startIndex;
                            const numToShow = Math.min(remaining, currentMaxImages);
                            
                            // Update info text
                            info.innerHTML = `Showing ${displayedCount + numToShow} of ${cluster.images.length} images from Cluster ${clusterIdx + 1}`;
                            
                            // Add images
                            for (let i = 0; i < numToShow; i++) {
                                const idx = startIndex + i;
                                if (idx >= cluster.images.length) break;
                                
                                const div = document.createElement('div');
                                div.className = 'image-container';
                                
                                const img = document.createElement('img');
                                img.src = cluster.images[idx];
                                img.className = 'gallery-image';
                                img.style.border = `3px solid ${cluster.color}`;
                                img.onclick = function() { this.classList.toggle('selected'); };
                                
                                const label = document.createElement('div');
                                label.className = 'image-label';
                                label.textContent = `Sample ${cluster.indices[idx]}`;
                                
                                div.appendChild(img);
                                div.appendChild(label);
                                gallery.appendChild(div);
                                displayedCount++;
                            }
                            
                            // Show/hide load more button
                            if (displayedCount < cluster.images.length) {
                                loadMoreBtn.style.display = 'block';
                                loadMoreBtn.onclick = function() {
                                    showClusterImages(clusterIdx, displayedCount);
                                };
                            } else {
                                loadMoreBtn.style.display = 'none';
                            }
                        }
                        
                        // Function to display selected images
                        function showSelectedImages(indices, startIndex = 0, reset = false) {
                            currentCluster = null;
                            currentSelection = indices;
                            
                            const gallery = document.getElementById('image-gallery');
                            const info = document.getElementById('gallery-info');
                            const loadMoreBtn = document.getElementById('load-more-btn');
                            
                            // Clear gallery if starting from beginning or resetting
                            if (startIndex === 0 || reset) {
                                gallery.innerHTML = '';
                                displayedCount = 0;
                            }
                            
                            // Calculate how many to show
                            const remaining = indices.length - startIndex;
                            const numToShow = Math.min(remaining, currentMaxImages);
                            
                            // Update info text
                            info.innerHTML = `Showing ${displayedCount + numToShow} of ${indices.length} selected samples`;
                            
                            // Display images for selected points
                            for (let i = 0; i < numToShow; i++) {
                                const idx = startIndex + i;
                                if (idx >= indices.length) break;
                                
                                const sampleIdx = indices[idx];
                                let clusterIdx, imageIdx;
                                
                                // Find which cluster this sample belongs to
                                for (const [cIdx, cluster] of Object.entries(clusterData)) {
                                    if (cIdx === 'additional') continue; // Skip additional images
                                    
                                    const localIdx = cluster.indices.indexOf(sampleIdx);
                                    if (localIdx !== -1) {
                                        clusterIdx = parseInt(cIdx);
                                        imageIdx = localIdx;
                                        break;
                                    }
                                }
                                
                                if (clusterIdx === undefined) continue;
                                
                                const div = document.createElement('div');
                                div.className = 'image-container';
                                
                                const img = document.createElement('img');
                                img.src = clusterData[clusterIdx].images[imageIdx];
                                img.className = 'gallery-image';
                                img.style.border = `3px solid ${clusterData[clusterIdx].color}`;
                                img.onclick = function() { this.classList.toggle('selected'); };
                                
                                const label = document.createElement('div');
                                label.className = 'image-label';
                                label.textContent = `Sample ${sampleIdx}`;
                                
                                div.appendChild(img);
                                div.appendChild(label);
                                gallery.appendChild(div);
                                displayedCount++;
                            }
                            
                            // Show/hide load more button
                            if (displayedCount < indices.length) {
                                loadMoreBtn.style.display = 'block';
                                loadMoreBtn.onclick = function() {
                                    showSelectedImages(indices, displayedCount);
                                };
                            } else {
                                loadMoreBtn.style.display = 'none';
                            }
                        }
                        
                        // Function to load more images
                        function loadMoreImages() {
                            if (currentCluster !== null) {
                                showClusterImages(currentCluster, displayedCount);
                            } else if (currentSelection.length > 0) {
                                showSelectedImages(currentSelection, displayedCount);
                            }
                        }
                        
                        // Handle selection events from the plot
                        document.getElementById('scatter-plot').on('plotly_selected', function(eventData) {
                            if (!eventData || !eventData.points || eventData.points.length === 0) {
                                return;
                            }
                            
                            // Collect selected points
                            const selectedIndices = eventData.points.map(pt => pt.customdata[0]);
                            
                            // Display the selected images
                            showSelectedImages(selectedIndices);
                        });
                        
                        // Show the first cluster by default
                        showClusterImages(0);
                    </script>
                </body>
                </html>
            
               """ 
                # Store the HTML for this layer
                results[layer_idx] = html
            
            # Save HTML files to disk
            output_dir = f"{output_path}/prototype_{prototype_idx}_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            for layer_idx, html_content in results.items():
                layer_name = layer_names.get(layer_idx, f"layer_{layer_idx}")
                file_path = os.path.join(output_dir, f"{layer_name}_visualization.html")
                
                with open(file_path, "w") as f:
                    f.write(html_content)
                
                print(f"Saved visualization for layer {layer_idx} to {file_path}")
            
            return results


