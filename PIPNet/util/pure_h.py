from typing import List, Dict, Union, Tuple, Optional
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from umap import UMAP

class MultiLayerPURE:
    """
    Enhanced PURE (Prototype Understanding through Real Examples) for PIPNet architecture
    with support for multi-layer attribution and analysis.
    """
    def __init__(self, model, device='cuda', num_ref_samples=100):
        """
        Initialize MultiLayerPURE for a PIPNet model.
        
        Args:
            model: The PIPNet model
            device: Device to run computations on ('cuda' or 'cpu')
            num_ref_samples: Number of reference samples to use for clustering
        """
        self.model = model
        self.device = device
        self.num_ref_samples = num_ref_samples
        self.model.to(device)
        
        
    def get_available_layers(self, include_shapes=True, sample_input=None):
        """
        Get all available layer names in the model for attribution analysis.
        
        Args:
            include_shapes: Whether to include output shapes in the result
            sample_input: Optional sample input to compute output shapes
                        If None and include_shapes=True, a dummy input will be created
                        
        Returns:
            Dictionary of layer information with hierarchical organization
        """
        available_layers = {}
        layer_hooks = {}
        handles = []
        
        # Create a hook function that captures output shapes
        def get_shape_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_hooks[name] = output.shape
                elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                    layer_hooks[name] = output[0].shape
            return hook
        
        # Recursively gather all modules
        def gather_modules(module, name=""):
            module_info = {
                "type": module.__class__.__name__,
                "params": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }
            
            # Register hook for shape capture if needed
            if include_shapes:
                handle = module.register_forward_hook(get_shape_hook(name))
                handles.append(handle)
            
            # Get child modules
            children = list(module.named_children())
            if children:
                module_info["children"] = {}
                for child_name, child_module in children:
                    # Build the full module path
                    if name:
                        child_full_name = f"{name}.{child_name}"
                    else:
                        child_full_name = child_name
                    
                    # Recursively gather child info
                    module_info["children"][child_name] = gather_modules(child_module, child_full_name)
            
            # Store in the overall structure
            if name:
                available_layers[name] = module_info
            
            return module_info
        
        try:
            # Start the recursive gathering
            model_info = gather_modules(self.model)
            
            # If shapes are requested, do a forward pass
            if include_shapes:
                self.model.eval()
                with torch.no_grad():
                    if sample_input is not None:
                        _ = self.model(sample_input.to(self.device))
                    else:
                        # Create a dummy input based on model input size
                        # This is a best-effort approach - might need adjustment for specific models
                        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                        try:
                            _ = self.model(dummy_input)
                        except Exception as e:
                            print(f"Error during forward pass with dummy input: {e}")
                            print("Shapes will not be available. Please provide a valid sample_input.")
                
                # Add shapes to the module info
                for name, shape in layer_hooks.items():
                    parts = name.split('.')
                    current = available_layers
                    
                    # Navigate to the right module
                    for i, part in enumerate(parts):
                        if i == 0:
                            if part in current:
                                current = current[part]
                            else:
                                # This is for the root model
                                continue
                        elif i < len(parts) - 1:
                            current = current["children"][part]
                    
                    # Add the shape
                    if parts:
                        if len(parts) == 1:
                            # Root level module
                            if parts[0] in available_layers:
                                available_layers[parts[0]]["output_shape"] = shape
                        else:
                            # Nested module
                            last_part = parts[-1]
                            if "children" in current and last_part in current["children"]:
                                current["children"][last_part]["output_shape"] = shape
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
        
        # Create a flattened version that's easier to select from
        flattened_layers = {}
        
        def flatten_layer_info(layers_dict, prefix=""):
            for name, info in layers_dict.items():
                full_name = f"{prefix}.{name}" if prefix else name
                layer_type = info["type"]
                params = info["params"]
                shape_info = info.get("output_shape", None)
                
                flattened_layers[full_name] = {
                    "type": layer_type,
                    "params": params,
                    "output_shape": shape_info
                }
                
                if "children" in info and info["children"]:
                    flatten_layer_info(info["children"], full_name)
        
        flatten_layer_info(available_layers)
        
        # Also include the root model
        root_info = {
            "type": self.model.__class__.__name__,
            "params": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        if "" in layer_hooks:
            root_info["output_shape"] = layer_hooks[""]
            
        flattened_layers["(root)"] = root_info
        
        return flattened_layers
        
    def compute_attributions(self, x, prototype_idx, target_layers=None):
        """
        Compute attributions (circuit) for a specific prototype across multiple layers
        using Gradient × Activation with recursive layer hooking.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype to compute attributions for
            target_layers: List of layer names to compute attributions for.
                        If None, attributions will be computed for all layers.
            
        Returns:
            Dictionary of attributions for each layer
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Get original features that will be attributed
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Dictionary to store intermediate activations
        layer_activations = {}
        
        # Register hooks to capture intermediate activations
        handles = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                # Only store activations that can have gradients
                if isinstance(output, torch.Tensor) and output.requires_grad:
                    layer_activations[name] = output
                    layer_activations[name].retain_grad()
                elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                    # For modules that return multiple tensors, store the first one
                    if output[0].requires_grad:
                        layer_activations[name] = output[0]
                        layer_activations[name].retain_grad()
            return hook
        
        # Recursively register hooks for all modules
        def register_hooks(module, name=""):
            # Register hook for this module
            if list(module.children()):
                # This is a container module, recurse into children
                for child_name, child_module in module.named_children():
                    # Build the full module path
                    if name:
                        child_full_name = f"{name}.{child_name}"
                    else:
                        child_full_name = child_name
                    
                    # Only register hooks for leaf modules with parameters
                    if not list(child_module.children()) or any(p.requires_grad for p in child_module.parameters()):
                        handle = child_module.register_forward_hook(get_activation_hook(child_full_name))
                        handles.append(handle)
                    
                    # Continue recursion
                    register_hooks(child_module, child_full_name)
        
        # Register hooks for all modules
        print("\n========= RECURSIVELY REGISTERING LAYER HOOKS =========")
        register_hooks(self.model)
        print(f"Registered hooks for all available layers")
        
        # Forward pass to get prototype presence
        try:
            _, pooled, _, features = self.model(x, inference=False, features_save=True)
            
            # Get the specific prototype activation
            target_activation = pooled[:, prototype_idx]
            
            # Print available layers
            print(f"\nCaptured {len(layer_activations)} layer activations:")
            for i, (name, activation) in enumerate(layer_activations.items()):
                if i < 10:  # Limit the output to first 10 layers to avoid excessive logging
                    print(f"  {name}: Shape {activation.shape}")
                elif i == 10:
                    print(f"  ... and {len(layer_activations) - 10} more layers")
            
            # Dictionary to store attributions for each layer
            attributions_dict = {}
            
            # Filter layers based on target_layers if specified
            layers_to_process = layer_activations
            if target_layers is not None:
                layers_to_process = {name: activation for name, activation in layer_activations.items() 
                                if any(target in name for target in target_layers)}
                print(f"\nSelected {len(layers_to_process)} layers based on target specifications: {target_layers}")
            
            # Compute gradients and attributions for each captured layer
            for name, layer_features in layers_to_process.items():
                try:
                    # Compute gradients with respect to features at this layer
                    gradients = torch.autograd.grad(target_activation.sum(), layer_features, 
                                                retain_graph=True, allow_unused=True)[0]
                    
                    if gradients is not None:
                        # Gradient × Activation attribution
                        layer_attributions = gradients * layer_features
                        
                        # Sum over spatial dimensions to get attribution per channel
                        # Handle different tensor shapes
                        if len(layer_attributions.shape) == 4:  # [B, C, H, W]
                            layer_attributions = layer_attributions.sum(dim=(2, 3))
                        elif len(layer_attributions.shape) == 3:  # [B, C, S]
                            layer_attributions = layer_attributions.sum(dim=2)
                        
                        attributions_dict[name] = layer_attributions
                        print(f"  Computed attributions for layer '{name}': Shape {layer_attributions.shape}")
                    else:
                        print(f"  No gradients available for layer '{name}'")
                except Exception as e:
                    print(f"  Error computing attributions for layer '{name}': {e}")
            
            # Include the default feature attribution if not already included
            if 'default' not in attributions_dict and features.requires_grad:
                try:
                    gradients = torch.autograd.grad(target_activation.sum(), features, 
                                                retain_graph=True, allow_unused=True)[0]
                    if gradients is not None:
                        attributions = gradients * features
                        attributions = attributions.sum(dim=(2, 3))
                        attributions_dict['default'] = attributions
                        # print(f"  Computed attributions for default features: Shape {attributions.shape}")
                except Exception as e:
                    print(f"  Error computing attributions for default features: {e}")
            
            # if not attributions_dict:
            #     print("Warning: No attributions were computed for any layer!")
            # else:
            #     print(f"Successfully computed attributions for {len(attributions_dict)} layers")
            
            return attributions_dict
            
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

    def compute_multi_layer_circuits(self, top_samples, prototype_idx, target_layers=None):
        """
        Compute circuits across multiple layers for each top activating sample.
        
        Args:
            top_samples: Tensor of top activating samples
            prototype_idx: Index of the prototype
            target_layers: List of layer indices to compute attributions for
            
        Returns:
            Dictionary of circuit attributions for each layer
        """
        # Dictionary to store circuits for each layer
        all_layer_circuits = {}
        
        for sample in top_samples:
            sample = sample.unsqueeze(0)  # Add batch dimension
            # Get attributions for all target layers
            layer_attributions = self.compute_attributions(sample, prototype_idx, target_layers)
            
            # Organize by layer
            for layer_idx, attribution in layer_attributions.items():
                if layer_idx not in all_layer_circuits:
                    all_layer_circuits[layer_idx] = []
                
                if attribution is not None:
                    all_layer_circuits[layer_idx].append(attribution.cpu().detach())
        
        # Stack circuits for each layer
        stacked_circuits = {}
        for layer_idx, circuits in all_layer_circuits.items():
            if circuits:  # Check if we have any valid attributions
                stacked_circuits[layer_idx] = torch.stack(circuits)
        
        return stacked_circuits
    
    def find_top_activating_samples(self, dataloader, prototype_idx):
        """
        Find the top activating samples for a prototype.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to find activating samples for
            
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
        top_indices = sorted_indices[:self.num_ref_samples]
        
        top_samples = torch.stack([images[i] for i in top_indices])
        top_activations = [activations[i] for i in top_indices]
        
        return top_samples, top_activations
        
    def cluster_multi_layer_circuits(self, layer_circuits, n_clusters=2, layer_weights=None):
        """
        Cluster circuits across multiple layers to identify different semantics.
        
        Args:
            layer_circuits: Dictionary of circuit attributions per layer
            n_clusters: Number of clusters (virtual neurons) to create
            layer_weights: Optional dictionary of weights for each layer
            
        Returns:
            Tuple of (cluster labels, centroids dict, weighted_features)
        """
        if layer_weights is None:
            # Default to equal weighting
            layer_weights = {layer_idx: 1.0 for layer_idx in layer_circuits.keys()}
        
        # Normalize weights
        total_weight = sum(layer_weights.values())
        normalized_weights = {k: v/total_weight for k, v in layer_weights.items()}
        
        # Prepare combined features for clustering
        combined_features = []
        layer_shapes = {}
        
        # Process each sample
        num_samples = next(iter(layer_circuits.values())).shape[0]
        
        for sample_idx in range(num_samples):
            # Flatten and concatenate features from all layers for this sample
            sample_features = []
            
            for layer_idx, circuits in layer_circuits.items():
                if sample_idx < circuits.shape[0]:  # Check if we have this sample for this layer
                    layer_features = circuits[sample_idx]
                    
                    # Store shape for later reconstruction
                    if layer_idx not in layer_shapes:
                        layer_shapes[layer_idx] = layer_features.shape
                    
                    # Flatten the features
                    flat_features = layer_features.reshape(-1).numpy()
                    
                    # Apply layer weight
                    if layer_idx in normalized_weights:
                        flat_features = flat_features * normalized_weights[layer_idx]
                    
                    sample_features.append(flat_features)
            
            # Combine all layer features for this sample
            if sample_features:
                combined_features.append(np.concatenate(sample_features))
        
        # Convert to numpy array for clustering
        combined_features = np.array(combined_features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(combined_features)
        
        # Get centroids and reconstruct them per layer
        centroids_dict = {}
        for layer_idx, shape in layer_shapes.items():
            # Initialize centroids for this layer
            layer_centroids = []
            
            for cluster_idx in range(n_clusters):
                # Find all samples in this cluster
                cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_idx]
                
                if cluster_samples:
                    # Average the features for this cluster and layer
                    cluster_layer_features = layer_circuits[layer_idx][cluster_samples]
                    centroid = cluster_layer_features.mean(dim=0)
                    layer_centroids.append(centroid)
                else:
                    # Fallback if cluster is empty
                    centroid = torch.zeros(shape)
                    layer_centroids.append(centroid)
            
            # Stack all centroids for this layer
            centroids_dict[layer_idx] = torch.stack(layer_centroids)
        
        return cluster_labels, centroids_dict, combined_features
    
    def assign_multi_layer_circuit(self, x, prototype_idx, centroids_dict, layer_weights=None):
        """
        Assign an input to the closest circuit across multiple layers for a prototype.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype
            centroids_dict: Dictionary of circuit centroids per layer
            layer_weights: Optional dictionary of weights for each layer
            
        Returns:
            Index of the closest centroid (circuit)
        """
        if layer_weights is None:
            # Default to equal weighting
            layer_weights = {layer_idx: 1.0 for layer_idx in centroids_dict.keys()}
        
        # Normalize weights
        total_weight = sum(layer_weights.values())
        normalized_weights = {k: v/total_weight for k, v in layer_weights.items()}
        
        # Compute attributions for all layers
        x = x.to(self.device)
        layer_attributions = self.compute_attributions(x, prototype_idx, target_layers=list(centroids_dict.keys()))
        
        # Compute weighted distance to each centroid
        n_clusters = next(iter(centroids_dict.values())).shape[0]
        distances = np.zeros(n_clusters)
        
        for layer_idx, centroids in centroids_dict.items():
            if layer_idx in layer_attributions and layer_attributions[layer_idx] is not None:
                attribution = layer_attributions[layer_idx]
                layer_weight = normalized_weights.get(layer_idx, 0.0)
                
                for i, centroid in enumerate(centroids):
                    centroid = centroid.to(self.device)
                    # Compute distance for this layer and add to total
                    dist = F.mse_loss(attribution, centroid.unsqueeze(0)).item() * layer_weight
                    distances[i] += dist
        
        # Return index of closest centroid
        return np.argmin(distances)
    
    def visualize_cluster_attributions(self, layer_circuits, cluster_labels, prototype_idx, 
                                    top_n=10, figsize=(15, 10), layer_names=None):
        """
        Visualize the attribution strength patterns for each cluster across layers.
        
        Args:
            layer_circuits: Dictionary of circuit attributions per layer
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
            top_n: Number of top channels to analyze per layer
            figsize: Size of the figure
            layer_names: Optional dictionary mapping layer indices to human-readable names
            
        Returns:
            Matplotlib figure showing attribution patterns per cluster
        """
        # Get number of clusters and layers
        n_clusters = len(np.unique(cluster_labels))
        layer_indices = sorted(layer_circuits.keys())
        
        # If layer names not provided, create default ones
        if layer_names is None:
            layer_names = {layer_idx: f"Layer {i+1}" for i, layer_idx in enumerate(layer_indices)}
        
        # Create figure
        fig, axes = plt.subplots(n_clusters, len(layer_indices), figsize=figsize, squeeze=False)
        
        # For each cluster and layer, visualize top attribution channels
        for cluster_idx in range(n_clusters):
            # Get samples in this cluster
            cluster_samples = np.where(cluster_labels == cluster_idx)[0]
            
            for i, layer_idx in enumerate(layer_indices):
                ax = axes[cluster_idx, i]
                
                if len(cluster_samples) > 0:
                    # Get circuits for these samples
                    cluster_circuits = layer_circuits[layer_idx][cluster_samples].squeeze(1)
                    
                    # Average attribution across samples
                    mean_attribution = torch.mean(cluster_circuits,dim=0).cpu().numpy()
                    
                    # Sort by absolute value for visualization
                    sorted_idx = np.argsort(np.abs(mean_attribution))[::-1]
                    
                    # Take top_n channels
                    top_idx = sorted_idx[:top_n]
                    
                    # Create bar chart of top attributions
                    colors = ['r' if mean_attribution[idx] < 0 else 'b' for idx in top_idx]
                    y_pos = np.arange(len(top_idx))
                    
                    ax.barh(y_pos, np.abs(mean_attribution[top_idx]), color=colors, alpha=0.7)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([str(idx) for idx in top_idx])
                    ax.invert_yaxis()  # Highest on top
                
                # Set titles and labels
                if cluster_idx == 0:
                    ax.set_title(layer_names.get(layer_idx, f"Layer {layer_idx}"))
                if i == 0:
                    ax.set_ylabel(f"Cluster {cluster_idx + 1}")
                
                # Clean up plot
                if i < len(layer_indices) - 1:
                    ax.set_xticks([])
                
                # Add a small legend for positive/negative values
                # if cluster_idx == 0 and i == 0:
                #     ax.legend(['Negative', 'Positive'], 
                #              loc='upper right', 
                #              labels=['Negative', 'Positive'],
                #              handler_map={str: HandlerPatch(patch_func=lambda x, y, radius: 
                #                                           plt.Rectangle((x, y), width=radius, height=radius, 
                #                                                        facecolor='r' if x == 'Negative' else 'b'))})
        
        plt.tight_layout()
        plt.suptitle(f"Top Channel Attributions per Cluster for Prototype {prototype_idx}", 
                    fontsize=16, y=1.02)
        
        return fig
    
    def disentangle_multi_layer_prototype(self, dataloader, prototype_idx, target_layers=None, 
                                          n_clusters=2, adaptive=True, max_clusters=5, 
                                          layer_weights=None):
        """
        Disentangle a prototype into multiple pure features using multi-layer attributions.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index or list of indices of the prototype(s) to disentangle
            target_layers: List of layer indices to compute attributions for
            n_clusters: Number of clusters (virtual neurons)
            adaptive: Whether to adaptively determine optimal number of clusters
            max_clusters: Maximum number of clusters to consider if adaptive=True
            layer_weights: Optional dictionary of weights for each layer
            
        Returns:
            Dictionary containing disentanglement results
        """
        print(f"Disentangling prototype {prototype_idx} across multiple layers into virtual neurons...")
        
        # Handle single or multiple prototype indices
        if isinstance(prototype_idx, List) and not isinstance(prototype_idx, int):
            # Process multiple prototypes
            overall_layer_circuits = {}
            overall_top_samples = []
            overall_saliency_maps = []
            overall_top_activations = []
            
            for proto in prototype_idx:
                # Find top activating samples for this prototype
                top_samples, top_activations = self.find_top_activating_samples(dataloader, proto)
                print(f"Found {len(top_samples)} top activating samples for prototype {proto}.")
                
                # Compute saliency maps
                saliency_maps = self.compute_saliency_maps(top_samples, proto)
                
                # Compute multi-layer circuits
                print(f"Computing multi-layer circuits for prototype {proto}...")
                layer_circuits = self.compute_multi_layer_circuits(top_samples, proto, target_layers)
                
                # Aggregate results
                for layer_idx, circuits in layer_circuits.items():
                    if layer_idx not in overall_layer_circuits:
                        overall_layer_circuits[layer_idx] = []
                    overall_layer_circuits[layer_idx].append(circuits)
                
                overall_top_samples.append(top_samples)
                overall_saliency_maps.extend(saliency_maps)
                overall_top_activations.extend(top_activations)
            
            # Combine circuits from all prototypes
            combined_layer_circuits = {}
            for layer_idx, circuits_list in overall_layer_circuits.items():
                combined_layer_circuits[layer_idx] = torch.cat(circuits_list, dim=0)
            
            # Combine top samples
            top_samples = torch.cat(overall_top_samples, dim=0)
            saliency_maps = overall_saliency_maps
            top_activations = overall_top_activations
            
            layer_circuits = combined_layer_circuits
        else:
            # Process single prototype
            top_samples, top_activations = self.find_top_activating_samples(dataloader, prototype_idx)
            print(f"Found {len(top_samples)} top activating samples.")
            
            # Compute saliency maps
            saliency_maps = self.compute_saliency_maps(top_samples, prototype_idx)
            
            # Compute multi-layer circuits
            print("Computing multi-layer circuits...")
            layer_circuits = self.compute_multi_layer_circuits(top_samples, prototype_idx, target_layers)
        
        print("Clustering multi-layer circuits...")
        if adaptive:
            # Determine optimal number of clusters
            n_clusters = self.determine_optimal_clusters_multi_layer(
                layer_circuits, max_clusters=max_clusters, layer_weights=layer_weights
            )
            print(f"Adaptively determined {n_clusters} as the optimal number of clusters.")
        
        # Cluster the circuits
        cluster_labels, centroids_dict, combined_features = self.cluster_multi_layer_circuits(
            layer_circuits, n_clusters=n_clusters, layer_weights=layer_weights
        )
        
        # Visualize clusters
        print("Generating visualizations...")
        cluster_vis = self.visualize_clusters(top_samples, cluster_labels, prototype_idx, saliency_maps=saliency_maps)
        
        # Visualize UMAP embedding of combined features
        umap_vis = self.visualize_umap_embedding(combined_features, cluster_labels, prototype_idx)
        
        # # Generate attribution path visualization
        path_vis = self.visualize_attribution_paths(
            layer_circuits, cluster_labels, centroids_dict, prototype_idx
        )

        # path_vis.show()
        
        # # Generate cluster attribution visualization
        attribution_vis = self.visualize_cluster_attributions(
            layer_circuits, cluster_labels, prototype_idx
        )
        plt.show()
        
        print(f"Successfully disentangled prototype {prototype_idx} into {n_clusters} virtual neurons across multiple layers.")
        
        # Calculate statistics per cluster
        cluster_stats = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_size = len(cluster_indices)
            
            if cluster_indices.size > 0:
                cluster_activations = [top_activations[idx] for idx in cluster_indices if idx < len(top_activations)]
                mean_activation = np.mean(cluster_activations) if cluster_activations else 0
                
                # Get samples belonging to this cluster
                cluster_samples = top_samples[cluster_indices] if cluster_indices.size > 0 else None
                
                cluster_stats.append({
                    'virtual_neuron': f"{prototype_idx}.{i+1}",
                    'size': cluster_size,
                    'mean_activation': mean_activation,
                    'samples': cluster_samples
                })
        
        # Return the disentangled prototype information
        return {
            'prototype_idx': prototype_idx,
            'top_samples': top_samples,
            'cluster_labels': cluster_labels,
            'centroids_dict': centroids_dict,
            'layer_circuits': layer_circuits,
            'cluster_visualization': cluster_vis,
            'umap_visualization': umap_vis,
            'attribution_path_visualization': path_vis,
            'cluster_attribution_visualization': attribution_vis,
            'cluster_stats': cluster_stats,
            'target_layers': target_layers
        }
    
    def compute_saliency_maps(self, samples, prototype_idx):
        """
        Compute saliency maps for a prototype on given samples.
        
        Args:
            samples: Tensor of samples
            prototype_idx: Index of the prototype
            
        Returns:
            List of saliency maps
        """
        self.model.eval()
        saliency_maps = []
        
        for sample in samples:
            sample = sample.unsqueeze(0).to(self.device)
            sample.requires_grad_(True)
            
            # Forward pass
            _, pooled, _ = self.model(sample, inference=False)
            
            # Get activation for the target prototype
            prototype_activation = pooled[:, prototype_idx]
            
            # Compute gradients w.r.t input
            gradients = torch.autograd.grad(prototype_activation, sample)[0]
            
            # Create saliency map
            saliency = gradients.abs().sum(dim=1).squeeze().cpu()
            
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            saliency_maps.append(saliency)
            
        return saliency_maps
    
    def visualize_clusters(self, top_samples, cluster_labels, prototype_idx, crop_images=False, saliency_maps=None):
        """
        Visualize the clusters of a prototype.
        
        Args:
            top_samples: Tensor of top activating samples
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
            crop_images: Whether to crop images based on saliency
            saliency_maps: Optional saliency maps for cropping
            
        Returns:
            Matplotlib figure
        """
        n_clusters = len(np.unique(cluster_labels))
        
        # Create a figure
        fig, axs = plt.subplots(n_clusters + 1, 1, figsize=(12, 2.5*(n_clusters+1)))
        
        # Visualize all samples
        axs[0].set_title(f"All top activating samples for prototype {prototype_idx}")
        self._plot_samples(axs[0], top_samples[:10], saliency_maps)
        
        # Visualize samples for each cluster
        for i in range(n_clusters):
            cluster_samples = top_samples[cluster_labels == i]
            cluster_saliency = None
            if saliency_maps is not None:
                cluster_saliency = [saliency_maps[j] for j, label in enumerate(cluster_labels) if label == i]
            
            axs[i+1].set_title(f"Cluster {i+1} samples (virtual neuron {prototype_idx}.{i+1})")
            self._plot_samples(axs[i+1], cluster_samples[:10], cluster_saliency)
        
        plt.tight_layout()
        return fig
    
    def _plot_samples(self, ax, samples, saliency_maps=None, max_samples=25):
        """
        Helper function to plot samples in a grid.
        
        Args:
            ax: Matplotlib axis
            samples: Tensor of samples to plot
            saliency_maps: Optional saliency maps for highlighting
            max_samples: Maximum number of samples to plot
        """
        samples = samples[:max_samples]
        grid_size = len(samples)
        
        # Define normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for i, sample in enumerate(samples):
            # Convert tensor to image
            img = sample.cpu().clone()
            
            # Handle different batch formats
            if img.dim() == 4 and img.size(0) == 1:
                img = img.squeeze(0)
                
            img = img * std + mean  # Denormalize
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)  # Ensure values are in valid range
            
            # Apply saliency map if provided
            if saliency_maps is not None and i < len(saliency_maps):
                saliency = saliency_maps[i]
                if isinstance(saliency, torch.Tensor):
                    saliency = saliency.cpu().numpy()
                
                # Resize saliency if needed
                if saliency.shape[:2] != img.shape[:2]:
                    from skimage.transform import resize
                    saliency = resize(saliency, img.shape[:2])
                
                # Create a mask from saliency
                mask = saliency > 0.1  # Threshold for visualization
                
                # Create an overlay
                overlay = img.copy()
                overlay[~mask] = overlay[~mask] * 0.3  # Dim non-salient regions
            else:
                overlay = img
                
            ax.imshow(overlay, extent=(i, i+1, 0, 1))
        
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def visualize_attribution_paths(self, layer_circuits, cluster_labels, centroids_dict, prototype_idx, 
                                   top_n=5, figsize=(15, 10), layer_names=None):
        """
        Visualize attribution paths across multiple layers and their cluster assignments.
        
        Args:
            layer_circuits: Dictionary of circuit attributions per layer
            cluster_labels: Numpy array of cluster labels for each sample
            centroids_dict: Dictionary of cluster centroids per layer
            prototype_idx: Index of the prototype
            top_n: Number of top attribution channels to display per layer
            figsize: Size of the figure (width, height)
            layer_names: Optional dictionary mapping layer indices to human-readable names
            
        Returns:
            Matplotlib figure showing attribution paths across layers colored by cluster
        """
        # Get the number of layers and clusters
        n_layers = len(layer_circuits)
        n_clusters = len(np.unique(cluster_labels))
        
        # If layer names not provided, create default ones
        if layer_names is None:
            layer_names = {layer_idx: f"Layer {i+1}" for i, layer_idx in enumerate(sorted(layer_circuits.keys()))}
        
        # Create a color map for clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create a grid for layer nodes
        layer_positions = {}
        layer_indices = sorted(layer_circuits.keys())
        
        # Track the max number of channels in any layer (for scaling)
        max_channels = max([centroids_dict[layer_idx].shape[1] for layer_idx in layer_indices])
        
        # Create horizontal layout
        for i, layer_idx in enumerate(layer_indices):
            x_pos = i / (n_layers - 1 if n_layers > 1 else 1)  # Normalize to [0, 1]
            layer_positions[layer_idx] = (x_pos, 0.5)  # Center vertically
            
        # Find top attribution channels for each cluster and layer
        top_channels = {}
        for layer_idx in layer_indices:
            top_channels[layer_idx] = []
            for cluster_idx in range(n_clusters):
                # Get all samples in this cluster
                cluster_samples = np.where(cluster_labels == cluster_idx)[0]
                
                if len(cluster_samples) > 0:
                    # Get circuits for these samples
                    cluster_circuits = layer_circuits[layer_idx][cluster_samples]
                    
                    # Average attribution across samples
                    mean_attribution = cluster_circuits.mean(dim=0)
                    
                    # Find top channels
                    values, indices = torch.topk(mean_attribution.abs(), min(top_n, mean_attribution.shape[0]))
                    
                    # Store for this cluster
                    top_channels[layer_idx].append({
                        'cluster': cluster_idx,
                        'indices': indices.cpu().numpy(),
                        'values': values.cpu().numpy()
                    })
                else:
                    # Handle case where cluster has no samples
                    top_channels[layer_idx].append({
                        'cluster': cluster_idx,
                        'indices': [],
                        'values': []
                    })
        
        # Draw nodes for each layer
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Node radius as fraction of figure
        node_radius = 0.01
        channel_spacing = 0.03
        
        # Draw layer nodes with channel markers
        for layer_idx in layer_indices:
            x, y = layer_positions[layer_idx]
            
            # Draw layer name
            ax.text(x, 0.05, layer_names.get(layer_idx, f"Layer {layer_idx}"), 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Draw base circle for layer
            circle = plt.Circle((x, y), node_radius * 2, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            # Draw channel nodes for each cluster's top channels
            for cluster_data in top_channels[layer_idx]:
                cluster_idx = cluster_data['cluster']
                
                for i, (channel_idx, value) in enumerate(zip(cluster_data['indices'], cluster_data['values'])):
                    # Position channels in a vertical line, centered on layer position
                    # Scale by the value's magnitude
                    magnitude = float(value) / max([c['values'].max() if len(c['values']) > 0 else 1.0 
                                                  for c in top_channels[layer_idx]])
                    
                    # Offset channels vertically
                    offset = (i - len(cluster_data['indices'])/2) * channel_spacing
                    
                    # Draw channel node colored by cluster
                    channel_x = x
                    channel_y = y + offset
                    
                    circle = plt.Circle(
                        (channel_x, channel_y), 
                        node_radius * (0.5 + magnitude), 
                        fill=True, 
                        facecolor=cluster_colors[cluster_idx],
                        edgecolor='black',
                        linewidth=1,
                        alpha=0.7
                    )
                    ax.add_patch(circle)
                    
                    # Add channel index text
                    ax.text(channel_x, channel_y, str(channel_idx), 
                           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Connect channels between adjacent layers
        for i in range(len(layer_indices) - 1):
            curr_layer = layer_indices[i]
            next_layer = layer_indices[i + 1]
            
            curr_x, curr_y = layer_positions[curr_layer]
            next_x, next_y = layer_positions[next_layer]
            
            # For each cluster, connect its channels between layers
            for cluster_idx in range(n_clusters):
                curr_channels = top_channels[curr_layer][cluster_idx]['indices']
                next_channels = top_channels[next_layer][cluster_idx]['indices']
                
                if len(curr_channels) > 0 and len(next_channels) > 0:
                    for ci, curr_channel in enumerate(curr_channels):
                        for ni, next_channel in enumerate(next_channels):
                            # Calculate positions
                            curr_offset = (ci - len(curr_channels)/2) * channel_spacing
                            next_offset = (ni - len(next_channels)/2) * channel_spacing
                            
                            # Draw connection line
                            ax.plot(
                                [curr_x, next_x], 
                                [curr_y + curr_offset, next_y + next_offset], 
                                '-', 
                                color=cluster_colors[cluster_idx],
                                alpha=0.5,
                                linewidth=1
                            )
        
        # Create legend for clusters
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     label=f'Cluster {i+1}',
                                     markerfacecolor=cluster_colors[i], 
                                     markersize=10) 
                          for i in range(n_clusters)]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and clean up axes
        ax.set_title(f'Attribution Paths Across Layers for Prototype {prototype_idx}', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def visualize_umap_embedding(self, features, cluster_labels, prototype_idx):
        """
        Visualize UMAP embedding of features colored by cluster.
        
        Args:
            features: Feature vectors to embed with UMAP
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
            
        Returns:
            Matplotlib figure
        """
        # Apply UMAP for dimensionality reduction
        umap_reducer = UMAP(n_components=2, random_state=42)
        embedding = umap_reducer.fit_transform(features)
        
        # Plot embedding
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            plt.scatter(
                embedding[cluster_labels == cluster, 0],
                embedding[cluster_labels == cluster, 1],
                c=[colors[i]],
                label=f'Cluster {cluster + 1}',
                alpha=0.7,
                s=100
            )
        
        plt.title(f"UMAP embedding of multi-layer circuit attributions for prototype {prototype_idx}")
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        return plt.gcf()
    
    def determine_optimal_clusters_multi_layer(self, layer_circuits, max_clusters=5, 
                                              layer_weights=None, random_state=42):
        """
        Determine the optimal number of clusters for prototype disentanglement
        using features from multiple layers.
        
        Args:
            layer_circuits: Dictionary of circuit attributions per layer
            max_clusters: Maximum number of clusters to consider
            layer_weights: Optional dictionary of weights for each layer
            random_state: Random seed for reproducibility
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.cluster import KMeans
        
        if layer_weights is None:
            # Default to equal weighting
            layer_weights = {layer_idx: 1.0 for layer_idx in layer_circuits.keys()}
        
        # Normalize weights
        total_weight = sum(layer_weights.values())
        normalized_weights = {k: v/total_weight for k, v in layer_weights.items()}
        
        # Prepare combined features for clustering
        combined_features = []
        
        # Get number of samples
        num_samples = next(iter(layer_circuits.values())).shape[0]
        
        for sample_idx in range(num_samples):
            # Flatten and concatenate features from all layers for this sample
            sample_features = []
            
            for layer_idx, circuits in layer_circuits.items():
                if sample_idx < circuits.shape[0]:  # Check if we have this sample for this layer
                    layer_features = circuits[sample_idx]
                    
                    # Flatten the features
                    flat_features = layer_features.reshape(-1).numpy()
                    
                    # Apply layer weight
                    if layer_idx in normalized_weights:
                        flat_features = flat_features * normalized_weights[layer_idx]
                    
                    sample_features.append(flat_features)
            
            # Combine all layer features for this sample
            if sample_features:
                combined_features.append(np.concatenate(sample_features))
        
        # Convert to numpy array for clustering
        combined_features = np.array(combined_features)
        
        # Must have at least max_clusters+1 samples
        n_samples = combined_features.shape[0]
        max_possible = min(max_clusters, n_samples - 1)
        
        if max_possible <= 1:
            return 1
        
        # Store scores for different numbers of clusters
        silhouette_scores = []
        ch_scores = []
        
        # Evaluate different numbers of clusters
        for n_clusters in range(1, max_possible + 1):
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(combined_features)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters exist
                s_score = silhouette_score(combined_features, cluster_labels)
                silhouette_scores.append(s_score)
                
                # Calculate Calinski-Harabasz score (variance ratio)
                ch_score = calinski_harabasz_score(combined_features, cluster_labels)
                ch_scores.append(ch_score)
            else:
                silhouette_scores.append(-1)
                ch_scores.append(0)
        
        # Normalize scores
        if silhouette_scores:
            norm_silhouette = np.array(silhouette_scores) / max(max(silhouette_scores), 1e-10)
            norm_ch = np.array(ch_scores) / max(max(ch_scores), 1e-10)
            
            # Combined score (weighted average)
            combined_scores = 0.1 * norm_silhouette + 0.9 * norm_ch
            combined_scores[0] = norm_ch[0]
            print(f"Combined cluster scores: {combined_scores}")
            
            # Get optimal number of clusters (add 1 because we started from 1)
            optimal_clusters = np.argmax(combined_scores) + 1
        else:
            optimal_clusters = 2  # Default if we couldn't compute scores
        
        return optimal_clusters

# Example usage function
def disentangle_multi_layer_prototypes(model, dataloader, target_layers=None, 
                                       device='cuda', prototypes=None, 
                                       num_ref_samples=100, layer_weights=None):
    """
    Identify and disentangle potentially polysemantic prototypes using multi-layer attribution.
    
    Args:
        model: PIPNet model
        dataloader: DataLoader containing dataset
        target_layers: List of layer indices to compute attributions for
        device: Device to run on
        prototypes: List of prototype indices to analyze
        num_ref_samples: Number of reference samples to use
        layer_weights: Optional dictionary of weights for each layer
        
    Returns:
        Dictionary of disentangled prototype information
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize MultiLayerPURE
    pure = MultiLayerPURE(model, device=device, num_ref_samples=num_ref_samples)
    # Find all available layers and print them in a readable format
    # pure.print_available_layers()

    # Filter to see only convolutional layers
    # conv_layers = pure.print_available_layers(filter_str='conv')
    
    target_layers = ['module._net.features.7.2', 'module._net.features.4.0','module._net.features.2.0']
    # layer_weights = 
    # Get total number of prototypes
    num_prototypes = model.module._num_prototypes if hasattr(model, 'module') else model._num_prototypes
    print(f"Analyzing {len(prototypes) if prototypes else num_prototypes} prototypes...")
    
    # Use specified prototypes or select some by default
    if not prototypes:
        # For demonstration, take a few prototypes evenly spaced
        prototypes = list(range(0, num_prototypes, num_prototypes//10))[:5]
    
    results = {}
    for proto_idx in prototypes:
        # Disentangle prototype using multi-layer attribution
        if isinstance(proto_idx, List) and len(proto_idx) >= 2 and all(isinstance(p, int) for p in proto_idx):
            print(f"Disentangling group of prototypes: {proto_idx}")
            result = pure.disentangle_multi_layer_prototype(
                dataloader, proto_idx, target_layers=target_layers, 
                n_clusters=3, adaptive=True, layer_weights=layer_weights
            )
            
            # Print statistics
            for cluster in result['cluster_stats']:
                print(f"Virtual neuron {cluster['virtual_neuron']} has {cluster['size']} samples "
                      f"with mean activation {cluster['mean_activation']}")
            
            # Store results for each prototype in the group
            for proto in proto_idx:
                results[proto] = result
        else:
            # Single prototype
            print(f"Disentangling prototype: {proto_idx}")
            result = pure.disentangle_multi_layer_prototype(
                dataloader, proto_idx, target_layers=target_layers, 
                n_clusters=3, adaptive=True, layer_weights=layer_weights
            )
            
            # Print statistics
            for cluster in result['cluster_stats']:
                print(f"Virtual neuron {cluster['virtual_neuron']} has {cluster['size']} samples "
                      f"with mean activation {cluster['mean_activation']}")
            
            results[proto_idx] = result
    
    return results