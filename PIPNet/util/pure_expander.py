import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from copy import deepcopy
import copy
import matplotlib.pyplot as plt

from pure import PURE


def expand_add_on_layer(original_add_on, pure_results_dict, original_num_prototypes, num_new_prototypes):
    """
    Directly uses circuit centroids as weights for new prototypes.
    
    Args:
        original_add_on: Original add-on layer
        pure_results_dict: PURE results dictionary
        num_new_prototypes: Total number of new prototypes
        
    Returns:
        New add-on layer with centroid-based weights
    """
    # Find the convolutional layer
    conv_layer = None
    for layer in reversed(original_add_on):
        if isinstance(layer, nn.Conv2d):
            conv_layer = layer
            break
            
    if conv_layer is None:
        raise ValueError("Could not find convolutional layer in add-on module")
        
    # Get original parameters
    original_weights = conv_layer.weight.data
    original_bias = conv_layer.bias.data if conv_layer.bias is not None else None
    
    # Create new weight tensor with expanded size
    in_channels = original_weights.shape[1]
    kernel_size = original_weights.shape[2:]
    new_weights = torch.zeros((num_new_prototypes, in_channels, *kernel_size), 
                            device=original_weights.device)
    
    if original_bias is not None:
        new_bias = torch.zeros(num_new_prototypes, device=original_bias.device)
    else:
        new_bias = None
        
    # Copy weights for non-disentangled prototypes
    next_idx = original_num_prototypes
    for proto_idx in range(original_num_prototypes):
        if proto_idx in pure_results_dict:
            # Get PURE results for this prototype
            result = pure_results_dict[proto_idx]
            unique_clusters = np.unique(result['cluster_labels'])
            
            # Get centroids representing each cluster
            centroids = result['centroids']
            
            # Process each cluster
            for i, cluster_idx in enumerate(unique_clusters):
                # Determine target index for this cluster
                target_idx = proto_idx if i == 0 else next_idx
                
                if i == 0:
                    # For the first cluster, keep original weights
                    new_weights[target_idx] = original_weights[proto_idx]
                else:
                    # For additional clusters, use centroid to create new weights
                    centroid = centroids[i]
                    
                    # Use centroid to directly shape the new weights
                    # Convert centroid to tensor if it's not already
                    if not isinstance(centroid, torch.Tensor):
                        centroid = torch.tensor(centroid, device=original_weights.device)
                    
                    # Reshape centroid to match in_channels dimension
                    # (this assumes the centroid captures feature importance)
                    if centroid.ndim == 1 and centroid.shape[0] == in_channels:
                        # Create weights that reflect the centroid's distribution
                        weight_template = original_weights[proto_idx].clone()
                        
                        # Normalize centroid for better numerical stability
                        centroid_norm = torch.abs(centroid)
                        if torch.sum(centroid_norm) > 0:
                            centroid_norm = centroid_norm / torch.sum(centroid_norm)
                        
                        # Use centroid to shape the weights across channels
                        for c in range(in_channels):
                            # Scale each input channel by its importance in the centroid
                            new_weights[target_idx, c] = weight_template[c] * centroid_norm[c]
                    else:
                        # If centroid shape doesn't match, fall back to using original weights
                        new_weights[target_idx] = original_weights[proto_idx]
                    
                    # Move to next new index
                    next_idx += 1
                    
                # Set bias (if present)
                if original_bias is not None:
                    new_bias[target_idx] = original_bias[proto_idx]
        else:
            # For non-disentangled prototypes, copy original weights
            new_weights[proto_idx] = original_weights[proto_idx]
            if original_bias is not None:
                new_bias[proto_idx] = original_bias[proto_idx]
    
    # Create new convolutional layer
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=num_new_prototypes,
        kernel_size=kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    
    # Set weights and bias
    new_conv.weight.data = new_weights
    if new_bias is not None:
        new_conv.bias.data = new_bias
        
    # Create new add-on layer
    new_add_on = nn.Sequential()
    for layer in original_add_on:
        if layer is conv_layer:
            new_add_on.append(new_conv)
        else:
            new_add_on.append(layer)
            
    return new_add_on

def expand_pipnet_with_pure_centroids(model, pure_results_dict, device='cuda', scaling=0.5):
    """
    Expand a PIPNet model by adding new prototypes based on PURE centroids.
    
    Args:
        model: Original PIPNet model
        pure_results_dict: Dictionary mapping prototype indices to PURE results
        device: Device to run computations on
        
    Returns:
        Expanded PIPNet model and mapping from original to new prototype indices
    """
    # Create a deep copy of the model to avoid modifying the original
    expanded_model = deepcopy(model)
    
    # Extract model parameters
    original_num_prototypes = model._num_prototypes
    num_classes = model._num_classes
    
    # Count new prototypes needed
    total_new_clusters = 0
    for proto_idx, result in pure_results_dict.items():
        n_clusters = len(np.unique(result['cluster_labels']))
        total_new_clusters += (n_clusters - 1)  # Subtract 1 as the first cluster uses original prototype
    
    num_new_prototypes = original_num_prototypes + total_new_clusters
    print(f"Expanding from {original_num_prototypes} to {num_new_prototypes} prototypes")
    
    # Create prototype mapping
    prototype_mapping = {}
    next_idx = original_num_prototypes
    
    for proto_idx in range(original_num_prototypes):
        if proto_idx in pure_results_dict:
            result = pure_results_dict[proto_idx]
            n_clusters = len(np.unique(result['cluster_labels']))
            new_indices = [proto_idx]  # First cluster keeps original index
            
            for _ in range(n_clusters - 1):
                new_indices.append(next_idx)
                next_idx += 1
                
            prototype_mapping[proto_idx] = new_indices
        else:
            prototype_mapping[proto_idx] = [proto_idx]
    
    # Extract add-on layer parameters
    add_on = model._add_on
    
    # Find the prototype layer (last convolutional layer in add-on)
    proto_layer = None
    for layer in list(add_on.children())[::-1]:
        if isinstance(layer, nn.Conv2d):
            proto_layer = layer
            break
    
    if proto_layer is None:
        raise ValueError("Could not find convolutional layer in add-on module")
    
    # Get key parameters from the prototype layer
    in_channels = proto_layer.in_channels
    kernel_size = proto_layer.kernel_size
    stride = proto_layer.stride
    padding = proto_layer.padding
    
    # Create new add-on layer with expanded output channels
    new_proto_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=num_new_prototypes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=proto_layer.bias is not None
    )
    
    # Copy original weights for existing prototypes
    with torch.no_grad():
        # Initialize weights from the original prototype layer
        new_proto_layer.weight.data[:original_num_prototypes] = proto_layer.weight.data
        
        if proto_layer.bias is not None:
            new_proto_layer.bias.data[:original_num_prototypes] = proto_layer.bias.data
        
        # Initialize weights for the new prototypes using centroids
        next_idx = original_num_prototypes
        
        for proto_idx, result in pure_results_dict.items():
            cluster_labels = result['cluster_labels']
            centroids = result['centroids']
            unique_clusters = np.unique(cluster_labels)

            # Get the centroid for the first cluster (original prototype)
            centroid = centroids[0][0]
            
            # Normalize the centroid
            # centroid_norm = centroid.abs()
            # centroid_norm = centroid_norm / centroid_norm.abs().sum()
            centroid_norm = centroid / torch.norm(centroid)
            # if centroid_norm.sum() > 0:
            #     centroid_norm = centroid_norm / centroid_norm.sum()
            
            # Apply channel-wise scaling based on the centroid
            # for c in range(in_channels):
            #     scale_factor = 1.0 + scaling * centroid_norm[c]  # Enhance important channels
            #     new_proto_layer.weight.data[proto_idx, c] *= scale_factor

            print(centroid_norm.size(), new_proto_layer.weight.data.size())
            for c in range(in_channels):
                new_proto_layer.weight.data[proto_idx,c] += scaling * centroid_norm[c] * new_proto_layer.weight.data[proto_idx,c]
            
            # Copy bias if present
            if proto_layer.bias is not None:
                new_proto_layer.bias.data[proto_idx] = proto_layer.bias.data[proto_idx]
            
            print(len(centroids), unique_clusters) 
            # Skip the first cluster (using original prototype)
            for i, cluster_idx in enumerate(unique_clusters[1:], 1):
                print(f"Applying centroid-based adaptation for prototype {proto_idx} {i}")
                # Get the centroid for this cluster
                centroid = centroids[i][0]
                
                # Convert to tensor if needed
                if not isinstance(centroid, torch.Tensor):
                    centroid = torch.tensor(centroid, device=device)
                
                # Initialize the new prototype weights based on the original
                new_proto_layer.weight.data[next_idx] = proto_layer.weight.data[proto_idx].clone()
                
                # Apply centroid-based adaptation
                # The centroid is a 1D vector of channel importance
                print(f"Applying centroid-based adaptation for prototype {proto_idx}")
                # Normalize the centroid
                centroid_norm = centroid / torch.norm(centroid)
                # centroid_norm = centroid.abs()
                # if centroid_norm.sum() > 0:
                #     centroid_norm = centroid_norm / centroid_norm.sum()
                
                # Apply channel-wise scaling based on the centroid
                # for c in range(in_channels):
                #     scale_factor = 1.0 + scaling * centroid_norm[c]  # Enhance important channels
                #     new_proto_layer.weight.data[next_idx, c] *= scale_factor
                for c in range(in_channels):
                    new_proto_layer.weight.data[next_idx,c] += scaling * centroid_norm[c] * new_proto_layer.weight.data[proto_idx,c]
                
                # Copy bias if present
                if proto_layer.bias is not None:
                    new_proto_layer.bias.data[next_idx] = proto_layer.bias.data[proto_idx]
                
                next_idx += 1
    
    # Replace the prototype layer in the add-on module
    new_add_on = nn.Sequential()
    replaced = False
    
    for layer in add_on.children():
        if layer is proto_layer:
            new_add_on.append(new_proto_layer)
            replaced = True
        else:
            new_add_on.append(layer)
    
    if not replaced:
        raise ValueError("Failed to replace the prototype layer")
    
    # Create new classification layer with expanded input size
    new_classification = nn.Linear(
        in_features=num_new_prototypes,
        out_features=num_classes,
        bias=model._classification.bias is not None
    )
    
    # Initialize classification weights
    with torch.no_grad():
        # For each original prototype
        for proto_idx, new_indices in prototype_mapping.items():
            # Copy weights to all new indices
            for new_idx in new_indices:
                new_classification.weight.data[:, new_idx] = model._classification.weight.data[:, proto_idx]
        
        # Copy bias if present
        if model._classification.bias is not None:
            new_classification.bias.data = model._classification.bias.data.clone()
        
        # If the original layer has a normalization multiplier attribute, preserve it
        if hasattr(model._classification, 'normalization_multiplier'):
            new_classification.normalization_multiplier = model._classification.normalization_multiplier
    
    # Update the model with the new layers
    expanded_model._add_on = new_add_on
    expanded_model._classification = new_classification
    expanded_model._num_prototypes = num_new_prototypes
    
    # Store the prototype mapping
    expanded_model._prototype_mapping = prototype_mapping
    
    return expanded_model, prototype_mapping


def expand_pipnet_with_pure_centroids_trajectory(original_model, results, device='cuda', scalings=[0.5, 1.0, 2.0], dataloader=None):
    """
    Create multiple expanded models with different scaling values to create a trajectory
    of prototype disentanglement.
    
    Args:
        original_model: The original PIPNet model
        results: Results from PURE disentanglement
        device: Device to run on
        scalings: List of scaling values to create trajectory
        
    Returns:
        dict: Dictionary mapping scaling values to (model, prototype_mapping) tuples
    """
    trajectory = {}
    
    for scaling in scalings:
        # Clone the original model for this scaling value
        model_copy = copy.deepcopy(original_model).to(device)
        
        # Original feature dimension
        original_features = model_copy._add_on[0].weight.shape[0]
        
        # Count new clusters to add
        total_new_clusters = 0
        for proto_idx, result in results.items():
            n_clusters = len(np.unique(result['cluster_labels']))
            if isinstance(proto_idx, list):
                total_new_clusters += n_clusters - len(proto_idx)
            else:
                total_new_clusters += (n_clusters - 1)  # First cluster uses original prototype
        
        # Create new model with expanded feature dimension
        new_feature_dim = original_features + total_new_clusters
        
        # Prepare expanded prototype layer
        new_add_on_layer = nn.Conv2d(
            model_copy._add_on[0].in_channels,
            new_feature_dim,
            kernel_size=1,
            bias=False
        )
        
        # Initialize with original weights
        with torch.no_grad():
            new_add_on_layer.weight[:original_features] = model_copy._add_on[0].weight
        
        # Replace the layer in the model
        model_copy._add_on[0] = new_add_on_layer
        model_copy._num_prototypes = new_feature_dim
        
        # Create new classification layer
        num_classes = model_copy._classification.weight.shape[0]
        new_classification = nn.Linear(new_feature_dim, num_classes, bias=model_copy._classification.bias is not None)
        
        # Copy original weights
        with torch.no_grad():
            new_classification.weight[:, :original_features] = model_copy._classification.weight
            if model_copy._classification.bias is not None:
                new_classification.bias = model_copy._classification.bias
        
        # Replace the classification layer
        model_copy._classification = new_classification
        
        # Keep track of the prototype mapping
        prototype_mapping = {}
        next_proto_idx = original_features
        model_copy= model_copy.to(device)
        
        # Add new prototypes based on PURE centroids
        for proto_idx, result in results.items():
            centroids = result['centroids']
            n_clusters = len(np.unique(result['cluster_labels']))
            
            # Handle both single and multiple prototype cases
            proto_indices = [proto_idx] if not isinstance(proto_idx, list) else proto_idx
            
            # Track which original prototype maps to which new prototypes
            for orig_idx in proto_indices:
                prototype_mapping[orig_idx] = []
            
            # For each cluster, add a new prototype
            for cluster_idx in range(n_clusters):

                centroid = centroids[cluster_idx] / torch.norm(centroids[cluster_idx] + 1e-6)
                centroid = project_centroids_to_manifold(original_model, dataloader, proto_idx, centroid, device)
                
                if cluster_idx == 0 and not isinstance(proto_idx, list):
                    # First cluster maps to original prototype for single prototype case
                    with torch.no_grad():
                        # Update original prototype weights with centroid (with scaling)
                        centroid_tensor = torch.tensor(centroid, device=device).reshape(
                            1, model_copy._add_on[0].in_channels, 1, 1
                        )

                        # Scale the centroid influence by the current scaling factor
                        diff = centroid_tensor - model_copy._add_on[0].weight[proto_idx:proto_idx+1]
                        model_copy._add_on[0].weight[proto_idx:proto_idx+1] += scaling * diff
                    
                    # Add to mapping
                    for orig_idx in proto_indices:
                        prototype_mapping[orig_idx].append(orig_idx)
                else:
                    # Add a new prototype
                    with torch.no_grad():
                        # Convert centroid to tensor
                        model_copy._add_on[0].weight[next_proto_idx:next_proto_idx+1] = model_copy._add_on[0].weight[proto_idx:proto_idx+1]
                        centroid_tensor = torch.tensor(centroid, device=device).reshape(
                            1, model_copy._add_on[0].in_channels, 1, 1
                        ).cuda()

                        diff = centroid_tensor - model_copy._add_on[0].weight[proto_idx:proto_idx+1]
                        model_copy._add_on[0].weight[next_proto_idx:next_proto_idx+1] += scaling * diff
                       
                        # Copy weights from classification layer with scaling
                        if isinstance(proto_idx, list):
                            # Average weights from all prototypes in the list
                            avg_weight = torch.zeros(num_classes, 1, device=device)
                            for idx in proto_idx:
                                avg_weight += model_copy._classification.weight[:, idx:idx+1]
                            avg_weight /= len(proto_idx)
                            
                            # Apply scaling factor
                            model_copy._classification.weight[:, next_proto_idx] = scaling * avg_weight.squeeze()
                        else:
                            # Scale from the original prototype
                            model_copy._classification.weight[:, next_proto_idx] = scaling * model_copy._classification.weight[:, proto_idx]
                    
                    # Add to mapping
                    for orig_idx in proto_indices:
                        prototype_mapping[orig_idx].append(next_proto_idx)
                    
                    # Increment prototype index
                    next_proto_idx += 1
        
        # Store the expanded model and its mapping
        trajectory[scaling] = (model_copy, prototype_mapping)
    
    return trajectory

def project_centroids_to_manifold(model, dataloader, prototype_idx, centroids, device='cuda'):
    """
    Project PURE centroids onto local PCA basis to ensure they stay on the data manifold.
    
    Args:
        model: The PIPNet model
        dataloader: DataLoader for accessing features
        prototype_idx: Index of the prototype being refined
        centroids: The centroids from PURE in attribution space
        device: Computing device
        
    Returns:
        Projected centroids that lie on the manifold
    """
    # Step 1: Get feature vectors from highly activating samples
    with torch.no_grad():
        # Find highly activating samples for this prototype
        pure_analyzer = PURE(model, device=device)
        top_samples, _ = pure_analyzer.find_top_activating_samples(dataloader, prototype_idx)
        
        # Extract feature vectors
        feature_vectors = []
        for sample in top_samples:
            sample = sample.unsqueeze(0).to(device)
            # Forward pass to get features
            _, _, _, features = model(sample, features_save=True)
            
            # Flatten features to get vectors
            b, c, h, w = features.shape
            flat_features = features.reshape(b, c, h*w).permute(0, 2, 1)  # [B, HW, C]
            
            # Add all feature vectors (each spatial location)
            for i in range(flat_features.shape[1]):
                feature_vectors.append(flat_features[0, i].cpu().numpy())
    
    # Step 2: Compute PCA basis from feature vectors
    from sklearn.decomposition import PCA
    
    # Center the feature vectors
    feature_vectors = np.array(feature_vectors)
    feature_mean = np.mean(feature_vectors, axis=0)
    centered_features = feature_vectors - feature_mean
    
    # Compute PCA to get principal components
    n_components = min(10, centered_features.shape[0]-1, centered_features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(centered_features)
    
    # Get principal components (basis vectors)
    principal_components = pca.components_  # Shape [n_components, feature_dim]
    
    # Step 3: Project centroids onto the PCA basis
    projected_centroids = []
    for centroid in centroids:
        # Reshape centroid to match feature vectors
        centroid_flat = centroid.reshape(-1)
        
        # Project onto PCA space (get coordinates in the new basis)
        pca_coords = pca.transform([centroid_flat])[0]
        
        # Project back to original space using only the PCA basis
        projected_centroid = pca.inverse_transform([pca_coords])[0]
        
        # Reshape back to original centroid shape
        projected_centroid = projected_centroid.reshape(centroid.shape)
        
        projected_centroids.append(projected_centroid)
    
    return projected_centroids

def expand_pipnet_with_pure(model, pure_results_dict, method='circuit', device='cuda'):
    """
    Expand a PIPNet model by replacing polysemantic prototypes with their
    disentangled versions discovered through PURE.
    
    Args:
        model: Original PIPNet model
        pure_results_dict: Dictionary mapping prototype indices to PURE results
        method: Expansion method ('basic', 'refined', or 'circuit')
        device: Device to run computations on
        
    Returns:
        Tuple of (expanded model, prototype mapping)
    """
    # Choose the appropriate expander based on the method
    num_new = 0
    for result in pure_results_dict.values():
        num_new += len(np.unique(result['cluster_labels']))
    # Expand the model
    num_old_prototypes = model.module._add_on[0].out_channels
    new_addon = expand_add_on_layer(model.module._add_on, pure_results_dict, num_old_prototypes,
                                    num_old_prototypes+ num_new)
    
    model.module._add_on = new_addon
    model.module._num_prototypes = num_old_prototypes + num_new
    return model