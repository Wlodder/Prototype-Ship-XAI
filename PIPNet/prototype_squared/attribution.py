import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class ForwardPURE:
    """
    Implementation of PURE using autograd during the forward pass.
    Uses the exact gradient × input calculation during inference.
    """
    def __init__(self, pipnet_model, device='cuda'):
        """
        Initialize with a trained PIPNet model and pre-computed centroids.
        
        Args:
            pipnet_model: Trained PIPNet model
            centroids_by_layer: Dictionary mapping layer indices to cluster centroids
                                computed by the original PURE method
            device: Computing device
        """
        self.model = pipnet_model
        self.device = device
        self.centroids_by_layer = {}
        
        # Extract classification weights for decision making
        self.classification_weights = pipnet_model.module._classification.weight.data
        
        # Track layers for attribution
        self.tracked_layers = {}
        self._register_hooks()

    def add_centroids(self, split_results):
        """
        Add pre-computed centroids for each layer to the model.
        
        Args:
            split_results: Dictionary mapping layer indices to cluster centroids
        """

        for proto_batch, _ in split_results.items(): 
            for layer, centroids in split_results[proto_batch]['centroids'].items():

                if layer not in self.centroids_by_layer:
                    self.centroids_by_layer[layer] = []

                self.centroids_by_layer[layer].append(centroids)
    
    def _register_hooks(self):
        """Register hooks to capture intermediate feature maps during forward pass."""
        # Clear existing hooks and tracked layers
        if hasattr(self, 'hooks'):
            for hook in self.hooks:
                hook.remove()
        self.hooks = []
        self.tracked_layers = {}
        
        # Helper function to create a hook for a specific layer
        def get_features(name):
            def hook(module, input, output):
                # Clone and require grad for backprop
                retained_output = output.clone().detach().requires_grad_(True)
                self.tracked_layers[name] = retained_output
                return retained_output
            return hook
        
        # Register hooks for the desired layers based on centroids_by_layer
        feature_net = self.model.module._net
        
        # Handle different network architectures
        if hasattr(feature_net, 'layer1'):
            # ResNet-like architecture
            for i in range(1, 7):  # Typically 4 layer groups
                if hasattr(feature_net, f'layer{i}'):
                    layer = getattr(feature_net, f'layer{i}')
                    hook = layer.register_forward_hook(get_features(f'layer{i}'))
                    self.hooks.append(hook)
        else:
            # General case - register hooks for various depths
            # This is just an example structure - adapt to your model
            idx = 0
            for name, module in feature_net.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)) and idx < 5:
                    hook = module.register_forward_hook(get_features(name))
                    self.hooks.append(hook)
                    idx += 1
    
    def compute_attributions(self, x, prototype_idx, require_intermediate=True):
        """
        Compute exact gradient × input attributions during inference.
        
        Args:
            x: Input tensor
            prototype_idx: Index of prototype to analyze
            require_intermediate: Whether to compute attributions for intermediate layers
            
        Returns:
            Dictionary of attributions for various layers
        """
        # Clear tracked layers from previous run
        self.tracked_layers = {}
        
        # Enable gradients for the input
        x.requires_grad_(True)
        
        # Forward pass through the network
        features = self.model.module._net(x)
        proto_features = self.model.module._add_on(features)
        pooled = self.model.module._pool(proto_features)
        
        # Get activation for the target prototype
        target_activation = pooled[:, prototype_idx]
        
        # Dictionary to store attributions
        attributions = {}
        
        # Compute gradient × input for the input
        input_grad = torch.autograd.grad(target_activation, x, 
                                        create_graph=False, 
                                        retain_graph=require_intermediate, allow_unused=True)[0]
        attributions['input'] = input_grad * x
        
        # Compute attributions for intermediate layers if requested
        if require_intermediate:
            for name, layer_output in self.tracked_layers.items():
                if layer_output.requires_grad:
                    layer_grad = torch.autograd.grad(target_activation, layer_output,
                                                  create_graph=False,
                                                  retain_graph=True)[0]
                    attributions[name] = layer_grad * layer_output
        
        # Add final feature layer attribution
        feature_grad = torch.autograd.grad(target_activation, features,
                                        create_graph=False,
                                        retain_graph=False)[0]
        attributions['final_features'] = feature_grad * features
        
        return attributions, target_activation
    
    def compute_attribution_distances(self, attributions, layer_mapping=None):
        """
        Compute distances between attributions and stored centroids.
        
        Args:
            attributions: Dictionary of attributions from compute_attributions
            layer_mapping: Optional mapping from attribution keys to centroid layer indices
            
        Returns:
            Dictionary of distances to centroids for each layer
        """
        # Default mapping from attribution layer names to centroid indices
        if layer_mapping is None:
            # This is an example mapping - adjust to match your model's layers and centroids
            layer_mapping = {
                'input': 0,           # Input layer to centroid layer 0
                'layer1': 1,          # layer1 to centroid layer 1
                'layer2': 2,          # etc.
                'layer3': 3,
                'layer4': 4,
                'final_features': -1  # Final features to centroid layer -1
            }
        
        # Dictionary to store distances for each layer
        distances = {}
        
        # Calculate distances for each layer
        for attr_name, attr_tensor in attributions.items():
            # Skip if this layer isn't in the mapping
            if attr_name not in layer_mapping:
                continue
                
            # Get corresponding centroid layer index
            centroid_layer_idx = layer_mapping[attr_name]
            
            # Skip if we don't have centroids for this layer
            if centroid_layer_idx not in self.centroids_by_layer:
                continue
                
            # Get centroids for this layer
            layer_centroids = self.centroids_by_layer[centroid_layer_idx]
            
            # Flatten attribution for distance calculation
            flat_attr = attr_tensor.flatten(start_dim=1)
            
            # Compute distances to all centroids
            layer_distances = []
            for centroid in layer_centroids:
                # Convert centroid to tensor if needed
                if not isinstance(centroid, torch.Tensor):
                    centroid = torch.tensor(centroid, device=self.device)
                
                # Make sure dimensions match for comparison
                if flat_attr.shape[1] != centroid.shape[0]:
                    # Handle dimension mismatch (e.g., by truncating or padding)
                    min_dim = min(flat_attr.shape[1], centroid.shape[0])
                    distance = torch.norm(flat_attr[:, :min_dim] - centroid[:min_dim].view(1, -1), dim=1)
                else:
                    # Standard distance calculation
                    distance = torch.norm(flat_attr - centroid.view(1, -1), dim=1)
                
                layer_distances.append(distance)
            
            # Stack distances to all centroids
            if layer_distances:
                distances[centroid_layer_idx] = torch.stack(layer_distances, dim=1)
        
        return distances
    
    def enhanced_classification(self, x, prototype_filter=None, distance_threshold=1.0):
        """
        Perform classification enhanced by centroid-based attribution matching.
        
        Args:
            x: Input tensor
            prototype_filter: Optional function to filter which prototypes to compute attributions for
            distance_threshold: Maximum distance to consider a valid centroid match
            
        Returns:
            Dictionary with enhanced classification results
        """
        start_time = time.time()
        
        # Regular forward pass to get initial predictions
        with torch.no_grad():
            proto_features, pooled, logits = self.model(x)
            initial_preds = torch.argmax(logits, dim=1)
        
        # Enhanced results container
        enhanced_results = {
            'initial_preds': initial_preds,
            'enhanced_confidence': torch.zeros_like(pooled),
            'centroid_matches': [[] for _ in range(x.shape[0])]
        }
        
        # Process each sample
        for i in range(x.shape[0]):
            sample_input = x[i:i+1].clone()
            
            # Determine which prototypes to analyze
            prototype_indices = []
            
            if prototype_filter is None:
                # Default: use prototypes with significant activation and weight
                active_mask = pooled[i] > 0.5  # Activation threshold
                
                # Get prototypes important for the predicted class
                pred_class = initial_preds[i].item()
                # important_protos = self.classification_weights[pred_class] > 0.1
                
                # Combine activation and importance
                significant_protos = active_mask #& important_protos
                prototype_indices = torch.nonzero(significant_protos, as_tuple=True)[0].tolist()
                
                # Limit to top 5 most activated for efficiency
                if len(prototype_indices) > 20:
                    proto_activations = pooled[i, prototype_indices]
                    _, top_indices = torch.topk(proto_activations, 20)
                    prototype_indices = [prototype_indices[j] for j in top_indices.tolist()]
            else:
                # Use custom filter function
                prototype_indices = prototype_filter(pooled[i], initial_preds[i])
            
            # Process each prototype for this sample
            for proto_idx in prototype_indices:
                # Compute exact attributions using autograd
                attributions, activation = self.compute_attributions(sample_input, proto_idx)
                
                # Compute distances to centroids
                distances = self.compute_attribution_distances(attributions)
                
                # Find best match across all layers
                best_matches = {}
                for layer_idx, layer_distances in distances.items():
                    # Get the closest centroid index and distance
                    min_idx = torch.argmin(layer_distances, dim=1).item()
                    min_distance = layer_distances[0, min_idx].item()
                    
                    best_matches[layer_idx] = {
                        'centroid_idx': min_idx,
                        'distance': min_distance,
                        'is_match': min_distance < distance_threshold
                    }
                
                # Store results for this prototype
                enhanced_results['centroid_matches'][i].append({
                    'prototype_idx': proto_idx,
                    'activation': activation.item(),
                    'centroid_matches': best_matches
                })
                
                # Adjust confidence based on match quality
                match_count = sum(1 for match in best_matches.values() if match['is_match'])
                if len(best_matches) > 0:
                    match_ratio = match_count / len(best_matches)
                    
                    # Boost activation by match quality
                    enhanced_results['enhanced_confidence'][i, proto_idx] = activation * (1.0 + match_ratio)
                else:
                    # No matches - just use the original activation
                    enhanced_results['enhanced_confidence'][i, proto_idx] = activation
        
        # Compute enhanced predictions using adjusted confidence
        enhanced_logits = torch.zeros_like(logits)
        
        for i in range(x.shape[0]):
            for class_idx in range(logits.shape[1]):
                # Apply classification weights to enhanced confidences
                enhanced_logits[i, class_idx] = torch.sum(
                    enhanced_results['enhanced_confidence'][i] * self.classification_weights[class_idx]
                )
        
        # Final predictions
        enhanced_preds = torch.argmax(enhanced_logits, dim=1)
        
        # Add final results
        enhanced_results['enhanced_logits'] = enhanced_logits
        enhanced_results['enhanced_preds'] = enhanced_preds
        enhanced_results['inference_time'] = time.time() - start_time
        
        return enhanced_results
    
    def generate_explanation(self, x, class_idx=None, top_k_prototypes=3):
        """
        Generate a detailed explanation for the classification decision.
        
        Args:
            x: Input tensor
            class_idx: Class to explain (if None, use predicted class)
            top_k_prototypes: Number of most important prototypes to include
            
        Returns:
            Explanation dictionary
        """
        # Get initial prediction
        with torch.no_grad():
            proto_features, pooled, logits = self.model(x)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = predicted_class
        
        # Get important prototypes for this class
        class_weights = self.classification_weights[class_idx]
        
        # Combine class weights with actual activations to get importance
        prototype_importance = pooled[0] * class_weights
        
        # Get top k prototypes for this class
        top_protos = torch.topk(prototype_importance, min(top_k_prototypes, len(prototype_importance)))
        top_indices = top_protos.indices.tolist()
        top_scores = top_protos.values.tolist()
        
        # Process each top prototype
        prototype_explanations = []
        
        for proto_idx, importance in zip(top_indices, top_scores):
            # Clone input to avoid modifying the original
            input_clone = x.clone()
            
            # Compute attributions for this prototype
            attributions, activation = self.compute_attributions(input_clone, proto_idx)
            
            # Compute distances to centroids
            distances = self.compute_attribution_distances(attributions)
            
            # Find best centroid match
            best_match = {'distance': float('inf'), 'layer': None, 'centroid': None}
            for layer_idx, layer_distances in distances.items():
                min_idx = torch.argmin(layer_distances, dim=1).item()
                min_dist = layer_distances[0, min_idx].item()
                
                if min_dist < best_match['distance']:
                    best_match = {
                        'distance': min_dist,
                        'layer': layer_idx,
                        'centroid': min_idx
                    }
            
            # Create heatmap by summing attribution across channels
            # Choose input attribution for the heatmap
            if 'input' in attributions:
                attribution_map = torch.sum(torch.abs(attributions['input']), dim=1)
                # Normalize for visualization
                attribution_map = attribution_map / (torch.max(attribution_map) + 1e-8)
            else:
                # Fallback to another layer if input attribution not available
                attr_key = next(iter(attributions.keys()))
                attr_tensor = attributions[attr_key]
                attribution_map = torch.sum(torch.abs(attr_tensor), dim=1)
                # Normalize
                attribution_map = attribution_map / (torch.max(attribution_map) + 1e-8)
            
            # Store explanation for this prototype
            prototype_explanations.append({
                'prototype_idx': proto_idx,
                'importance': importance,
                'activation': activation.item(),
                'attribution_map': attribution_map,
                'best_match': best_match,
                'distances': {k: v[0].tolist() for k, v in distances.items()}
            })
        
        # Create complete explanation
        explanation = {
            'input': x.clone().cpu().detach(),
            'predicted_class': predicted_class,
            'explained_class': class_idx,
            'class_score': logits[0, class_idx].item(),
            'prototype_explanations': prototype_explanations
        }
        
        return explanation


def visualize_explanation(explanation, save_path=None):
    """
    Visualize the explanation with attribution maps.
    
    Args:
        explanation: Explanation dictionary from generate_explanation
        save_path: Optional path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Extract data from explanation
    input_image = explanation['input'][0].permute(1, 2, 0).numpy()
    # Normalize for visualization
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min() + 1e-8)
    
    prototype_explanations = explanation['prototype_explanations']
    n_prototypes = len(prototype_explanations)
    
    # Create figure
    fig, axes = plt.subplots(n_prototypes + 1, 2, figsize=(10, 3 * (n_prototypes + 1)))
    
    # Show original image in the first row
    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title(f"Input Image\nPredicted Class: {explanation['predicted_class']}")
    axes[0, 0].axis('off')
    
    # Show explanation text in the first row, second column
    info_text = f"Explaining Class: {explanation['explained_class']}\n" \
               f"Class Score: {explanation['class_score']:.3f}"
    axes[0, 1].text(0.5, 0.5, info_text, 
                   ha='center', va='center', fontsize=12)
    axes[0, 1].axis('off')
    
    # Create colormap for attribution visualization
    cmap = LinearSegmentedColormap.from_list('attribution', [(0, 'white'), (1, 'red')])
    
    # Show each prototype explanation
    for i, proto_exp in enumerate(prototype_explanations):
        row_idx = i + 1
        
        # Get attribution map
        attr_map = proto_exp['attribution_map'][0].cpu().numpy()
        
        # Show attribution overlay
        axes[row_idx, 0].imshow(input_image)
        axes[row_idx, 0].imshow(attr_map, cmap=cmap, alpha=0.6)
        axes[row_idx, 0].set_title(f"Prototype {proto_exp['prototype_idx']}\n" \
                                 f"Importance: {proto_exp['importance']:.3f}")
        axes[row_idx, 0].axis('off')
        
        # Show match information
        best_match = proto_exp['best_match']
        match_text = f"Best Centroid Match:\n" \
                    f"Layer: {best_match['layer']}\n" \
                    f"Centroid: {best_match['centroid']}\n" \
                    f"Distance: {best_match['distance']:.3f}\n" \
                    f"Activation: {proto_exp['activation']:.3f}"
        
        axes[row_idx, 1].text(0.5, 0.5, match_text,
                            ha='center', va='center', fontsize=10)
        axes[row_idx, 1].axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


class BatchForwardPURE:
    """
    Optimized implementation for batch processing with autograd-based attribution.
    """
    def __init__(self, forward_pure):
        """
        Initialize with a ForwardPURE instance.
        
        Args:
            forward_pure: ForwardPURE instance
        """
        self.pure = forward_pure
        self.model = forward_pure.model
        self.device = forward_pure.device
        self.centroids_by_layer = forward_pure.centroids_by_layer
        self.classification_weights = forward_pure.classification_weights
    
    def batch_inference(self, x, prototype_threshold=0.5, weight_threshold=0.1, 
                      max_prototypes_per_sample=5, distance_threshold=1.0):
        """
        Process a batch of inputs with optimized attribution computation.
        
        Args:
            x: Batch of input tensors
            prototype_threshold: Minimum activation to consider a prototype
            weight_threshold: Minimum classification weight to consider a prototype
            max_prototypes_per_sample: Maximum prototypes to analyze per sample
            distance_threshold: Maximum distance for a valid centroid match
            
        Returns:
            Dictionary with enhanced classification results
        """
        start_time = time.time()
        
        # Get initial predictions with a single forward pass
        with torch.no_grad():
            proto_features, pooled, logits = self.model(x)
            initial_preds = torch.argmax(logits, dim=1)
        
        batch_size = x.shape[0]
        
        # Enhanced results container
        enhanced_results = {
            'initial_preds': initial_preds,
            'enhanced_confidence': torch.zeros_like(pooled),
            'batch_centroid_matches': [[] for _ in range(batch_size)]
        }
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Clone input to avoid modifying the original
            sample_input = x[i:i+1].clone()
            
            # Select important prototypes
            active_mask = pooled[i] > prototype_threshold
            pred_class = initial_preds[i].item()
            important_protos = self.classification_weights[pred_class] > weight_threshold
            
            # Combine activation and importance
            significant_protos = active_mask & important_protos
            prototype_indices = torch.nonzero(significant_protos, as_tuple=True)[0].tolist()
            
            # Limit to top k most activated prototypes
            if len(prototype_indices) > max_prototypes_per_sample:
                proto_activations = pooled[i, prototype_indices]
                _, top_indices = torch.topk(proto_activations, max_prototypes_per_sample)
                prototype_indices = [prototype_indices[j] for j in top_indices.tolist()]
            
            # Process each important prototype
            for proto_idx in prototype_indices:
                # Compute attributions with autograd
                attributions, activation = self.pure.compute_attributions(sample_input, proto_idx)
                
                # Compute distances to centroids
                distances = self.pure.compute_attribution_distances(attributions)
                
                # Find best match across all layers
                best_match = {'distance': float('inf'), 'layer': None, 'centroid': None}
                for layer_idx, layer_distances in distances.items():
                    min_idx = torch.argmin(layer_distances, dim=1).item()
                    min_dist = layer_distances[0, min_idx].item()
                    
                    if min_dist < best_match['distance']:
                        best_match = {
                            'distance': min_dist,
                            'layer': layer_idx,
                            'centroid': min_idx
                        }
                
                # Store match information
                enhanced_results['batch_centroid_matches'][i].append({
                    'prototype_idx': proto_idx,
                    'activation': activation.item(),
                    'best_match': best_match
                })
                
                # Adjust confidence based on match quality
                match_quality = 1.0 / (1.0 + best_match['distance'])  # Convert distance to similarity score
                enhanced_results['enhanced_confidence'][i, proto_idx] = activation * match_quality
        
        # Compute enhanced predictions using adjusted confidence
        enhanced_logits = torch.zeros_like(logits)
        
        for i in range(batch_size):
            for class_idx in range(logits.shape[1]):
                enhanced_logits[i, class_idx] = torch.sum(
                    enhanced_results['enhanced_confidence'][i] * self.classification_weights[class_idx]
                )
        
        # Final predictions
        enhanced_results['enhanced_logits'] = enhanced_logits
        enhanced_results['enhanced_preds'] = torch.argmax(enhanced_logits, dim=1)
        enhanced_results['inference_time'] = time.time() - start_time
        
        return enhanced_results


# Example usage
def example_usage(pipnet_model, dataloader, pre_computed_centroids):
    """Example of how to use ForwardPURE with autograd."""
    # Initialize ForwardPURE
    forward_pure = ForwardPURE(
        pipnet_model=pipnet_model,
        centroids_by_layer=pre_computed_centroids
    )
    
    # Initialize batch processor for faster inference
    batch_processor = BatchForwardPURE(forward_pure)
    
    # Get a batch of data
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        inputs, labels = batch[0], batch[1]
    else:
        inputs, labels = batch, None
    
    # Move to device
    inputs = inputs.to(forward_pure.device)
    
    # Get regular predictions
    with torch.no_grad():
        _, _, logits = pipnet_model(inputs)
        preds = torch.argmax(logits, dim=1)
    
    print(f"Regular predictions: {preds[:4]}")
    
    # Enhanced classification for a single sample
    enhanced_result = forward_pure.enhanced_classification(inputs[0:1])
    print(f"Enhanced prediction: {enhanced_result['enhanced_preds']}")
    
    # Batch processing for multiple samples
    batch_results = batch_processor.batch_inference(inputs[:4])
    print(f"Batch enhanced predictions: {batch_results['enhanced_preds']}")
    
    # Generate explanation for the first sample
    explanation = forward_pure.generate_explanation(inputs[0:1])
    
    # Visualize the explanation
    fig = visualize_explanation(explanation)
    
    return {
        'explanation': explanation,
        'enhanced_result': enhanced_result,
        'batch_results': batch_results
    }