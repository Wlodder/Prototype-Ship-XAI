import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from util.attribution_analyzer import MultiLayerAttributionAnalyzer

class EnhancedForwardPURE:
    """
    Enhanced implementation of PURE using the same layer structure and attribution
    method as MultiLayerAttributionAnalyzer.
    """
    def __init__(self, pipnet_model, device='cuda'):
        """
        Initialize with a trained PIPNet model.
        
        Args:
            pipnet_model: Trained PIPNet model
            device: Computing device
        """
        self.model = pipnet_model
        self.device = device
        self.centroids_by_layer = {}
        
        # Extract classification weights for decision making
        self.classification_weights = pipnet_model.module._classification.weight.data
        
        # Create a MultiLayerAttributionAnalyzer instance to utilize its layer handling
        self.analyzer = MultiLayerAttributionAnalyzer(pipnet_model, device=device)
        
        # Will store mapping between analyzer's layer indices and centroid layer indices
        self.layer_mapping = None

    def add_centroids(self, split_results):
        """
        Add pre-computed centroids for each layer to the model.
        
        Args:
            split_results: Dictionary mapping layer indices to cluster centroids
        """
        # Process and store centroids
        for proto_batch, _ in split_results.items(): 
            for layer, centroids in split_results[proto_batch]['centroids'].items():
                if layer not in self.centroids_by_layer:
                    self.centroids_by_layer[layer] = []
                self.centroids_by_layer[layer].append(centroids)
        
        # Compute automatic layer mapping now that we have centroids
        self.layer_mapping = self.compute_automatic_layer_mapping()
        
        print(f"Added centroids for layers: {list(self.centroids_by_layer.keys())}")
        print(f"Created mapping between analyzer layers and centroid layers: {self.layer_mapping}")

    def compute_automatic_layer_mapping(self):
        """
        Compute mapping between analyzer's layer indices and centroid layer indices.
        
        Returns:
            Dictionary mapping analyzer layer indices to centroid layer indices
        """
        # Get available centroid layer indices
        centroid_layer_indices = sorted(list(self.centroids_by_layer.keys()))
        
        if not centroid_layer_indices:
            print("Warning: No centroids available for mapping")
            return {}
        
        # Get analyzer's layer indices
        analyzer_layer_indices = list(range(len(self.analyzer.layer_modules)))
        
        # If the number of layers matches exactly, create a direct mapping
        if len(analyzer_layer_indices) == len(centroid_layer_indices):
            return {idx: centroid_idx for idx, centroid_idx in zip(analyzer_layer_indices, centroid_layer_indices)}
        
        # Otherwise, we need to create a mapping based on layer depth
        mapping = {}
        
        # Add special cases: input layer and final layer
        mapping[0] = centroid_layer_indices[0]  # First analyzer layer -> first centroid layer
        mapping[analyzer_layer_indices[-1]] = centroid_layer_indices[-1]  # Last analyzer layer -> last centroid layer
        
        # For the intermediate layers, distribute them evenly
        if len(analyzer_layer_indices) > 2 and len(centroid_layer_indices) > 2:
            # Get intermediate layers (excluding first and last)
            intermediate_analyzer_layers = analyzer_layer_indices[1:-1]
            intermediate_centroid_layers = centroid_layer_indices[1:-1]
            
            # Create evenly spaced mapping
            for i, analyzer_idx in enumerate(intermediate_analyzer_layers):
                # Find nearest centroid layer
                if intermediate_centroid_layers:
                    relative_pos = i / max(1, len(intermediate_analyzer_layers) - 1)
                    centroid_pos = int(relative_pos * (len(intermediate_centroid_layers) - 1))
                    mapping[analyzer_idx] = intermediate_centroid_layers[min(centroid_pos, len(intermediate_centroid_layers) - 1)]
        
        # For analyzer layers that weren't mapped, assign to nearest mapped layer
        for analyzer_idx in analyzer_layer_indices:
            if analyzer_idx not in mapping:
                # Find nearest mapped layer
                distances = [(abs(analyzer_idx - mapped_idx), mapped_idx, centroid_idx) 
                           for mapped_idx, centroid_idx in mapping.items()]
                _, _, nearest_centroid_idx = min(distances)
                mapping[analyzer_idx] = nearest_centroid_idx
        
        return mapping

    def compute_attributions(self, x, prototype_idx, layer_indices=None):
        """
        Compute attributions using the MultiLayerAttributionAnalyzer's method.
        
        Args:
            x: Input tensor
            prototype_idx: Index of prototype to analyze
            layer_indices: Optional list of specific layer indices to analyze
            
        Returns:
            Dictionary of layer attributions and prototype activation
        """
        # Use the analyzer's method to compute attributions
        attributions = self.analyzer.compute_layer_attributions(
            x, prototype_idx, layer_indices
        )
        
        # Get the prototype activation for this input
        with torch.no_grad():
            _, pooled, _ = self.model(x, inference=False)
            activation = pooled[0, prototype_idx]
        
        # Return both the attributions and the activation
        return attributions, activation

    def compute_attribution_distances(self, attributions):
        """
        Compute distances between attributions and centroids.
        
        Args:
            attributions: Dictionary of layer attributions from compute_attributions
            
        Returns:
            Dictionary of distances to centroids for each analyzer layer
        """
        distances = {}
        
        # For each layer in the attributions
        for analyzer_layer_idx, attribution in attributions.items():
            # Skip if attribution is None
            if attribution is None:
                continue
            
            # Map analyzer layer to centroid layer using our mapping
            if self.layer_mapping and analyzer_layer_idx in self.layer_mapping:
                centroid_layer_idx = self.layer_mapping[analyzer_layer_idx]
            else:
                # Skip if we can't map this layer
                continue
            
            # Skip if we don't have centroids for this layer
            if centroid_layer_idx not in self.centroids_by_layer:
                continue
            
            # Get centroids for this layer
            layer_centroids = self.centroids_by_layer[centroid_layer_idx]
            if not layer_centroids:
                continue
            
            # Compute distances to each centroid group
            layer_distances = []
            
            for centroid_group in layer_centroids:
                # Convert attribution to a flat vector for comparison
                flat_attr = attribution.flatten()
                
                # Compute distances to all centroids in this group
                centroid_distances = []
                
                for centroid in centroid_group:
                    # Convert centroid to tensor if needed
                    if not isinstance(centroid, torch.Tensor):
                        centroid = torch.tensor(centroid, device=self.device)
                    
                    # Flatten centroid for comparison
                    flat_centroid = centroid.flatten().cuda()
                    
                    
                    # Handle dimension mismatch
                    if flat_attr.shape[0] != flat_centroid.shape[0]:
                        min_dim = min(flat_attr.shape[0], flat_centroid.shape[0])
                        distance = torch.norm(flat_attr[:min_dim] - flat_centroid[:min_dim])
                    else:
                        distance = torch.norm(flat_attr - flat_centroid)
                    
                    centroid_distances.append(distance.item())
                
                # Store the minimum distance to any centroid in this group
                if centroid_distances:
                    layer_distances.append(min(centroid_distances))
            
            # Store distances for this layer
            if layer_distances:
                distances[analyzer_layer_idx] = layer_distances
        
        return distances

    def enhanced_classification(self, x, layer_indices=None, distance_threshold=3.0):
        """
        Perform classification enhanced by centroid-based attribution matching.
        
        Args:
            x: Input tensor
            layer_indices: Optional list of specific layer indices to analyze
            distance_threshold: Maximum distance to consider a valid centroid match
            
        Returns:
            Dictionary with enhanced classification results
        """
        # If layer_indices is not provided, use layers corresponding to available centroids
        if layer_indices is None:
            if self.layer_mapping:
                layer_indices = list(self.layer_mapping.keys())
            else:
                # Default to using first 5 layers if no mapping available
                layer_indices = list(range(min(5, len(self.analyzer.layer_modules))))
        
        # Get initial predictions
        with torch.no_grad():
            _, pooled, logits = self.model(x)
            initial_preds = torch.argmax(logits, dim=1)
        
        batch_size = x.shape[0]
        
        # Results container
        enhanced_results = {
            'initial_preds': initial_preds,
            'enhanced_confidence': torch.zeros_like(pooled),
            'centroid_matches': [[] for _ in range(batch_size)],
            'matches': [[] for _ in range(batch_size)]
        }
        
        # Process each sample
        for i in range(batch_size):
            sample = x[i:i+1]
            
            # Find active prototypes for this sample
            active_protos = torch.where(pooled[i] > 0.3)[0].tolist()
            
            # Limit to top 10 most activated prototypes for efficiency
            if len(active_protos) > 10:
                _, top_indices = torch.topk(pooled[i], 10)
                active_protos = top_indices.tolist()
            
            # Process each active prototype
            for proto_idx in active_protos:
                # Compute attributions using the analyzer
                attributions, activation = self.compute_attributions(
                    sample, proto_idx, layer_indices
                )

                
                # Compute distances to centroids
                distances = self.compute_attribution_distances(attributions)
                
                # Skip if no valid distances
                if not distances:
                    continue
                
                # Find best matching centroid
                matches = []
                best_match = {'layer': None, 'distance': float('inf')}
                
                for layer_idx, layer_distances in distances.items():
                    matches.append([{
                        'layer':layer_idx,
                        'distance':layer_distances,
                        'layer_name': self.analyzer.layer_names[layer_idx] if layer_idx < len(self.analyzer.layer_names) else f"layer_{layer_idx}"
                    }])
                    min_distance = min(layer_distances)
                    if min_distance < best_match['distance']:
                        best_match = {
                            'layer': layer_idx,
                            'distance': min_distance,
                            'layer_name': self.analyzer.layer_names[layer_idx] if layer_idx < len(self.analyzer.layer_names) else f"layer_{layer_idx}"
                        }
                
                # Store match information
                enhanced_results['centroid_matches'][i].append({
                    'prototype_idx': proto_idx,
                    'activation': activation.item(),
                    'best_match': best_match
                })
                enhanced_results['matches'][i].append(matches)
                
                # Adjust confidence based on match quality
                # If distance is below threshold, boost confidence
                if best_match['distance'] < distance_threshold:
                    confidence_boost = 1.0 / (1.0 + best_match['distance'])
                    enhanced_results['enhanced_confidence'][i, proto_idx] = activation * (1.0 + confidence_boost)
                else:
                    # Otherwise, slightly reduce confidence
                    enhanced_results['enhanced_confidence'][i, proto_idx] = activation * 0.9
                
        
        # Compute enhanced predictions
        enhanced_logits = torch.zeros_like(logits)
        
        for i in range(batch_size):
            for class_idx in range(logits.shape[1]):
                enhanced_logits[i, class_idx] = torch.sum(
                    enhanced_results['enhanced_confidence'][i] * self.classification_weights[class_idx]
                )
        
        # Get final predictions and store in results
        enhanced_results['enhanced_logits'] = enhanced_logits
        enhanced_results['enhanced_preds'] = torch.argmax(enhanced_logits, dim=1)
        
        return enhanced_results

