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
    def __init__(self, pipnet_model, layer_weights, device='cuda'):
        """
        Initialize with a trained PIPNet model.
        
        Args:
            pipnet_model: Trained PIPNet model
            device: Computing device
        """
        self.model = pipnet_model
        self.device = device
        self.centroids = {}
        
        # Extract classification weights for decision making
        self.classification_weights = pipnet_model.module._classification.weight.data
        
        # Create a MultiLayerAttributionAnalyzer instance to utilize its layer handling
        self.analyzer = MultiLayerAttributionAnalyzer(pipnet_model, device=device)
        
        # Will store mapping between analyzer's layer indices and centroid layer indices
        self.layer_mapping = None

        # Store layer weights for distance computation
        total_weight = sum(layer_weights.values())
        layer_weights = {idx: w/total_weight for idx, w in layer_weights.items()}
        self._layer_weights = layer_weights
        self._layer_indicies = list(layer_weights.keys())

        # Store centroids for each layer
        self.centroids_by_layer = {}
        self.centroids = []

    def add_centroids(self, split_results):
        """
        Add pre-computed centroids for each layer to the model.
        
        Args:
            split_results: Dictionary mapping layer indices to cluster centroids
        """
        # Process and store centroids
        for proto_batch, _ in split_results.items(): 
            # for layer_idx, centroid in split_results[proto_batch]['centroids'].items():

            #     if layer_idx not in self.centroids_by_layer:
            #         self.centroids_by_layer[layer_idx] = []

            #     self.centroids_by_layer[layer_idx].append(centroid)

            for concat_centroid in split_results[proto_batch]['overall_centroids']:
                self.centroids.append(concat_centroid)
        
        # Compute automatic layer mapping now that we have centroids
        print(f"Added centroids : {len(self.centroids)}")
        print(f"Added centroids : {self.centroids_by_layer.keys()}")


    def compute_attributions(self, x, prototype_idx):
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
            x, prototype_idx, self._layer_indicies
        )
        
        # Get the prototype activation for this input
        with torch.no_grad():
            _, pooled, _ = self.model(x, inference=False)
            activation = pooled[0, prototype_idx]
        
        # Return both the attributions and the activation
        return attributions, activation

    def _process_attribution(self, attribution):
        """
        Process the attribution to ensure it's in the correct format.
        
        Args:
            attribution: Attribution tensor
        """
        
        # Combine attributions from all layers
    
        # Collect attributions for this sample from all layers
        sample_features = {}
        
        
        for layer_idx, weight in self._layer_weights.items():
            if layer_idx in attribution:
                # Get attribution for this layer and sample
                layer_attr = attribution[layer_idx][0]
                
                # Flatten and normalize the attribution
                flat_attr = layer_attr.flatten()
                norm = torch.norm(flat_attr)
                if norm > 0:
                    flat_attr = flat_attr / norm
                
                # Weight by layer importance
                flat_attr = flat_attr * weight
                
                sample_features[layer_idx] = flat_attr
        
        # Concatenate all layer features
        if sample_features:
            combined_attr = torch.cat(list(sample_features.values()))
        
        return combined_attr, sample_features

    def compute_attribution_distances(self, attributions):
        """
        Compute distances between attributions and centroids.
        
        Args:
            attributions: Dictionary of layer attributions from compute_attributions
            
        Returns:
            Dictionary of distances to centroids for each analyzer layer
        """
        combined_attributions, sample_features = self._process_attribution(attributions)
        distances = {}
        # for i, centroid in enumerate(self.centroids):
        #     distances[i] = torch.nn.functional.pairwise_distance(combined_attributions, centroid)
        
        for layer in sample_features.keys():
            # Compute distances to centroids for this layer
            layer_distances = []
            combined_distances = []

            for concatenated_centroid in self.centroids:
                centroid = concatenated_centroid.to(self.device)
                dist = torch.nn.functional.pairwise_distance(combined_attributions, centroid)
                combined_distances.append(dist)

            # for centroid in self.centroids_by_layer[layer]:
            #     # Compute distance
            #     centroid = centroid.to(self.device)
            #     dist = torch.nn.functional.pairwise_distance(sample_features[layer], centroid)
                # layer_distances.append(dist)
            
            # Store distances for this layer
            distances[layer] = layer_distances
        
        return distances, combined_distances


    def enhanced_classification(self, x, layer_indices=None, distance_threshold=3.0, custom_prototypes=None):
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
            if custom_prototypes is not None:
                active_protos = custom_prototypes
            elif len(active_protos) > 10:
                _, top_indices = torch.topk(pooled[i], 10)
                active_protos = top_indices.tolist()
            
            # Process each active prototype
            for proto_idx in active_protos:
                # Compute attributions using the analyzer
                attributions, activation = self.compute_attributions(
                    sample, proto_idx
                )

                if activation < 0.3:
                    continue

                # print(f"Attributions for prototype {proto_idx}: {attributions}")
                
                # Compute distances to centroids
                distances, real_distances = self.compute_attribution_distances(attributions)
                for layer_idx, layer_distances in distances.items():
                    distances[layer_idx] = [dist.flatten() for dist in layer_distances]
                
                real_distances = torch.tensor(real_distances)
                real_distances = torch.softmax(1/real_distances, dim=0)
                print(f"Real Distances for prototype {proto_idx}: {real_distances}")
                
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
                    # min_distance = min(layer_distances)
                    # if min_distance < best_match['distance']:
                    #     best_match = {
                    #         'layer': layer_idx,
                    #         'distance': min_distance,
                    #         'layer_name': self.analyzer.layer_names[layer_idx] if layer_idx < len(self.analyzer.layer_names) else f"layer_{layer_idx}"
                    #     }
                
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

