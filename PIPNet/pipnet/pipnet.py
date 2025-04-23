import argparse
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
import glob
from torchvision import transforms
import os
from PIL import Image
from torch import Tensor


# heads.py  –– new file
import torch
import torch.nn as nn
import numpy as np


# Dirichlet allocation
class CRPPrototypeAllocation(nn.Module):
    """
    Chinese Restaurant Process inspired prototype allocation mechanism.
    Uses the "rich get richer" property of CRP but balances it with
    exploration of new features.
    """
    def __init__(self, 
                 num_prototypes: int,
                 alpha: float = 5.0,  # Concentration parameter
                 beta: float = 0.5,   # Power-law parameter 
                 device='cuda'):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.alpha = alpha  # Controls probability of new "tables" (prototypes)
        self.beta = beta    # Controls power-law exponent for rare prototype emphasis
        self.device = device
        
        # Initialize prototype count tensor
        # Initialize with small non-zero values instead of zeros
        self.register_buffer('prototype_counts', 
                            torch.ones(num_prototypes, device=device) * 0.1)
        self.register_buffer('total_count', 
                            torch.tensor([num_prototypes * 0.1], device=device))
        
        # Initialize allocation probabilities uniformly
        self.register_buffer('allocation_probs', 
                            torch.ones(num_prototypes, device=device) / num_prototypes)
    
    def update_allocation(self, pooled_activations: torch.Tensor):
        """
        Update prototype allocation probabilities based on CRP
        
        Args:
            pooled_activations: Tensor of prototype activations [batch_size, num_prototypes]
        """
        
        # Count "customer assignments" in this batch (activations above threshold)
        assignments = (pooled_activations > 0.2).float().sum(dim=0)
        
        # Update counts
        self.prototype_counts += assignments
        self.total_count += assignments.sum()
        
        # CRP probability calculation:
        # p(existing table) = n_k / (n + alpha)
        # p(new table) = alpha / (n + alpha)
        normalized_counts = self.prototype_counts / (self.total_count + self.alpha)
        
        # Apply power-law transformation to emphasize rare prototypes
        # Lower beta emphasizes rare prototypes more strongly
        transformed_probs = normalized_counts.pow(self.beta)
        
        # Normalize to get probabilities
        self.allocation_probs = transformed_probs / transformed_probs.sum()
    
    def forward(self, pooled_activations: torch.Tensor):
        """
        Apply CRP-inspired weighting to prototype activations
        
        Args:
            pooled_activations: Tensor of prototype activations [batch_size, num_prototypes]
            
        Returns:
            Reweighted activations that emphasize rare prototypes
        """
        # Update allocation probabilities (no gradient)
        with torch.no_grad():
            self.update_allocation(pooled_activations.detach())
        
        # Inverse probability weighting (give more weight to rare prototypes)
        inverse_probs = 1.0 / (self.allocation_probs + 1e-6)
        normalized_weights = inverse_probs / inverse_probs.mean()
        
        # Apply weights to activations
        weighted_activations = pooled_activations * normalized_weights.unsqueeze(0)
        
        return weighted_activations
    
    def get_allocation_stats(self):
        """Get statistics about prototype allocation"""
        # Get counts and probabilities
        counts = self.prototype_counts.cpu().numpy()
        probs = self.allocation_probs.cpu().numpy()
        
        # Calculate entropy of the allocation (higher = more diverse)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Calculate Gini coefficient (lower = more equal)
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        index = np.arange(1, n+1)
        gini = 1 - 2 * np.sum((n + 1 - index) * sorted_probs) / (n * np.sum(sorted_probs))
        
        return {
            'counts': counts,
            'probabilities': probs,
            'entropy': entropy,
            'gini': gini
        }


class CompetingHead(nn.Module):
    """
    One weight matrix per hypothesis; best hypothesis wins (max‑pool).
    weight : (C, H, D)   C = classes, H = hypotheses, D = prototypes
    """
    def __init__(self, num_classes: int,
                 num_prototypes: int,
                 num_hypotheses: int = 3,
                 normalization_multiplier: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(
            0.01 * torch.randn(num_classes, num_hypotheses, num_prototypes))
        self.normalization_multiplier = normalization_multiplier
        self._num_hyp = num_hypotheses

    def forward(self, pooled):
        # Original code
        logits = torch.einsum('bd,chd->bch', pooled, self.weight)
        
        # Add competition between heads
        # Each head gets to "claim" certain prototypes more strongly
        prototype_importance = torch.softmax(self.weight.abs().sum(dim=0), dim=0)  # (H, D)
        
        # Scale prototypes by their importance to each head
        scaled_pooled = pooled.unsqueeze(1) * prototype_importance  # (B, H, D)
        
        # Use scaled pooled values for each head
        head_logits = torch.einsum('bhd,chd->bch', scaled_pooled, self.weight)
        
        # Max over hypotheses as before
        logits, _ = head_logits.max(dim=2)
        return logits * self.normalization_multiplier

class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module,
                 dropout: float = 0.7,
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._pool = pool_layer
        # old classifcation 
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

        # Diriclet allocation
        self.crp_allocation = CRPPrototypeAllocation(self._num_features, alpha=8, beta=0.3, device='cuda')

        # Head clasification
        # self._num_hypotheses = 16
        # self._classification = CompetingHead(num_classes, num_prototypes, num_hypotheses=self._num_hypotheses)

    def forward(self, xs,  inference=False, features_save=False):
        features = self._net(xs) 
        
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)

        

        if inference:
            pooled = pooled.double()
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            clamped_pooled = clamped_pooled.float()
            out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            if features_save:
                return proto_features, clamped_pooled, out, features
            return proto_features, clamped_pooled, out
        else:
            self.crp_allocation.update_allocation(pooled)
            weighted_pooled = self.crp_allocation(pooled)
            out = self._classification(weighted_pooled) #shape (bs*2, num_classes) 

            if features_save:
                return proto_features, weighted_pooled, out, features

            return proto_features, weighted_pooled, out

    def extract_features(self, xs,  inference=False, features_save=False):
        features = self._net(xs) 
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        return proto_features, pooled, features


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'convnext_tiny_26': convnext_tiny_26_features,
                                 'convnext_tiny_13': convnext_tiny_13_features}

# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)


def get_network(num_classes: int, args: argparse.Namespace): 
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    pool_layer = nn.Sequential(
                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, pool_layer, classification_layer, num_prototypes


    