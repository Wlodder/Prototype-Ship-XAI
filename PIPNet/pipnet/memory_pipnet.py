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

class PrototypeMemoryBank:
    """
    Memory bank for storing feature representations in MaSSL.
    
    This buffer maintains a non-parametric distribution of previously seen
    representations using a FIFO queue. It enables the core MaSSL training approach 
    of comparing current feature representations with previous ones.
    """
    def __init__(self, size=65536, feature_dim=256):
        """
        Initialize the memory bank.
        
        Args:
            size: Maximum number of feature vectors to store
            feature_dim: Dimensionality of each feature vector
        """
        # Memory storage
        self.size = size
        self.feature_dim = feature_dim
        
        # Initialize memory with random unit vectors
        self.memory = torch.randn(size, feature_dim)
        self.memory = F.normalize(self.memory, dim=1)
        
        # Pointer for FIFO updates
        self.ptr = 0
        
        # Track whether memory has wrapped around once
        self.is_full = False
    
    def update(self, features):
        """
        Update memory with new features, removing oldest ones.
        
        Args:
            features: Tensor of shape [batch_size, feature_dim] containing normalized feature vectors
        """
        # Ensure features are normalized
        features = F.normalize(features, dim=1)
        
        # Number of features to add
        batch_size = features.size(0)
        
        # Handle case where batch is larger than memory
        if batch_size > self.size:
            # If batch is larger than entire memory, just use most recent features
            features = features[-self.size:]
            batch_size = self.size
        
        # Calculate indices for updating memory
        if self.ptr + batch_size <= self.size:
            # Simple case: enough space at current position
            self.memory[self.ptr:self.ptr+batch_size] = features
            self.ptr = (self.ptr + batch_size) % self.size
        else:
            # Need to wrap around to beginning of memory
            # First part: fill to the end of memory
            first_part = self.size - self.ptr
            self.memory[self.ptr:] = features[:first_part]
            
            # Second part: wrap around to beginning
            second_part = batch_size - first_part
            self.memory[:second_part] = features[first_part:]
            
            # Update pointer
            self.ptr = second_part
            
            # Mark memory as full since we've wrapped around
            self.is_full = True
    
    def get_memory_blocks(self, num_blocks=4, block_size=4096):
        """
        Sample random blocks of memory representations for MaSSL training.
        
        This is a key component of MaSSL that enables stochastic learning from
        multiple memory subsets, improving training stability.
        
        Args:
            num_blocks: Number of memory blocks to create
            block_size: Size of each memory block
            
        Returns:
            List of memory blocks, each containing block_size features,
            or None if insufficient data is available
        """
        # Determine actual memory size (full size or up to current pointer)
        actual_size = self.size if self.is_full else self.ptr
        
        # Check if we have enough data for at least one block
        if actual_size < block_size:
            return None
        
        # Create blocks
        blocks = []
        for _ in range(num_blocks):
            # Sample indices without replacement for each block
            indices = torch.randperm(actual_size)[:block_size]
            block = self.memory[indices]
            blocks.append(block)
        
        return blocks
    
    def to(self, device):
        """
        Move memory bank to specified device.
        
        Args:
            device: Target device (CPU or GPU)
            
        Returns:
            Self for chaining
        """
        self.memory = self.memory.to(device)
        return self

from torch import Tensor

class PIPNetMember(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 memory_layer: nn.Module,
                 softmax: nn.Module,
                 pool_layer: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        # self._add_on = add_on_layers
        self._add_on = memory_layer
        self._softmax = softmax
        self._pool = pool_layer
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs,  inference=False):
        features = self._net(xs) 
        comparisons = self._add_on(features) / torch.norm(features)
        proto_features = self._softmax(comparisons)

        # proto_features = self._add_on(softmaxes)
        # pooled = self._pool(proto_features)
        if inference:
            # pooled = pooled.double()
            # clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            # clamped_pooled = clamped_pooled.float()
            # out = self._classification(clamped_pooled) #shape (bs*2, num_classes)
            return proto_features
        else:
            # out = self._classification(pooled) #shape (bs*2, num_classes) 
            return proto_features



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
    
    
    num_prototypes = args.num_features
    print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
    memory_layer = nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=False)
    softmax = nn.Softmax(dim=1) #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    pool_layer = nn.Sequential(

                nn.AdaptiveMaxPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
                nn.Flatten() #outputs (bs, ps)
                ) 
    
    if args.bias:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = NonNegLinear(num_prototypes, num_classes, bias=False)
        
    return features, memory_layer, softmax, pool_layer, classification_layer, num_prototypes


    