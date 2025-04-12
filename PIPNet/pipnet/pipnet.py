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

    def forward(self, pooled):                  # pooled : (B, D)
        # logits  (B, C, H)
        logits = torch.einsum('bd,chd->bch', pooled, self.weight)
        logits, _ = logits.max(dim=2)           # max over hypotheses
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
        # self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier
        self._num_hypotheses = 4
        self._classification = CompetingHead(num_classes, num_prototypes, num_hypotheses=self._num_hypotheses)

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
            out = self._classification(pooled) #shape (bs*2, num_classes) 
            if features_save:
                return proto_features, pooled, out, features
            return proto_features, pooled, out

    def extract_features(self, xs,  inference=False, features_save=False):
        features = self._net(xs) 
        proto_features = self._add_on(features)
        pooled = self._pool(proto_features)
        return proto_features, pooled, features

class PIPNetSampler(nn.Module):
    def __init__(self,
                pipnet: PIPNet,
                num_classes: int,
                args: argparse.Namespace,
                heatmap_size=(30, 30),
                num_patches_to_select=5
                ):
        super().__init__()
        assert num_classes > 0
        self.pipnet = pipnet
        self.patch_location_network = PatchLocalizationHeatmap(heatmap_size).cuda()
        self.num_patches_to_select = num_patches_to_select
        self.initialize_patch_folders(args)
        # Initialize patch folders

    def initialize_patch_folders(self, args):
        self.patch_folders = []
        for i in range(args.num_features):
            folder_path = os.path.join(args.feature_folder, f"prototype_{i}")
            if os.path.exists(folder_path):
                patches = self._load_patches_from_folder(folder_path)
                self.patch_folders.append(patches)
            else:
                self.patch_folders.append([])
    
    def _load_patches_from_folder(self, folder_path):
        patch_files = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                      glob.glob(os.path.join(folder_path, "*.png"))
        patches = []
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for file in patch_files:
            try:
                img = Image.open(file).convert('RGB')
                patch = transform(img)
                patches.append(patch)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        return patches

    def gumbel_sigmoid_normalized(self, probabilities, temperature=1.0, hard=False, eps=1e-10):
        probabilities = torch.clamp(probabilities, eps, 1-eps)
        logits = torch.log(probabilities) - torch.log(1 - probabilities)
        
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, device=logits.device)))
        y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
        
        if hard:
            y_hard = (y_soft > 0.5).float()
            y = y_hard.detach() - y_soft.detach() + y_soft
        else:
            y = y_soft
            
        return y

    def select_top_k_prototypes(self, prototype_activations, k=5):
        """Select top-k activated prototypes from existing folders only"""
        batch_size = prototype_activations.shape[0]
        selected_indices = []
        
        for b in range(batch_size):
            # Create mask for prototypes with existing folders
            existing_folders_mask = torch.zeros_like(prototype_activations[b])
            for i in range(len(self.patch_folders)):
                if len(self.patch_folders[i]) > 0:  # Folder has patches
                    existing_folders_mask[i] = 1.0
            
            # Apply mask to activations
            masked_activations = prototype_activations[b] * existing_folders_mask
            
            # Sort masked activations
            sorted_act, indices = torch.sort(masked_activations, descending=True)
            
            # Get non-zero activations
            valid_indices = [idx for idx, val in zip(indices.tolist(), sorted_act.tolist()) 
                                if val > 0 and len(self.patch_folders[idx]) > 0]
            
            # Take top-k or as many as available
            top_indices = valid_indices[:min(k, len(valid_indices))]
            
            # If we don't have enough, pad with zeros (will be handled in patch sampling)
            while len(top_indices) < k:
                top_indices.append(0)  # Dummy index
                
            selected_indices.append(top_indices)
        
        return selected_indices
    
    def sample_patches_from_folders(self, prototype_indices):
        """Sample random patches from the folders of selected prototypes"""
        batch_size = len(prototype_indices)
        sampled_patches = []
        
        print(prototype_indices)
        for b in range(batch_size):
            batch_patches = []
            
            for idx in prototype_indices[b]:
                if idx < len(self.patch_folders) and len(self.patch_folders[idx]) > 0:
                    # Randomly select a patch from this prototype's folder
                    patch_idx = torch.randint(0, len(self.patch_folders[idx]), (1,)).item()
                    patch = self.patch_folders[idx][patch_idx]
                    batch_patches.append(patch)
                else:
                    # If no patches available, create a blank patch
                    patch = torch.zeros(3, 224, 224, device=prototype_indices[0][0].device)
                    batch_patches.append(patch)
            
            # Ensure we have exactly num_patches_to_select
            while len(batch_patches) < self.num_patches_to_select:
                batch_patches.append(torch.zeros(3, 224, 224, device=prototype_indices[0][0].device))
            
            # If we have too many, truncate
            batch_patches = batch_patches[:self.num_patches_to_select]
            sampled_patches.append(torch.stack(batch_patches))
        
        return torch.stack(sampled_patches)  # [batch_size, num_patches, 3, H, W]
    
    def forward(self, xs, inference=False, features_save=False):
        # 1. Run the image through PIPNet to get prototype activations
        proto_features, pooled, out = self.pipnet(xs, inference, features_save)
        
        # 2. Select top-k prototype indices (differentiable)
        selections = self.gumbel_sigmoid_normalized(pooled, temperature=0.1, hard=True)
        selected_indices = self.select_top_k_prototypes(pooled, k=self.num_patches_to_select)
        
        # 3. Sample random patches from the folders of selected prototypes
        sampled_patches = self.sample_patches_from_folders(selected_indices).cuda()
        
        # 4. Get patch locations from the PIPNet feature map
        patch_locations = self.get_patch_locations(proto_features, selected_indices)
        
        # 5. Generate ground truth heatmaps from the patch locations
        gt_heatmaps = self.generate_patch_heatmaps(xs, patch_locations, heatmap_size=(64, 64))
        
        # 6. Use the patch location network to predict where these patches are located
        patch_heatmaps, combined_heatmap = self.patch_location_network(sampled_patches, xs)
        
        return {
            'proto_features': proto_features,
            'pooled': pooled,
            'classification': out,
            'selections': selections,
            'selected_indices': selected_indices,
            'sampled_patches': sampled_patches,
            'patch_locations': patch_locations,
            'gt_heatmaps': gt_heatmaps,
            'patch_heatmaps': patch_heatmaps,
            'combined_heatmap': combined_heatmap
        }

    def get_patch_locations(self, proto_features, selected_indices):
        """Extract patch locations from PIPNet feature map for selected prototypes"""
        batch_size = proto_features.shape[0]
        feature_size = proto_features.shape[2:]  # (H, W)
        patch_locations = []
        
        for b in range(batch_size):
            batch_locations = []
            for prototype_idx in selected_indices[b]:
                # Skip dummy indices
                if prototype_idx >= proto_features.shape[1]:
                    batch_locations.append((0, 0, 0, 0))  # Dummy location
                    continue
                    
                # Get activation map for this prototype
                activation_map = proto_features[b, prototype_idx]
                
                # Find location with highest activation
                max_val, max_idx = torch.max(activation_map.view(-1), dim=0)
                h_idx, w_idx = max_idx // feature_size[1], max_idx % feature_size[1]
                
                # Convert to original image coordinates 
                # Assuming feature map is proportionally sized to input image
                img_h, img_w = 224, 224  # Standard input size
                scale_h, scale_w = img_h / feature_size[0], img_w / feature_size[1]
                
                # Define patch size (e.g., 1/8 of image size)
                patch_h, patch_w = img_h // 8, img_w // 8
                
                # Calculate patch coordinates
                x = w_idx.item() * scale_w - patch_w // 2
                y = h_idx.item() * scale_h - patch_h // 2
                
                # Ensure coordinates are within image bounds
                x = max(0, min(img_w - patch_w, x))
                y = max(0, min(img_h - patch_h, y))
                
                batch_locations.append((x, y, patch_w, patch_h))
                
            patch_locations.append(batch_locations)
        
        return patch_locations

    def generate_patch_heatmaps(self, images, patch_locations, heatmap_size=(56, 56)):
        """Generate ground truth heatmaps from patch locations"""
        batch_size = images.shape[0]
        num_patches = len(patch_locations[0])
        img_h, img_w = images.shape[2:]
        h_scale, w_scale = heatmap_size[0] / img_h, heatmap_size[1] / img_w
        
        heatmaps = torch.zeros(batch_size, num_patches, 1, *heatmap_size, device=images.device)
        
        for b in range(batch_size):
            for p in range(len(patch_locations[b])):
                x, y, w, h = patch_locations[b][p]
                
                # Skip invalid locations
                if w == 0 or h == 0:
                    continue
                    
                # Scale to heatmap size
                x1, y1 = int(x * w_scale), int(y * h_scale)
                x2, y2 = int((x + w) * w_scale), int((y + h) * h_scale)
                
                # Ensure within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(heatmap_size[1]-1, x2), min(heatmap_size[0]-1, y2)
                
                # Create Gaussian-like heatmap
                sigma = min(w, h) * w_scale * 0.2
                cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                
                y_indices = torch.arange(0, heatmap_size[0], device=images.device).float()
                x_indices = torch.arange(0, heatmap_size[1], device=images.device).float()
                
                y_grid, x_grid = torch.meshgrid(y_indices, x_indices)
                
                dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
                heatmap = torch.exp(-dist_sq / (2 * sigma**2))
                
                heatmaps[b, p, 0] = heatmap
        
        return heatmaps    




class PatchLocalizationHeatmap(nn.Module):
    def __init__(self, heatmap_size=(56, 56)):
        super().__init__()
        # Feature extractor
        resnet = models.resnet18(pretrained=True)
        self.patch_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.target_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512
        
        # Correlation module
        self.correlation_module = nn.Sequential(
            nn.Conv2d(self.feature_dim*2, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Heatmap decoder
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, selected_patches, target_images):
        batch_size, num_patches = selected_patches.shape[0], selected_patches.shape[1]
        
        # Extract features from target images
        target_features = self.target_encoder(target_images)  # [B, 512, h, w]
        
        # Process each patch and predict heatmap
        all_heatmaps = []
        
        for b in range(batch_size):
            batch_heatmaps = []
            
            for i in range(num_patches):
                # Get this patch
                patch = selected_patches[b, i].unsqueeze(0)  # [1, 3, H, W]
                
                # Skip if this is a zero patch (placeholder)
                if torch.sum(torch.abs(patch)) < 1e-6:
                    heatmap = torch.zeros(1, 1, 56, 56, device=patch.device)
                    batch_heatmaps.append(heatmap)
                    continue
                
                # Extract features
                patch_features = self.patch_encoder(patch)  # [1, 512, h, w]
                
                # Global pooling for patch representation
                patch_features_pooled = F.adaptive_avg_pool2d(patch_features, (1, 1))
                patch_features_expanded = patch_features_pooled.expand_as(
                    target_features[b:b+1])  # Match target size
                
                # Concatenate patch and target features
                combined = torch.cat([target_features[b:b+1], patch_features_expanded], dim=1)
                
                # Process through correlation module
                corr_features = self.correlation_module(combined)
                
                # Generate heatmap
                heatmap = self.heatmap_decoder(corr_features)
                batch_heatmaps.append(heatmap)
            
            # Stack batch heatmaps [num_patches, 1, H, W]
            batch_heatmaps = torch.cat(batch_heatmaps, dim=0)
            all_heatmaps.append(batch_heatmaps)
        
        # Stack all results [B, num_patches, 1, H, W]
        stacked_heatmaps = torch.stack(all_heatmaps)
        
        # Combined visualization heatmap (sum across patches)
        combined_heatmap = torch.sum(stacked_heatmaps, dim=1)
        
        # Normalize for visualization
        eps = 1e-8
        max_vals = torch.max(
            torch.max(
                combined_heatmap.view(batch_size, -1), dim=1, keepdim=True
            )[0], 
            torch.tensor(eps, device=combined_heatmap.device)
        ).view(batch_size, 1, 1, 1)
        normalized_combined = combined_heatmap / max_vals
        
        return stacked_heatmaps, normalized_combined

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


    