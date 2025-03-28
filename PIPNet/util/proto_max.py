import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pywt
import torchvision
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def multi_resolution_prototype_activation_localize(model, prototype_idx, device='cuda', iterations=500, 
                                         learning_rate=0.1, img_size=224, alpha_l1=0.0001,
                                         alpha_tv=0.001, blur_every=10, blur_sigma=0.5,
                                         normalize_mean=(0.485, 0.456, 0.406), 
                                         normalize_std=(0.229, 0.224, 0.225),
                                         resolutions=None, localization_lambda=0.2,
                                         spatial_concentration=0.5):
    """
    Generate an image that maximally activates a specific prototype part using curriculum learning
    with additional localization constraints.
    
    Args:
        model: The PIPNet model
        prototype_idx: Index of the prototype to maximize
        device: Device to run computations on
        iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        img_size: Size of the generated image
        alpha_l1: L1 regularization strength (sparsity)
        alpha_tv: Total variation regularization strength (smoothness)
        blur_every: Apply Gaussian blur every N iterations
        blur_sigma: Sigma parameter for Gaussian blur
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization
        resolutions: List of resolutions to use. If None, defaults to [1, 2, 4, 8, 16, 32, 64, 128, img_size]
        localization_lambda: Weight for the localization loss component
        spatial_concentration: Target concentration ratio for spatial activations (smaller = more focused)
        
    Returns:
        Tuple containing the final composite image and a dictionary of resolution components
    """
    import torch
    import torch.nn.functional as F
    import torchvision
    from tqdm import tqdm
    
    model.eval()
    model = model.to(device)
    
    # Set up resolutions if not provided
    if resolutions is None:
        # Create a progression of resolutions
        resolutions = [1, 2, 4, 8, 16, 32, 64, 128]
        if img_size not in resolutions:
            resolutions.append(img_size)
    
    # Sort resolutions from low to high for curriculum learning
    resolutions = sorted(resolutions)
    
    # Create multi-resolution components
    resolution_components = {}
    for res in resolutions:
        # Initialize with small random values for stability
        resolution_components[res] = torch.randn(1, 3, res, res, device=device) * 0.01
        resolution_components[res].requires_grad_(True)
    
    # Create optimizers for each resolution component
    optimizers = {
        res: torch.optim.Adam([resolution_components[res]], lr=0.0)  # Start with zero learning rate
        for res in resolutions
    }
    
    # Create Gaussian blur function
    def gaussian_blur(img, kernel_size, sigma):
        padding = kernel_size // 2
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        
        x = torch.arange(-padding, padding+1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        channels = img.shape[1]
        kernel_h = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        kernel_v = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
        
        img_blurred = F.conv2d(img_padded, kernel_h, groups=channels)
        img_blurred = F.conv2d(img_blurred, kernel_v, groups=channels)
        
        return img_blurred
    
    # Define transforms with curriculum learning (starting with milder transforms)
    def get_transforms(progress):
        """Get transforms with intensity based on optimization progress"""
        # Scale between 0 and 1
        p = min(1.0, progress / 0.7)  # Full strength at 70% of iterations
        
        jitter_strength = 0.05 + 0.3 * p
        translate_strength = 0.02 + 0.08 * p
        blur_min = 0.1
        blur_max = 0.5 + 1.5 * p
        
        jitter = torchvision.transforms.ColorJitter(
            brightness=jitter_strength, 
            contrast=jitter_strength, 
            saturation=jitter_strength, 
            hue=jitter_strength/2
        )
        translate = torchvision.transforms.RandomAffine(
            degrees=0, 
            translate=(translate_strength, translate_strength)
        )
        blur = torchvision.transforms.GaussianBlur(
            3, 
            sigma=(blur_min, blur_max)
        )
        
        return jitter, translate, blur
    
    # Calculate curriculum milestones
    # Lower resolutions start earlier, higher resolutions later
    curriculum_schedule = {}
    active_resolutions = []
    
    # Create curriculum schedule for introducing resolutions
    for i, res in enumerate(resolutions):
        # Progressive introduction of resolutions
        # Start with lowest resolution, gradually add higher ones
        start_iteration = int((i / len(resolutions)) * (iterations * 0.7))  # Introduce all by 70% of iterations
        curriculum_schedule[res] = start_iteration
    
    # Create curriculum for regularization strengths
    l1_schedule = lambda t: alpha_l1 * min(1.0, (t / iterations) * 2)  # Gradually increase L1
    tv_schedule = lambda t: alpha_tv * min(1.0, (t / iterations) * 3)   # Gradually increase TV
    
    # Localization weight schedule (increase over time)
    loc_schedule = lambda t: localization_lambda * min(1.0, (t / iterations) * 4)
    
    # Progress bar
    progress_bar = tqdm(range(iterations))
    
    # Track best activation and corresponding image
    best_activation = -float('inf')
    best_image = None
    
    # Retrieve prototype weight if possible for direct similarity
    # This helps target the specific prototype features
    try:
        prototype_weight = model.module._add_on[0].weight.data[0, prototype_idx].detach()
    except (AttributeError, IndexError):
        try:
            # Try alternative model structures
            prototype_weight = model._add_on[0].weight.data[0, prototype_idx].detach()
        except (AttributeError, IndexError):
            prototype_weight = None
            print("Warning: Could not retrieve prototype weight directly.")
    
    # Optimization loop
    for i in progress_bar:
        # Update curriculum - activate resolutions according to schedule
        for res in resolutions:
            if i >= curriculum_schedule[res] and res not in active_resolutions:
                active_resolutions.append(res)
                # Start with higher learning rate for lower resolutions
                scale_factor = 1.0 - (0.5 * (resolutions.index(res) / len(resolutions)))
                optimizers[res].param_groups[0]['lr'] = learning_rate * scale_factor
        
        # Apply Gaussian blur to components occasionally
        if i % blur_every == 0 and i > 0:
            with torch.no_grad():
                for res in active_resolutions:
                    if res > 1:  # No need to blur 1x1 components
                        kernel_size = min(int(blur_sigma * 4) + 1, res)
                        if kernel_size % 2 == 0:  # Make sure kernel size is odd
                            kernel_size += 1
                        kernel_size = max(3, min(kernel_size, res))  # Ensure kernel is at least 3 and fits in the image
                        if kernel_size < res:  # Only blur if kernel fits
                            resolution_components[res].data = gaussian_blur(
                                resolution_components[res].data, kernel_size, blur_sigma
                            )
        
        # Get transforms based on curriculum progress
        jitter, translate, blur = get_transforms(i / iterations)
        
        # Create composite image by upsampling and summing active components
        composite_image = torch.zeros(1, 3, img_size, img_size, device=device)
        for res in active_resolutions:
            if res < img_size:
                upsampled = F.interpolate(resolution_components[res], size=(img_size, img_size), 
                                          mode='bilinear', align_corners=False)
            else:
                upsampled = resolution_components[res]
            composite_image += upsampled
        
        # Clamp composite image to valid range
        composite_image = torch.clamp(composite_image, 0, 1)
        
        # Apply transforms with probability increasing over time
        p_transform = min(0.9, (i / iterations) * 1.5)  # Max 90% chance, reach full prob after 60% iterations
        
        transformed_image = composite_image
        if torch.rand(1).item() < p_transform:
            transformed_image = blur(transformed_image)
        if torch.rand(1).item() < p_transform:
            transformed_image = jitter(transformed_image)
        if torch.rand(1).item() < p_transform:
            transformed_image = translate(transformed_image)
        
        # Normalize input image for the model
        mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        normalized_image = (transformed_image - mean) / std
        
        # Forward pass through the model
        features, pooled, additional_outputs = model(normalized_image, inference=False)
        
        # ------- KEY CHANGE: Modified activation computation for part visualization -------
        
        # Different activation options depending on model structure
        if prototype_weight is not None:
            # Direct similarity to prototype weight (more focused on the specific prototype)
            similarity_map = F.conv2d(features, prototype_weight.unsqueeze(0).unsqueeze(0))
            
            # Get the maximum similarity and its location
            max_similarity, _ = torch.max(similarity_map.view(similarity_map.size(0), -1), dim=1)
            activation = max_similarity.mean()
            
            # For localization: Get spatial activation map
            activation_map = similarity_map.squeeze()
        else:
            # Fallback to using the pooled features
            target = torch.zeros_like(pooled[0])
            target[prototype_idx] = 1
            
            # Cross-entropy for classification (less focused)
            activation = -torch.nn.functional.cross_entropy(pooled[0], target[0].long())
            
            # Try to get an activation map from last conv features if available
            try:
                if hasattr(model, 'get_activations'):
                    activation_map = model.get_activations(normalized_image, prototype_idx)
                else:
                    # Create a proxy activation map using gradients
                    pooled_output = pooled[0][prototype_idx]
                    grads = torch.autograd.grad(pooled_output, features, 
                                              retain_graph=True, create_graph=True)[0]
                    activation_map = torch.mean(grads * features, dim=1).squeeze()
            except Exception as e:
                # If we can't get an activation map, use a dummy one
                print(f"Warning: Could not get activation map: {e}")
                activation_map = torch.ones((features.shape[2], features.shape[3]), device=device)
        
        # ------- Localization Loss to Encourage Part-Specific Visualization -------
        
        # 1. Spatial concentration loss (encourage activations to be concentrated)
        if activation_map is not None:
            # Normalize the activation map to sum to 1
            norm_act_map = F.softmax(activation_map.view(-1), dim=0).view(activation_map.shape)
            
            # Compute the spatial concentration (smaller is more concentrated)
            # We want activations to be focused on a small region, not spread out
            concentration = torch.sum(norm_act_map > 0.01) / float(norm_act_map.numel())
            
            # The loss encourages concentration below the target value
            loc_loss = F.relu(concentration - spatial_concentration)
            
            # Weight the localization loss based on the schedule
            current_loc_weight = loc_schedule(i)
            localization_loss = current_loc_weight * loc_loss
        else:
            localization_loss = 0
        
        # The total activation loss (negative because we want to maximize)
        activation_loss = -activation
        
        # Add regularization with curriculum
        # L1 regularization - weighted by resolution (stronger penalty for high-frequency components)
        current_l1_strength = l1_schedule(i)
        l1_losses = 0
        for res in active_resolutions:
            # Stronger regularization for higher resolutions
            resolution_factor = (res / img_size) ** 0.75  # Increased exponent for more aggressive scaling
            frequency_factor = 1.0 + (resolutions.index(res) / len(resolutions)) * (i / iterations) * 4
            
            l1_losses += current_l1_strength * resolution_factor * frequency_factor * torch.abs(resolution_components[res]).sum()
        
        # Total variation regularization to encourage smoothness
        current_tv_strength = tv_schedule(i)
        tv_h = torch.abs(composite_image[:, :, 1:, :] - composite_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(composite_image[:, :, :, 1:] - composite_image[:, :, :, :-1]).sum()
        tv_loss = current_tv_strength * (tv_h + tv_w)
        
        # Total loss (now including localization)
        total_loss = activation_loss + l1_losses + tv_loss + localization_loss
        
        # Zero all gradients for active optimizers
        for res in active_resolutions:
            optimizers[res].zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Step active optimizers
        for res in active_resolutions:
            optimizers[res].step()
        
        # Clamp component values to valid range
        with torch.no_grad():
            for res in active_resolutions:
                resolution_components[res].data.clamp_(0, 1)
        
        # Track best result based on activation score
        if activation.item() > best_activation:
            best_activation = activation.item()
            best_image = composite_image.detach().clone()
        
        # Update progress bar with more useful information
        progress_bar.set_description(
            f"Prototype {prototype_idx} | Activation: {activation.item():.4f} | " +
            f"Active Res: {len(active_resolutions)}/{len(resolutions)} | " +
            (f"Concentration: {concentration.item():.4f}" if 'concentration' in locals() else "")
        )
    
    # In the final step, visualize only the most significant regions
    # This helps focus the visualization on just the part we care about
    if activation_map is not None:
        try:
            with torch.no_grad():
                # Normalize activation map
                norm_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
                
                # Create a threshold mask to keep only the highly activated regions
                threshold = 0.7  # Keep only top 30% activations
                mask = (norm_map > threshold).float()
                
                # Upsample the mask to image size if needed
                if mask.shape != best_image.shape[2:]:
                    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                         size=(best_image.shape[2], best_image.shape[3]), 
                                         mode='bilinear', align_corners=False).squeeze()
                
                # Apply soft masking to focus on the part
                # Softly fade out less relevant regions
                soft_mask = F.interpolate(norm_map.unsqueeze(0).unsqueeze(0), 
                                         size=(best_image.shape[2], best_image.shape[3]), 
                                         mode='bilinear', align_corners=False).squeeze()
                
                # Apply the soft mask to focus the visualization
                # Keep some background context but highlight the important part
                best_image = best_image * (0.2 + 0.8 * soft_mask.view(1, 1, best_image.shape[2], best_image.shape[3]))
        except Exception as e:
            print(f"Warning: Could not apply final part-focusing: {e}")
    
    # Recreate final composite image from components
    final_image = torch.zeros(1, 3, img_size, img_size, device=device)
    components_dict = {}
    
    for res in resolutions:
        component = resolution_components[res].detach().cpu()
        
        # Upsample to img_size for the final dict
        if res < img_size:
            upsampled = F.interpolate(component, size=(img_size, img_size), 
                                      mode='bilinear', align_corners=False)
        else:
            upsampled = component
        
        components_dict[res] = upsampled
        final_image += upsampled.to(device)
    
    # Clamp final image
    final_image = torch.clamp(final_image, 0, 1).cpu()
    
    # Return best image if it's better than the final one
    if best_image is not None:
        final_image = best_image.cpu()
    
    return final_image, components_dict

def multi_resolution_prototype_activation(model, prototype_idx, device='cuda', iterations=500, 
                                         learning_rate=0.1, img_size=224, alpha_l1=0.0001,
                                         alpha_tv=0.001, blur_every=10, blur_sigma=0.5,
                                         normalize_mean=(0.485, 0.456, 0.406), 
                                         normalize_std=(0.229, 0.224, 0.225),
                                         resolutions=None):
    """
    Generate an image that maximally activates a specific prototype using curriculum learning.
    
    Args:
        model: The PIPNet model
        prototype_idx: Index of the prototype to maximize
        device: Device to run computations on
        iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        img_size: Size of the generated image
        alpha_l1: L1 regularization strength (sparsity)
        alpha_tv: Total variation regularization strength (smoothness)
        blur_every: Apply Gaussian blur every N iterations
        blur_sigma: Sigma parameter for Gaussian blur
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization
        resolutions: List of resolutions to use. If None, defaults to [1, 2, 4, 8, 16, 32, 64, 128, img_size]
        
    Returns:
        Tuple containing the final composite image and a dictionary of resolution components
    """
    import torch
    import torch.nn.functional as F
    import torchvision
    from tqdm import tqdm
    
    model.eval()
    model = model.to(device)
    
    # Set up resolutions if not provided
    if resolutions is None:
        # Create a progression of resolutions
        resolutions = [1, 2, 4, 8, 16, 32, 64, 128]
        if img_size not in resolutions:
            resolutions.append(img_size)
    
    # Sort resolutions from low to high for curriculum learning
    resolutions = sorted(resolutions)
    
    # Create multi-resolution components
    resolution_components = {}
    for res in resolutions:
        # Initialize with small random values for stability
        resolution_components[res] = torch.randn(1, 3, res, res, device=device) * 0.01
        resolution_components[res].requires_grad_(True)
    
    # Create optimizers for each resolution component
    optimizers = {
        res: torch.optim.Adam([resolution_components[res]], lr=0.0)  # Start with zero learning rate
        for res in resolutions
    }
    
    # Create Gaussian blur function
    def gaussian_blur(img, kernel_size, sigma):
        padding = kernel_size // 2
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        
        x = torch.arange(-padding, padding+1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        channels = img.shape[1]
        kernel_h = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        kernel_v = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
        
        img_blurred = F.conv2d(img_padded, kernel_h, groups=channels)
        img_blurred = F.conv2d(img_blurred, kernel_v, groups=channels)
        
        return img_blurred
    
    # Define transforms with curriculum learning (starting with milder transforms)
    def get_transforms(progress):
        """Get transforms with intensity based on optimization progress"""
        # Scale between 0 and 1
        p = min(1.0, progress / 0.7)  # Full strength at 70% of iterations
        
        jitter_strength = 0.05 + 0.3 * p
        translate_strength = 0.02 + 0.08 * p
        blur_min = 0.1
        blur_max = 0.5 + 1.5 * p
        
        jitter = torchvision.transforms.ColorJitter(
            brightness=jitter_strength, 
            contrast=jitter_strength, 
            saturation=jitter_strength, 
            hue=jitter_strength/2
        )
        translate = torchvision.transforms.RandomAffine(
            degrees=0, 
            translate=(translate_strength, translate_strength)
        )
        blur = torchvision.transforms.GaussianBlur(
            3, 
            sigma=(blur_min, blur_max)
        )
        
        return jitter, translate, blur
    
    # Calculate curriculum milestones
    # Lower resolutions start earlier, higher resolutions later
    curriculum_schedule = {}
    active_resolutions = []
    
    # Create curriculum schedule for introducing resolutions
    for i, res in enumerate(resolutions):
        # Progressive introduction of resolutions
        # Start with lowest resolution, gradually add higher ones
        start_iteration = int((i / len(resolutions)) * (iterations * 0.7))  # Introduce all by 70% of iterations
        curriculum_schedule[res] = start_iteration
    
    # Create curriculum for regularization strengths
    l1_schedule = lambda t: alpha_l1 * min(1.0, (t / iterations) * 2)  # Gradually increase L1
    tv_schedule = lambda t: alpha_tv * min(1.0, (t / iterations) * 3)   # Gradually increase TV
    
    # Progress bar
    progress_bar = tqdm(range(iterations))
    
    # Track best activation and corresponding image
    best_activation = -float('inf')
    best_image = None
    
    # Optimization loop
    for i in progress_bar:
        # Update curriculum - activate resolutions according to schedule
        for res in resolutions:
            if i >= curriculum_schedule[res] and res not in active_resolutions:
                active_resolutions.append(res)
                # Start with higher learning rate for lower resolutions
                scale_factor = 1.0 - (0.5 * (resolutions.index(res) / len(resolutions)))
                optimizers[res].param_groups[0]['lr'] = learning_rate * scale_factor
        
        # Apply Gaussian blur to components occasionally
        if i % blur_every == 0 and i > 0:
            with torch.no_grad():
                for res in active_resolutions:
                    if res > 1:  # No need to blur 1x1 components
                        kernel_size = min(int(blur_sigma * 4) + 1, res)
                        if kernel_size % 2 == 0:  # Make sure kernel size is odd
                            kernel_size += 1
                        kernel_size = max(3, min(kernel_size, res))  # Ensure kernel is at least 3 and fits in the image
                        if kernel_size < res:  # Only blur if kernel fits
                            resolution_components[res].data = gaussian_blur(
                                resolution_components[res].data, kernel_size, blur_sigma
                            )
        
        # Get transforms based on curriculum progress
        jitter, translate, blur = get_transforms(i / iterations)
        
        # Create composite image by upsampling and summing active components
        composite_image = torch.zeros(1, 3, img_size, img_size, device=device)
        for res in active_resolutions:
            if res < img_size:
                upsampled = F.interpolate(resolution_components[res], size=(img_size, img_size), 
                                          mode='bilinear', align_corners=False)
            else:
                upsampled = resolution_components[res]
            composite_image += upsampled
        
        # Clamp composite image to valid range
        composite_image = torch.clamp(composite_image, 0, 1)
        
        # Apply transforms in curriculum fashion
        # Early iterations: minimal transforms
        # Later iterations: stronger transforms for robustness
        transformed_image = composite_image
        
        # Apply transforms with probability increasing over time
        p_transform = min(0.9, (i / iterations) * 1.5)  # Max 90% chance, reach full prob after 60% iterations
        
        if torch.rand(1).item() < p_transform:
            transformed_image = blur(transformed_image)
        if torch.rand(1).item() < p_transform:
            transformed_image = jitter(transformed_image)
        if torch.rand(1).item() < p_transform:
            transformed_image = translate(transformed_image)
        
        # Normalize input image for the model
        mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        normalized_image = (transformed_image - mean) / std
        
        # Forward pass through the model
        features, pooled, _ = model(normalized_image, inference=False)
        
        # Get activation for target prototype
        target = torch.zeros_like(pooled[0])
        target[prototype_idx] = 1
        activation = torch.nn.functional.cross_entropy(pooled[0], target[0].long())
        
        # Loss is negative activation (since we want to maximize)
        loss = -activation
        
        # Add regularization with curriculum (progressive strengthening)
        # L1 regularization - weighted by resolution (stronger penalty for high-frequency components)
        current_l1_strength = l1_schedule(i)
        l1_losses = 0
        for res in active_resolutions:
            # Higher penalty for higher resolutions - curriculum within curriculum
            resolution_factor = (res / img_size) ** 0.5  # Square root to moderate the effect
            # Frequency factor - stronger regularization for high frequencies in later iterations
            frequency_factor = 1.0 + (resolutions.index(res) / len(resolutions)) * (i / iterations) * 3
            
            l1_losses += current_l1_strength * resolution_factor * frequency_factor * torch.abs(resolution_components[res]).sum()
        
        # Total variation regularization on the composite image
        # Gradually increase TV regularization as iterations progress
        current_tv_strength = tv_schedule(i)
        tv_h = torch.abs(composite_image[:, :, 1:, :] - composite_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(composite_image[:, :, :, 1:] - composite_image[:, :, :, :-1]).sum()
        tv_loss = current_tv_strength * (tv_h + tv_w)
        
        # Total loss
        total_loss = loss + l1_losses + tv_loss
        
        # Zero all gradients for active optimizers
        for res in active_resolutions:
            optimizers[res].zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Step active optimizers
        for res in active_resolutions:
            optimizers[res].step()
        
        # Clamp component values to valid range
        with torch.no_grad():
            for res in active_resolutions:
                resolution_components[res].data.clamp_(0, 1)
        
        # Track best result
        if activation.item() > best_activation:
            best_activation = activation.item()
            # Create a detached copy of the composite image
            best_image = composite_image.detach().clone()
        
        # Update progress bar
        progress_bar.set_description(f"Prototype {prototype_idx} | Activation: {activation.item():.4f} | Active Res: {len(active_resolutions)}/{len(resolutions)}")
    
    # Recreate final composite image from components
    final_image = torch.zeros(1, 3, img_size, img_size, device=device)
    components_dict = {}
    
    for res in resolutions:
        component = resolution_components[res].detach().cpu()
        
        # Upsample to img_size for the final dict
        if res < img_size:
            upsampled = F.interpolate(component, size=(img_size, img_size), 
                                      mode='bilinear', align_corners=False)
        else:
            upsampled = component
        
        components_dict[res] = upsampled
        final_image += upsampled.to(device)
    
    # Clamp final image
    final_image = torch.clamp(final_image, 0, 1).cpu()
    
    # Return best image if it's better than the final one
    if best_image is not None:
        final_image = best_image.cpu()
    
    return final_image, components_dict

def visualize_multi_resolution_prototype(model, prototype_idx, device='cuda', **kwargs):
    """
    Generate and visualize a multi-resolution image that maximally activates a specific prototype.
    
    Args:
        model: The PIPNet model
        prototype_idx: Index of the prototype to visualize
        device: Device to run computations on
        **kwargs: Additional arguments for multi_resolution_prototype_activation
        
    Returns:
        Matplotlib figure with the visualization
    """
    # Generate image using multi-resolution approach
    composite_image, components_dict = multi_resolution_prototype_activation(
        model, prototype_idx, device, **kwargs
    )
    
    # Get sorted resolutions for visualization
    resolutions = sorted(components_dict.keys())
    
    # Select a subset of resolutions to display if there are too many
    if len(resolutions) > 6:
        # Choose representative resolutions
        indices = np.linspace(0, len(resolutions)-1, 6, dtype=int)
        selected_resolutions = [resolutions[i] for i in indices]
    else:
        selected_resolutions = resolutions
    
    # Create figure for visualization
    n_cols = min(4, len(selected_resolutions) + 1)  # +1 for composite image
    n_rows = (len(selected_resolutions) + 1 + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot composite image first
    img_np = composite_image[0].permute(1, 2, 0).numpy()
    axes[0].imshow(img_np)
    axes[0].set_title(f"Composite Image\nPrototype {prototype_idx}")
    axes[0].axis('off')
    
    # Plot selected resolution components
    for i, res in enumerate(selected_resolutions):
        component = components_dict[res]
        img_np = component[0].permute(1, 2, 0).numpy()
        axes[i+1].imshow(img_np)
        axes[i+1].set_title(f"Resolution {res}x{res}")
        axes[i+1].axis('off')
    
    # Hide any unused axes
    for i in range(len(selected_resolutions) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig, composite_image, components_dict

def visualize_multi_resolution_evolution(model, prototype_idx, device='cuda', 
                                         n_steps=5, iterations_per_step=100, **kwargs):
    """
    Visualize the evolution of the multi-resolution components over time.
    
    Args:
        model: The PIPNet model
        prototype_idx: Index of the prototype to visualize
        device: Device to run computations on
        n_steps: Number of steps to visualize
        iterations_per_step: Number of iterations per step
        **kwargs: Additional arguments for multi_resolution_prototype_activation
        
    Returns:
        Matplotlib figure with the visualization
    """
    # Set total iterations
    total_iterations = n_steps * iterations_per_step
    kwargs['iterations'] = total_iterations
    
    # Track intermediate results
    intermediate_results = []
    original_iterations = kwargs.get('iterations', 500)
    
    # Define a hook function to capture intermediate results
    def hook_fn(i, composite_image, components_dict):
        if (i+1) % iterations_per_step == 0:
            # Make a deep copy of the current state
            current_composite = composite_image.detach().clone().cpu()
            current_components = {
                res: comp.detach().clone().cpu() 
                for res, comp in components_dict.items()
            }
            intermediate_results.append((current_composite, current_components))
            return True  # Continue optimization
        return False  # Continue optimization
    
    # Run optimization with hook
    final_image, final_components = multi_resolution_prototype_activation_with_hook(
        model, prototype_idx, device, hook_fn=hook_fn, **kwargs
    )
    
    # Ensure we have exactly n_steps results
    if len(intermediate_results) != n_steps:
        print(f"Warning: Expected {n_steps} intermediate results, got {len(intermediate_results)}")
    
    # Get key resolutions to display
    all_resolutions = sorted(final_components.keys())
    
    # Create figure
    n_cols = min(len(all_resolutions) + 1, 5)  # +1 for composite image, max 5 cols
    fig, axes = plt.subplots(n_steps, n_cols, figsize=(4*n_cols, 4*n_steps))
    
    # Handle single row/column case
    if n_steps == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_steps == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot evolution
    for step, (composite, components) in enumerate(intermediate_results):
        # Plot composite image
        img_np = composite[0].permute(1, 2, 0).numpy()
        axes[step, 0].imshow(img_np)
        axes[step, 0].set_title(f"Step {step+1}\nComposite")
        axes[step, 0].axis('off')
        
        # Plot key resolutions
        for i, res in enumerate(all_resolutions[:n_cols-1]):
            if i+1 < n_cols:
                component = components[res]
                img_np = component[0].permute(1, 2, 0).numpy()
                axes[step, i+1].imshow(img_np)
                axes[step, i+1].set_title(f"Res {res}x{res}")
                axes[step, i+1].axis('off')
    
    plt.tight_layout()
    return fig, final_image, final_components

def multi_resolution_prototype_activation_with_hook(model, prototype_idx, device='cuda', hook_fn=None, **kwargs):
    """
    Extended version of multi_resolution_prototype_activation that calls a hook function
    during optimization to track intermediate results.
    
    Args:
        model: The PIPNet model
        prototype_idx: Index of the prototype to maximize
        device: Device to run computations on
        hook_fn: Hook function that takes (iteration, composite_image, components_dict) and returns 
                a boolean indicating whether to continue optimization
        **kwargs: Additional arguments as in multi_resolution_prototype_activation
        
    Returns:
        Same as multi_resolution_prototype_activation
    """
    model.eval()
    model = model.to(device)
    
    iterations = kwargs.get('iterations', 500)
    learning_rate = kwargs.get('learning_rate', 0.1)
    img_size = kwargs.get('img_size', 224)
    alpha_l1 = kwargs.get('alpha_l1', 0.0001)
    alpha_tv = kwargs.get('alpha_tv', 0.001)
    blur_every = kwargs.get('blur_every', 10)
    blur_sigma = kwargs.get('blur_sigma', 0.5)
    normalize_mean = kwargs.get('normalize_mean', (0.485, 0.456, 0.406))
    normalize_std = kwargs.get('normalize_std', (0.229, 0.224, 0.225))
    resolutions = kwargs.get('resolutions')
    
    # Set up resolutions if not provided
    if resolutions is None:
        resolutions = [1, 2, 4, 8, 16, 32, 64, 128]
        # Add img_size if not already in the list
        if img_size not in resolutions:
            resolutions.append(img_size)
        # Sort and filter resolutions that are too large
        resolutions = sorted([r for r in resolutions if r <= img_size])
    
    # Create multi-resolution components
    resolution_components = {}
    for res in resolutions:
        resolution_components[res] = torch.randn(1, 3, res, res, device=device) * 0.01
        resolution_components[res].requires_grad_(True)
    
    # Create optimizers for each resolution component
    optimizers = {
        res: torch.optim.Adam([resolution_components[res]], lr=learning_rate)
        for res in resolutions
    }
    
    # Gaussian blur function
    def gaussian_blur(img, kernel_size, sigma):
        # Same function as before...
        padding = kernel_size // 2
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        
        x = torch.arange(-padding, padding+1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        channels = img.shape[1]
        kernel_h = kernel.view(1, 1, 1, kernel_size).repeat(channels, 1, 1, 1)
        kernel_v = kernel.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
        
        img_blurred = F.conv2d(img_padded, kernel_h, groups=channels)
        img_blurred = F.conv2d(img_blurred, kernel_v, groups=channels)
        
        return img_blurred
    
    # Progress bar
    progress_bar = tqdm(range(iterations))
    
    # Track best activation
    best_activation = -float('inf')
    best_image = None
    
    # Optimization loop
    for i in progress_bar:
        # Apply blur occasionally
        if i % blur_every == 0 and i > 0:
            with torch.no_grad():
                for res in resolutions:
                    if res > 1:
                        kernel_size = min(blur_sigma * 4 + 1, res)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        kernel_size = max(3, min(kernel_size, res))
                        sigma = blur_sigma
                        resolution_components[res].data = gaussian_blur(
                            resolution_components[res].data, 
                            int(kernel_size), 
                            sigma
                        )
        
        # Create composite image
        composite_image = torch.zeros(1, 3, img_size, img_size, device=device)
        for res in resolutions:
            if res < img_size:
                upsampled = F.interpolate(resolution_components[res], size=(img_size, img_size), 
                                          mode='bilinear', align_corners=False)
            else:
                upsampled = resolution_components[res]
            composite_image += upsampled
        
        # Clamp composite image
        composite_image = torch.clamp(composite_image, 0, 1)
        
        # Call hook if provided
        if hook_fn is not None:
            # Create clean copies for the hook
            current_composite = composite_image.detach().clone()
            current_components = {
                res: comp.detach().clone()
                for res, comp in resolution_components.items()
            }
            
            # Call the hook
            should_continue = hook_fn(i, current_composite, current_components)
            if not should_continue:
                break
        
        # Normalize for the model
        mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        normalized_image = (composite_image - mean) / std
        
        # Forward pass
        _, pooled, _ = model(normalized_image, inference=False)
        activation = pooled[0, prototype_idx]
        
        # Loss calculation
        loss = -activation
        
        # Add regularization
        l1_losses = 0
        for res in resolutions:
            weight = (res / img_size) ** 0.5
            l1_losses += alpha_l1 * weight * torch.abs(resolution_components[res]).sum()
        
        tv_h = torch.abs(composite_image[:, :, 1:, :] - composite_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(composite_image[:, :, :, 1:] - composite_image[:, :, :, :-1]).sum()
        tv_loss = alpha_tv * (tv_h + tv_w)
        
        total_loss = loss + l1_losses + tv_loss
        
        # Backward and optimize
        for opt in optimizers.values():
            opt.zero_grad()
        
        total_loss.backward()
        
        for opt in optimizers.values():
            opt.step()
        
        # Clamp component values
        with torch.no_grad():
            for res in resolutions:
                resolution_components[res].data.clamp_(0, 1)
        
        # Track best result
        if activation.item() > best_activation:
            best_activation = activation.item()
            best_image = composite_image.detach().clone()
        
        # Update progress bar
        progress_bar.set_description(f"Prototype {prototype_idx} | Activation: {activation.item():.4f}")
    
    # Prepare final output
    final_image = torch.zeros(1, 3, img_size, img_size, device=device)
    components_dict = {}
    
    for res in resolutions:
        component = resolution_components[res].detach().cpu()
        
        if res < img_size:
            upsampled = F.interpolate(component, size=(img_size, img_size), 
                                      mode='bilinear', align_corners=False)
        else:
            upsampled = component
        
        components_dict[res] = upsampled
        final_image += upsampled.to(device)
    
    final_image = torch.clamp(final_image, 0, 1).cpu()
    
    # Use best image if better
    if best_image is not None:
        final_image = best_image.cpu()
    
    return final_image, components_dict
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pywt

def wavelet_prototype_activation(model, prototype_idx, device='cuda', iterations=500, 
                                learning_rate=0.1, img_size=224, alpha_l1=0.0001,
                                alpha_tv=0.001, blur_every=10, blur_sigma=0.5,
                                normalize_mean=(0.485, 0.456, 0.406), 
                                normalize_std=(0.229, 0.224, 0.225),
                                wavelet_type='db4',    # Wavelet family (db1/haar, db4, sym4, coif3, etc.)
                                wavelet_levels=4,      # Number of wavelet decomposition levels
                                wavelet_mode='zero',   # Padding mode for wavelet transform
                                optimize_approx=True,  # Whether to optimize approximation coefficients
                                optimize_details=True, # Whether to optimize detail coefficients
                                band_weights=None,     # Weights for different coefficient bands
                                color_correlation=True, # Encourage correlation between color channels
                                alpha_color=0.01):     # Weight for color correlation loss
    """
    Generate an image that maximally activates a specific prototype using wavelet decomposition.
    
    Args:
        model: The model to visualize
        prototype_idx: Index of the prototype to maximize
        device: Device to run computations on
        iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        img_size: Size of the generated image
        alpha_l1: L1 regularization strength (sparsity)
        alpha_tv: Total variation regularization strength (smoothness)
        blur_every: Apply Gaussian blur every N iterations
        blur_sigma: Sigma parameter for Gaussian blur
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization
        wavelet_type: Type of wavelet to use (haar, db4, sym8, etc.)
        wavelet_levels: Number of decomposition levels
        wavelet_mode: Padding mode for wavelet transform
        optimize_approx: Whether to optimize approximation coefficients
        optimize_details: Whether to optimize detail coefficients
        band_weights: Custom weights for different coefficient bands
        color_correlation: Whether to encourage correlation between color channels
        alpha_color: Weight for color correlation loss
        
    Returns:
        Tuple containing the final image and a dictionary with wavelet coefficients
    """
    model.eval()
    model = model.to(device)
    
    # Ensure image size is compatible with wavelet levels
    # Each level divides dimensions by 2, so we need img_size to be divisible by 2^wavelet_levels
    min_size = 2 ** wavelet_levels
    if img_size % min_size != 0:
        original_size = img_size
        img_size = ((img_size // min_size) + 1) * min_size
        print(f"Adjusting image size from {original_size} to {img_size} for compatibility with {wavelet_levels} wavelet levels")
    
    # Set up wavelet band weights if not provided
    if band_weights is None:
        # Default weights that emphasize mid-frequency details
        band_weights = {}
        # Approximation coefficient (lowest frequency)
        band_weights['approx'] = 1.0
        
        # Detail coefficients (horizontal, vertical, diagonal) at each level
        for level in range(1, wavelet_levels + 1):
            # Scale factor: higher weight for middle levels (mid-frequencies)
            # Lower weight for very low and very high frequencies
            if level <= wavelet_levels // 2:
                # Increasing weights for lower levels (higher frequencies)
                scale = 0.5 + 0.5 * (level / (wavelet_levels // 2))
            else:
                # Decreasing weights for higher levels (lower frequencies)
                scale = 1.0 - 0.5 * ((level - wavelet_levels // 2) / (wavelet_levels - wavelet_levels // 2))
            
            # Apply weights to each orientation (horizontal, vertical, diagonal)
            band_weights[f'detail_h_{level}'] = scale
            band_weights[f'detail_v_{level}'] = scale
            band_weights[f'detail_d_{level}'] = scale * 0.8  # Slightly lower weight for diagonal details
    
    # Create PyWavelets wavelet object
    wavelet = pywt.Wavelet(wavelet_type)
    
    # Initialize wavelet coefficients with small random values
    wavelet_coeffs = {}
    wavelet_coeffs_params = []  # List to store parameters for optimizer
    
    # Image will be optimized through its wavelet representation
    # We'll have coefficients for each color channel
    for channel in range(3):  # R, G, B channels
        # Initialize approximation coefficients (lowest frequency)
        approx_size = img_size // (2 ** wavelet_levels)
        if optimize_approx:
            wavelet_coeffs[f'approx_{channel}'] = torch.randn(1, approx_size, approx_size, device=device) * 0.01
            wavelet_coeffs[f'approx_{channel}'].requires_grad_(True)
            wavelet_coeffs_params.append(wavelet_coeffs[f'approx_{channel}'])
        else:
            # If not optimizing approximation, initialize with zeros
            wavelet_coeffs[f'approx_{channel}'] = torch.zeros(1, approx_size, approx_size, device=device)
        
        # Initialize detail coefficients at each level
        if optimize_details:
            for level in range(1, wavelet_levels + 1):
                # Size of detail coefficients at this level
                detail_size = img_size // (2 ** (wavelet_levels - level + 1))
                
                # Create detail coefficients for horizontal, vertical, and diagonal components
                for orientation in ['h', 'v', 'd']:  # horizontal, vertical, diagonal
                    coeff_name = f'detail_{orientation}_{level}_{channel}'
                    wavelet_coeffs[coeff_name] = torch.randn(1, detail_size, detail_size, device=device) * 0.01
                    wavelet_coeffs[coeff_name].requires_grad_(True)
                    wavelet_coeffs_params.append(wavelet_coeffs[coeff_name])
    
    # Create optimizer for all wavelet coefficients
    optimizer = torch.optim.Adam(wavelet_coeffs_params, lr=learning_rate)
    
    # Define a custom wavelet reconstruction that maintains gradients
    def wavelet_reconstruction(coeffs_dict, channel):
        """
        Reconstruct image channel from wavelet coefficients while preserving gradients.
        This custom implementation avoids the PyWavelets CPU-only limitation.
        """
        # Create a starting tensor for reconstruction
        reconstructed = torch.zeros((img_size, img_size), device=device)
        
        # We'll implement a simplified version of the wavelet reconstruction
        # that preserves gradients by using PyTorch operations
        
        # First, handle the approximation coefficients (lowest frequency)
        approx_coeff = coeffs_dict[f'approx_{channel}'].squeeze(0)
        # Upsample to full size
        scale_factor = img_size // approx_coeff.shape[0]
        upsampled_approx = F.interpolate(
            approx_coeff.unsqueeze(0).unsqueeze(0),
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        ).squeeze(0).squeeze(0)
        
        # Add approximation component to the reconstructed image
        reconstructed = reconstructed + upsampled_approx
        
        # Add detail coefficients at each level
        for level in range(1, wavelet_levels + 1):
            # Get detail coefficients for this level
            h_coeff = coeffs_dict[f'detail_h_{level}_{channel}'].squeeze(0)
            v_coeff = coeffs_dict[f'detail_v_{level}_{channel}'].squeeze(0)
            d_coeff = coeffs_dict[f'detail_d_{level}_{channel}'].squeeze(0)
            
            # Calculate scale factor for this level
            detail_size = h_coeff.shape[0]
            scale_factor = img_size // detail_size
            
            # Upsample detail coefficients
            upsampled_h = F.interpolate(
                h_coeff.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            upsampled_v = F.interpolate(
                v_coeff.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            upsampled_d = F.interpolate(
                d_coeff.unsqueeze(0).unsqueeze(0),
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
            
            # Weight the coefficients based on level (higher levels contribute less)
            level_weight = 1.0 / (2 ** (wavelet_levels - level))
            
            # Add detail components to the reconstructed image
            reconstructed = reconstructed + level_weight * (upsampled_h + upsampled_v + upsampled_d)
        
        return reconstructed
    
    # Function to apply Gaussian blur
    def gaussian_blur(img, kernel_size, sigma):
        padding = kernel_size // 2
        img_padded = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        
        x = torch.arange(-padding, padding+1, dtype=torch.float32, device=device)
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        kernel_h = kernel.view(1, 1, kernel_size, 1)
        kernel_v = kernel.view(1, 1, 1, kernel_size)
        
        img_blurred = F.conv2d(img_padded.unsqueeze(1), kernel_h, padding=0)
        img_blurred = F.conv2d(img_blurred, kernel_v, padding=0)
        
        return img_blurred.squeeze(1)
    
    # Transforms for augmentation
    jitter = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    translate = torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
    
    # Progress bar
    progress_bar = tqdm(range(iterations))
    
    # Track best activation and corresponding coefficients
    best_activation = -float('inf')
    best_coeffs = None
    
    # Track optimized image for logging
    current_img = None
    
    # Optimization loop
    for i in progress_bar:
        # Apply blur to coefficients occasionally
        if i % blur_every == 0 and i > 0:
            with torch.no_grad():
                for key in wavelet_coeffs:
                    if 'detail' in key:  # Only blur detail coefficients
                        level = int(key.split('_')[2])
                        # Apply stronger blur to higher frequencies
                        effective_sigma = blur_sigma * (wavelet_levels - level + 1) / wavelet_levels
                        
                        # Kernel size must be odd and at least 3
                        kernel_size = max(3, min(int(effective_sigma * 4 + 1), wavelet_coeffs[key].shape[2]))
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        
                        # Apply Gaussian blur while maintaining tensor shape
                        coeff_data = wavelet_coeffs[key].data
                        blurred = gaussian_blur(
                            coeff_data,
                            kernel_size,
                            effective_sigma
                        )
                        
                        # Ensure the shape is preserved
                        if blurred.dim() == 2:
                            blurred = blurred.unsqueeze(0)
                        if blurred.dim() == 3 and wavelet_coeffs[key].dim() == 4:
                            blurred = blurred.unsqueeze(1)
                            
                        wavelet_coeffs[key].data = blurred
        
        # Reconstruct image from wavelet coefficients
        # Perform reconstruction for each channel separately
        channels = []
        for channel in range(3):
            channel_img = wavelet_reconstruction(wavelet_coeffs, channel)
            channels.append(channel_img.unsqueeze(0))  # Add channel dimension
        
        # Combine channels into an RGB image
        composite_image = torch.cat(channels, dim=0).unsqueeze(0)  # Shape: [1, 3, H, W]
        
        # Apply band weights
        # This is done by scaling the respective coefficients before reconstruction
        # Already handled in the wavelet_coeffs initialization and updates
        
        # Store the image for visualization
        current_img = composite_image.detach()
        
        # Clamp to valid range
        composite_image = torch.clamp(composite_image, 0, 1)
        
        # Apply augmentation occasionally
        if i % 5 == 0:
            composite_image = jitter(composite_image)
            composite_image = translate(composite_image)
        
        # Normalize input image for the model
        mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        normalized_image = (composite_image - mean) / std
        
        # Forward pass through the model
        features, pooled, _ = model(normalized_image, inference=False)
        
        # Get activation for target prototype
        # activation = pooled[0, prototype_idx]
        target = torch.zeros_like(pooled[0])
        target[ prototype_idx] = 1
        activation = torch.nn.functional.cross_entropy(pooled[0], target[0].long())
        
        # Loss is negative activation (since we want to maximize)
        loss = -activation
        
        # Add regularization
        
        # L1 regularization on wavelet coefficients
        l1_loss = 0
        for key, coeff in wavelet_coeffs.items():
            if coeff.requires_grad:
                # Extract level for weighting if available
                if 'detail' in key:
                    level = int(key.split('_')[2])
                    # Stronger regularization for higher frequencies (lower levels)
                    level_weight = 1.0 - (level / wavelet_levels)
                else:
                    level_weight = 0.5  # Lower weight for approximation coefficients
                
                # Add weighted L1 loss
                weight = alpha_l1 * level_weight
                l1_loss += weight * torch.abs(coeff).sum()
        
        # Total variation regularization on the composite image
        tv_h = torch.abs(composite_image[:, :, 1:, :] - composite_image[:, :, :-1, :]).sum()
        tv_w = torch.abs(composite_image[:, :, :, 1:] - composite_image[:, :, :, :-1]).sum()
        tv_loss = alpha_tv * (tv_h + tv_w)
        
        # Color correlation loss to encourage natural colors
        color_loss = 0
        if color_correlation:
            # Encourage correlation between color channels for natural images
            r = composite_image[:, 0]
            g = composite_image[:, 1]
            b = composite_image[:, 2]
            
            # Calculate correlation between channels
            # This encourages the RGB channels to have similar structures
            # which is common in natural images
            rg_diff = torch.abs(r - g).mean()
            rb_diff = torch.abs(r - b).mean()
            gb_diff = torch.abs(g - b).mean()
            
            # We want some correlation but not perfect correlation
            # So we use a target correlation value
            target_diff = 0.1
            color_loss = alpha_color * (
                torch.abs(rg_diff - target_diff) + 
                torch.abs(rb_diff - target_diff) + 
                torch.abs(gb_diff - target_diff)
            )
        
        # Total loss
        total_loss = loss + l1_loss + tv_loss + color_loss
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        total_loss.backward()
        
        # Step optimizer
        optimizer.step()
        
        # Clamp coefficient values to reasonable range
        with torch.no_grad():
            for key, coeff in wavelet_coeffs.items():
                if coeff.requires_grad:
                    # Different clamping for approximation and detail coefficients
                    if 'approx' in key:
                        # Approximation coefficients can have a wider range
                        coeff.data.clamp_(-1.0, 1.0)
                    else:
                        # Detail coefficients typically have smaller magnitudes
                        level = int(key.split('_')[2])
                        # Tighter bounds for higher frequencies
                        bound = 0.5 / (wavelet_levels - level + 1)
                        coeff.data.clamp_(-bound, bound)
        
        # Track best result
        if activation.item() > best_activation:
            best_activation = activation.item()
            # Create a deep copy of the current coefficients
            best_coeffs = {k: v.detach().clone() for k, v in wavelet_coeffs.items()}
        
        # Update progress bar
        progress_bar.set_description(f"Prototype {prototype_idx} | Activation: {activation.item():.4f}")
    
    # Use best coefficients if available
    if best_coeffs is not None:
        wavelet_coeffs = best_coeffs
    
    # Final reconstruction
    channels = []
    for channel in range(3):
        channel_img = wavelet_reconstruction(wavelet_coeffs, channel)
        channels.append(channel_img.unsqueeze(0))
    
    final_image = torch.cat(channels, dim=0).unsqueeze(0)
    final_image = torch.clamp(final_image, 0, 1)
    
    return final_image.cpu(), wavelet_coeffs

def visualize_wavelet_prototype(model, prototype_idx, device='cuda', **kwargs):
    """
    Generate and visualize a wavelet-based image that maximally activates a specific prototype.
    
    Args:
        model: The model to visualize
        prototype_idx: Index of the prototype to visualize
        device: Device to run computations on
        **kwargs: Additional arguments for wavelet_prototype_activation
        
    Returns:
        Matplotlib figure with the visualization
    """
    # Generate image using wavelet approach
    final_image, wavelet_coeffs = wavelet_prototype_activation(
        model, prototype_idx, device, **kwargs
    )
    
    # Get wavelet levels from coefficients
    keys = list(wavelet_coeffs.keys())
    levels = set()
    for key in keys:
        if 'detail' in key:
            level = int(key.split('_')[2])
            levels.add(level)
    wavelet_levels = max(levels) if levels else 0
    
    # Helper function for visualization
    def tensor_to_display(tensor):
        """Convert tensor to numpy array suitable for display"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        return tensor.numpy()
    
    # Normalize function for visualization
    def normalize(x):
        if isinstance(x, torch.Tensor):
            x = tensor_to_display(x)
        if np.max(x) == np.min(x):
            return np.zeros_like(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the final image
    ax1 = fig.add_subplot(2, 3, 1)
    img_np = tensor_to_display(final_image[0].permute(1, 2, 0))
    ax1.imshow(np.clip(img_np, 0, 1))
    ax1.set_title(f"Prototype {prototype_idx}")
    ax1.axis('off')
    
    # Plot approximation coefficients if available
    ax2 = fig.add_subplot(2, 3, 2)
    if all(f'approx_{c}' in wavelet_coeffs for c in range(3)):
        # Combine RGB channels for approximation
        try:
            approx_r = tensor_to_display(wavelet_coeffs[f'approx_0'])
            approx_g = tensor_to_display(wavelet_coeffs[f'approx_1'])
            approx_b = tensor_to_display(wavelet_coeffs[f'approx_2'])
            
            # Normalize each channel for visualization
            approx_rgb = np.stack([normalize(approx_r), normalize(approx_g), normalize(approx_b)], axis=2)
            ax2.imshow(approx_rgb)
            ax2.set_title("Approximation Coefficients")
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error displaying approx: {str(e)}", 
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No approximation coefficients", 
                ha='center', va='center', transform=ax2.transAxes)
    ax2.axis('off')
    
    # Plot detail coefficients at different levels
    # Choose a few representative levels to display
    if wavelet_levels > 3:
        display_levels = [1, wavelet_levels // 2, wavelet_levels]
    else:
        display_levels = list(range(1, wavelet_levels + 1)) if wavelet_levels > 0 else []
    
    for i, level in enumerate(display_levels):
        ax = fig.add_subplot(2, 3, i + 3)
        
        try:
            # Combine horizontal, vertical, and diagonal details for visualization
            # Using only one channel (red) for clarity
            h_coeff = tensor_to_display(wavelet_coeffs[f'detail_h_{level}_0'])
            v_coeff = tensor_to_display(wavelet_coeffs[f'detail_v_{level}_0'])
            d_coeff = tensor_to_display(wavelet_coeffs[f'detail_d_{level}_0'])
            
            # Normalize for visualization
            h_coeff = normalize(h_coeff)
            v_coeff = normalize(v_coeff)
            d_coeff = normalize(d_coeff)
            
            # Create RGB image where R=horizontal, G=vertical, B=diagonal
            detail_rgb = np.stack([h_coeff, v_coeff, d_coeff], axis=2)
            ax.imshow(detail_rgb)
            ax.set_title(f"Level {level} Detail Coefficients")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error displaying level {level}: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Plot frequency spectrum for comparison
    ax6 = fig.add_subplot(2, 3, 6)
    try:
        # Convert image to grayscale for FFT
        gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted) + 1)
        # Normalize for visualization
        magnitude_spectrum = normalize(magnitude_spectrum)
        ax6.imshow(magnitude_spectrum, cmap='viridis')
        ax6.set_title("Frequency Spectrum")
    except Exception as e:
        ax6.text(0.5, 0.5, f"Error computing spectrum: {str(e)}", 
               ha='center', va='center', transform=ax6.transAxes)
    ax6.axis('off')
    
    plt.tight_layout()
    return fig, final_image, wavelet_coeffs

def analyze_wavelet_components(wavelet_coeffs, wavelet_type='db4'):

    """
    Analyze the energy distribution in different wavelet components.
    
    Args:
        wavelet_coeffs: Dictionary of wavelet coefficients
        wavelet_type: Type of wavelet used
        
    Returns:
        Dictionary with energy statistics
    """
    # Extract wavelet levels
    keys = list(wavelet_coeffs.keys())
    levels = set()
    for key in keys:
        if 'detail' in key:
            level = int(key.split('_')[2])
            levels.add(level)
    wavelet_levels = max(levels)
    
    # Calculate energy in each component
    energy = {}
    
    # Approximation energy
    approx_energy = 0
    for channel in range(3):
        approx_energy += torch.sum(wavelet_coeffs[f'approx_{channel}']**2).item()
    energy['approximation'] = approx_energy
    
    # Detail energy by level
    for level in range(1, wavelet_levels + 1):
        level_energy = 0
        # Sum energy across all orientations and channels
        for orientation in ['h', 'v', 'd']:
            for channel in range(3):
                coeff_name = f'detail_{orientation}_{level}_{channel}'
                level_energy += torch.sum(wavelet_coeffs[coeff_name]**2).item()
        energy[f'level_{level}'] = level_energy
    
    # Total energy
    total_energy = sum(energy.values())
    
    # Normalized energy (percentage)
    energy_percent = {k: 100 * v / total_energy for k, v in energy.items()}
    
    # Add normalized energy to results
    for k, v in energy_percent.items():
        energy[f'{k}_percent'] = v
    
    return energy


def fourier_prototype_activation(model, prototype_idx, device='cuda', iterations=500, 
                              learning_rate=0.1, img_size=224, alpha_tv=0.001,
                              normalize_mean=(0.485, 0.456, 0.406), 
                              normalize_std=(0.229, 0.224, 0.225),
                              frequency_reg=0.01,          # Regularization for frequency coefficients
                              frequency_bias='mid',        # 'low', 'mid', or 'high' frequency bias
                              normalize_spectrum=True,     # Keep the spectrum normalized
                              color_correlation=True,      # Whether to encourage color correlation
                              alpha_color=0.01,            # Weight for color correlation loss
                              phase_conservation=0.01,     # Phase conservation regularization
                              band_sparsity=0.05):         # Sparsity within frequency bands
    """
    Generate an image that maximally activates a specific prototype by optimizing in the Fourier domain.
    
    Args:
        model: The model to visualize
        prototype_idx: Index of the prototype to maximize
        device: Device to run computations on
        iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        img_size: Size of the generated image
        alpha_tv: Total variation regularization strength (smoothness)
        normalize_mean: Mean values for normalization
        normalize_std: Standard deviation values for normalization
        frequency_reg: Regularization for frequency coefficients
        frequency_bias: Whether to bias toward 'low', 'mid', or 'high' frequencies
        normalize_spectrum: Keep the spectrum normalized
        color_correlation: Whether to encourage correlation between color channels
        alpha_color: Weight for color correlation loss
        phase_conservation: Phase conservation regularization
        band_sparsity: Sparsity within frequency bands
        
    Returns:
        Tuple containing the final image and frequency components
    """
    model.eval()
    model = model.to(device)
    
    # Ensure image size is a power of 2 for more efficient FFT
    if img_size & (img_size - 1) != 0:
        # Find the next power of 2
        img_size = 2 ** math.ceil(math.log2(img_size))
        print(f"Adjusting image size to {img_size} for more efficient FFT")
    
    # Initialize frequency-domain representation
    # We need to create a complex tensor for each color channel
    frequency_components = []
    for c in range(3):  # RGB channels
        # Create real and imaginary parts
        real_part = torch.randn(img_size, img_size // 2 + 1, device=device) * 0.01
        imag_part = torch.randn(img_size, img_size // 2 + 1, device=device) * 0.01
        
        # For the DC component and Nyquist frequencies, the imaginary part must be zero
        imag_part[0, 0] = 0
        if img_size % 2 == 0:
            imag_part[img_size // 2, 0] = 0
        
        # Create complex tensor
        complex_tensor = torch.complex(real_part, imag_part)
        
        # Set up for gradient tracking
        real_part.requires_grad_(True)
        imag_part.requires_grad_(True)
        
        frequency_components.append((real_part, imag_part))
    
    # Create optimizer
    params = []
    for real_part, imag_part in frequency_components:
        params.extend([real_part, imag_part])
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Set up frequency weighting based on bias
    h, w = img_size, img_size // 2 + 1
    freq_weights = torch.zeros(h, w, device=device)
    
    for y in range(h):
        for x in range(w):
            # Calculate normalized frequency (0 to 1)
            y_freq = min(y, h - y) / (h / 2)
            x_freq = x / w
            
            # Normalized distance from DC component (0,0)
            dist = torch.norm(torch.tensor([y_freq, x_freq]))
            # dist = torch.sqrt(y_freq**2 + x_freq**2)
            
            # Apply frequency bias
            if frequency_bias == 'low':
                # Favor low frequencies with exponential decay
                weight = torch.exp(-3.0 * dist)
            elif frequency_bias == 'high':
                # Favor high frequencies
                weight = 1.0 - torch.exp(-3.0 * dist)
            else:  # 'mid' (default)
                # Gaussian bump centered at mid frequencies
                weight = torch.exp(-(dist - 0.3)**2 / 0.1)
            
            freq_weights[y, x] = weight
    
    # Functions for FFT/IFFT operations
    def to_complex(real, imag):
        return torch.complex(real, imag)
    
    def from_complex(complex_tensor):
        return complex_tensor.real, complex_tensor.imag
    
    def create_image_from_frequency(frequency_components):
        channels = []
        for c, (real_part, imag_part) in enumerate(frequency_components):
            # Create complex tensor
            complex_tensor = to_complex(real_part, imag_part)
            
            # Normalize spectrum if requested
            if normalize_spectrum:
                # Calculate current magnitude
                magnitude = torch.abs(complex_tensor)
                
                # Skip DC component in normalization
                mean_magnitude = torch.mean(magnitude[1:])
                if mean_magnitude > 0:
                    scale = 1.0 / mean_magnitude
                    
                    # Keep phase but normalize magnitude
                    normalized = complex_tensor * scale
                    
                    # Preserve DC component
                    normalized[0, 0] = complex_tensor[0, 0]
                    
                    complex_tensor = normalized
            
            # Inverse FFT to get spatial image
            spatial = torch.fft.irfft2(complex_tensor, s=(img_size, img_size))
            
            # Optionally process the spatial domain image
            # (e.g., add constraints, clipping, etc.)
            
            channels.append(spatial)
        
        # Stack channels to create RGB image
        image = torch.stack(channels, dim=0).unsqueeze(0)
        
        # Normalize pixel values to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    # Function to apply Gaussian blur in frequency domain
    def frequency_domain_blur(real_part, imag_part, sigma=1.0):
        complex_tensor = to_complex(real_part, imag_part)
        
        # Create Gaussian filter in frequency domain
        h, w = real_part.shape
        filter_mask = torch.zeros(h, w, device=device)
        
        for y in range(h):
            for x in range(w):
                # Calculate normalized frequency (0 to 1)
                y_freq = min(y, h - y) / (h / 2)
                x_freq = x / w
                
                # Normalized distance from DC component (0,0)
                dist = torch.norm(torch.tensor([y_freq, x_freq]))
                # dist = torch.sqrt(y_freq**2 + x_freq**2)
                
                # Gaussian filter (inverse - high frequencies get attenuated more)
                filter_value = torch.exp(-dist**2 / (2 * sigma**2))
                filter_mask[y, x] = filter_value
        
        # Apply filter
        filtered = complex_tensor * filter_mask
        
        return from_complex(filtered)
    
    # Create transformations for data augmentation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        torchvision.transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02))
    ])
    
    # Progress bar
    progress_bar = tqdm(range(iterations))
    
    # Track best activation and corresponding coefficients
    best_activation = -float('inf')
    best_coefficients = None
    
    # Optimization loop
    for i in progress_bar:
        # Apply occasional frequency-domain blur
        if i % 50 == 0 and i > 0:
            with torch.no_grad():
                for c in range(3):
                    real_blurred, imag_blurred = frequency_domain_blur(
                        frequency_components[c][0],
                        frequency_components[c][1],
                        sigma=1.0
                    )
                    frequency_components[c][0].data = real_blurred
                    frequency_components[c][1].data = imag_blurred
        
        # Create image from frequency components
        image = create_image_from_frequency(frequency_components)
        
        # Apply transformations occasionally
        if i % 5 == 0:
            image = transform(image)
        
        # Ensure valid range
        image = torch.clamp(image, 0, 1)
        
        # Normalize input for the model
        mean = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        normalized_image = (image - mean) / std
        
        # Forward pass
        features, pooled, _ = model(normalized_image, inference=False)
        
        target = torch.zeros_like(pooled[0])
        target[ prototype_idx] = 1
        activation = torch.nn.functional.cross_entropy(pooled[0], target[0].long())
        
        # Loss is negative activation (since we want to maximize)
        loss = -activation
        
        # Add regularization terms
        
        # 1. Frequency regularization (penalize unwanted frequencies)
        freq_loss = 0
        for c in range(3):
            real_part, imag_part = frequency_components[c]
            complex_tensor = to_complex(real_part, imag_part)
            magnitude = torch.abs(complex_tensor)
            
            # Apply frequency weighting (bias towards desired frequencies)
            weighted_magnitude = magnitude * (1.0 - freq_weights)
            freq_loss += frequency_reg * torch.sum(weighted_magnitude)
        
        # 2. Total variation loss (spatial smoothness)
        tv_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).sum()
        tv_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).sum()
        tv_loss = alpha_tv * (tv_h + tv_w)
        
        # 3. Color correlation loss
        color_loss = 0
        if color_correlation:
            # Encourage correlation between color channels for natural images
            r = image[:, 0]
            g = image[:, 1]
            b = image[:, 2]
            
            # Calculate correlation between channels
            rg_diff = torch.abs(r - g).mean()
            rb_diff = torch.abs(r - b).mean()
            gb_diff = torch.abs(g - b).mean()
            
            # We want some correlation but not perfect correlation
            # So we use a target correlation value
            target_diff = 0.1
            color_loss = alpha_color * (
                torch.abs(rg_diff - target_diff) + 
                torch.abs(rb_diff - target_diff) + 
                torch.abs(gb_diff - target_diff)
            )
        
        # 4. Phase conservation - encourage similar phases across color channels
        phase_loss = 0
        if phase_conservation > 0:
            # Extract phases
            phases = []
            for c in range(3):
                complex_tensor = to_complex(frequency_components[c][0], frequency_components[c][1])
                phase = torch.angle(complex_tensor)
                phases.append(phase)
            
            # Calculate phase differences between channels
            phase_diff_rg = torch.abs(phases[0] - phases[1]).mean()
            phase_diff_rb = torch.abs(phases[0] - phases[2]).mean()
            phase_diff_gb = torch.abs(phases[1] - phases[2]).mean()
            
            # We want some phase alignment for structural coherence
            phase_loss = phase_conservation * (phase_diff_rg + phase_diff_rb + phase_diff_gb)
        
        # 5. Band sparsity - encourage fewer active frequencies within bands
        sparsity_loss = 0
        if band_sparsity > 0:
            for c in range(3):
                complex_tensor = to_complex(frequency_components[c][0], frequency_components[c][1])
                magnitude = torch.abs(complex_tensor)
                
                # Define bands (low, mid, high)
                h, w = magnitude.shape
                low_band = magnitude[:h//8, :w//8]
                mid_band = magnitude[h//8:h//2, w//8:w//2]
                high_band = magnitude[h//2:, w//2:]
                
                # L1 regularization within each band
                sparsity_loss += band_sparsity * (
                    0.2 * torch.sum(low_band) +  # Less sparsity for low frequencies
                    0.5 * torch.sum(mid_band) +
                    1.0 * torch.sum(high_band)   # More sparsity for high frequencies
                )
        
        # Total loss
        total_loss = loss + freq_loss + tv_loss + color_loss + phase_loss + sparsity_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track best result
        if activation.item() > best_activation:
            best_activation = activation.item()
            # Store a copy of the current frequency components
            best_coefficients = [
                (real.detach().clone(), imag.detach().clone())
                for real, imag in frequency_components
            ]
        
        # Update progress bar
        progress_bar.set_description(f"Prototype {prototype_idx} | Activation: {activation.item():.4f}")
    
    # Use best coefficients if available
    if best_coefficients is not None:
        frequency_components = best_coefficients
    
    # Generate final image
    final_image = create_image_from_frequency(frequency_components)
    final_image = torch.clamp(final_image, 0, 1)
    
    return final_image.cpu(), frequency_components

def visualize_fourier_components(image, frequency_components, num_bands=5):
    """
    Visualize the Fourier components of the optimized image.
    
    Args:
        image: The final image tensor [1, 3, H, W]
        frequency_components: List of (real, imaginary) tensors for each channel
        num_bands: Number of frequency bands to visualize
        
    Returns:
        Matplotlib figure with the visualization
    """
    # Convert tensors to CPU for visualization
    if image.is_cuda:
        image = image.cpu()
    
    frequency_components = [
        (real.cpu().detach(), imag.cpu().detach())
        for real, imag in frequency_components
    ]
    
    # Get image dimensions
    img_size = image.shape[2]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot the final image
    ax1 = fig.add_subplot(2, 3, 1)
    img_np = image[0].permute(1, 2, 0).numpy()
    ax1.imshow(np.clip(img_np, 0, 1))
    ax1.set_title("Optimized Image")
    ax1.axis('off')
    
    # Plot the full magnitude spectrum (average of RGB channels)
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Combine magnitude spectra from all channels
    magnitude_spectrum = torch.zeros(img_size, img_size // 2 + 1)
    for c in range(3):
        complex_tensor = torch.complex(frequency_components[c][0], frequency_components[c][1])
        magnitude_spectrum += torch.abs(complex_tensor)
    magnitude_spectrum /= 3.0
    
    # Apply log scaling for better visualization
    log_spectrum = torch.log(magnitude_spectrum + 1)
    
    # Shift DC component to center for visualization
    h, w = log_spectrum.shape
    shifted_spectrum = torch.zeros(h, h)
    
    shifted_spectrum[:h//2, h//2:] = log_spectrum[:h//2, :w-h//2] if w > h//2 else log_spectrum[:h//2, :w]
    if w > h//2:
        shifted_spectrum[:h//2, :h//2] = log_spectrum[:h//2, w-h//2:]
    if h > 1:
        shifted_spectrum[h//2:, h//2:] = log_spectrum[h//2:, :w-h//2] if w > h//2 else log_spectrum[h//2:, :w]
        if w > h//2:
            shifted_spectrum[h//2:, :h//2] = log_spectrum[h//2:, w-h//2:]
    
    ax2.imshow(shifted_spectrum.numpy(), cmap='viridis')
    ax2.set_title("Magnitude Spectrum (log scale)")
    ax2.axis('off')
    
    # Plot the phase spectrum
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Combine phase spectra from all channels
    phase_spectrum = torch.zeros(img_size, img_size // 2 + 1)
    for c in range(3):
        complex_tensor = torch.complex(frequency_components[c][0], frequency_components[c][1])
        phase_spectrum += torch.angle(complex_tensor)
    phase_spectrum /= 3.0
    
    # Shift DC component to center for visualization
    h, w = phase_spectrum.shape
    shifted_phase = torch.zeros(h, h)
    
    shifted_phase[:h//2, h//2:] = phase_spectrum[:h//2, :w-h//2] if w > h//2 else phase_spectrum[:h//2, :w]
    if w > h//2:
        shifted_phase[:h//2, :h//2] = phase_spectrum[:h//2, w-h//2:]
    if h > 1:
        shifted_phase[h//2:, h//2:] = phase_spectrum[h//2:, :w-h//2] if w > h//2 else phase_spectrum[h//2:, :w]
        if w > h//2:
            shifted_phase[h//2:, :h//2] = phase_spectrum[h//2:, w-h//2:]
    
    ax3.imshow(shifted_phase.numpy(), cmap='hsv')
    ax3.set_title("Phase Spectrum")
    ax3.axis('off')
    
    # Visualize different frequency bands
    # Create evenly spaced frequency bands
    band_edges = np.linspace(0, 1, num_bands + 1)
    
    for i in range(min(3, num_bands)):
        ax = fig.add_subplot(2, 3, i + 4)
        
        # Filter to only keep frequencies in this band
        low_freq = band_edges[i]
        high_freq = band_edges[i + 1]
        
        # Create filtered version for each channel
        channels = []
        for c in range(3):
            # Get original complex tensor
            complex_tensor = torch.complex(frequency_components[c][0], frequency_components[c][1])
            
            # Create band-pass filter mask
            h, w = complex_tensor.shape
            filter_mask = torch.zeros(h, w, dtype=torch.float32)
            
            for y in range(h):
                for x in range(w):
                    # Calculate normalized frequency (0 to 1)
                    y_freq = min(y, h - y) / (h / 2)
                    x_freq = x / w
                    
                    # Normalized distance from DC component (0,0)
                    dist = math.sqrt(y_freq**2 + x_freq**2)
                    
                    # Check if in current band
                    if low_freq <= dist < high_freq:
                        filter_mask[y, x] = 1.0
            
            # Apply filter (keep only frequencies in this band)
            filtered = complex_tensor * filter_mask
            
            # Inverse FFT to get spatial image for this band
            band_image = torch.fft.irfft2(filtered, s=(img_size, img_size))
            channels.append(band_image)
        
        # Stack channels
        band_spatial = torch.stack(channels, dim=0)
        
        # Normalize for visualization
        band_spatial = (band_spatial - band_spatial.min()) / (band_spatial.max() - band_spatial.min() + 1e-8)
        
        # Plot
        band_np = band_spatial.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(band_np, 0, 1))
        ax.set_title(f"Band {i+1}: {low_freq:.1f}-{high_freq:.1f}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def frequency_band_contribution(frequency_components, img_size, num_bands=10):
    """
    Analyze the contribution of different frequency bands to the image.
    
    Args:
        frequency_components: List of (real, imaginary) tensors for each channel
        img_size: Size of the image
        num_bands: Number of frequency bands to analyze
        
    Returns:
        Dictionary with energy statistics by frequency band
    """
    # Move components to CPU
    frequency_components = [
        (real.cpu().detach(), imag.cpu().detach())
        for real, imag in frequency_components
    ]
    
    # Calculate total energy in the spectrum
    total_energy = 0
    for c in range(3):
        complex_tensor = torch.complex(frequency_components[c][0], frequency_components[c][1])
        magnitude = torch.abs(complex_tensor)
        total_energy += torch.sum(magnitude**2).item()
    
    # Create evenly spaced frequency bands
    band_edges = np.linspace(0, 1, num_bands + 1)
    
    # Calculate energy in each band
    band_energy = {}
    
    for i in range(num_bands):
        low_freq = band_edges[i]
        high_freq = band_edges[i + 1]
        band_name = f"band_{i+1}_{low_freq:.2f}_{high_freq:.2f}"
        
        energy = 0
        for c in range(3):
            complex_tensor = torch.complex(frequency_components[c][0], frequency_components[c][1])
            h, w = complex_tensor.shape
            
            # Count energy in this band
            for y in range(h):
                for x in range(w):
                    # Calculate normalized frequency (0 to 1)
                    y_freq = min(y, h - y) / (h / 2)
                    x_freq = x / w
                    
                    # Normalized distance from DC component (0,0)
                    dist = math.sqrt(y_freq**2 + x_freq**2)
                    
                    # Check if in current band
                    if low_freq <= dist < high_freq:
                        energy += abs(complex_tensor[y, x])**2
        
        band_energy[band_name] = energy
        band_energy[f"{band_name}_percent"] = 100 * energy / total_energy
    
    return band_energy