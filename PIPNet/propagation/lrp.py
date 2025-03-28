import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity  # For cosine similarity

# Import the ConvNeXt model definition (replace with your actual import)
# You can get this from timm library or build it from scratch
# For this example, let's assume you have a function called `convnext_tiny`
# or you can adapt this to any ConvNeXt variant from timm
# Example using timm (you might need to install it: `pip install timm`)
import timm

# --- LRP Functions for different layer types ---
def lrp_linear_spatial(layer, a_in, R_out): # New function for spatially applied linear layers
    """LRP for Linear layers applied spatially (like in CNBlock)."""
    W = layer.weight
    b = layer.bias
    R_in = torch.zeros_like(a_in) # Initialize relevance for input

    print(W.size(), a_in.size(), R_out.size())
    # Iterate over spatial dimensions (assuming a_in is [B, C, H, W] or similar)
    for b in range(a_in.shape[0]):      # Batch dimension
        for h in range(a_in.shape[1]):  # Height dimension
            for w in range(a_in.shape[2]):  # Width dimension
                current_a_in = a_in[b, h, w, :] # [C] - Feature vector at this spatial location
                current_R_out = R_out[b, :, h, w] # [C_out] - Relevance at this spatial location

                # Apply standard linear LRP rule locally at this spatial location
                z = torch.matmul(current_a_in.unsqueeze(0), W.T) + b # z = a_in * W + b
                s = current_R_out / z # Stabilize division if needed
                c = s @ W
                R_in[b, :, h, w] = current_a_in * c # Distribute relevance back to input location

    return R_in

def lrp_linear(layer, a_in, R_out):
    """LRP for Linear layers."""
    W = layer.weight
    b = layer.bias
    a_in = a_in.flatten(1) # Flatten input activations

    z = torch.matmul(a_in, W.T) + b
    s = R_out / z # Stabilize division - can add small epsilon if needed
    c = s @ W
    R_in = a_in * c
    return R_in

def lrp_conv2d(layer, a_in, R_out):
    """LRP for Conv2D layers (including DepthwiseConv2d)."""
    W = layer.weight
    b = layer.bias
    padding = layer.padding
    stride = layer.stride
    dilation = layer.dilation
    groups = layer.groups

    z = F.conv2d(a_in, W, b, stride, padding, dilation, groups)
    s = R_out / z # Stabilize division - can add small epsilon if needed
    c = F.conv_transpose2d(s, W, None, stride, padding, dilation, groups)
    R_in = a_in * c
    return R_in

def lrp_flatten(layer, a_in, R_out):
    return R_out.reshape(a_in.shape)

def lrp_permute(layer, a_in, R_out):
    """LRP for Permute layer: Invert the permutation for relevance."""
    dims = layer.dims # Get the permutation dimensions from the layer
    print(dims)
    return R_out.permute(dims) # Apply the *same* permutation to relevance in reverse

def lrp_layernorm(layer, a_in, R_out):
    """LRP for LayerNorm (treat as identity for relevance flow)."""
    return R_out # Relevance passes through unchanged

def lrp_gelu(layer, a_in, R_out):
    """LRP for GELU (can be approximated by ReLU for simplicity or more complex rules).
       Here we use a simple approach: Pass relevance through where activation is positive.
       More refined rules exist for Gelu if needed for better accuracy."""

    # Simple approximation, refine if needed based on LRP rule you want to use for Gelu
    R_in = R_out * (a_in > 0).float()
    return R_in


def lrp_avgpool(layer, a_in, R_out):
    """LRP for AvgPool (distribute relevance evenly)."""
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding

    z = F.avg_pool2d(a_in, kernel_size, stride, padding)
    s = R_out / z # Stabilize division - can add small epsilon if needed
    c = F.interpolate(s, scale_factor=stride, mode='nearest') # Simple upscale
    R_in = a_in * c
    return R_in


def lrp_convnext_prototype(model, input_tensor, prototype_layer_name, prototype_vector, cosine_similarity_threshold=0.8):
    activations = {}
    x = input_tensor

    def forward_hook(module, input, output):
        activations[module] = output

    hooks = []
    def apply_hooks_recursively(module, prefix=""):
        for name, child_module in module.named_children():
            module_name = prefix + name
            if isinstance(child_module, (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.GELU, nn.AvgPool2d)):
                hooks.append(child_module.register_forward_hook(forward_hook))
            elif isinstance(child_module, nn.Sequential):
                apply_hooks_recursively(child_module, prefix=module_name + ".")
            elif isinstance(child_module, nn.Module):
                apply_hooks_recursively(child_module, prefix=module_name + ".")
    apply_hooks_recursively(model)

    _,output,_ = model(x) # Forward pass to collect activations

    # 1. Get activations of the prototype layer
    prototype_layer_activation = activations[dict(model.named_modules())[prototype_layer_name]]
    prototype_layer_activation = prototype_layer_activation.squeeze(0) # Remove batch dim [C, H, W]
    print(prototype_layer_activation.size())


    # 2. Find prototypical location based on cosine similarity
    max_similarity = -1.0
    prototypical_location = None

    for h in range(prototype_layer_activation.shape[1]):
        for w in range(prototype_layer_activation.shape[2]):
            current_vector = prototype_layer_activation[:,h, w].flatten().detach().cpu().numpy() # [C]
            similarity = cosine_similarity(prototype_vector.reshape(1, -1), current_vector.reshape(1, -1))[0][0] # Compare with prototype
            if similarity > max_similarity: # Looking for maximum similarity
                max_similarity = similarity
                prototypical_location = (h, w)

    print(f"Max Cosine Similarity: {max_similarity:.4f} at location: {prototypical_location}")

    # if max_similarity < cosine_similarity_threshold:
    #     print(f"Max similarity is below threshold ({cosine_similarity_threshold}). Prototype not strongly activated.")
    #     return torch.zeros_like(input_tensor.unsqueeze(0)) # Return zero relevance if not prototypical enough

    # 3. Initialize Relevance for the prototype layer
    R_prototype_layer = torch.zeros_like(prototype_layer_activation) # [C, H, W]
    if prototypical_location:
        proto_h, proto_w = prototypical_location
        R_prototype_layer[:, proto_h, proto_w] = 1.0 # Set relevance to 1 at the prototypical location, all channels


    relevance_dict = {prototype_layer_name: R_prototype_layer.unsqueeze(0)} # Add batch dimension back

    # 4. Backpropagate Relevance from the prototype layer backwards
    R_in = R_prototype_layer.unsqueeze(0) # Add batch dimension back
    modules_reversed = list(model.named_modules())[1:][::-1] # Reverse modules

    reached_prototype_layer = False # Flag to stop backpropagation
    num_layers = 0
    for name, module in modules_reversed:
        if name == prototype_layer_name: # Start backpropagation from just before the prototype layer
            reached_prototype_layer = True
            continue # Skip the prototype layer itself, start from layers before it

        if not reached_prototype_layer:
            continue # Keep skipping until we reach the prototype layer in reverse

        print('r: ', R_in.size(), 'a:', module)
        # print(activations[module].size())
        if isinstance(module, nn.Linear):
            print(R_in.size(),activations[module].size())
            
            R_in = lrp_linear_spatial(module, activations[module], R_in) # Use lrp_linear_spatial for Linear layers in CNBlocks!
        elif isinstance(module, nn.Conv2d):
            R_in = lrp_conv2d(module, activations[module], R_in)
        elif isinstance(module, nn.LayerNorm):
            R_in = lrp_layernorm(module, activations[module], R_in)
        elif isinstance(module, nn.GELU):
            R_in = lrp_gelu(module, activations[module], R_in)
        elif isinstance(module, nn.AvgPool2d):
            R_in = lrp_avgpool(module, activations[module], R_in)
        elif isinstance(module, nn.Flatten):
            R_in = lrp_flatten(module, activations[module], R_in)
        elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
            continue
        else:
            continue

        relevance_dict[name] = R_in
        if name == list(model.named_modules())[1:][::-1][-1][0]: # Stop at the input layer
            break

        if num_layers >= 1:
            break


    for hook in hooks:
        hook.remove()
    relevance_map = R_in.sum(dim=1, keepdim=True).squeeze(0) # Sum over channels
    return relevance_map.squeeze().cpu().detach(), 1.0