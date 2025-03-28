from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import torchvision
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from util.vis_pipnet import get_img_coordinates, get_patch_size
import math
from PIL import Image

class HeatmapAlignmentLoss(torch.nn.Module):
   def __init__(self, alpha=0.5, beta=0.5):
       super().__init__()
       self.alpha = alpha  # Weight for BCE loss
       self.beta = beta    # Weight for SSIM loss
       
   def forward(self, pred_heatmaps, gt_heatmaps, mask=None):
       """
       Compute alignment loss between predicted and ground truth heatmaps
       
       Args:
           pred_heatmaps: Tensor of shape [B, num_patches, 1, H, W]
           gt_heatmaps: Tensor of shape [B, num_patches, 1, H, W]
           mask: Optional tensor indicating valid patches to consider
       
       Returns:
           Loss tensor
       """
       batch_size, num_patches = pred_heatmaps.shape[:2]
       
       # Initialize loss
       total_loss = 0
       valid_count = 0
       
       for b in range(batch_size):
           for p in range(num_patches):
               # Skip if mask indicates invalid patch
               if mask is not None and not mask[b, p]:
                   continue
                   
               pred = pred_heatmaps[b, p]
               target = gt_heatmaps[b, p,0 ]
               
               # Skip if ground truth is all zeros
               if torch.sum(target) < 1e-6:
                   continue
               
               # Binary cross entropy loss
               bce_loss = F.binary_cross_entropy(pred, target)
               
               # SSIM loss for structural similarity (1 - SSIM to make it a loss)
               ssim_loss = 1 - ssim(pred.unsqueeze(0), target.unsqueeze(0))
               
               # Combined loss
               patch_loss = self.alpha * bce_loss + self.beta * ssim_loss
               total_loss += patch_loss
               valid_count += 1
       
       # Return average loss
       if valid_count > 0:
           return total_loss / valid_count
       else:
           return torch.tensor(0.0, device=pred_heatmaps.device)

# SSIM implementation
def ssim(img1, img2, window_size=11, size_average=True):
   """Calculate SSIM between two images"""
   # Get device and dtype
   device = img1.device
   dtype = img1.dtype
   
   # Create Gaussian window
   window = gaussian_window(window_size, 1.5, device, dtype)
   window = window.expand(1, 1, window_size, window_size)
   
   # Mean calculations
   mu1 = F.conv2d(img1, window, padding=window_size//2)
   mu2 = F.conv2d(img2, window, padding=window_size//2)
   
   mu1_sq = mu1.pow(2)
   mu2_sq = mu2.pow(2)
   mu1_mu2 = mu1 * mu2
   
   # Variance calculations
   sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
   sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
   sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2
   
   # Constants for stability
   C1 = 0.01 ** 2
   C2 = 0.03 ** 2
   
   # SSIM calculation
   ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
              ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
   
   if size_average:
       return ssim_map.mean()
   else:
       return ssim_map.mean(1).mean(1).mean(1)

def gaussian_window(window_size, sigma, device, dtype):
   """Create 1D Gaussian window"""
   x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
   gauss = torch.exp(-(x.pow(2) / (2 * sigma ** 2)))
   return gauss / gauss.sum()


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def cross_batch_similarity_loss(current_prototypes, memory_samples, temperature=0.1):
    """
    Compute cross-batch similarity loss to promote diversity.
    current_prototypes: tensor of shape [batch_size, prototype_dim]
    memory_samples: tensor of shape [num_samples, prototype_dim]
    """
    if memory_samples is None or memory_samples.size(0) == 0:
        return torch.tensor(0.0, device=current_prototypes.device)
    
    # Normalize current prototypes
    current_prototypes = F.normalize(current_prototypes, dim=1)
    
    # Compute cosine similarity
    similarities = torch.mm(current_prototypes, memory_samples.t()) / temperature
    
    # We want to minimize similarity with previous batch prototypes to increase diversity
    # Lower similarity means more diverse prototypes
    loss = torch.mean(torch.logsumexp(similarities, dim=1))
    
    return loss

def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight,
                    net_normalization_multiplier, pretrain, finetune, 
                    criterion, train_iter, shared_features_loss=True, print=True, EPS=1e-10):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.


    ## New loss to encourage prototype diversity
    prototype_diversity_loss = torch.mean(torch.relu(torch.sum(pooled1,dim=0) - 1)) * 0.05

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        loss += t_weight * tanh_loss

        if not shared_features_loss:
            loss += prototype_diversity_loss
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
        if finetune:
            loss= cl_weight * class_loss
        else:
            loss+= cl_weight * class_loss

    
    # # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc=0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    if print: 
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)            
    return loss, acc

def train_pipnet_gumbel(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, 
                                  scheduler_classifier, criterion, epoch, nr_epochs, 
                                  device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch', args=None):

    # Make sure the model is in train mode
    net.train()
    net.initialize_patch_folders(args)
    # if pretrain:
    #     # Disable training of classification layer
    #     net.module._classification.requires_grad = False
    #     progress_prefix = 'Pretrain Epoch'
    # else:
    #     # Enable training of classification layer (disabled in case of pretraining)
    #     net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
        # Add memory bank weights for pretraining
        cross_batch_weight = 0.5
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.
        # Add memory bank weights for fine-tuning
        cross_batch_weight = 0.2
    
    # Initialize memory bank for prototypes
    # Use the dimension of pooled prototypes from the model
    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, 
          "Class weight:", cl_weight, "Cross-batch weight:", cross_batch_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    

    heatmaploss = HeatmapAlignmentLoss()
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        stats = net(torch.cat([xs1, xs2]))
        proto_features, pooled, out = stats['proto_features'], stats['pooled'], stats['classification']
        # patch_heatmaps = stats['patch_heatmaps']
        # combined_heatmaps = stats['combined_heatmap']

        # plt.imshow(stats['gt_heatmaps'][0][0][0].detach().cpu().numpy())
        # plt.show()
        
        
        # Calculate standard PIP-Net loss
        standard_loss, acc = calculate_loss(proto_features, pooled, out, ys, 
                                          align_pf_weight, t_weight, unif_weight, cl_weight, 
                                          net.pipnet.module._classification.normalization_multiplier, 
                                          pretrain, finetune, criterion, train_iter, 
                                          print=True, EPS=1e-8)
        
        # Combine losses
        loss = standard_loss +  heatmaploss(stats['combined_heatmap'], stats['gt_heatmaps'])
        
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        if not pretrain:
            with torch.no_grad():
                net.pipnet.module._classification.weight.copy_(torch.clamp(net.pipnet.module._classification.weight.data - 1e-3, min=0.)) 
                net.pipnet.module._classification.normalization_multiplier.copy_(torch.clamp(net.pipnet.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.pipnet.module._classification.bias is not None:
                    net.pipnet.module._classification.bias.copy_(torch.clamp(net.pipnet.module._classification.bias.data, min=0.))  
    
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info