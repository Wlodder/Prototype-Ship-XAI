from tqdm import tqdm
import torch.nn.functional as F
from prototype_squared.adaptive import AdaptivePrototypeRotation, replace_classification_with_rotation
import numpy as np
import torch
from util.vis_pipnet import get_img_coordinates, get_patch_size
import math
from PIL import Image
from pipnet.losses import align_loss, budget_loss, sharing_loss, calculate_loss, calculate_loss_with_crp

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
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
        align_pf_weight = (epoch/nr_epochs)*(1.5)
        unif_weight = 1.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 10. 
        t_weight = 8.
        unif_weight = 0.
        cl_weight = (epoch/nr_epochs)*2.0

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)

    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))

        # Standard loss
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, 
                                   unif_weight, cl_weight, net.module._classification.normalization_multiplier, 
                                   pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)

        usage_dist = net.module.crp_allocation.prototype_counts / net.module.crp_allocation.total_count
        target_dist = usage_dist.pow(-0.5)  # Power-law transformation
        target_dist = target_dist / target_dist.sum()  # Normalize
        
        # KL divergence encouraging uniform prototype usage
        diversity_loss = F.kl_div(
            F.log_softmax(pooled.mean(dim=0).unsqueeze(0)+1e-6, dim=1),
            target_dist.unsqueeze(0),
            reduction='batchmean'
        )
        
        loss+=  1e-3 * diversity_loss
        
        # Mulit head budget losses
        # head = net.module._classification
        # budget_weight=1e-2
        # share_weight=1e-2
        # budget_l = budget_loss(head, budget_weight)
        # share_l = sharing_loss(head, share_weight, p=2.0)
        # loss = loss + budget_l + share_l
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
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            with torch.no_grad():
                # net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  

                # head.normalization_multiplier = min(head.normalization_multipier, 1.0)

                # if head.bias is not None:
                #     head.bias = min(head.bias,0.0)

    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info
