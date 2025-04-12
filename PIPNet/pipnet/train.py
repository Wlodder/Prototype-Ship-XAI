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


def budget_loss(head: "CompetingHead",
                lmb_budget: float = 1e-4) -> torch.Tensor:
    """
    L1 budget per head  (encourages sparsity).
    Scales with lmb_budget.
    """
    return lmb_budget * head.weight.abs().sum(dim=2).mean()  # mean over (C, H)


def sharing_loss(head: "CompetingHead",
                 lmb_share: float = 1e-3,
                 p: float = 2.0) -> torch.Tensor:
    """
    Penalise a prototype being used by ≥2 heads of the *same* class.

    For class c and prototype d:
        overlap = sum_h |w_chd|
        penalty  = (overlap ** p)

    * p=2   → quadratic, smooth
    * lmb_share  controls strength
    """
    w = head.weight.abs()                       # (C, H, D)
    overlap = w.sum(dim=1)                      # (C, D)
    return lmb_share * (overlap ** p).mean()    # mean over (C, D)

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
        align_pf_weight = (epoch/nr_epochs)*1
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 3. 
        t_weight = 3.
        unif_weight = 0.
        cl_weight = 2.
    
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
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, 
                                   unif_weight, cl_weight, net.module._classification.normalization_multiplier, 
                                   pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
        
        head = net.module._classification
        budget_weight=1e-4
        share_weight=1e-3
        budget_l = budget_loss(head, budget_weight)
        share_l = sharing_loss(head, share_weight, p=2.0)
        loss = loss + budget_l + share_l
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
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                # net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                # if net.module._classification.bias is not None:
                #     net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
                # head.normalization_multiplier = min(head.normalization_multipier, 1.0)

                # if head.bias is not None:
                #     head.bias = min(head.bias,0.0)

    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info


def train_pipnet_cutmix(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier,
                         criterion, epoch, nr_epochs, device, proto_dir, args, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch',
                         prototype_buffer=None):
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

    toTensor = torchvision.transforms.ToTensor()

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
        unif_weight = 0.5 #currently used
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    patchsize, skip = get_patch_size(args)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # For visuzliation purposes
    def unnormalize(image):
        return image * torch.tensor(std).view(-1,1,1) + torch.tensor(mean).view(-1,1,1)
    normalize = torchvision.transforms.Normalize(mean=mean,std=std)

    if pretrain:
        for i, (xs1, xs2, ys) in train_iter:       

            xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
            if os.path.exists(proto_dir) and not pretrain: 

                # Initial pass to get prototype allocations
                with torch.no_grad():
                
                    # Reset the gradients
                    optimizer_classifier.zero_grad(set_to_none=True)
                    optimizer_net.zero_grad(set_to_none=True)
                
                    # Perform a forward pass through the network
                    softmaxes, pooled, _ = net(torch.cat([xs1, xs2]), inference=True)
                    xs1 = xs1.cpu()

                    _, sorted_pooled_indices = torch.sort(pooled, descending=True, dim=1)
                    sorted_pooled_indices = sorted_pooled_indices[:,:10]

                    for index in range(xs1.size(0)):
                        # image = xs1[index]
                        for prototype_idx in sorted_pooled_indices[index]:
                            max_h, max_idx_h = torch.max(softmaxes[index, prototype_idx, :, :], dim=0)
                            max_w, max_idx_w = torch.max(max_h, dim=0)

                            # coordinates for replacement
                            max_idx_h = max_idx_h[max_idx_w].item()
                            max_idx_w = max_idx_w.item()


                            # We are cut mixing
                            prototype_image_path_dir  = os.path.join(proto_dir,f'prototype_{prototype_idx.item()}')
                            if not os.path.exists(prototype_image_path_dir):
                                continue

                            prototypes = os.listdir(prototype_image_path_dir)
                            prototype_image_path = os.path.join(prototype_image_path_dir, prototypes[random.randint(0,len(prototypes)-1)])

                            # Remember to normalize 
                            image_tensor = toTensor(Image.open(prototype_image_path).convert("RGB"))
                            patch_height, patch_width = image_tensor.size()[1:]
                            
                            # hmin, hmax, wmin, wmax = get_img_coordinates(image_tensor.size()[1],softmaxes[0].size(), image_tensor.size()[1], 0, max_idx_h, max_idx_w)
                            ratio = xs1[0].size()[1] / image_tensor.size()[1] 
                            patch_size = image_tensor.size()[1]
                            hmin, hmax = int(max_idx_h * ratio) - patch_size // 2, int(max_idx_h * ratio) + patch_size // 2
                            wmin, wmax = int(max_idx_w * ratio) - patch_size // 2, int(max_idx_w * ratio) + patch_size // 2

                            xs1[index,:,hmin:hmax, wmin:wmax] = image_tensor
                            # plt.imshow(unnormalize(xs1[index]).permute(1,2,0))
                            # plt.show()
                            # plt.imshow(image_tensor.permute(1,2,0))
                            # plt.show()

                    xs1 = xs1.to(device)

            proto_features, pooled, out = net(torch.cat([xs1, xs2]))
            loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, 
                                    unif_weight, cl_weight, net.module._classification.normalization_multiplier, 
                                    pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
            
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
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
    else:
        for i, (xs1, ys) in enumerate(train_iter):       
            xs1, ys = ys

            # Clone the original image
            xs2 = xs1.clone()
            xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

            print(args.buffer_type.split(':'))
            if not pretrain and ((os.path.exists(proto_dir) and args.buffer_type.split(':')[0] == 'folder') or args.buffer_type.split(':')[0] == 'buffer'): 
                # Initial pass to get prototype allocations
                with torch.no_grad():
                
                    # Reset the gradients
                    optimizer_classifier.zero_grad(set_to_none=True)
                    optimizer_net.zero_grad(set_to_none=True)
                
                    # Perform a forward pass through the network
                    softmaxes, pooled, _ = net(torch.cat([xs1, xs2]), inference=True)
                    xs1 = xs1.cpu()

                    _, sorted_pooled_indices = torch.sort(pooled, descending=True, dim=1)

                    amount = int(args.buffer_type.split(':')[-1])
                    
                    sorted_pooled_indices = sorted_pooled_indices[:,:amount]
                    # For every element of a batch
                    for index in range(xs1.size(0)):
                        # image = xs1[index]
                        for prototype_idx in sorted_pooled_indices[index]:
                            max_h, max_idx_h = torch.max(softmaxes[index, prototype_idx, :, :], dim=0)
                            max_w, max_idx_w = torch.max(max_h, dim=0)

                            # coordinates for replacement
                            max_idx_h = max_idx_h[max_idx_w].item()
                            max_idx_w = max_idx_w.item()

                            # if we are cutmixing based on saving images, to view later or are we using a buffer?
                            if args.buffer_type.split(':')[0] == "folder":
                                prototype_image_path_dir  = os.path.join(proto_dir,f'prototype_{prototype_idx.item()}')

                                # if the prototype does not exist
                                # just skip the prototype
                                if not os.path.exists(prototype_image_path_dir):
                                    continue

                                prototypes = os.listdir(prototype_image_path_dir)
                                prototype_image_path = os.path.join(prototype_image_path_dir, prototypes[random.randint(0,len(prototypes)-1)])

                                # Remember to normalize 
                                image_tensor = normalize(toTensor(Image.open(prototype_image_path).convert("RGB")))
                            else:
                                id = prototype_idx.item()
                                if len(prototype_buffer[id])  == 0:
                                    continue
                                    
                                image_tensor = normalize(prototype_buffer[id][random.randint(0, len(prototype_buffer[id])-1)])


                            hmin, hmax, wmin, wmax = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, max_idx_h, max_idx_w)
                            # jitter_w = random.randint(-1,1)
                            # jitter_h = random.randint(-1,1)
                            # xs1[index,:,hmin+jitter_h:hmax+jitter_h, wmin+jitter_w:wmax+jitter_w] = image_tensor
                            xs1[index,:,hmin:hmax, wmin:wmax] = image_tensor


                            # For displaying purposes
                            # plt.imshow(unnormalize(xs1[index]).permute(1,2,0))
                            # plt.show()

                    xs1 = xs1.to(device)

            proto_features, pooled, out = net(torch.cat([xs1, xs2]))
            loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, 
                                    unif_weight, cl_weight, net.module._classification.normalization_multiplier, 
                                    pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
            
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
                    net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                    net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                    if net.module._classification.bias is not None:
                        net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info




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
    # prototype_diversity_loss = torch.mean(torch.relu(torch.sum(pooled1,dim=0) - 1)) * 0.05

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        loss += t_weight * tanh_loss

        # if not shared_features_loss:
        #     loss += prototype_diversity_loss
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
        if finetune:
            loss= cl_weight * class_loss
        else:
            loss+= cl_weight * class_loss

    
    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    else:
        uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
        loss += unif_weight * uni_loss

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



# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

def align_loss_cluster_loss(inputs, targets, neg_samples=None, pos_samples=None, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    assert neg_samples.requires_grad == False
    assert pos_samples.requires_grad == False

    

    base_loss = torch.einsum("nc,nc->n", [inputs, targets])
    # Dot product
    if neg_samples != None:
        neg_loss = torch.einsum("nc,bnc->bn",[inputs, neg_samples])
        neg_loss = torch.sum(torch.exp(neg_loss))

    if pos_samples != None:
        pos_loss = torch.einsum("nc,bnc->bn",[inputs, pos_samples])
        pos_loss = torch.sum(torch.exp(pos_loss))

    loss = -torch.log((torch.exp(base_loss) + EPS + pos_loss) / (neg_loss + EPS)).mean()
    return loss

def train_pipnet_memory_bank(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):
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
    
    # Initialize or retrieve prototype utilization memory bank
    if not hasattr(net.module, 'prototype_memory_bank'):
        # Initialize memory bank for tracking prototype utilization
        # Shape: [num_prototypes]
        num_prototypes = net.module.num_features
        net.module.register_buffer('prototype_memory_bank', torch.zeros(num_prototypes, device=device))
        net.module.register_buffer('prototype_class_assoc', torch.zeros(num_prototypes, net.module._classification.weight.size(0), device=device))
        
        # Set target utilization based on dataset size and number of prototypes
        # This can be adjusted based on expected concept frequency
        dataset_size = len(train_loader.dataset)
        net.module.target_utilization = min(10.0 / dataset_size, 0.05)  # At least activate on 10 samples, but not more than 5% of data
        print(f"Target prototype utilization: {net.module.target_utilization:.5f}", flush=True)
        
        # Initialize prototype scheduling parameters
        net.module.prototype_schedule_base = min(500, num_prototypes)  # Start with this many prototypes
        
    # Calculate number of active prototypes based on curriculum schedule
    total_prototypes = net.module.num_features
    if pretrain:
        # During pretraining, gradually increase from base to maximum
        schedule_progress = min(1.0, epoch / (nr_epochs * 0.7))  # Reach max at 70% of pretraining
        active_prototypes = min(
            int(net.module.prototype_schedule_base + schedule_progress * (total_prototypes - net.module.prototype_schedule_base)),
            total_prototypes
        )
    else:
        # During main training, use all prototypes
        active_prototypes = total_prototypes
    
    # Create active prototype mask
    prototype_mask = torch.zeros(total_prototypes, device=device)
    prototype_mask[:active_prototypes] = 1.0
    
    # Set loss weights based on training phase
    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5  # ignored
        memory_weight = 5.0  # Weight for memory bank utilization loss
        indep_weight = 1.0  # Weight for prototype independence loss
        t_weight = 0.0  # Original tanh weight set to zero
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        memory_weight = 2.0
        indep_weight = 0.5
        t_weight = 0.0  # Replace original tanh loss with memory-based loss
        unif_weight = 0.
        cl_weight = 2.
    
    print(f"Align weight: {align_pf_weight}, Memory weight: {memory_weight}, Independence weight: {indep_weight}, Class weight: {cl_weight}", flush=True)
    print(f"Active prototypes: {active_prototypes}/{total_prototypes}", flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    
    # EMA momentum coefficient for memory bank updates
    momentum = 0.99
    
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        
        # Apply the active prototype mask to focus only on scheduled prototypes
        masked_pooled = pooled * prototype_mask.unsqueeze(0)
        
        # Calculate standard loss components
        loss, acc = calculate_loss(
            proto_features, masked_pooled, out, ys, 
            align_pf_weight, t_weight, unif_weight, cl_weight, 
            net.module._classification.normalization_multiplier, 
            pretrain, finetune, criterion, train_iter, 
            print=True, EPS=1e-8
        )
        
        # Add memory bank utilization loss if enabled
        if memory_weight > 0:
            # Calculate current batch utilization (binary: whether each prototype activates above threshold)
            batch_size = pooled.size(0) // 2  # Because we concatenated xs1 and xs2
            activation_threshold = 0.2  # Threshold for considering a prototype "activated"
            
            # Use pooled features to determine prototype activations
            batch_activations = (pooled[:batch_size] > activation_threshold).float()
            batch_utilization = batch_activations.mean(dim=0)  # Average over batch dimension
            
            # Update prototype-class associations
            with torch.no_grad():
                for c in range(net.module._classification.weight.size(0)):
                    class_mask = (ys == c).float().unsqueeze(1)  # Shape: [batch_size, 1]
                    class_activations = batch_activations * class_mask  # Only count activations for this class
                    class_presence = class_mask.sum()
                    if class_presence > 0:
                        class_utilization = class_activations.sum(dim=0) / class_presence
                        net.module.prototype_class_assoc[:, c] = momentum * net.module.prototype_class_assoc[:, c] + (1 - momentum) * class_utilization
            
            # Calculate prototype "purity" - how exclusively a prototype activates for one class
            with torch.no_grad():
                # Add a small epsilon to avoid division by zero
                total_activations = net.module.prototype_class_assoc.sum(dim=1, keepdim=True) + 1e-8
                max_class_activations, _ = net.module.prototype_class_assoc.max(dim=1)
                prototype_purity = max_class_activations / total_activations.squeeze()
            
            # Update memory bank with exponential moving average
            with torch.no_grad():
                net.module.prototype_memory_bank = momentum * net.module.prototype_memory_bank + (1 - momentum) * batch_utilization
            
            # Calculate memory bank utilization loss
            # We want utilization to be close to target utilization for active prototypes
            utilization_ratio = net.module.prototype_memory_bank / net.module.target_utilization
            # Cap utilization ratio at 1.0 for well-utilized prototypes
            capped_ratio = torch.min(utilization_ratio, torch.ones_like(utilization_ratio))
            # Only apply to active prototypes
            masked_ratio = capped_ratio * prototype_mask
            
            # Memory bank utilization loss
            memory_loss = -torch.mean(torch.log(masked_ratio + 1e-8))
            loss = loss + memory_weight * memory_loss
            
            # Print some utilization statistics periodically
            if i % 50 == 0:
                # Calculate percent of prototypes meeting utilization target
                meeting_target = (net.module.prototype_memory_bank >= net.module.target_utilization).float().mean().item() * 100
                avg_utilization = net.module.prototype_memory_bank.mean().item()
                avg_purity = prototype_purity.mean().item()
                print(f"Prototype stats: {meeting_target:.1f}% meet target, avg utilization: {avg_utilization:.4f}, avg purity: {avg_purity:.4f}", flush=True)
        
        # Add prototype independence loss if enabled
        if indep_weight > 0 and not pretrain:
            # Calculate cosine similarity between all pairs of prototype weight vectors
            proto_weights = net.module._classification.weight.t()  # [num_prototypes, num_classes]
            normalized_weights = proto_weights / (proto_weights.norm(dim=1, keepdim=True) + 1e-8)
            
            # Compute pairwise cosine similarities
            similarities = torch.mm(normalized_weights, normalized_weights.t())
            
            # Mask out self-similarities
            mask = torch.eye(similarities.size(0), device=device) * 10.0
            similarities = similarities - mask
            
            # Only penalize positive similarities above a threshold
            threshold = 0.3
            independence_loss = torch.mean(torch.clamp(similarities - threshold, min=0.0)**2)
            
            loss = loss + indep_weight * independence_loss
        
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
                # Apply regularization to classification weights
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.))  # Set weights < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                
                # Apply purity-based regularization
                if hasattr(net.module, 'prototype_memory_bank') and i % 100 == 0:
                    # Reduce weights for impure prototypes
                    purity_threshold = 0.7
                    low_purity_mask = (prototype_purity < purity_threshold).float().unsqueeze(1)
                    weight_reduction = 0.9  # Reduce by 10%
                    net.module._classification.weight.copy_(
                        net.module._classification.weight * (1.0 - 0.1 * low_purity_mask.t())
                    )
                
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))
    
    # At the end of the epoch, sort prototypes by utilization if we're in pretraining
    # This ensures the active prototypes are the most utilized ones
    if pretrain and active_prototypes < total_prototypes:
        with torch.no_grad():
            # Sort prototypes by utilization
            sorted_indices = torch.argsort(net.module.prototype_memory_bank, descending=True)
            
            # Reorder prototype_memory_bank
            net.module.prototype_memory_bank = net.module.prototype_memory_bank[sorted_indices]
            
            # Reorder prototype_class_assoc
            net.module.prototype_class_assoc = net.module.prototype_class_assoc[sorted_indices]
            
            # Reorder weights in classification layer
            if hasattr(net.module._classification, 'weight'):
                reordered_weights = net.module._classification.weight.clone()
                for new_idx, old_idx in enumerate(sorted_indices):
                    reordered_weights[:, new_idx] = net.module._classification.weight[:, old_idx]
                net.module._classification.weight.copy_(reordered_weights)
    
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info

def calculate_loss_with_memory_bank(proto_features, pooled, out, ys, memory_bank, class_assoc, target_utilization, momentum=0.99, activation_threshold=0.2):
    """
    Calculate memory bank based utilization loss and prototype purity statistics.
    
    Args:
        proto_features: Prototype features from network
        pooled: Pooled prototype activations [batch_size, num_prototypes]
        out: Network output logits [batch_size, num_classes]
        ys: Ground truth labels [batch_size]
        memory_bank: Prototype utilization tracking tensor [num_prototypes]
        class_assoc: Prototype-class association matrix [num_prototypes, num_classes]
        target_utilization: Target activation rate for prototypes
        momentum: EMA momentum coefficient
        activation_threshold: Threshold to consider a prototype activated
    
    Returns:
        memory_loss: Memory bank utilization loss
        prototype_purity: Purity score for each prototype
    """
    batch_size = pooled.size(0)
    num_prototypes = pooled.size(1)
    num_classes = class_assoc.size(1)
    
    # Calculate current batch utilization (binary: whether each prototype activates above threshold)
    batch_activations = (pooled > activation_threshold).float()
    batch_utilization = batch_activations.mean(dim=0)  # Average over batch dimension
    
    # Update prototype-class associations
    for c in range(num_classes):
        class_mask = (ys == c).float().unsqueeze(1)  # Shape: [batch_size, 1]
        class_activations = batch_activations * class_mask  # Only count activations for this class
        class_presence = class_mask.sum()
        if class_presence > 0:
            class_utilization = class_activations.sum(dim=0) / class_presence
            class_assoc[:, c] = momentum * class_assoc[:, c] + (1 - momentum) * class_utilization
    
    # Calculate prototype "purity" - how exclusively a prototype activates for one class
    total_activations = class_assoc.sum(dim=1, keepdim=True) + 1e-8
    max_class_activations, _ = class_assoc.max(dim=1)
    prototype_purity = max_class_activations / total_activations.squeeze()
    
    # Update memory bank with exponential moving average
    memory_bank = momentum * memory_bank + (1 - momentum) * batch_utilization
    
    # Calculate memory bank utilization loss
    # We want utilization to be close to target utilization for all prototypes
    utilization_ratio = memory_bank / target_utilization
    # Cap utilization ratio at 1.0 for well-utilized prototypes
    capped_ratio = torch.min(utilization_ratio, torch.ones_like(utilization_ratio))
    
    # Memory bank utilization loss
    memory_loss = -torch.mean(torch.log(capped_ratio + 1e-8))
    
    return memory_loss, prototype_purity, memory_bank, class_assoc