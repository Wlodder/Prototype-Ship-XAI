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


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
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

def train_pipnet_with_memory_bank(net, patch_encoder, memory_bank, train_loader, optimizer_net, optimizer_classifier, scheduler_net, 
                                  scheduler_classifier, criterion, epoch, nr_epochs,  
                                  device, args, pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    patch_encoder.train()
    
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

        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 0#
        cl_weight = 0.
        # Add memory bank weights for pretraining
        cross_batch_weight = 0.5
        memory_weight = 0.5
    else:
        align_pf_weight = 0.0
        t_weight = 0#2
        unif_weight = 0.
        cl_weight = 2.
        # Add memory bank weights for fine-tuning
        cross_batch_weight = 0.2
        memory_weight=1.5
    
    # Initialize memory bank for prototypes
    # Use the dimension of pooled prototypes from the model
    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, 
          "Class weight:", cl_weight, "Cross-batch weight:", cross_batch_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
        
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
        
        patches = memory_bank.get_image_features()
    
        # ===== Student Forward Pass =====
        student_features1, _, _ = net(xs1)
        student_features2, _, _ = net(xs2)

        patch_features = patch_encoder(patches)
        
        # Cosine distance between patch features and student features
        student_features1 = F.normalize(student_features1, p=2, dim=1) @ F.normalize(patch_features, p=2, dim=1).T
        student_features2 = F.normalize(student_features2, p=2, dim=1) @ F.normalize(patch_features, p=2, dim=1).T

        # Softmax over the features student features
        student_pooled1 = F.softmax(student_features1, dim=1)
        student_pooled2 = F.softmax(student_features2, dim=1)

        student_out1 = net.module._classification(student_pooled1)
        student_out2 = net.module._classification(student_pooled2)
        
        # Combine outputs from both views for classification
        pooled = torch.cat([student_pooled1, student_pooled2])
        out = torch.cat([student_out1, student_out2])
        proto_features = torch.cat([student_features1, student_features2])
        
        # ===== Standard PIPNet Loss =====
        standard_loss, acc = calculate_loss(
            proto_features, pooled, out, ys, 
            align_pf_weight, t_weight, 0.0, cl_weight,  # Uniformity weight set to 0
            net.module._classification.normalization_multiplier, 
            pretrain, finetune, criterion, train_iter, 
            print=False, EPS=1e-8
        )
        
        # ===== Combine Losses =====
        loss = standard_loss 

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
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) 
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  
    
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info