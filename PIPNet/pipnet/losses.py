import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def calculate_loss_with_crp( pooled, crp_allocation, rare_feature_weight=1.0):
    
    # Add CRP-inspired rare feature loss
    # Get importance weights based on rarity
    # importance_weights = crp_allocation.get_importance_weights()
    
    # Compute diversity loss based on current prototype usage distribution
    usage_dist = crp_allocation.prototype_counts / crp_allocation.total_count
    target_dist = usage_dist.pow(-0.5)  # Power-law transformation
    target_dist = target_dist / target_dist.sum()  # Normalize
    
    # KL divergence encouraging uniform prototype usage
    diversity_loss = F.kl_div(
        F.log_softmax(pooled.mean(dim=0).unsqueeze(0), dim=1),
        target_dist.unsqueeze(0),
        reduction='batchmean'
    )
    
    return rare_feature_weight * diversity_loss

# Bio inspired losses
def budget_loss(head: "CompetingHead",
                lmb_budget: float = 1e-4) -> torch.Tensor:
    """
    L1 budget per head  (encourages sparsity).
    Scales with lmb_budget.
    """
    return lmb_budget * head.weight.abs().sum(dim=2).mean()  # mean over (C, H)


def off_diagonal(x):
    # return a flat view of all off-diagonal elements of a square matrix x
    n = x.size(0)
    assert x.dim() == 2 and x.size(1) == n
    return x.flatten()[:-1].view(n - 1, n + 1)[:,1:].flatten()

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

## Uniformity loss
def coral_loss(pooled, target_scale=1.0):
    B, D = pooled.shape
    # 1) center
    mean = pooled.mean(dim=0, keepdim=True)           # [1×D]
    P_centered = pooled - mean                        # [B×D]
    # 2) cov matrix
    C = (P_centered.T @ P_centered) / (B - 1)          # [D×D]
    # 3) target covariance
    C_target = torch.eye(D, device=pooled.device) * (target_scale / D)
    # 4) Frobenius norm
    loss = torch.norm(C - C_target, p='fro')**2
    return loss


def gaussian_kernel(x, y, sigma=1.0):
    # x,y: [B×D], returns [B×B]
    dist2 = torch.cdist(x, y, p=2).pow(2).cuda()
    return torch.exp(-dist2 / (2 * sigma**2))

def mmd_loss(pooled, ref, sigma=1.0):
    Kxx = gaussian_kernel(pooled, pooled, sigma)
    Kyy = gaussian_kernel(ref, ref, sigma)
    Kxy = gaussian_kernel(pooled, ref, sigma)
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

def robust_emd_loss(pooled, alpha=0.2, beta=0.2, threshold=0.1, epsilon=1e-6):
    """
    Numerically stable implementation of Earth Mover's Distance loss
    with beta distribution target
    """
    batch_size, num_prototypes = pooled.shape
    
    # 1. Compute activation frequency with smoothing to avoid zeros
    act_freq = (pooled > threshold).float().mean(dim=0) + epsilon
    act_freq = act_freq / act_freq.sum()  # Re-normalize
    
    # 2. Generate target beta distribution - robust version
    x = torch.linspace(epsilon, 1-epsilon, num_prototypes, device=pooled.device)
    
    # 3. Clip alpha/beta to avoid extreme values
    alpha_safe = max(0.05, alpha)
    beta_safe = max(0.05, beta)
    
    # 4. Compute PDF values directly without beta function
    # Note: We don't need exact normalization since we normalize again after
    log_pdf = (alpha_safe-1)*torch.log(x) + (beta_safe-1)*torch.log(1-x)
    target_dist = torch.exp(log_pdf - log_pdf.max())  # Subtract max for numerical stability
    target_dist = target_dist / (target_dist.sum() + epsilon)
    
    # 5. Sort distributions (required for 1D Wasserstein distance)
    sorted_act_freq, _ = torch.sort(act_freq)
    sorted_target, _ = torch.sort(target_dist)
    
    # 6. Compute EMD with careful cumulative sum
    cdf_actual = torch.cumsum(sorted_act_freq, dim=0)
    cdf_target = torch.cumsum(sorted_target, dim=0)
    
    # 7. Add checks against NaN
    if torch.isnan(cdf_actual).any() or torch.isnan(cdf_target).any():
        print("Warning: NaN in CDFs before EMD calculation")
        # Fall back to a simpler loss
        return torch.tensor(1.0, device=pooled.device, requires_grad=True)
    
    emd = torch.abs(cdf_actual - cdf_target).mean()
    
    # 8. Add minimum activation constraint (ensures each prototype is used)
    min_act = 1.0 / batch_size  # At least one activation per batch
    min_act_penalty = torch.relu(min_act - act_freq).sum()
    
    # 9. Final safety check
    final_loss = emd + min_act_penalty
    if torch.isnan(final_loss):
        print("Warning: NaN in final EMD loss")
        return torch.tensor(1.0, device=pooled.device, requires_grad=True)
        
    return final_loss

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

    # Barlow like loss
    B, C = pooled1.shape
    z1 = (pooled1 - pooled1.mean(0)) / (pooled1.std(0) + EPS)
    z2 = (pooled2 - pooled2.mean(0)) / (pooled2.std(0) + EPS)
    C = (z1.T @ z2) / B
    on_diag  = torch.diagonal(C).add_(-1).pow(2).sum()
    off_diag = off_diagonal(C).pow(2).sum()
    bt_loss  = on_diag - off_diag


   # marginal uniformity loss (push each proto's batch-mean toward 1/D) - first order
    B, D = pooled1.shape                 # D = number of prototypes
    # per-prototype means across the batch
    m1 = pooled1.mean(dim=0)             # shape [D]
    m2 = pooled2.mean(dim=0)             # shape [D]
    # target for each prototype is 1/D
    target = torch.full_like(m1, 1.0 / D)
    # L2 toward that target
    u1 = ((m1 - target) ** 2).mean()
    u2 = ((m2 - target) ** 2).mean()
    

    emd_loss= 0.5 * robust_emd_loss(pooled1, alpha=0.2, beta=0.2, threshold=0.1)


    uni_proto_loss = 0.5 * (u1 + u2)


    # Maginal uniformity loss (push each proto's batch-mean toward 1/D) - second order
    c_loss = 0.5 * (coral_loss(pooled1) + coral_loss(pooled2))

    # inside calculate_loss
    # draw ref_samples ∼ Uniform on sphere or use a held‑out set
    ref = torch.randn(B, D)  # e.g. torch.randn(B,D); ref = F.normalize(ref,dim=1)
    ref = torch.nn.functional.normalize(ref, dim=1).to(device=pooled.device)


    mmd_l = 0.5 * (mmd_loss(pooled1, ref) + mmd_loss(pooled2, ref)) / 2

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
        #  barlow loss
        # loss += bt_loss * 5e-4
        uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
        loss += unif_weight * uni_loss

    # loss += 1e-4 * emd_loss
    # loss += 1e-4 * uni_proto_loss 
    # loss += c_loss * 1e-4
    # loss += mmd_l * 1e-4
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