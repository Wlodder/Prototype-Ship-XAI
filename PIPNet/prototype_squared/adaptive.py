from torch import nn
import torch
from tqdm import tqdm
from pipnet.losses import align_loss, budget_loss, sharing_loss, calculate_loss

from matplotlib import pyplot as plt


class AdaptivePrototypeRotation(nn.Module):
    """
    Implements prototype rotation by temporarily deactivating recently used prototypes.
    
    The mechanism works by:
    1. Tracking which prototypes contribute significantly to classification decisions
    2. Applying a temporary suppression to these prototypes using an exponential decay
    3. Gradually reactivating prototypes after they've been unused for some time
    """
    def __init__(self, 
                 num_prototypes, 
                 num_classes, 
                 tau=5.0,  # Decay rate parameter (higher = slower reactivation)
                 threshold=0.1,  # Threshold for considering a prototype "used"
                 device='cuda'):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.tau = tau
        self.threshold = threshold
        self.device = device
        
        # Register cooldown timer for each prototype
        self.register_buffer('cooldown_timers', torch.zeros(num_prototypes, device=device))
        
        # Track influence of each prototype on classification decisions
        self.register_buffer('prototype_influence', torch.zeros(num_prototypes, device=device))
        
        # Flag to enable/disable the rotation mechanism
        self.enabled = True
    
    def forward(self, pooled, class_weights):
        """
        Apply adaptive rotation by scaling prototype activations based on cooldown timers.
        
        Args:
            pooled: Prototype activations [batch_size, num_prototypes]
            class_weights: Classification weights [num_classes, num_hypotheses, num_prototypes]
                           or [num_classes, num_prototypes] depending on architecture
        
        Returns:
            Modified pooled activations with cooldown applied
        """
        if not self.enabled or not self.training:
            return pooled
        
        # Calculate suppression factor based on timers: (1 - e^(-t/tau))
        # This means:
        # - When t=0 (just activated): factor = 0 (fully suppressed)
        # - As t increases: factor approaches 1 (fully active)
        suppression_factor = 1.0 - torch.exp(-self.cooldown_timers / self.tau)
        
        # Apply suppression to prototype activations
        modified_pooled = pooled * suppression_factor.unsqueeze(0)
        
        return modified_pooled
    
    def update_timers(self, pooled, logits, targets):
        """
        Update cooldown timers based on which prototypes were influential
        for correct classifications in this batch.
        
        Args:
            pooled: Original prototype activations [batch_size, num_prototypes]
            logits: Model output logits [batch_size, num_classes]
            targets: Ground truth targets [batch_size]
        """
        if not self.enabled or not self.training:
            return
        
        # Increment all timers by 1 (passage of time)
        self.cooldown_timers += 1
        
        batch_size = pooled.size(0)
        
        # Get predictions
        _, predictions = torch.max(logits, dim=1)
        
        # Identify correctly classified samples
        correct_mask = (predictions == targets).float()
        
        # Find prototypes that contributed significantly to correct classifications
        # by examining their activations
        significant_activations = (pooled > self.threshold).float()
        
        # Weight by correctness of prediction
        weighted_activations = significant_activations * correct_mask.unsqueeze(1)
        
        # Sum over batch to get total influence of each prototype
        batch_influence = weighted_activations.sum(dim=0)
        
        # Update prototype influence with exponential moving average
        alpha = 0.2  # EMA weight
        self.prototype_influence = (1-alpha) * self.prototype_influence + alpha * batch_influence
        
        # Reset timers for prototypes that were significantly used in this batch
        used_prototype_mask = (batch_influence > 0).float()
        self.cooldown_timers = self.cooldown_timers * (1 - used_prototype_mask)
    
    def get_most_influential_prototypes(self, top_k=10):
        """
        Get the indices of the most influential prototypes based on
        their contribution to correct classifications.
        
        Args:
            top_k: Number of top prototypes to return
            
        Returns:
            Indices of top_k most influential prototypes
        """
        values, indices = torch.topk(self.prototype_influence, k=min(top_k, self.num_prototypes))
        return indices.cpu().numpy(), values.cpu().numpy()
    
    def visualize_prototype_status(self, k=20):
        """
        Create a visualization of prototype status showing cooldown times
        and influence scores.
        
        Args:
            k: Number of top prototypes to highlight
            
        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get prototype data
        cooldown = self.cooldown_timers.cpu().numpy()
        influence = self.prototype_influence.cpu().numpy()
        
        # Create suppression factors for visualization
        suppression = 1.0 - np.exp(-cooldown / self.tau)
        
        # Sort prototypes by influence
        sorted_indices = np.argsort(influence)[::-1]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot cooldown status
        bars = ax1.bar(range(self.num_prototypes), cooldown[sorted_indices])
        ax1.set_xlabel('Prototype Index (sorted by influence)')
        ax1.set_ylabel('Cooldown Timer')
        ax1.set_title('Prototype Cooldown Status')
        
        # Highlight top-k prototypes
        for i in range(min(k, self.num_prototypes)):
            bars[i].set_color('red')
        
        # Plot activation suppression
        ax2.bar(range(self.num_prototypes), suppression[sorted_indices], color='green')
        ax2.set_xlabel('Prototype Index (sorted by influence)')
        ax2.set_ylabel('Activation Factor')
        ax2.set_title('Prototype Activation Factor (1=fully active, 0=suppressed)')
        ax2.set_ylim(0, 1.1)
        
        plt.tight_layout()
        return fig

    # In the AdaptivePrototypeRotation class
    def set_warmup_schedule(self, current_epoch, warmup_epochs=10, max_epochs=100):
        """Gradually increase rotation strength based on training progress"""
        if current_epoch < warmup_epochs:
            # Disable during warmup
            self.enabled = False
        else:
            # Enable after warmup
            self.enabled = True
            
            # Gradually increase tau (slower reactivation) as training progresses
            progress = min(1.0, (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs))
            self.tau = 5.0 + progress * 15.0  # Increase from 5 to 20

class CompetingHeadWithRotation(nn.Module):
    """
    Extension of CompetingHead that incorporates adaptive prototype rotation.
    This directly integrates the rotation mechanism into the classification layer.
    """
    def __init__(self, num_classes, num_prototypes, num_hypotheses=3, 
                 normalization_multiplier=1.0, tau=5.0):
        super().__init__()
        self.weight = nn.Parameter(
            0.01 * torch.randn(num_classes, num_hypotheses, num_prototypes))
        self.normalization_multiplier = nn.Parameter(torch.tensor([normalization_multiplier]))
        self._num_hyp = num_hypotheses
        
        # Create the rotation mechanism
        self.rotation = AdaptivePrototypeRotation(
            num_prototypes=num_prototypes,
            num_classes=num_classes,
            tau=tau
        )
        
        # Track whether last forward was for inference or training
        self.last_pooled = None
        self.last_logits = None

    def enable(self):
        """Enable the rotation mechanism"""
        self.rotation.enabled = True

    def disable(self):
        """Enable the rotation mechanism"""
        self.rotation.enabled = False
    
    def forward(self, pooled):
        # Store original pooled for later use in update_timers
        self.last_pooled = pooled
        
        # Apply rotation mechanism to pooled activations
        modified_pooled = self.rotation(pooled, self.weight)
        
        # Standard competing head logic with modified activations
        logits = torch.einsum('bd,chd->bch', modified_pooled, self.weight)
        logits, _ = logits.max(dim=2)  # max over hypotheses
        
        # Store logits for later use in update_timers
        self.last_logits = logits.detach()
        
        return logits * self.normalization_multiplier
    
    def update_rotation_timers(self, targets):
        """
        Update the rotation timers based on the results of the last forward pass.
        
        Args:
            targets: Ground truth targets [batch_size]
        """
        if self.last_pooled is not None and self.last_logits is not None:
            self.rotation.update_timers(self.last_pooled, self.last_logits, targets)


def replace_classification_with_rotation(model, tau=5.0):
    """
    Replace the model's classification layer with a version that has
    adaptive prototype rotation built in.
    
    Args:
        model: PIPNet model
        tau: Decay parameter for prototype rotation
        
    Returns:
        Modified model with prototype rotation
    """
    # Get current classification layer
    old_head = model.module._classification
    
    # Create new head with rotation
    head_with_rotation = CompetingHeadWithRotation(
        num_classes=old_head.weight.size(0),
        num_prototypes=old_head.weight.size(2),
        num_hypotheses=old_head.weight.size(1),
        normalization_multiplier=old_head.normalization_multiplier,
        tau=tau
    )
    
    # Copy weights
    head_with_rotation.weight.data.copy_(old_head.weight.data)
    
    # Replace head
    model.module._classification = head_with_rotation
    
    return model


def train_with_prototype_rotation(net, train_loader, optimizer_net, optimizer_classifier, 
                                 scheduler_net, scheduler_classifier, criterion, epoch, 
                                 nr_epochs, device, pretrain=False, finetune=False):
    """
    Training function incorporating prototype rotation.
    
    This is based on the original train_pipnet function with added
    prototype rotation functionality.
    """
    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer
        net.module._classification.requires_grad = True
        progress_prefix = 'Train Epoch'
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"{progress_prefix} {epoch}",
                    mininterval=2.,
                    ncols=0)
    
    # Set loss weights based on training phase
    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1
        unif_weight = 0.5
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 3. 
        t_weight = 3.
        unif_weight = 0.
        cl_weight = 2.
    
    print(f"Align weight: {align_pf_weight}, U_tanh weight: {t_weight}, Class weight: {cl_weight}", flush=True)
    print(f"Pretrain? {pretrain}, Finetune? {finetune}", flush=True)
    
    lrs_net = []
    lrs_class = []
    
    # Iterate through the data set
    for i, (xs1, xs2, ys) in train_iter:       
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        
        # Calculate loss
        loss, acc = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, 
                                   unif_weight, cl_weight, net.module._classification.normalization_multiplier, 
                                   pretrain, finetune, criterion, train_iter, print=True, EPS=1e-8)
        
        # Add budget and sharing loss if using CompetingHead
        head = net.module._classification
        if hasattr(head, 'weight') and len(head.weight.shape) == 3:
            budget_weight = 1e-4
            share_weight = 1e-3
            budget_l = budget_loss(head, budget_weight)
            share_l = sharing_loss(head, share_weight, p=2.0)
            loss = loss + budget_l + share_l
        
        # Compute the gradient
        loss.backward()

        # Optimization steps
        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step()
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        # Update tracking metrics
        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        # Apply weight regularization
        if not pretrain:
            with torch.no_grad():
                # Standard weight regularization
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)
                )
                
                # Update prototype rotation timers
                if hasattr(net.module._classification, 'update_rotation_timers'):
                    net.module._classification.update_rotation_timers(torch.cat([ys, ys]))
    
    # Record training info
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    # Visualize prototype rotation status at end of epoch
    if hasattr(net.module._classification, 'rotation') and epoch % 5 == 0:
        rotation = net.module._classification.rotation
        fig = rotation.visualize_prototype_status()
        fig.savefig(f'prototype_rotation_epoch_{epoch}.png')
        plt.close(fig)
    
    return train_info