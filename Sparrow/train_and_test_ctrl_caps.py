import time
import torch
import torch.nn.functional as F
import pdb
import math

from helpers import list_of_distances, make_one_hot
from settings import sep_cost_filter, sep_cost_cutoff, use_cap, debug, sub_mean, ltwo

def angular_similarity_loss_fn(model):
    """
    Loss to decorrelate prototypes (especially within the same class).
    This encourages prototypes to represent different semantic concepts.
    
    Formula: L_AS = -1/C * ∑_ν=1^C max_(i,j∈I_ν) log(1 - AS(p_i, p_j))
    """
    # Get prototype vectors
    prototype_vectors = model.module.prototype_vectors
    num_prototypes = model.module.num_prototypes
    num_classes = model.module.num_classes
    
    # Reshape vectors to 2D for easier computation
    p = prototype_vectors.view(num_prototypes, -1)
    
    # Get class identity for each prototype
    prototype_class_identity = model.module.prototype_class_identity
    
    # Compute cosine similarity between all prototypes
    p_normalized = F.normalize(p, p=2, dim=1)  # Normalize to unit vectors
    cosine_sim = torch.mm(p_normalized, p_normalized.t())
    
    # Convert to angular similarity: AS(p_i, p_j) = 1 - 1/π * arccos(CS(p_i, p_j))
    # Clamp cosine_sim to avoid numerical issues with arccos
    cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angular_sim = 1.0 - torch.acos(cosine_sim) / math.pi
    
    # Set diagonal elements to 0 (ignore self-similarity)
    angular_sim.fill_diagonal_(0)
    
    # Compute loss for each class
    total_loss = 0.0
    for c in range(num_classes):
        # Get indices of prototypes for this class
        class_mask = prototype_class_identity[c].bool()
        if torch.sum(class_mask) <= 1:
            continue  # Skip if class has 0 or 1 prototype
            
        # Get angular similarities between all prototypes of this class
        intra_class_sim = angular_sim[class_mask][:, class_mask]
        
        # Take maximum similarity (most correlated pair)
        max_sim = torch.max(intra_class_sim)
        
        # Add to loss: -log(1 - max_sim) to minimize high correlations
        # Add small epsilon to avoid log(0)
        class_loss = -torch.log(1.0 - max_sim + 1e-7)
        total_loss += class_loss
    
    # Average over number of classes
    return total_loss / num_classes

def prototype_sample_distance_loss_fn(model, min_distances):
    """
    Loss to keep prototypes close to samples in latent space.
    
    Formula: L_PSD = -1/m * ∑_j=1^m log(1 - PSD_j(X, p_j) / dist_max)
    """
    num_prototypes = model.module.num_prototypes
    
    # Maximum possible distance in the latent space
    max_dist = (model.module.prototype_shape[1] * 
                model.module.prototype_shape[2] * 
                model.module.prototype_shape[3])
    
    # For each prototype, find the minimum distance to any sample patch
    # min_distances already has this information from the forward pass
    # Shape of min_distances: [batch_size, num_prototypes]
    
    # Get minimum distance across the batch for each prototype
    min_dist_per_prototype = torch.min(min_distances, dim=0)[0]
    
    # Normalize distances
    normalized_dist = min_dist_per_prototype / max_dist
    
    # Compute loss: -log(1 - normalized_dist)
    # Clamp to avoid numerical issues
    normalized_dist = torch.clamp(normalized_dist, 0.0, 1.0 - 1e-7)
    loss = -torch.log(1.0 - normalized_dist).mean()
    
    return loss


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, clst_k=1,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        #print('current index for loader is',i)
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req: #, torch.autograd.detect_anomaly():
            # nn.Module has implemented __call__() function
            # so no need to call .forward

            output, min_distances = model(input)
            cap = torch.linalg.norm(model.module.cap_width_l2)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            
            max_dist = (model.module.prototype_shape[1]
                        * model.module.prototype_shape[2]
                        * model.module.prototype_shape[3])

            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
 
            #calculate cluster cost
            if clst_k == 1: 
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            else: 
                # clst_k is a hyperparameter that lets the cluster cost apply in a "top-k" fashion: the original cluster cost is equivalent to the k = 1 casse
                inverted_distances, _ = torch.topk((max_dist - min_distances) * prototypes_of_correct_class, k = clst_k, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)
            #calculate separation cost
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg separation cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)

            # Sparrow loss components
            angular_similarity_loss = angular_similarity_loss_fn(model)
            prototype_sample_distance_loss = prototype_sample_distance_loss_fn(model, min_distances)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            foo = total_cross_entropy / n_batches
            if math.isnan(foo): 
                print('iteration: ', i)
                print('detected nan!')
                print('total cross entropy: ', total_cross_entropy)
                print('n_batches: ', n_batches)
                print('cross entropy for this batch: ', cross_entropy.item())
                print('target: ', target)
                print('output: ', output)
                torch.save(model, 'nanmodel.pth')
                exit()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                              + coefs['clst'] * cluster_cost
                              + coefs['sep'] * separation_cost
                              + coefs['l1'] * l1
                              +coefs['cap']*cap 
                              + coefs.get('ang_sim', 1.0) * angular_similarity_loss
                              + coefs.get('proto_sample_dist', 1.0) * prototype_sample_distance_loss)
                    if math.isnan(loss): 
                        print('loss is nan!')
                        print('cross_entropy: ', cross_entropy)
                        print('cluster_cost: ', cluster_cost)
                        print('separation_cost: ', separation_cost)
                        print('l1: ', l1)
                        print('angular_similarity_loss: ', angular_similarity_loss)
                        print('prototype_sample_distance_loss: ', prototype_sample_distance_loss)
                        exit()
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1.0 * angular_similarity_loss + 100 * prototype_sample_distance_loss
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1
                          + coefs.get('ang_sim', 1.0) * angular_similarity_loss
                          + coefs.get('proto_sample_dist', 1.0) * prototype_sample_distance_loss
                          )
                else:
                    loss = cross_entropy + 1e-4 * l1 + 1.0 * angular_similarity_loss + 100 * prototype_sample_distance_loss
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    log('\tcap loss: \t\t{0}'.format(cap))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, clst_k=1, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, clst_k = clst_k, coefs=coefs, log=log)

def test(model, dataloader, class_specific=False, clst_k=1, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, clst_k=clst_k, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')
        

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')

def stable_log(p): 
    one_mask = (p == 0.).nonzero()
    return torch.log(p + one_mask)