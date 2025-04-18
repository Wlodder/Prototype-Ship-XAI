from sklearn.cluster import KMeans
import umap
from sklearn.cluster import HDBSCAN
from typing import List

from sklearn.decomposition import PCA
# from k_means_contrained import KMeansConstrained
from pipnet.pipnet import PIPNet, get_network
from tqdm import tqdm
from util.log import Log
import torch.nn as nn
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from util.vis_pipnet import visualize, visualize_topk, visualize_prototypes, visualize_prototype
from util.pure_expander import expand_pipnet_with_pure, expand_pipnet_with_pure_centroids
from util.proto_max import *
from util.proto_program import visualize_prototype_with_program
# from pipnet.train import train_pipnet_memory
import torch
from util.pure_vis import *
import sys, os
import random
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from util.visualization_tool import *

from util.pure import disentangle_polysemantic_prototypes
# from util.pure_h import  disentangle_multi_layer_prototypes

import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_pairs(number_pairs):
    """
    Creates a graph visualization from a list of number pairs.
    
    Parameters:
    -----------
    number_pairs : list of tuples
        List of pairs of numbers, where each pair represents an edge between nodes
        
    Returns:
    --------
    matplotlib figure
    """
    # Create an empty graph
    G = nx.Graph()
    
    pairs = {}


    # Add edges (and nodes automatically) from the number pairs
    for pair in number_pairs:
        G.add_edge(pair[0], pair[1])


        pair_key = f'{pair[0]}_{pair[1]}'
        if not pair_key in pairs:
            pairs[pair_key] = 0

        pairs[pair_key] += 1

    node_list = [] 
    for node in G.nodes:
        # print(node, G.degree[node])
        node_list.append((node, G.degree[node]))

    node_list.sort(key = lambda x : x[1], reverse=True)
    
    
    # other_removes1 = G.adj[node_list[0][0]].copy()
    # other_removes2 = G.adj[node_list[1][0]].copy()

    # other_removes = set(list(other_removes1.keys()) + list(other_removes2.keys()))

    # for i in other_removes:
    #     G.remove_node(i)

    # Remove the sea and the sky nodes

    # G.remove_node(node_list[0][0])
    # G.remove_node(node_list[1][0])

    adj_list = []
    for n in node_list[:12]:
        print(n, G.adj[n[0]])

        adj_list.append(G.adj[n[0]])

    print(nx.community.louvain_communities(G))
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Generate layout for node positions
    pos = nx.spring_layout(G, seed=42)  # for reproducible layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G


def run_pipnet(args=None):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification")
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.")
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids)
    
    # Obtain the dataset and dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if len(classes)<=20:
        if args.validation_size == 0.:
            print("Classes: ", testloader.dataset.class_to_idx)
        else:
            print("Classes: ", str(classes))
    
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, addon_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
   
    
    net = PIPNet(num_classes=len(classes),
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers=addon_layers,
                    pool_layer=pool_layer,
                    classification_layer = classification_layer
                    )
    
    net = net.to(device=device)
    net = nn.DataParallel(net,device_ids=device_ids)
    #pure = PURE(net, device=device)
    
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded")
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*len(classes)): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained backbone only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...")
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item())
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
            
 


    #         print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item())

    # prototypes = net.module._add_on[0].weight.data.squeeze(2).squeeze(2)
    
    # distances = np.zeros((prototypes.size(0), prototypes.size(0)))

    # for i in range(600):
    #     for j in range(i, 600):
    #         distances[i,j] = torch.dot(prototypes[i], prototypes[j]).cpu()
        
    # plt.imshow(distances)
    # plt.show()


    # Forward one batch through the backbone to get the latent output size
    distances = [ 0.4, 0.45]
    # distances = [0.35]
    prototypes = net.module._add_on[0].weight.data.squeeze(2).squeeze(2)
    occs = [[] for x in distances]

    

    # target vectors
    target_vector = 106
    neighbours = [set() for x in distances]
    high_prototypes = [[] for x in distances]

    with torch.no_grad():
        for x in range(10):
            xs1,  _ = next(iter(trainloader_normal))
            xs1 = xs1.to(device)
            proto_features, _, _, features = net(xs1, features_save=True)

            wshape = proto_features.shape[-1]
            args.wshape = wshape #needed for calculating image patch size
            print("Output shape: ", proto_features.shape)
            batch_shape = proto_features.size()
            for b in range(batch_shape[0]):
                new_img = proto_features[b].permute(1,2,0)
                image_shape = new_img.size()

                for h in range(image_shape[0]):
                    for w in range(image_shape[1]):

                        for occ, dist, neighbour, hp in zip(occs, distances, neighbours, high_prototypes):

                            vector = new_img[h,w]
                            indicies = torch.nonzero(vector > dist)
                            
                            if len(indicies) > 1:
                                torch.sort(indicies.flatten())
                                occ.append(indicies)

                                x, y = prototypes[indicies[0][0]], prototypes[indicies[1][0]]
                                feature_vector = features[b].permute(1,2,0)[h,w].cpu()
                                # print(indicies,feature_vector-x,y-feature_vector, torch.norm(x-y))

                                if indicies[0][0] == target_vector:
                                    neighbour.add(indicies[1][0].item())
                                    # hp.append(feature_vector)
                                elif indicies[1][0] == target_vector:
                                    neighbour.add(indicies[0][0].item())
                                    # hp.append(feature_vector)
                                hp.append(feature_vector)
                                # if len(indicies)> 2:
                                #     print(indicies)

        # print(high_prototypes[0].size())
        reducer = umap.UMAP(2)
        emb_prot = reducer.fit_transform(prototypes.cpu())
        emb_samples = reducer.fit_transform(np.array(high_prototypes[0])[:len(high_prototypes[0])//2 ])
        # kmeans = KMeans(20)
        # predictions = kmeans.fit_predict(emb_prot)

        # full = np.concatenate([emb_prot, emb_samples])

        # plt.scatter(full[:,0], full[:,1])
        fig, ax = plt.subplots(figsize=(10,8))

        ax.scatter(emb_prot[:,0], emb_prot[:,1], color='red', s=100,label='prototypes')
        for i, (x, y) in enumerate(emb_prot):
            ax.text(x+0.01, y+0.01, f"P_{i}", fontsize=9)
        plt.scatter(emb_samples[:,0], emb_samples[:,1], color='lightgray', s=15, label='samples')
        plt.show()


        # for dist in hp:
        #     pca = PCA(10)
        #     dist = torch.cat(dist).cpu().numpy()
        #     pca = pca.fit(dist)
            
        print(neighbours)
        # modal_occs = torch.cat(modal_occs, dim=0)
        # print(modal_occs)
        # for mod in modal_occs:
        #     print(modal_occs)
        node_lists = []

        for occ in occs:
            modal_occs = [(x[0].item(), x[1].item())  for x in occ]
            modal_hist = {}

            unique_num = set()
            for x, y in modal_occs:
                unique_num.add(x)
                unique_num.add(y)

                hist_label = f'{x}_{y}'
                if hist_label not in modal_hist:
                    modal_hist[hist_label] = 0
                modal_hist[hist_label] += 1

            graph = create_graph_from_pairs(modal_occs)
            
            # visualizer = run_prototype_visualization(
            #     model=net, 
            #     dataset=trainloader_normal.dataset,
            #     reducer=reducer,
            #     prototype_graph=graph
            # )
            # plt.plot(list(modal_hist.keys()), list(modal_hist.values()))
            # plt.show()

        # for key, tm in modals.items():
        #     print(key, tm)
        allocations = torch.argmax(proto_features, dim=1)
        entropies = -torch.sum(proto_features * torch.log2(proto_features), dim=1)
        f, axes = plt.subplots(3, allocations.size(0))

        for i in range(len(allocations)):
            axes[0, i].imshow(allocations[i].detach().cpu())
            axes[1, i].imshow(xs1[i].permute(1,2,0).detach().cpu())
            axes[2, i].imshow(entropies[i].detach().cpu())


        plt.show()

        # visualize_prototypes(net, projectloader, len(classes), device, f'visualised_prototypes_after_split', args)

    full_vis_path = args.state_dict_dir_net.split('/')[-1]
    visualize(net, test_projectloader, len(classes), device, f'visualised_prototypes_test_{full_vis_path}', args)


    # Polysemantic <- this already works
    # prototypes = [67,105,227, 553]  # 

    # Same concept <- needs some work
    # prototypes = [[15, 32], [67, 15],[359, 516]]
    # prototypes = [67,[15, 32], [359, 516]]
    # prototypes = [26, [3, 12], 67, [359, 516]]#, [15,32]]
    #print(node_lists[0])
    # prototypes = [list(node_lists[0][0].keys())[:6]]
    prototypes = list(range(10))
    # prototypes = [522, [58, 522]]
    # prototypes = [[102, 12, 49], [359, 516]]#, [15,32]]
    # scalings = [1.0, 2.0, 5.0, 10.0]
    print(prototypes)
    scalings = [0.7]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # model = net
    # for prototype_idx in range(0,400):
    #     # prototype_idx = 26
    #     # Basic usage

    #     image, components = multi_resolution_prototype_activation(
    #         model, 
    #         prototype_idx=prototype_idx,  # Specify which prototype to visualize
    #         device='cuda',
    #         iterations=1000,    # Number of optimization iterations
    #         learning_rate=0.003, # Learning rate for optimization
    #         alpha_l1=0.0001,   # L1 regularization strength
    #         alpha_tv=0.0001,    # Total variation regularization
    #         blur_sigma=0.2,     # Blur sigma for smoothing
    #         resolutions = resolutions
    #     )
    #     torchvision.utils.save_image(image, f'{args.log_dir}/prototype_{prototype_idx}_multi_resolution.png')



           ## Save pytorch image
    
    results = disentangle_polysemantic_prototypes(
        model=net,
        dataloader=trainloader_normal,
        prototypes=prototypes,
        n_clusters=2
    )

    old_net = net

    all_centroids = []
    for ps, result in results.items():
        
        for centroid in result['centroids']:
            
            all_centroids.append(centroid)
    

    for scaling in scalings:
        net = old_net
        #new_model = expand_pipnet_with_pure(net, results, device=device)
        new_model, prototype_mapping = expand_pipnet_with_pure_centroids(net.module, results, device=device, scaling=scaling)
        new_model = new_model.to(device=device)
        net.module = new_model

        print(prototype_mapping)
        
        count_param=0
        for name, param in net.named_parameters():
            if param.requires_grad:
                count_param+=1           
        print("Number of parameters that require gradient: ", count_param, flush=True)

        
        total_new_clusters = 0 
        for proto_idx, result in results.items():
            n_clusters = len(np.unique(result['cluster_labels']))
            if isinstance(proto_idx, List):
                total_new_clusters += n_clusters - len(proto_idx)
            else:
                total_new_clusters += (n_clusters - 1)  # Subtract 1 as the first cluster uses original prototype

        prototypes = prototypes + list(range(args.num_features, args.num_features + total_new_clusters))

        print(prototypes)
        for prototype in prototypes:
            
            if isinstance(prototype, List):
                for subprototype in prototype:

                    visualize_prototype(net, projectloader, len(classes), device, f'visualised_prototypes_after_split_{scaling}', args, subprototype)
            else:
                visualize_prototype(net, projectloader, len(classes), device, f'visualised_prototypes_after_split_{scaling}', args, prototype)
        # visualize_prototypes(new_model, projectloader, len(classes), device, f'visualised_prototypes_after_split', args)
  


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_pipnet(args)
    
    sys.stdout.close()
    sys.stderr.close()
