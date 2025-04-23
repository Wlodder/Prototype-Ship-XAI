import torch
import random
import os
import argparse
from prototype_management import PrototypeManager
from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from prototype_squared import attribution
from util.vis_pipnet import  visualize_prototypes, visualize_prototype
from scipy.stats import beta
from util.evaluate_janes import check_prototype_locations

def split_and_merge_prototypes(args=None):
    """
    Split polysemantic prototypes and merge similar prototypes in a PIPNet model.
    """
    args = args or get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    
    # Create model
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(len(classes), args)
    
    net = PIPNet(num_classes=len(classes),
                 num_prototypes=num_prototypes,
                 feature_net=feature_net,
                 args=args,
                 add_on_layers=add_on_layers,
                 pool_layer=pool_layer,
                 classification_layer=classification_layer)
    
    net = net.to(device=device)
    net = torch.nn.DataParallel(net)
    
    # Load a pretrained model
    if args.state_dict_dir_net != '':
        print(f"Loading pretrained model from {args.state_dict_dir_net}")
        checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        raise ValueError("Please provide a pretrained model using --state_dict_dir_net")


    with torch.no_grad():
        xs1,  _ = next(iter(trainloader_normal))
        xs1 = xs1.to(device)
        proto_features, _, _, features = net(xs1, features_save=True)

        wshape = proto_features.shape[-1]
        args.wshape = wshape #needed for calculating image patch size
        
    if args.analyze_all_prototypes:
        for example_prototype in range(args.num_features):
            visualize_prototype(net, test_projectloader, len(classes), device, f'visualised_prototypes_represent',
                                args, prototype=example_prototype)

    protos_to_check = []
    # Check prototype locations
    if args.check_prototype_locations:
        print("Checking prototype locations...")
        img_stats = check_prototype_locations(
            net,
            device,
            args,
        )



        # Check the top overlaps for each image
        classification_weights = net.module._classification.weight
        for stat in img_stats:
            for gt_idx in stat.top_overlaps.keys():
                print(f"GT {gt_idx}: {stat.top_overlaps[gt_idx]}")
                for gt in stat.top_overlaps[gt_idx]:
                    for overlap in gt:
                        proto_overlap = overlap[0]
                        overlap_score = overlap[1]['box_overlap']
                        activation = overlap[1]['activation']
                        c_weight = torch.max(classification_weights[:,proto_overlap]) #ignore prototypes that are not relevant to any class
                        if overlap_score > 0.1 and activation > 0.1 and c_weight > 0.01:
                            print(f"Proto {proto_overlap}: {overlap_score}, Activation: {activation}")
                            protos_to_check.append(proto_overlap)

    protos_to_check = list(set(protos_to_check))
    # Initialize PrototypeManager
    prototype_manager = PrototypeManager(net, device=device)
    
    # Step 1: Analyze all prototypes to find polysemantic ones
    print("Analyzing prototypes to find polysemantic ones...")
    
    # Option 1: Analyze all prototypes (can be slow)
    prototype_indices = args.prototype_indices if hasattr(args, 'prototype_indices') else [3, 29, 30]  # Example prototypes
    
    # prototype_indices  = prototype_indices + [[3,22]]
    random_samples = 2


    prototype_indices = []
    final_pos = 0
    group_size=1
    for i in range(0,len(protos_to_check),group_size):
        prototype_indices.append(protos_to_check[i:i+group_size])
        final_pos = i
    prototype_indices.append(protos_to_check[final_pos:])
    
    # for i in range(random_samples):
    #     prototype_indices.append(list(np.random.randint(0, args.num_features, size=(3))))
    # prototype_indices  = [[3,22,102,103]]#, 135, 140]] 
    # prototype_indices  = [[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29],[30,31,32,33,34],[35,36,37,38,39],[40,41,42,43]]
    # example_prototypes = set()
    # prototype_indices  = [[0],[0,1],[0,1,2],[0,1,2,3]]
    # for i in range(len(prototype_indices)):
    #     example_prototypes = example_prototypes.union(set(prototype_indices[i]))


    # print(f"Visualizing prototypes...{example_prototypes}")
    # for example_prototype in example_prototypes:
    #     visualize_prototype(net, test_projectloader, len(classes), device, f'visualised_prototypes_represent',
    #                         args, prototype=example_prototype)
    # for i in range():
    #     prototype_indices.append(list(np.random.choice(args.num_features, size=(5), replace=False)))
    # prototype_indices  = [[3,22]]
    print(f"Analyzing {len(prototype_indices)} prototypes: {prototype_indices}")
    save_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images),"pure_prototypes")
    
    # Step 2: Split polysemantic prototypes
    if args.split_prototypes:
        print("\n--- SPLITTING POLYSEMANTIC PROTOTYPES ---")
        
        # Analyze the selected prototypes
        # split_results = prototype_manager.split_multiple_prototypes(
        #     trainloader_normal,
        #     prototype_indices,
        #     n_clusters=args.n_clusters,
        #     adaptive=args.adaptive_clustering,
        #     visualize=args.visualize_results,
        #     algorithm=args.clustering_algorithm  # Use the specified clustering algorithm
        # )

        # Set layer weights as a beta distribution
        values = beta.pdf(np.linspace(1, 0, 7), a=args.a, b=args.b)
        layer_weights = {}

        for i in range(1,8,1):
            layer_weights[i] = values[i-1]
        # layer_weights = { 7:1.0,
        #             6:1.1,
        #             5:1.6,
        #             4:1.0,
        #             3:0.3,
        #             2:0.0,
        #             1:0.0}
        split_results = prototype_manager.split_multiple_prototypes_multi_depth(
            trainloader_normal,
            prototype_indices,
            adaptive=args.adaptive_clustering,
            visualize=args.visualize_results,
            algorithm=args.clustering_algorithm,  # Use the specified clustering algorithm
            layer_weights=layer_weights,
            output_path=save_dir
        )

        # We return the new cluster centroids, for each level and place the centroids into a new model
        # Adatper that does the propagation in the backwards pass.

        print(split_results.keys())
        # for key, value in split_results.items():
        #     for x, item  in split_results[key]['centroids'].items():
        #         print(item.size())
        #         os.makedirs(f'pure_prototypes/{key}', exist_ok=True)
        #         for grad_prototype in item:
        #             torch.save(grad_prototype, os.path.join(f"pure_prototypes/{key}/centroid_{key}_{x}.pt"))
                
        p2model = attribution.EnhancedForwardPURE(
            net,
            device=device,
            layer_weights=layer_weights,
        )

        p2model.add_centroids(split_results)
        

        # Testing against the new model for inference mode
        xs1,  _ = next(iter(trainloader_normal))
        xs1 = xs1.to(device)
        results = p2model.enhanced_classification(xs1, custom_prototypes=prototype_indices[0])


        for centroid_match in results['centroid_matches']:
            for match in centroid_match:
                print(match)

        # for centroid_match in results['matches']:
        #     for match in centroid_match:
        #         print(match)

    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIPNet Prototype Management')
    
    # Add arguments specific to this script
    parser.add_argument('--analyze_all_prototypes', action='store_true', 
                        help='Analyze all prototypes in the model')
    parser.add_argument('--prototype_indices', type=int, nargs='+', default=[3, 29, 30],
                        help='List of prototype indices to analyze')
    
    # Splitting options
    parser.add_argument('--split_prototypes', action='store_true', 
                        help='Analyze and split polysemantic prototypes')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for splitting (if None, determined automatically)')
    parser.add_argument('--adaptive_clustering', action='store_true', default=False,
                        help='Adaptively determine number of clusters')
    parser.add_argument('--apply_splitting', action='store_true',
                        help='Actually modify the model by splitting prototypes')
    parser.add_argument('--splitting_scale', type=float, default=0.7,
                        help='Scaling factor for split prototypes (controls how different they are)')
    parser.add_argument('--save_expanded_model', action='store_true', 
                        help='Save the model with expanded prototypes')
    parser.add_argument('--clustering_algorithm', type=str, default='kmeans',
                        choices=['kmeans', 'hdbscan', 'gmm', 'spectral','hdbscan_kmeans'],
                        help='Clustering algorithm to use for prototype splitting')
    parser.add_argument('--use_adaptive_expansion', action='store_true', default=True,
                        help='Use adaptive step size when expanding prototypes')
    parser.add_argument('--use_manifold_projection', action='store_true', default=True,
                        help='Project new prototypes onto the prototype manifold')
    
    # Merging options
    parser.add_argument('--merge_prototypes', action='store_true',
                        help='Identify and merge similar prototypes')
    parser.add_argument('--merge_threshold', type=float, default=0.85,
                        help='Similarity threshold for merging prototypes')
    parser.add_argument('--similarity_type', type=str, default='combined', 
                        choices=['feature', 'weight', 'combined'],
                        help='Type of similarity to use for identifying merge candidates')
    parser.add_argument('--min_weight', type=float, default=0.01,
                        help='Minimum classification weight to consider a prototype for merging')
    parser.add_argument('--max_pairs_to_merge', type=int, default=10,
                        help='Maximum number of prototype pairs to merge')
    parser.add_argument('--merge_strategy', type=str, default='weighted_average',
                        choices=['weighted_average', 'max'],
                        help='Strategy for merging prototypes')
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('-b', type=int, default=2)
    parser.add_argument('-a', type=int, default=10)

    
    # Visualization options
    parser.add_argument('--visualize_results', action='store_true', default=True,
                        help='Visualize results of analysis')
    parser.add_argument('--visualization_dir', type=str, default=None,
                        help='Custom directory for saving visualizations')
    parser.add_argument('--vis_samples', type=int, default=10,
                        help='Number of samples to show per prototype in visualizations')
    parser.add_argument('--vis_max_prototypes', type=int, default=50,
                        help='Maximum number of prototypes to visualize')
    parser.add_argument('--check_prototype_locations', action='store_true', default=True,
                        help='Do we want to evaluate which prototpyes are useful?')
    
    # Get default PIPNet arguments
    args = get_args(parser)
    
    # Run the script
    split_and_merge_prototypes(args)