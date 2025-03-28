import torch
import argparse
from prototype_management import PrototypeManager
from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Initialize PrototypeManager
    prototype_manager = PrototypeManager(net, device=device)
    
    # Step 1: Analyze all prototypes to find polysemantic ones
    print("Analyzing prototypes to find polysemantic ones...")
    
    # Option 1: Analyze all prototypes (can be slow)
    if args.analyze_all_prototypes:
        all_protos = np.arange(net.module._num_prototypes)
        # Filter to only include prototypes with non-zero class weights
        used_protos = []
        class_weights = net.module._classification.weight.data
        for p in all_protos:
            if torch.max(class_weights[:, p]) > 0.01:  # Threshold for considering a prototype used
                used_protos.append(p)
        
        # Analyze a subset of the used prototypes to save time
        sample_size = min(50, len(used_protos))  # Limit to 50 prototypes for analysis
        prototype_indices = np.random.choice(used_protos, size=sample_size, replace=False)
    
    # Option 2: Use predefined list of prototypes
    else:
        prototype_indices = args.prototype_indices if hasattr(args, 'prototype_indices') else [3, 29, 30]  # Example prototypes
    
    print(f"Analyzing {len(prototype_indices)} prototypes: {prototype_indices}")
    
    # Step 2: Split polysemantic prototypes
    if args.split_prototypes:
        print("\n--- SPLITTING POLYSEMANTIC PROTOTYPES ---")
        
        # Analyze the selected prototypes
        split_results = prototype_manager.split_multiple_prototypes(
            trainloader_normal,
            prototype_indices,
            n_clusters=args.n_clusters,
            adaptive=args.adaptive_clustering,
            visualize=args.visualize_results
        )
        
        # Find prototypes that are actually polysemantic
        polysemantic_prototypes = [
            proto_idx for proto_idx, result in split_results.items() 
            if result['is_polysemantic']
        ]
        
        print(f"Found {len(polysemantic_prototypes)} polysemantic prototypes: {polysemantic_prototypes}")
        
        if len(polysemantic_prototypes) > 0 and args.apply_splitting:
            # Filter results to only include polysemantic prototypes
            poly_results = {idx: split_results[idx] for idx in polysemantic_prototypes}
            
            # Expand the model with new prototypes
            print(f"Expanding model with split prototypes (scaling={args.splitting_scale})...")
            expanded_model, prototype_mapping = prototype_manager.expand_model_with_split_prototypes(
                poly_results, scaling=args.splitting_scale
            )
            
            print(f"Model expanded from {num_prototypes} to {expanded_model.module._num_prototypes} prototypes")
            print("Prototype mapping:", prototype_mapping)
            
            # Save the expanded model if requested
            if args.save_expanded_model:
                save_path = f"{args.log_dir}/expanded_model.pt"
                torch.save({
                    'model_state_dict': expanded_model.state_dict(),
                    'prototype_mapping': prototype_mapping,
                    'polysemantic_prototypes': polysemantic_prototypes
                }, save_path)
                print(f"Expanded model saved to {save_path}")
    
    # Step 3: Identify and merge similar prototypes
    if args.merge_prototypes:
        print("\n--- MERGING SIMILAR PROTOTYPES ---")
        
        # Find candidate pairs for merging
        merge_candidates = prototype_manager.identify_merge_candidates(
            similarity_threshold=args.merge_threshold,
            similarity_type=args.similarity_type,
            min_weight=args.min_weight
        )
        
        print(f"Found {len(merge_candidates)} candidate pairs for merging")
        
        # Show the top candidates
        if len(merge_candidates) > 0:
            print("\nTop merge candidates:")
            for p1, p2, sim in merge_candidates[:5]:  # Show top 5
                print(f"  Prototypes {p1} and {p2}: similarity = {sim:.4f}")
            
            # Visualize a few merge candidates if requested
            if args.visualize_results and len(merge_candidates) > 0:
                for i in range(min(3, len(merge_candidates))):
                    p1, p2, _ = merge_candidates[i]
                    prototype_manager.compare_prototypes([p1, p2], trainloader_normal)
            
            # Apply merging if requested
            if args.apply_merging:
                # Convert to list of pairs
                pairs_to_merge = [(p1, p2) for p1, p2, _ in merge_candidates[:args.max_pairs_to_merge]]
                
                # Merge the prototypes
                print(f"Merging {len(pairs_to_merge)} prototype pairs...")
                merged_model = prototype_manager.merge_prototypes(
                    pairs_to_merge, merge_strategy=args.merge_strategy
                )
                
                # Save the merged model if requested
                if args.save_merged_model:
                    save_path = f"{args.log_dir}/merged_model.pt"
                    torch.save({
                        'model_state_dict': merged_model.state_dict(),
                        'merged_pairs': pairs_to_merge
                    }, save_path)
                    print(f"Merged model saved to {save_path}")
    
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
    parser.add_argument('--adaptive_clustering', action='store_true', default=True,
                        help='Adaptively determine number of clusters')
    parser.add_argument('--apply_splitting', action='store_true',
                        help='Actually modify the model by splitting prototypes')
    parser.add_argument('--splitting_scale', type=float, default=0.7,
                        help='Scaling factor for split prototypes (controls how different they are)')
    parser.add_argument('--save_expanded_model', action='store_true', 
                        help='Save the model with expanded prototypes')
    
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
    parser.add_argument('--apply_merging', action='store_true',
                        help='Actually modify the model by merging prototypes')
    parser.add_argument('--save_merged_model', action='store_true',
                        help='Save the model with merged prototypes')
    
    # Visualization options
    parser.add_argument('--visualize_results', action='store_true', default=True,
                        help='Visualize results of analysis')
    
    # Get default PIPNet arguments
    args = get_args(parser)
    
    # Run the script
    split_and_merge_prototypes(args)