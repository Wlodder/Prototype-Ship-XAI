from typing import List
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.functional as F
from umap import UMAP

class PURE:
    """
    in PIPNet architecture.
    """
    def __init__(self, model, device='cuda', num_ref_samples=100):
        """
        Initialize PURE for a PIPNet model.
        
        Args:
            model: The PIPNet model
            device: Device to run computations on ('cuda' or 'cpu')
            num_ref_samples: Number of reference samples to use for clustering
        """
        self.model = model
        self.device = device
        self.num_ref_samples = num_ref_samples
        self.model.to(device)
        
    def compute_attributions(self, x, prototype_idx):
        """
        Compute attributions (circuit) for a specific prototype using Gradient × Activation.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype to compute attributions for
            
        Returns:
            Tensor of attributions for lower-level features
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Get original features that will be attributed
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass to get prototype presence
        # with torch.no_grad():
        _, pooled, _, features = self.model(x, inference=False, features_save=True)
        # print(pooled)
        # input()
        
        # Get the specific prototype activation
        target_activation = pooled[:, prototype_idx]
        

        
        # Compute gradients with respect to features
        gradients = torch.autograd.grad(target_activation.sum(), features, 
                                        retain_graph=True, allow_unused=True)[0]

        
        # Gradient × Activation attribution
        attributions = gradients * features
        
        # Sum over spatial dimensions to get attribution per channel
        attributions = attributions.sum(dim=(2, 3))
        
        return attributions

   
        
    def find_top_activating_samples(self, dataloader, prototype_idx):
        """
        Find the top activating samples for a prototype.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to find activating samples for
            
        Returns:
            Tuple of (top samples, top activations)
        """
        self.model.eval()
        activations = []
        images = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different dataloader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass
                _, pooled, _ = self.model(inputs, inference=False)
                
                # Get activations for the target prototype
                batch_activations = pooled[:, prototype_idx].cpu()
                
                for i, activation in enumerate(batch_activations):
                    activations.append(activation.item())
                    images.append(inputs[i].cpu())
                
        # Sort by activation and get top samples
        sorted_indices = np.argsort(activations)[::-1]  # Descending order
        top_indices = sorted_indices[:self.num_ref_samples]
        
        top_samples = torch.stack([images[i] for i in top_indices])
        top_activations = [activations[i] for i in top_indices]
        
        return top_samples, top_activations

    def compute_circuits(self, top_samples, prototype_idx):
        """
        Compute circuits for each top activating sample.
        
        Args:
            top_samples: Tensor of top activating samples
            prototype_idx: Index of the prototype
            
        Returns:
            Tensor of circuit attributions for each sample
        """
        circuits = []
        
        for sample in top_samples:
            sample = sample.unsqueeze(0)  # Add batch dimension
            attribution = self.compute_attributions(sample, prototype_idx)
            circuits.append(attribution.cpu().detach())
        
        # Stack all circuits
        all_circuits = torch.stack(circuits)
        
        return all_circuits
    
    def cluster_circuits(self, circuits, n_clusters=2):
        """
        Cluster circuits to identify different semantics.
        
        Args:
            circuits: Tensor of circuit attributions
            n_clusters: Number of clusters (virtual neurons) to create
            
        Returns:
            Tuple of (cluster labels, centroids)
        """
        # Reshape for clustering
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42 )

        # hdbscan = HDBSCAN(allow_single_cluster=True, store_centers="centroid" )
        # hdbscan = DBSCAN()
        cluster_labels = kmeans.fit_predict(flat_circuits)
        
        # Get centroids
        # centroids = hdbscan.centroids_
        # n_clusters = centroids.shape[0]
        centroids = kmeans.cluster_centers_
        centroids = torch.tensor(centroids).reshape(n_clusters, *circuits.shape[1:])
        return cluster_labels, centroids, n_clusters
    
    def assign_circuit(self, x, prototype_idx, centroids):
        """
        Assign an input to the closest circuit for a prototype.
        
        Args:
            x: Input image tensor
            prototype_idx: Index of the prototype
            centroids: Tensor of circuit centroids
            
        Returns:
            Index of the closest centroid (circuit)
        """
        x = x.to(self.device)
        attribution = self.compute_attributions(x, prototype_idx)
        
        # Compute distances to each centroid
        distances = []
        for centroid in centroids:
            centroid = centroid.to(self.device)
            dist = F.mse_loss(attribution, centroid).item()
            distances.append(dist)
        
        # Return index of closest centroid
        return np.argmin(distances)
    
    def visualize_clusters(self, top_samples, cluster_labels, prototype_idx, crop_images=False, saliency_maps=None):
        """
        Visualize the clusters of a prototype.
        
        Args:
            top_samples: Tensor of top activating samples
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
            crop_images: Whether to crop images based on saliency
            saliency_maps: Optional saliency maps for cropping
            
        Returns:
            Matplotlib figure
        """
        n_clusters = len(np.unique(cluster_labels))
        
        # Create a figure
        fig, axs = plt.subplots(n_clusters + 1, 1, figsize=(12, 2.5*(n_clusters+1)))
        
        # Visualize all samples
        axs[0].set_title(f"All top activating samples for prototype {prototype_idx}")
        self._plot_samples(axs[0], top_samples[:10], saliency_maps)
        
        # Visualize samples for each cluster
        for i in range(n_clusters):
            cluster_samples = top_samples[cluster_labels == i]
            cluster_saliency = None
            if saliency_maps is not None:
                cluster_saliency = [saliency_maps[j] for j, label in enumerate(cluster_labels) if label == i]
            
            axs[i+1].set_title(f"Cluster {i+1} samples (virtual neuron {prototype_idx}.{i+1})")
            self._plot_samples(axs[i+1], cluster_samples[:10], cluster_saliency)
        
        plt.tight_layout()
        return fig
    
    def _plot_samples(self, ax, samples, saliency_maps=None, max_samples=25):
        """
        Helper function to plot samples in a grid.
        
        Args:
            ax: Matplotlib axis
            samples: Tensor of samples to plot
            saliency_maps: Optional saliency maps for highlighting
            max_samples: Maximum number of samples to plot
        """
        samples = samples[:max_samples]
        grid_size = len(samples)
        
        # Define normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        for i, sample in enumerate(samples):
            # Convert tensor to image
            sample = sample[0]
            img = sample.cpu().clone()
            img = img * std + mean  # Denormalize
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)  # Ensure values are in valid range
            
            # Apply saliency map if provided
            if saliency_maps is not None and i < len(saliency_maps):
                saliency = saliency_maps[i]
                if isinstance(saliency, torch.Tensor):
                    saliency = saliency.cpu().numpy()
                
                
                # Resize saliency if needed
                if saliency.shape[:2] != img.shape[:2]:
                    from skimage.transform import resize
                    saliency = resize(saliency, img.shape[:2])
                
                # Create a mask from saliency
                mask = saliency > 0.1  # Threshold for visualization
                
                # Create an overlay
                overlay = img.copy()
                overlay[~mask] = overlay[~mask] * 0.3  # Dim non-salient regions
            else:
                overlay = img
                
            ax.imshow(overlay, extent=(i, i+1, 0, 1))#, aspect='auto')
        
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def compute_saliency_maps(self, samples, prototype_idx):
        """
        Compute saliency maps for a prototype on given samples.
        
        Args:
            samples: Tensor of samples
            prototype_idx: Index of the prototype
            
        Returns:
            List of saliency maps
        """
        self.model.eval()
        saliency_maps = []
        
        for sample in samples:
            sample = sample.unsqueeze(0).to(self.device)
            sample.requires_grad_(True)
            
            # Forward pass
            _, pooled, _ = self.model(sample, inference=False)
            
            # Get activation for the target prototype
            prototype_activation = pooled[:, prototype_idx]
            
            # Compute gradients w.r.t input
            gradients = torch.autograd.grad(prototype_activation, sample)[0]
            
            # Create saliency map
            saliency = gradients.abs().sum(dim=1).squeeze().cpu()
            
            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            saliency_maps.append(saliency)
            
        return saliency_maps
    
    def visualize_umap_embedding(self, circuits, cluster_labels, prototype_idx):
        """
        Visualize UMAP embedding of circuit attributions colored by cluster.
        
        Args:
            circuits: Tensor of circuit attributions
            cluster_labels: Numpy array of cluster labels
            prototype_idx: Index of the prototype
            
        Returns:
            Matplotlib figure
        """
        # Reshape circuits for UMAP
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Apply UMAP for dimensionality reduction
        umap_reducer = UMAP(n_components=2, n_neighbors=10, random_state=42)
        embedding = umap_reducer.fit_transform(flat_circuits)
        
        # Plot embedding
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            plt.scatter(
                embedding[cluster_labels == cluster, 0],
                embedding[cluster_labels == cluster, 1],
                c=[colors[i]],
                label=f'Cluster {cluster + 1}',
                alpha=0.7,
                s=100
            )
        
        plt.title(f"UMAP embedding of circuit attributions for prototype {prototype_idx}")
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        
        return plt.gcf()

    def determine_optimal_clusters(self, circuits, max_clusters=5, random_state=42):
        """
        Determine the optimal number of clusters for prototype disentanglement.
        
        Args:
            circuits: Tensor of circuit attributions
            max_clusters: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
            
        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from sklearn.cluster import KMeans
    
        # Reshape circuits for clustering
        flat_circuits = circuits.reshape(circuits.shape[0], -1).numpy()
        
        # Must have at least max_clusters+1 samples
        n_samples = flat_circuits.shape[0]
        max_possible = min(max_clusters, n_samples - 1)
        
        # print("The hopkins statistic: ", len(flat_circuits),self._hopkins_statistic(flat_circuits))

        if max_possible <= 1:
            return 1
        
        # Store scores for different numbers of clusters
        silhouette_scores = []
        ch_scores = []
        
        # Evaluate different numbers of clusters
        for n_clusters in range(1, max_possible + 1):
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = kmeans.fit_predict(flat_circuits)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters exist
                s_score = silhouette_score(flat_circuits, cluster_labels)
                silhouette_scores.append(s_score)
                
                # Calculate Calinski-Harabasz score (variance ratio)
                ch_score = calinski_harabasz_score(flat_circuits, cluster_labels)
                ch_scores.append(ch_score)
            else:
                silhouette_scores.append(-1)
                ch_scores.append(0)
        
        # Normalize scores
        if silhouette_scores:
            norm_silhouette = np.array(silhouette_scores) / max(max(silhouette_scores), 1e-10)
            norm_ch = np.array(ch_scores) / max(max(ch_scores), 1e-10)
            
            # Combined score (weighted average)
             
            combined_scores = 0.1 * norm_silhouette + 0.9 * norm_ch
            combined_scores[0] = norm_ch[0]
            print(combined_scores)
            
            # Get optimal number of clusters (add 2 because we started from 2)
            optimal_clusters = np.argmax(combined_scores) + 1
        else:
            optimal_clusters = 2  # Default if we couldn't compute scores
        
        return optimal_clusters

  

    def disentangle_prototype(self, dataloader, prototype_idx, n_clusters=2, adaptive=True, max_clusters=5):
        """
        Disentangle a prototype into multiple pure features.
        
        Args:
            dataloader: DataLoader containing the dataset
            prototype_idx: Index of the prototype to disentangle
            n_clusters: Number of clusters (virtual neurons)
            
        Returns:
            Dictionary containing disentanglement results
        """
        print(f"Disentangling prototype {prototype_idx} into {n_clusters} virtual neurons...")
        if isinstance(prototype_idx, List):
            # Find top activating samples
            overall_circuits = []
            overall_top_samples = []
            overall_saliency_maps = []
            overall_top_activations = []
            for prototype in prototype_idx:
                top_samples, top_activations = self.find_top_activating_samples(dataloader, prototype)
                print(f"Found {len(top_samples)} top activating samples.")
                
                # Compute saliency maps
                saliency_maps = self.compute_saliency_maps(top_samples, prototype)
                
                # Compute circuits
                print("Computing circuits...")
                circuits = self.compute_circuits(top_samples, prototype)

                # Cluster circuits
                overall_circuits.append(circuits)
                overall_top_samples.append(top_samples)
                for sm in saliency_maps:
                    overall_saliency_maps.append(sm)

                for activation in top_activations:
                    overall_top_activations.append(activation)                


            circuits = torch.cat(overall_circuits, dim=0).squeeze(0)
            top_samples = torch.cat(overall_top_samples, dim=0).squeeze(0)
            # saliency_maps = [sm for sm in overall_saliency_maps]
            saliency_maps = overall_saliency_maps
            top_activations = overall_top_activations
            # Cluster circuits
        else:
            top_samples, top_activations = self.find_top_activating_samples(dataloader, prototype_idx)
            print(f"Found {len(top_samples)} top activating samples.")
            
            # Compute saliency maps
            saliency_maps = self.compute_saliency_maps(top_samples, prototype_idx)
            
            # Compute circuits
            print("Computing circuits...")
            circuits = self.compute_circuits(top_samples, prototype_idx)



        print("Clustering circuits...")
        cluster_labels, centroids, n_clusters = self.cluster_circuits(circuits, n_clusters=n_clusters)

        
        # Visualize clusters
        print("Generating visualizations...")
        cluster_vis = self.visualize_clusters(top_samples, cluster_labels, prototype_idx, saliency_maps=saliency_maps)
        
        # # Visualize UMAP embedding
        umap_vis = self.visualize_umap_embedding(circuits, cluster_labels, prototype_idx)
        
        # plt.show()
        print(f"Successfully disentangled prototype {prototype_idx} into {n_clusters} virtual neurons.")
        
        # Calculate statistics per cluster
        cluster_stats = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_size = len(cluster_indices)
            cluster_activations = [top_activations[idx] for idx in cluster_indices]
            mean_activation = np.mean(cluster_activations) if cluster_activations else 0
            
            cluster_stats.append({
                'virtual_neuron': f"{prototype_idx}.{i+1}",
                'size': cluster_size,
                'mean_activation': mean_activation,
                'samples': top_samples[cluster_indices]
            })
        
        # Return the disentangled prototype information
        return {
            'prototype_idx': prototype_idx,
            'top_samples': top_samples,
            'cluster_labels': cluster_labels,
            'centroids': centroids,
            'circuits': circuits,
            'cluster_visualization': cluster_vis,
            'umap_visualization': umap_vis,
            'cluster_stats': cluster_stats
        }




# Example usage function
def disentangle_polysemantic_prototypes(model, dataloader, device='cuda', prototypes=None, num_ref_samples=100, n_clusters=2):
    """
    Identify and disentangle potentially polysemantic prototypes.
    
    Args:
        model: PIPNet model
        dataloader: DataLoader containing dataset
        device: Device to run on
        
    Returns:
        Dictionary of disentangled prototype information
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize PURE
    pure = PURE(model, device=device, num_ref_samples=num_ref_samples)
    
    # Get total number of prototypes
    num_prototypes = model.module._num_prototypes
    print(f"Analyzing {num_prototypes} prototypes...")
    
    # Find potentially polysemantic prototypes
    # For demonstration, we'll just take a few prototypes 
    # In practice, you might want to analyze all prototypes or use a heuristic
    # candidate_prototypes = list(range(0, num_prototypes, num_prototypes//10))[:5]  # Take 5 prototypes evenly spaced

    # The first 3 are not polysemantic whilst the result are polysemantic
    # candidate_prototypes = [217, 221, 224, 227,262,332,516,553]  # 
    candidate_prototypes = prototypes#[56,57,90, 158]  # 

    
    results = {}
    for proto_idx in candidate_prototypes:
        # Disentangle prototype
        if isinstance(proto_idx, List) :
            # result = pure.test_patch_substitution(dataloader, proto_idx[0], proto_idx[1], num_samples=25, patch_size=64)
            result = pure.disentangle_prototype(dataloader, proto_idx, n_clusters=n_clusters, adaptive=True)
            for cluster in result['cluster_stats']:
                print(f"Virtual neuron {cluster['virtual_neuron']} has {cluster['size']} samples with mean activation {cluster['mean_activation']}")
            # results[proto_idx] = result
            for proto in proto_idx:
                results[proto] = result
        else:
            result = pure.disentangle_prototype(dataloader, proto_idx, n_clusters=n_clusters, adaptive=True)

        # print(result['cluster_stats'])
            for cluster in result['cluster_stats']:
                print(f"Virtual neuron {cluster['virtual_neuron']} has {cluster['size']} samples with mean activation {cluster['mean_activation']}")
            results[proto_idx] = result
    
        # plt.show()

    return results