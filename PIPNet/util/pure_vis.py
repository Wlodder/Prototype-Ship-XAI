import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import shutil


def save_cluster_images(pure_results, output_dir, save_visualization=True):
    """
    Save images from different clusters to separate directories.
    
    Args:
        pure_results: Dictionary containing results from PURE disentanglement
        output_dir: Base directory to save the cluster images
        save_visualization: Whether to save a visualization of all clusters
        
    Returns:
        Dictionary with paths to saved images for each cluster
    """
    # Extract needed data from pure_results
    prototype_idx = pure_results['prototype_idx']
    top_samples = pure_results['top_samples']
    cluster_labels = pure_results['cluster_labels']
    
    # Create base output directory
    output_path = Path(output_dir)
    prototype_dir = output_path / f"prototype_{prototype_idx}"
    
    # Remove existing directory if it exists
    if prototype_dir.exists():
        shutil.rmtree(prototype_dir)
    
    # Create the directory structure
    prototype_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a denormalization transform for better visualization
    denorm = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    # Identify unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Create separate directory for each cluster
    cluster_dirs = {}
    for cluster_idx in unique_clusters:
        cluster_path = prototype_dir / f"cluster_{cluster_idx+1}"
        cluster_path.mkdir(exist_ok=True)
        cluster_dirs[cluster_idx] = cluster_path
    
    # Save images for each cluster
    saved_paths = {cluster_idx: [] for cluster_idx in unique_clusters}
    
    # Process and save each image
    for i, (sample, cluster_idx) in enumerate(zip(top_samples, cluster_labels)):
        # Denormalize the image
        img_tensor = denorm(sample.cpu())
        
        # Convert to PIL image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Define save path
        img_path = cluster_dirs[cluster_idx] / f"sample_{i:03d}.png"
        
        # Save the image
        img_pil.save(img_path)
        saved_paths[cluster_idx].append(str(img_path))
    
    # Save a visualization of all clusters if requested
    if save_visualization:
        visualize_clusters(top_samples, cluster_labels, prototype_idx, prototype_dir, denorm)
        
    # Save cluster metadata
    # save_cluster_metadata(pure_results, prototype_dir)
    
    print(f"Saved {len(top_samples)} images across {n_clusters} clusters for prototype {prototype_idx}")
    print(f"Output directory: {prototype_dir}")
    
    return {
        "base_dir": str(prototype_dir),
        "cluster_dirs": {f"cluster_{k+1}": str(v) for k, v in cluster_dirs.items()},
        "saved_images": saved_paths
    }


def visualize_clusters(samples, cluster_labels, prototype_idx, output_dir, denorm):
    """
    Create and save a visualization of all clusters.
    
    Args:
        samples: Tensor of samples
        cluster_labels: Cluster assignments
        prototype_idx: Index of the prototype
        output_dir: Directory to save visualization
        denorm: Denormalization transform
    """
    # Identify unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Determine samples per row and number of rows
    n_cols = min(8, len(samples) // n_clusters)
    
    # Create figure
    plt.figure(figsize=(n_cols * 2, n_clusters * 2))
    
    # For each cluster
    for i, cluster_idx in enumerate(unique_clusters):
        # Get samples for this cluster
        cluster_samples = samples[cluster_labels == cluster_idx]
        
        # Determine how many samples to display
        n_display = min(n_cols, len(cluster_samples))
        
        # Plot each sample
        for j in range(n_display):
            # Calculate subplot position
            plt_idx = i * n_cols + j + 1
            
            # Create subplot
            plt.subplot(n_clusters, n_cols, plt_idx)
            
            # Process and display image
            img = denorm(cluster_samples[j].cpu())
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            
            # Add cluster label to first image in row
            if j == 0:
                plt.title(f"Cluster {cluster_idx+1}")
            
            plt.axis('off')
    
    # Add overall title
    plt.suptitle(f"Prototype {prototype_idx} - Clusters Visualization", fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(output_dir / "clusters_visualization.png", dpi=200, bbox_inches='tight')
    plt.close()


def save_cluster_metadata(pure_results, output_dir):
    """
    Save metadata about the clusters.
    
    Args:
        pure_results: Dictionary containing results from PURE disentanglement
        output_dir: Directory to save metadata
    """
    # Extract data
    prototype_idx = pure_results['prototype_idx']
    cluster_labels = pure_results['cluster_labels']
    
    # Count samples per cluster
    unique_clusters = np.unique(cluster_labels)
    cluster_counts = {f"cluster_{k+1}": np.sum(cluster_labels == k) for k in unique_clusters}
    
    # Create metadata
    metadata = {
        "prototype_idx": prototype_idx,
        "num_clusters": len(unique_clusters),
        "total_samples": len(cluster_labels),
        "samples_per_cluster": cluster_counts
    }
    
    # Add cluster statistics if available
    if 'cluster_stats' in pure_results:
        metadata["cluster_stats"] = [{
            "virtual_neuron": stats['virtual_neuron'],
            "size": stats['size'],
            "mean_activation": float(stats['mean_activation'])
        } for stats in pure_results['cluster_stats']]
    
    # Save metadata as JSON
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)


# Function to batch process multiple prototypes
def save_all_prototype_clusters(pure_results_dict, output_dir):
    """
    Save images from multiple disentangled prototypes.
    
    Args:
        pure_results_dict: Dictionary mapping prototype indices to PURE results
        output_dir: Base directory to save the cluster images
        
    Returns:
        Dictionary with saved paths for each prototype
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each prototype
    results = {}
    for prototype_idx, pure_result in pure_results_dict.items():
        results[prototype_idx] = save_cluster_images(
            pure_result, 
            output_dir=output_path
        )
    
    # Create an index HTML file for easy browsing
    create_html_index(pure_results_dict, output_path)
    
    return results


def create_html_index(pure_results_dict, output_dir):
    """
    Create an HTML index for browsing all prototypes and clusters.
    
    Args:
        pure_results_dict: Dictionary mapping prototype indices to PURE results
        output_dir: Base directory where images are saved
    """
    output_path = Path(output_dir)
    
    # Start building HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PURE Prototype Disentanglement Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            .prototype { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .clusters { display: flex; flex-wrap: wrap; }
            .cluster { margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }
            .cluster-title { font-weight: bold; margin-bottom: 10px; }
            .image-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
            .thumbnail { width: 100%; height: auto; border: 1px solid #eee; }
            .stats { margin-top: 15px; font-size: 0.9em; color: #555; }
        </style>
    </head>
    <body>
        <h1>PURE Prototype Disentanglement Results</h1>
    """
    
    # Add each prototype
    for prototype_idx, pure_result in pure_results_dict.items():
        html += f"""
        <div class="prototype">
            <h2>Prototype {prototype_idx}</h2>
            <img src="prototype_{prototype_idx}/clusters_visualization.png" alt="Clusters visualization" style="max-width: 100%;">
            <div class="clusters">
        """
        
        # Get unique clusters
        cluster_labels = pure_result['cluster_labels']
        unique_clusters = np.unique(cluster_labels)
        
        # Add each cluster
        for cluster_idx in unique_clusters:
            # Get cluster samples
            cluster_samples = np.where(cluster_labels == cluster_idx)[0]
            
            html += f"""
            <div class="cluster">
                <div class="cluster-title">Cluster {cluster_idx + 1}</div>
                <div class="image-grid">
            """
            
            # Add thumbnails for this cluster (limit to 12 per cluster)
            max_display = min(12, len(cluster_samples))
            for i in range(max_display):
                sample_idx = cluster_samples[i]
                html += f"""
                <a href="prototype_{prototype_idx}/cluster_{cluster_idx + 1}/sample_{sample_idx:03d}.png" target="_blank">
                    <img class="thumbnail" src="prototype_{prototype_idx}/cluster_{cluster_idx + 1}/sample_{sample_idx:03d}.png" alt="Sample {sample_idx}">
                </a>
                """
            
            # Add statistics if available
            if 'cluster_stats' in pure_result:
                stats = pure_result['cluster_stats'][cluster_idx]
                html += f"""
                <div class="stats">
                    <div>Virtual neuron: {stats['virtual_neuron']}</div>
                    <div>Size: {stats['size']} samples</div>
                    <div>Mean activation: {stats['mean_activation']:.3f}</div>
                </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path / "index.html", 'w') as f:
        f.write(html)