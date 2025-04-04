import torch
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename

# Import PIPNet components
from prototype_management import PrototypeManager
from pipnet.pipnet import PIPNet, get_network
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from prototype_squared import attribution
from util.attribution_analyzer import MultiLayerAttributionAnalyzer
from sklearn.preprocessing import StandardScaler
import umap

# Create Flask app
os.environ["CUDA_VISIBLE_DEVICES"]="0"
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pt', 'pth', ''}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model, data, and analysis state
global_state = {
    'model': None,
    'device': None,
    'trainloader': None,
    'testloader': None,
    'classes': None,
    'prototype_manager': None,
    'selected_prototypes': [],
    'analyzer': None,
    'split_results': {},
    'umap_embeddings': {},
    'custom_clusters': {},
    'args': None
}

def allowed_file(filename):
    # return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    return True

def tensor_to_base64(tensor):
    """Convert a tensor to a base64 encoded image."""
    # Normalize and convert to numpy
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    
    # Convert to PIL Image and then to base64
    pil_img = Image.fromarray(img)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/load_model', methods=['GET', 'POST'])
def load_model():
    """Load a model."""
    if request.method == 'POST':
        # Check if model file is provided
        if 'model_file' not in request.files:
            return jsonify({"error": "No model file provided"})
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Parse arguments
            args = get_args()
            args.state_dict_dir_net = filepath
            
            # Set custom dataset path if provided
            dataset_path = request.form.get('dataset_path', '')
            if dataset_path:
                args.dataset = dataset_path
                args.custom_dataset = True
            else:
                return jsonify({"error": "Custom dataset path is required"})

            num_prototypes = int(request.form.get('num_prototypes', ''))
            if num_prototypes:
                args.num_features = num_prototypes
            else:
                return jsonify({"error": "You must provide the number of prototypes"})
            
            # Initialize model
            try:
                initialize_model(args)
                return jsonify({"success": "Model loaded successfully"})
            except Exception as e:
                return jsonify({"error": f"Error loading model: {str(e)}"})
        
        return jsonify({"error": "Invalid file type"})
    
    return render_template('load_model.html', global_state=global_state)

def initialize_model(args):

    """Initialize the PIPNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
    
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
    
    # Update global state
    global_state['model'] = net
    global_state['device'] = device
    global_state['trainloader'] = trainloader
    global_state['trainloader_normal'] = trainloader_normal
    global_state['testloader'] = testloader
    global_state['classes'] = classes
    global_state['prototype_manager'] = PrototypeManager(net, device=device)
    global_state['analyzer'] = MultiLayerAttributionAnalyzer(net, device=device)
    global_state['args'] = args
    
    print(f"Model initialized on {device} with {num_prototypes} prototypes")
    return net

def run_prototype_analysis(prototype_indices, n_clusters=None, adaptive=True, max_clusters=5, algorithm='kmeans'):
    """Analyze prototype behavior using MultiLayerAttributionAnalyzer."""
    analyzer = global_state['analyzer']
    trainloader = global_state['trainloader_normal']
    
    # Normalize indices to list format
    normalized_indices = []
    for proto in prototype_indices:
        if isinstance(proto, int):
            normalized_indices.append([proto])
        else:
            normalized_indices.append(proto)
    
    results = {}
    
    # Analyze each prototype or prototype group
    for protogroup in normalized_indices:
        results['_'.join(map(str, protogroup))] = analyzer.analyze_related_prototypes(
            dataloader=trainloader,
            prototype_groups=[protogroup],
            layer_indices=[7, 6, 5, 4, 3],
            adaptive=adaptive,
            max_clusters=max_clusters,
            clustering_method=algorithm,
            visualize=True,
            max_samples=100,
            layer_weights={
                7: 0.5,
                6: 1.0,
                5: 1.1,
                4: 1.2,
                3: 0.6,
            }
        )
    
    # Store results in global state
    global_state['split_results'].update(results)
    
    # Compute UMAP embeddings for the first prototype group
    print("Computing UMAP embeddings...")
    compute_umap_embeddings(list(results.keys())[0], results)
    
    return results

def compute_umap_embeddings(proto_key, results=None):
    """Compute UMAP embeddings for visualization."""
    if results is None:
        results = global_state['split_results']
    
    if proto_key not in results:
        return {}
    
    # Get attributions and cluster labels
    circuits = results[proto_key].get('circuits', {})
    cluster_labels = results[proto_key].get('cluster_labels', [])
    
    # Compute embeddings for each layer
    embeddings = {}
    for layer_idx, layer_circuits in circuits.items():
        # Reshape for UMAP
        flat_circuits = layer_circuits.reshape(layer_circuits.shape[0], -1).numpy()
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(flat_circuits)
        
        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_components=2,
            min_dist=0.1,
            n_neighbors=min(15, len(flat_circuits)-1),
            random_state=42
        )
        
        embedding = reducer.fit_transform(scaled_data)
        embeddings[layer_idx] = embedding
    
    # Store in global state
    global_state['umap_embeddings'][proto_key] = embeddings
    
    return embeddings

def create_custom_centroid(proto_key, layer_idx, selected_indices, cluster_name):
    """Create a custom centroid from selected points in UMAP."""
    if proto_key not in global_state['split_results']:
        return {"error": "Prototype group not found"}
    
    results = global_state['split_results'][proto_key]
    
    if 'circuits' not in results or layer_idx not in results['circuits']:
        return {"error": "Circuit data not found for this layer"}
    
    # Get the circuit data for selected points
    circuits = results['circuits'][layer_idx]
    selected_circuits = circuits[selected_indices]
    
    # Compute centroid
    centroid = torch.mean(selected_circuits, dim=0)
    
    # Store as custom cluster
    if proto_key not in global_state['custom_clusters']:
        global_state['custom_clusters'][proto_key] = {}
    
    global_state['custom_clusters'][proto_key][cluster_name] = {
        'layer_idx': layer_idx,
        'centroid': centroid,
        'sample_indices': selected_indices
    }
    
    return {
        "status": "success",
        "message": f"Created custom cluster '{cluster_name}' with {len(selected_indices)} points"
    }

def run_custom_prototype_pass(proto_key, sample_indices=None):
    """Run inference with custom prototype centroids."""
    if proto_key not in global_state['custom_clusters']:
        return {"error": "No custom clusters found for this prototype"}
    
    model = global_state['model']
    device = global_state['device']
    
    # Create ForwardPURE model
    pure_model = attribution.ForwardPURE(model, device=device)
    
    # Add centroids from analysis results
    if proto_key in global_state['split_results']:
        pure_model.add_centroids({proto_key: global_state['split_results'][proto_key]})
    
    # Add custom centroids
    custom_clusters = global_state['custom_clusters'][proto_key]
    for cluster_name, cluster_data in custom_clusters.items():
        layer_idx = cluster_data['layer_idx']
        centroid = cluster_data['centroid']
        
        if layer_idx not in pure_model.centroids_by_layer:
            pure_model.centroids_by_layer[layer_idx] = []
        
        pure_model.centroids_by_layer[layer_idx].append(centroid)
    
    # Get samples
    if sample_indices is None:
        # Use first batch from dataloader
        batch = next(iter(global_state['testloader']))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            samples, labels = batch[0][:8].to(device), batch[1][:8]  # Use first 8 samples
        else:
            samples = batch[:8].to(device)
            labels = None
    else:
        # TODO: Implement sample selection from dataset
        batch = next(iter(global_state['testloader']))
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            samples, labels = batch[0][:8].to(device), batch[1][:8]  # Use first 8 samples
        else:
            samples = batch[:8].to(device)
            labels = None
    
    # Run enhanced classification
    results = pure_model.enhanced_classification(samples)
    
    # Process results for display
    processed_results = {
        'original_predictions': results['initial_preds'].cpu().numpy().tolist(),
        'enhanced_predictions': results['enhanced_preds'].cpu().numpy().tolist(),
        'sample_images': [tensor_to_base64(sample) for sample in samples.cpu()],
        'class_names': global_state['classes'],
        'matched_centroids': []
    }
    
    # Extract matched centroids info
    for i, matches in enumerate(results['centroid_matches']):
        sample_matches = []
        for match_info in matches:
            proto_idx = match_info['prototype_idx']
            best_matches = {}
            
            for layer_idx, match_data in match_info['centroid_matches'].items():
                if match_data['is_match']:
                    best_matches[layer_idx] = {
                        'centroid_idx': match_data['centroid_idx'],
                        'distance': float(match_data['distance'])
                    }
            
            if best_matches:
                sample_matches.append({
                    'prototype_idx': proto_idx,
                    'activation': float(match_info['activation']),
                    'matches': best_matches
                })
        
        processed_results['matched_centroids'].append(sample_matches)
    
    return processed_results

# Flask routes

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html', global_state=global_state)


@app.route('/prototypes')
def prototypes():
    """Prototype visualization and selection page."""
    if global_state['model'] is None:
        return redirect(url_for('load_model'))
    
    # Get prototype info
    class_weights = global_state['model'].module._classification.weight.data
    max_class_indices = torch.argmax(class_weights, axis=0).cpu().numpy()
    max_class_weights = torch.max(class_weights, axis=0)[0].cpu().numpy()
    
    num_prototypes = global_state['model'].module._num_prototypes
    
    # Create prototype list with class info
    prototype_info = []
    for p in range(num_prototypes):
        if max_class_weights[p] > 0.01:  # Only include significant prototypes
            prototype_info.append({
                'id': p,
                'class': int(max_class_indices[p]),
                'class_name': global_state['classes'][int(max_class_indices[p])],
                'weight': float(max_class_weights[p])
            })
    
    # Sort by weight
    prototype_info.sort(key=lambda x: x['weight'], reverse=True)
    
    return render_template('prototypes.html', 
                         prototypes=prototype_info[:100],  # Limit to top 100
                         classes=global_state['classes'], global_state=global_state,
                         enumerate=enumerate)

@app.route('/get_prototype_samples/<int:prototype_id>')
def get_prototype_samples(prototype_id):
    """Get top activating samples for a prototype."""
    if global_state['model'] is None:
        return jsonify({"error": "No model loaded"})
    
    try:
        prototype_manager = global_state['prototype_manager']
        top_samples, top_activations = prototype_manager.find_top_activating_samples(
            global_state['trainloader_normal'], prototype_id, num_samples=10)
        
        # Convert to base64 images
        sample_images = [tensor_to_base64(sample) for sample in top_samples]
        
        return jsonify({
            "prototype_id": prototype_id,
            "samples": sample_images,
            "activations": [float(act) for act in top_activations]
        })
    except Exception as e:
        return jsonify({"error": f"Error getting samples: {str(e)}"})

@app.route('/analyze_prototypes', methods=['POST'])
def analyze_prototypes():
    """Run prototype analysis."""
    if global_state['model'] is None:
        return jsonify({"error": "No model loaded"})
    
    try:
        # Get prototype indices from request
        prototype_indices = request.json.get('prototype_indices', [])
        if not prototype_indices:
            return jsonify({"error": "No prototypes selected"})
        
        # Convert to expected format for multi-layer analyzer
        indices_to_analyze = []
        for idx_str in prototype_indices:
            if '_' in idx_str:
                # Handle prototype groups
                indices_to_analyze.append([int(i) for i in idx_str.split('_')])
            else:
                indices_to_analyze.append(int(idx_str))
        
        # Run analysis
        results = run_prototype_analysis(
            indices_to_analyze,
            n_clusters=request.json.get('n_clusters'),
            adaptive=request.json.get('adaptive', True),
            max_clusters=request.json.get('max_clusters', 5),
            algorithm=request.json.get('algorithm', 'kmeans')
        )
        
        # Return success
        return jsonify({
            "success": "Analysis completed",
            "prototype_keys": list(results.keys())
        })
    except Exception as e:
        print(e)
        return jsonify({"error": f"Analysis error: {str(e)}"})

@app.route('/get_umap_data/<string:proto_key>/<int:layer_idx>')
def get_umap_data(proto_key, layer_idx):
    """Get UMAP embedding data for visualization."""
    if proto_key not in global_state['umap_embeddings']:
        return jsonify({"error": "No UMAP data for this prototype"})
    
    embeddings = global_state['umap_embeddings'][proto_key]
    if layer_idx not in embeddings:
        return jsonify({"error": "No UMAP data for this layer"})
    
    embedding = embeddings[layer_idx]
    
    # Get cluster labels
    results = global_state['split_results'][proto_key]
    cluster_labels = results['cluster_labels']
    
    # Get sample images for hover display
    samples = results['samples']
    
    # Prepare data for plot
    plot_data = []
    for i in range(len(embedding)):
        sample_img = tensor_to_base64(samples[i])
        plot_data.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'cluster': int(cluster_labels[i]),
            'sample_idx': i,
            'image': sample_img
        })
    
    return jsonify({
        'plot_data': plot_data,
        'num_clusters': int(np.max(cluster_labels) + 1)
    })

@app.route('/create_cluster', methods=['POST'])
def create_cluster():
    """Create a custom prototype cluster from selected points."""
    proto_key = request.json.get('proto_key')
    layer_idx = request.json.get('layer_idx')
    selected_indices = request.json.get('selected_indices', [])
    cluster_name = request.json.get('cluster_name')
    
    if not all([proto_key, layer_idx is not None, selected_indices, cluster_name]):
        return jsonify({"error": "Missing required parameters"})
    
    result = create_custom_centroid(proto_key, layer_idx, selected_indices, cluster_name)
    return jsonify(result)

@app.route('/view_clusters/<string:proto_key>')
def view_clusters(proto_key):
    """View existing clusters for a prototype."""
    if proto_key not in global_state['split_results']:
        return jsonify({"error": "No analysis results for this prototype"})
    
    results = global_state['split_results'][proto_key]
    
    # Get cluster info from results
    cluster_labels = results['cluster_labels']
    unique_clusters = np.unique(cluster_labels)
    
    # Count samples per cluster
    cluster_counts = {int(c): int(np.sum(cluster_labels == c)) for c in unique_clusters}
    
    # Get custom clusters
    custom_clusters = {}
    if proto_key in global_state['custom_clusters']:
        custom_clusters = {
            name: len(data['sample_indices']) 
            for name, data in global_state['custom_clusters'][proto_key].items()
        }
    
    return jsonify({
        "auto_clusters": cluster_counts,
        "custom_clusters": custom_clusters
    })

@app.route('/run_inference/<string:proto_key>', methods=['POST'])
def run_inference(proto_key):
    """Run inference with custom prototype centroids."""
    sample_indices = request.json.get('sample_indices')
    results = run_custom_prototype_pass(proto_key, sample_indices)
    return jsonify(results)

@app.route('/export_clusters/<string:proto_key>')
def export_clusters(proto_key):
    """Export custom clusters to file."""
    if proto_key not in global_state['custom_clusters']:
        return jsonify({"error": "No custom clusters for this prototype"})
    
    # Prepare data for export
    export_data = {
        'prototype_key': proto_key,
        'clusters': {}
    }
    
    for name, data in global_state['custom_clusters'][proto_key].items():
        export_data['clusters'][name] = {
            'layer_idx': int(data['layer_idx']),
            'centroid': data['centroid'].cpu().numpy().tolist(),
            'sample_indices': data['sample_indices']
        }
    
    # Save to file
    import json
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], f"clusters_{proto_key}.json")
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    return send_file(export_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=2000)#, threaded=True)