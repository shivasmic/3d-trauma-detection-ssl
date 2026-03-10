import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from tqdm import tqdm
from encoder import UNet3D 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from contextlib import redirect_stdout
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import silhouette_score, davies_bouldin_score



# --- Configuration ---
LOGS_DIR = 'logs'
WEIGHTS_FILE = os.path.join('unet_weights', 'v2', '3D_unet_ssl_weights.pth')
FULL_DATA_DIR = 'preprocessed_data'
HOLDOUT_DATA_DIR = 'preprocessed_holdout'

LOG_FILE_PATH = os.path.join(LOGS_DIR, 'evaluation_results.log')
RESULTS_DIR = os.path.join(LOGS_DIR, 'evaluation_outputs')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_patch_mask(volume: torch.Tensor, mask_ratio: float = 0.75, patch_size: int = 8):
    """
    Creates a patch-based mask on the input 3D volume for reconstruction tasks.

    It selects a random subset of patches to mask out (set to -1.0) based on
    the mask_ratio and returns the masked volume and a boolean mask indicating 
    the masked-out regions. This simulates a self-supervised masking task.

    Args:
        volume (torch.Tensor): Input volume of shape (B, C, D, H, W).
        mask_ratio (float): Fraction of patches to mask (e.g., 0.75 for 75%).
        patch_size (int): Size of the cubic patch (e.g., 8x8x8).

    Returns:
        tuple: (masked_volume, mask_bool)
    """
    B, C, D, H, W = volume.shape
    
    # Calculate number of patches along each dimension
    num_patches_d = D // patch_size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_patches = num_patches_d * num_patches_h * num_patches_w
    
    # Determine which patches to mask
    num_masked = int(total_patches * mask_ratio)
    patch_indices = torch.randperm(total_patches)[:num_masked].to(volume.device)
    
    masked_volume = volume.clone()
    mask = torch.zeros_like(volume, dtype=torch.bool).to(volume.device)
    
    for patch_idx in patch_indices:
        # Calculate 3D coordinates of the patch
        pd = patch_idx // (num_patches_h * num_patches_w)
        remainder = patch_idx % (num_patches_h * num_patches_w)
        ph = remainder // num_patches_w
        pw = remainder % num_patches_w
        
        d_start = pd * patch_size
        h_start = ph * patch_size
        w_start = pw * patch_size
        
        # Apply mask (set pixel values to -1.0 and set boolean mask to True)
        masked_volume[:, :,
                      d_start:d_start + patch_size,
                      h_start:h_start + patch_size,
                      w_start:w_start + patch_size] = -1.0 
        
        mask[:, :,
             d_start:d_start + patch_size,
             h_start:h_start + patch_size,
             w_start:w_start + patch_size] = True
    
    return masked_volume, mask


def extract_encoder_features(model, patch_tensor, device):
    """
    Extracts the bottleneck feature vector from the UNet encoder.

    This function simulates the forward pass of the encoder portion of the UNet 
    up to the bottleneck layer and then performs Global Average Pooling (GAP) 
    to get a fixed-size feature vector.

    Args:
        model (UNet3D): The loaded UNet model.
        patch_tensor (torch.Tensor): Input volume tensor.
        device (torch.device): The device the model is on.

    Returns:
        tuple: (feature_vector, bottleneck_map)
    """
    with torch.no_grad():
        c1 = model.down1(patch_tensor)
        p1 = model.pool1(c1)
        c2 = model.down2(p1)
        p2 = model.pool2(c2)
        c3 = model.down3(p2)
        p3 = model.pool3(c3)
        c4 = model.down4(p3)
        p4 = model.pool4(c4)
        bottleneck = model.bottleneck(p4)
        
        # Global Average Pooling across spatial dimensions (D, H, W)
        features = bottleneck.mean(dim=[2, 3, 4]) 
    
    return features, bottleneck


def visualize_features_2d(features, labels, save_path):
    """
    Reduces the dimensionality of the feature vectors and plots them.

    Uses t-SNE (for non-linear separation) and PCA (for linear separation) 
    to compress the high-dimensional features into 2D for visualization,
    colored by their cluster/label assignments. 

    Args:
        features (np.ndarray): N x D feature matrix.
        labels (np.ndarray): 1D array of cluster/class labels for coloring.
        save_path (str): Path to save the PNG visualization.
    """
    print("\n  Generating feature visualizations...")
    
    tsne_perplexity = min(30, len(features) - 1)
    if tsne_perplexity < 1:
        print("Too few samples for t-SNE/PCA visualization. Skipping.")
        return
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
    features_tsne = tsne.fit_transform(features)
    
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    unique_labels = np.unique(labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # t-SNE Plot
    scatter1 = axes[0].scatter(
        features_tsne[:, 0], 
        features_tsne[:, 1],
        c=labels,
        cmap='coolwarm' if len(unique_labels) == 2 else 'tab10',
        alpha=0.6,
        s=20
    )
    axes[0].set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0], label='Label/Cluster ID')
    
    # PCA Plot
    scatter2 = axes[1].scatter(
        features_pca[:, 0],
        features_pca[:, 1],
        c=labels,
        cmap='coolwarm' if len(unique_labels) == 2 else 'tab10',
        alpha=0.6,
        s=20
    )
    var_explained = pca.explained_variance_ratio_.sum()
    axes[1].set_title(f'PCA (Var: {var_explained:.1%})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    plt.colorbar(scatter2, ax=axes[1], label='Label/Cluster ID')
    
    plt.tight_layout()
    save_path_with_ext = save_path.replace('.png', f'_n{len(features)}.png')
    plt.savefig(save_path_with_ext, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path_with_ext}")


def visualize_attention_map(model, patch_tensor, save_path):
    """
    Generates a Grad-CAM (Gradient-weighted Class Activation Map) visualization 
    to show where the model 'looks' to compute its features.

    It calculates gradients flowing from the feature output back to the final
    convolutional layer (bottleneck) and uses them to weigh the feature maps,
    creating a heat map that highlights the most important regions in the 
    original 3D volume. 

    Args:
        model (UNet3D): The trained UNet model.
        patch_tensor (torch.Tensor): Input volume tensor (must have requires_grad=True).
        save_path (str): Path to save the multi-slice attention map image.
    """
    print("\n  Generating attention map...")
    
    gradients = None
    activations = None
    hooks = []
    
    # Hook functions to capture forward activations and backward gradients
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    # Target the final convolutional layer before the bottleneck feature aggregation
    # Assumes model.bottleneck is a Sequential block
    target_layer = model.bottleneck[-1] 
    hooks.append(target_layer.register_forward_hook(forward_hook))
    hooks.append(target_layer.register_full_backward_hook(backward_hook))
    
    try:
        patch_tensor = patch_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass to get output and populate the activations hook
        output = model(patch_tensor)
        
        # Calculate gradients (using mean output as the 'class' score)
        model.zero_grad()
        output.mean().backward() 
        
        # Compute CAM (Class Activation Map)
        weights = gradients.mean(dim=[2, 3, 4], keepdim=True) # Global average of gradients
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam) 
        
        # Upsample CAM to original volume size for overlay
        cam = F.interpolate(
            cam,
            size=patch_tensor.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        
        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Prepare for plotting (slicing midpoints)
        volume_np = patch_tensor[0, 0].detach().cpu().numpy()
        cam_np = cam[0, 0].detach().cpu().numpy()
        
        mid_z = volume_np.shape[0] // 2
        mid_y = volume_np.shape[1] // 2
        mid_x = volume_np.shape[2] // 2
        
        # Plotting logic for Axial, Coronal, and Sagittal views (omitted for brevity)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        # ... (plotting code here)
        
        # Axial Slice (Z)
        axes[0, 0].imshow(volume_np[mid_z], cmap='gray')
        axes[0, 0].set_title('Axial - Original', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(volume_np[mid_z], cmap='gray')
        axes[1, 0].imshow(cam_np[mid_z], cmap='jet', alpha=0.5)
        axes[1, 0].set_title('Axial - Attention', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Coronal Slice (Y)
        axes[0, 1].imshow(volume_np[:, mid_y], cmap='gray')
        axes[0, 1].set_title('Coronal - Original', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[1, 1].imshow(volume_np[:, mid_y], cmap='gray')
        axes[1, 1].imshow(cam_np[:, mid_y], cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Coronal - Attention', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Sagittal Slice (X)
        axes[0, 2].imshow(volume_np[:, :, mid_x], cmap='gray')
        axes[0, 2].set_title('Sagittal - Original', fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 2].imshow(volume_np[:, :, mid_x], cmap='gray')
        axes[1, 2].imshow(cam_np[:, :, mid_x], cmap='jet', alpha=0.5)
        axes[1, 2].set_title('Sagittal - Attention', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Grad-CAM Attention Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path}")

    finally:
        # Crucial step to clean up memory and prevent errors in subsequent runs
        for hook in hooks:
            hook.remove()


# --- Main Evaluation Script ---

print(f"Starting SSL Encoder Evaluation. Logging to: {LOG_FILE_PATH}")
print("-" * 60 + "\n", file=sys.stderr)

with open(LOG_FILE_PATH, 'w') as f, redirect_stdout(f):
    
    print("="*80)
    print("SSL ENCODER COMPREHENSIVE EVALUATION (Series-Level)")
    print("="*80)
    print(f"Start Time: {pd.Timestamp.now()}")
    print("="*80)
    
    ## 1/6: LOADING DATA & MODEL 
    print("\n[1/6] LOADING DATA & MODEL")
    print("-"*80)
    
    # --- CSV Loading (Only for Linear Probe) ---
    try:
        series_df = pd.read_csv('train_series_labeled.csv')
        print(f"Loaded Series CSV: {len(series_df)} series (For Linear Probe)")
        print(f"Unique patients: {series_df['patient_id'].nunique()}")
    except FileNotFoundError:
        print("✗ Error: 'train_series_labeled.csv' not found! Cannot run Linear Probe.")
        sys.exit(1)
    
    # We find holdout series IDs ONLY from the files in the holdout directory
    holdout_files = [f for f in os.listdir(HOLDOUT_DATA_DIR) if f.endswith('.npz')]
    
    # Store tuples of (series_id, full_path) for reconstruction and feature extraction
    holdout_series_data = []
    for f_name in holdout_files:
        try:
            series_id = int(f_name.split('_')[0])
            holdout_series_data.append((series_id, os.path.join(HOLDOUT_DATA_DIR, f_name)))
        except ValueError:
            # Skip files that don't start with an integer series ID
            continue 

    HOLDOUT_SIZE = len(holdout_series_data)
    print(f"Defined Holdout Dataset: {HOLDOUT_SIZE} series (from {HOLDOUT_DATA_DIR})")
    
    # --- Model Loading ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=1, out_channels=1, n_filters=16) 
    
    try:
        if not os.path.exists(WEIGHTS_FILE):
            raise FileNotFoundError(f"Weights file not found at: {WEIGHTS_FILE}")
        
        checkpoint = torch.load(WEIGHTS_FILE, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✓ Model loaded from: {WEIGHTS_FILE}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    model = model.to(device)
    
    use_fp16 = device.type == 'cuda'
    if use_fp16:
        model = model.half()
        print("Using FP16 for faster inference")

    
    ## 2/6: RECONSTRUCTION QUALITY EVALUATION (Holdout Set)
    print("\n[2/6] RECONSTRUCTION QUALITY EVALUATION (Holdout Set)")
    print("-"*80)
    
    reconstruction_results = {}
    mask_ratios = [0.25, 0.50, 0.75]
    num_recon_samples = HOLDOUT_SIZE
    
    print(f"Testing reconstruction on {num_recon_samples} samples at mask ratios: {mask_ratios}")
    
    if num_recon_samples == 0:
        print("No holdout data found for reconstruction.")
        for mask_ratio in mask_ratios:
            reconstruction_results[f"mask_{int(mask_ratio*100)}%"] = {
                'mse_mean': float('nan'), 'mse_std': float('nan'),
                'psnr_mean': float('nan'), 'psnr_std': float('nan')
            }
    else:
        for mask_ratio in mask_ratios:
            losses = []
            psnrs = []
            
            # --- ITERATE DIRECTLY OVER THE FOUND HOLDOUT FILES ---
            for series_id, npz_path in tqdm(holdout_series_data, total=num_recon_samples, desc=f"Recon R={mask_ratio}"):
                
                # Check file existence again just in case, though it should exist
                if not os.path.exists(npz_path):
                    continue
                
                data = np.load(npz_path)
                volume = data['volume']
                Z, Y, X = volume.shape
                
                # Center crop and padding logic (Unchanged)
                target_size = 128
                start_z = max(0, (Z - target_size) // 2)
                start_y = max(0, (Y - target_size) // 2)
                start_x = max(0, (X - target_size) // 2)
                patch = volume[start_z:start_z+target_size, start_y:start_y+target_size, start_x:start_x+target_size]
                
                if patch.shape != (target_size, target_size, target_size):
                    pad_z = target_size - patch.shape[0]
                    pad_y = target_size - patch.shape[1]
                    pad_x = target_size - patch.shape[2]
                    patch = np.pad(patch, [(0, pad_z), (0, pad_y), (0, pad_x)])
                
                if use_fp16:
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).half().to(device)
                else:
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                
                masked_patch, mask_bool = create_patch_mask(patch_tensor, mask_ratio, patch_size=8)
                
                with torch.no_grad():
                    reconstructed = model(masked_patch)
                
                # Calculate metrics only on the masked region
                mse = F.mse_loss(reconstructed[mask_bool], patch_tensor[mask_bool]).item()
                psnr = 10 * np.log10(1.0**2 / (mse + 1e-8)) 
                
                losses.append(mse)
                psnrs.append(psnr)
            
            reconstruction_results[f"mask_{int(mask_ratio*100)}%"] = {
                'mse_mean': float(np.mean(losses)),
                'mse_std': float(np.std(losses)),
                'psnr_mean': float(np.mean(psnrs)),
                'psnr_std': float(np.std(psnrs))
            }
            
            print(f"  Mask {int(mask_ratio*100)}%: MSE={np.mean(losses):.6f}±{np.std(losses):.6f}, PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} dB")
    
    
    ## 3/6: FEATURE EXTRACTION (Full Data for Linear Probe)
    print("\n[3/6] FEATURE EXTRACTION (Full Data for Linear Probe)")
    print("-"*80)
    
    all_features = []
    all_series_ids = []
    
    # Get all available files in the FULL_DATA_DIR
    full_data_files = [f for f in os.listdir(FULL_DATA_DIR) if f.endswith('.npz')]
    full_series_ids_from_files = set([int(f.split('_')[0]) for f in full_data_files if f.split('_')[0].isdigit()])

    # Filter series_df to only include series with files (Ensures valid labels)
    series_df_process = series_df[series_df['series_id'].isin(full_series_ids_from_files)].copy()

    # Create a list of (series_id, path) for the full dataset matching the CSV
    full_series_data = []
    for series_id in series_df_process['series_id']:
        f_labeled = os.path.join(FULL_DATA_DIR, f'{series_id}_labeled.npz')
        f_unlabeled = os.path.join(FULL_DATA_DIR, f'{series_id}_unlabeled.npz')
        
        if os.path.exists(f_labeled):
            full_series_data.append((series_id, f_labeled))
        elif os.path.exists(f_unlabeled):
            full_series_data.append((series_id, f_unlabeled))

    total_series = len(full_series_data)
    print(f"Extracting features from {total_series} series (Full Data for Linear Probe)...")
    
    for series_id, npz_path in tqdm(full_series_data, total=total_series, desc="Extracting"):
        data = np.load(npz_path)
        volume = data['volume']
        Z, Y, X = volume.shape
        
        # Center crop or pad to 128x128x128
        target_size = 128
        start_z = max(0, (Z - target_size) // 2)
        start_y = max(0, (Y - target_size) // 2)
        start_x = max(0, (X - target_size) // 2)
        patch = volume[start_z:start_z+target_size, start_y:start_y+target_size, start_x:start_x+target_size]
        
        if patch.shape != (target_size, target_size, target_size):
            pad_z = target_size - patch.shape[0]
            pad_y = target_size - patch.shape[1]
            pad_x = target_size - patch.shape[2]
            patch = np.pad(patch, [(0, pad_z), (0, pad_y), (0, pad_x)])
        
        if use_fp16:
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).half().to(device)
        else:
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        
        features, _ = extract_encoder_features(model, patch_tensor, device)
        
        all_features.append(features.cpu().float().numpy())
        all_series_ids.append(series_id)
    
    print(f"✓ Extracted features from {len(all_features)} series (Full Set)")
    
    
    ## 4/6: FEATURE QUALITY ANALYSIS (Holdout Set)
    print("\n[4/6] FEATURE QUALITY ANALYSIS (Holdout Set)")
    print("-"*80)
    
    X_holdout = []
    
    # --- Feature Extraction for Holdout (No CSV lookup) ---
    if HOLDOUT_SIZE > 0:
        for series_id, npz_path in tqdm(holdout_series_data, total=HOLDOUT_SIZE, desc="Holdout Features"):
            # Same feature extraction logic as Section 3 (omitted for brevity)
            data = np.load(npz_path)
            volume = data['volume']
            Z, Y, X = volume.shape
            
            # Center crop or pad to 128x128x128
            target_size = 128
            start_z = max(0, (Z - target_size) // 2)
            start_y = max(0, (Y - target_size) // 2)
            start_x = max(0, (X - target_size) // 2)
            patch = volume[start_z:start_z+target_size, start_y:start_y+target_size, start_x:start_x+target_size]
            
            if patch.shape != (target_size, target_size, target_size):
                pad_z = target_size - patch.shape[0]
                pad_y = target_size - patch.shape[1]
                pad_x = target_size - patch.shape[2]
                patch = np.pad(patch, [(0, pad_z), (0, pad_y), (0, pad_x)])
            
            if use_fp16:
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).half().to(device)
            else:
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            
            features, _ = extract_encoder_features(model, patch_tensor, device)
            X_holdout.append(features.cpu().float().numpy())

        X_holdout = np.vstack(X_holdout)
        print(f"Holdout Feature matrix shape: {X_holdout.shape}")
        
        # ... (Feature analysis and clustering logic remains unchanged)
        if len(X_holdout) < 2:
            print("Too few samples in holdout set for analysis.")
            feature_quality_results = {'error': 'Not enough holdout samples'}
        else:
            # Feature Statistics (variance, correlation, sparsity)
            # Clustering Analysis (K-Means, Silhouette, Davies-Bouldin)
            
            feature_variance = np.var(X_holdout, axis=0)
            avg_variance = np.mean(feature_variance)
            feature_corr = np.corrcoef(X_holdout.T)
            if feature_corr.ndim == 2:
                avg_corr = np.mean(np.abs(feature_corr[np.triu_indices_from(feature_corr, k=1)]))
            else:
                avg_corr = 0.0
            sparsity = np.mean(X_holdout == 0)
            
            print(f"\nFeature Statistics (Holdout):")
            print(f"  Average variance: {avg_variance:.6f}")
            print(f"  Average correlation: {avg_corr:.4f}")
            print(f"  Sparsity: {sparsity:.4f}")
            
            # Clustering Analysis
            print(f"\nClustering Analysis (Holdout):")
            n_clusters = min(5, len(X_holdout) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_holdout)
            
            silhouette = silhouette_score(X_holdout, cluster_labels)
            davies_bouldin = davies_bouldin_score(X_holdout, cluster_labels)
            
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
            
            feature_quality_results = {
                'avg_variance': float(avg_variance),
                'avg_correlation': float(avg_corr),
                'sparsity': float(sparsity),
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin)
            }


    else:
        print("No holdout data found. Skipping feature quality analysis.")
        feature_quality_results = {'error': 'No holdout data found'}
    
    # ... (Sections 5 and 6 remain mostly the same, ensuring they use the right data)
    
    ## 5/6: LINEAR PROBE EVALUATION (Full Dataset)
    print("\n[5/6] LINEAR PROBE EVALUATION (Full Dataset)")
    print("-"*80)
    # This section remains unchanged as it correctly uses all_features (from Full Data) 
    # and series_df_filtered (CSV labels matched to Full Data series IDs).

    X_full = np.vstack(all_features)
    print(f"Feature matrix shape (Full): {X_full.shape}")
    
    # Match labels to extracted features by series_id (Requires CSV)
    label_columns = [col for col in series_df.columns if col not in ['series_id', 'patient_id']]
    
    # Filter original CSV to only contain rows for which features were successfully extracted
    series_df_filtered = series_df[series_df['series_id'].isin(all_series_ids)].copy()
    # Reindex to ensure labels match the order of all_series_ids
    series_df_filtered = series_df_filtered.set_index('series_id').loc[all_series_ids].reset_index()
    
    y_full = series_df_filtered[label_columns].values
    print(f"Label matrix shape (Full): {y_full.shape}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_full, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    print(f"\nTraining Linear Probes:")
    print("-"*80)
    
    linear_probe_results = {}
    all_accuracies = []
    all_aucs = []
    
    for i, label_name in enumerate(label_columns):
        clf = LogisticRegression(max_iter=5000, random_state=42)
        
        if len(np.unique(y_train[:, i])) < 2:
            print(f"{label_name:30s} | SKIPPED (only one class in training set)")
            continue
        
        clf.fit(X_train, y_train[:, i])
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val[:, i], y_pred)
        
        if len(np.unique(y_val[:, i])) > 1:
            y_pred_proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val[:, i], y_pred_proba)
        else:
            auc = -1
        
        linear_probe_results[label_name] = {
            'accuracy': float(acc),
            'auc': float(auc) if auc > 0 else None
        }
        
        all_accuracies.append(acc)
        if auc > 0:
            all_aucs.append(auc)
        
        auc_str = f"{auc:.3f}" if auc > 0 else "N/A"
        print(f"{label_name:30s} | Acc: {acc:.3f} | AUC: {auc_str:>6s}")
    
    print("-"*80)
    avg_acc = np.mean(all_accuracies)
    avg_auc = np.mean(all_aucs) if all_aucs else -1
    
    print(f"\nOVERALL PERFORMANCE (FULL DATA):")
    print(f"  Average Accuracy: {avg_acc:.3f}")
    if avg_auc > 0:
        print(f"  Average AUC: {avg_auc:.3f}")
    
    linear_probe_summary = {
        'avg_accuracy': float(avg_acc),
        'avg_auc': float(avg_auc) if avg_auc > 0 else None,
        'per_label': linear_probe_results
    }


    
    ## 6/6: GENERATING VISUALIZATIONS (Holdout Set)
    print("\n[6/6] GENERATING VISUALIZATIONS (Holdout Set)")
    print("-"*80)
    
    # Feature Visualization (t-SNE/PCA)
    if 'silhouette_score' in feature_quality_results and 'error' not in feature_quality_results and len(X_holdout) >= 2:
        # Since holdout features have no labels from CSV, color by K-Means cluster
        vis_labels = cluster_labels
        print(" Coloring t-SNE/PCA by K-Means cluster")
        
        visualize_features_2d(
            X_holdout,
            vis_labels,
            os.path.join(RESULTS_DIR, 'feature_visualization.png')
        )
    else:
        print(" Not enough holdout samples for feature visualization.")
    
    # Attention map (Grad-CAM)
    if holdout_series_data:
        # Use the first series from the holdout data for visualization
        first_series_id, npz_path = holdout_series_data[0]
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            volume = data['volume']
            
            # Center crop or pad to 128x128x128
            target_size = 128
            Z, Y, X = volume.shape
            start_z = max(0, (Z - target_size) // 2)
            start_y = max(0, (Y - target_size) // 2)
            start_x = max(0, (X - target_size) // 2)
            patch = volume[start_z:start_z+target_size, start_y:start_y+target_size, start_x:start_x+target_size]
            
            if patch.shape != (target_size, target_size, target_size):
                pad_z = target_size - patch.shape[0]
                pad_y = target_size - patch.shape[1]
                pad_x = target_size - patch.shape[2]
                patch = np.pad(patch, [(0, pad_z), (0, pad_y), (0, pad_x)])
            
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
            if use_fp16:
                patch_tensor = patch_tensor.half()
            
            visualize_attention_map(
                model,
                patch_tensor,
                os.path.join(RESULTS_DIR, 'attention_map.png')
            )
        else:
            print("  ✗ Could not load data for attention map.")
    else:
        print("  ✗ No series processed for attention map.")

    
    # --- Evaluation Summary & Scoring (Unchanged) ---
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # ... (Summary and scoring logic remains the same, using the collected results)
    
    print("\nRECONSTRUCTION QUALITY (Holdout):")
    best_recon_mse = reconstruction_results['mask_75%']['mse_mean']
    for mask_level, metrics in reconstruction_results.items():
        print(f"  {mask_level}: MSE={metrics['mse_mean']:.6f}, PSNR={metrics['psnr_mean']:.2f} dB")
    
    print("\nFEATURE QUALITY (Holdout):")
    score = 0
    silhouette = feature_quality_results.get('silhouette_score')
    avg_acc = linear_probe_summary['avg_accuracy']
    
    if 'silhouette_score' in feature_quality_results and 'error' not in feature_quality_results:
        avg_variance = feature_quality_results['avg_variance']
        
        print(f"  Silhouette Score: {silhouette:.4f}")
        
        # Scoring based on Silhouette Score
        if silhouette is not None and silhouette > 0.5:
            score += 2
        elif silhouette is not None and silhouette > 0.3:
            score += 1
    else:
        print("  ✗ Skipping feature quality metrics.")

    print("\nLINEAR PROBE (FULL DATA):")
    print(f"  Average Accuracy: {avg_acc:.3f}")
    if avg_auc > 0:
        print(f"  Average AUC: {avg_auc:.3f}")

    # Scoring based on Reconstruction MSE
    if not np.isnan(best_recon_mse):
        if best_recon_mse < 0.01:
            score += 2
        elif best_recon_mse < 0.05:
            score += 1

    # Scoring based on Linear Probe Accuracy
    if avg_acc > 0.70:
        score += 2
    elif avg_acc > 0.60:
        score += 1

    # ... (Interpretation and final score printout)
    
    print("\n" + "="*80)
    print(f"OVERALL SCORE: {score}/6")
    print("="*80)

    # ... (Final JSON saving)
    final_results = {
        'timestamp': str(pd.Timestamp.now()),
        'model_path': WEIGHTS_FILE,
        'reconstruction': reconstruction_results,
        'feature_quality': feature_quality_results,
        'linear_probe': linear_probe_summary,
        'overall_score': f"{score}/6"
    }

    results_json_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_json_path, 'w') as jf:
        json.dump(final_results, jf, indent=2)

    print(f"\nSaved results to JSON: {results_json_path}")
