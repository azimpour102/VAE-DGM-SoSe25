import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# Try to import FID, but make it optional
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ModuleNotFoundError:
    print("Warning: FrechetInceptionDistance not available. Install with: pip install torchmetrics[image]")
    FID_AVAILABLE = False

from config import *

# def read_eval_config(device):
#     print("Loading the model ...")
#     checkpoint = {}
    
#     model = MODEL(device)
#     model.load_state_dict(torch.load(SAVED_MODEL_PATH)['model_state_dict'])
#     model.eval()

#     return model

def plot_losses(train_losses, val_losses, data_flag, reconstruction_losses=None, kl_losses=None):
    """Enhanced loss plotting with reconstruction and KL divergence tracking"""
    
    # Create subplots for comprehensive loss visualization
    if reconstruction_losses is not None and kl_losses is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main losses
        axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
        axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(reconstruction_losses, label='Reconstruction Loss', color='orange', alpha=0.8)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL divergence
        axes[1, 0].plot(kl_losses, label='KL Divergence', color='red', alpha=0.8)
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss ratio
        ratio = np.array(reconstruction_losses) / (np.array(kl_losses) + 1e-8)
        axes[1, 1].plot(ratio, label='Recon/KL Ratio', color='purple', alpha=0.8)
        axes[1, 1].set_title('Reconstruction to KL Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Training Analysis - {data_flag}")
        plt.tight_layout()
        
    else:
        # Fallback to simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.title("Model Loss During Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.savefig(data_flag + "_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def reconstruct_images(num, model, dataset, device, label=-1):
    """Enhanced reconstruction with better sampling"""
    actual_images = []
    reconstructed_images = []

    # Better sampling strategy
    if hasattr(dataset, 'targets') or hasattr(dataset, 'labels'):
        # Sample from each class if available
        try:
            targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
            unique_labels = torch.unique(targets)
            samples_per_class = max(1, num // len(unique_labels))
            
            indices = []
            for label_val in unique_labels:
                label_indices = torch.where(targets == label_val)[0]
                selected = torch.randperm(len(label_indices))[:samples_per_class]
                indices.extend(label_indices[selected].tolist())
            
            indices = indices[:num]  # Ensure we don't exceed num
        except:
            indices = np.random.choice(range(len(dataset)), num, replace=False)
    else:
        indices = np.random.choice(range(len(dataset)), num, replace=False)
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for actual_images, y in loader:
            actual_images = actual_images.to(device)
            y = y.to(device)
            if label == -1:
                reconstructed_images = model(actual_images)[0]
            else:
                c = torch.nn.functional.one_hot(y, num_classes=len(DATASET)).float().to(device)
                reconstructed_images = model(actual_images, c)[0]

    return actual_images.detach().cpu().numpy(), reconstructed_images.detach().cpu().numpy()

def generate_images(num, model, device, label=-1, temperature=1.0):
    """Enhanced generation with temperature control"""
    model.eval()
    with torch.no_grad():
        # Determine latent dimension more robustly
        if hasattr(model, 'mean_layer'):
            if hasattr(model.mean_layer, 'in_features'):
                latent_dim = model.mean_layer.in_features
            elif hasattr(model.mean_layer, '0') and hasattr(model.mean_layer[0], 'in_features'):
                latent_dim = model.mean_layer[0].in_features
            else:
                latent_dim = 128
        else:
            latent_dim = 128
        
        z_sample = torch.randn(num, latent_dim).to(device)
        z_sample = z_sample * temperature  # Temperature scaling
        
        if label == -1:
            x_decoded = model.decode(z_sample)
        else:
            labels = torch.full((num,), label, dtype=torch.long).to(device)
            c = torch.nn.functional.one_hot(labels, num_classes=len(DATASET)).float().to(device)
            x_decoded = model.decode(z_sample, c)
    
    return x_decoded.detach().cpu().numpy()

def interpolate_in_latent_space(model, dataset, device, num_steps=10, num_pairs=3):
    """Generate interpolations between pairs of images in latent space"""
    model.eval()
    
    # Get random pairs of images
    indices = np.random.choice(range(len(dataset)), num_pairs * 2, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num_pairs * 2, shuffle=False)
    
    interpolations = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Encode images to latent space
            if len(DATASET) == 1:
                mean, log_var = model.encode(images)
            else:
                c = torch.nn.functional.one_hot(labels, num_classes=len(DATASET)).float().to(device)
                mean, log_var = model.encode(images, c)
            
            # Sample from latent distribution
            z = model.reparameterization(mean, log_var)
            
            # Create interpolations between pairs
            for i in range(0, len(z), 2):
                if i + 1 < len(z):
                    z1, z2 = z[i], z[i + 1]
                    
                    # Linear interpolation in latent space
                    alphas = torch.linspace(0, 1, num_steps).to(device)
                    interp_z = torch.stack([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])
                    
                    # Decode interpolated latent codes
                    if len(DATASET) == 1:
                        interp_images = model.decode(interp_z)
                    else:
                        # Use label from first image for conditional generation
                        c_interp = c[i].unsqueeze(0).repeat(num_steps, 1)
                        interp_images = model.decode(interp_z, c_interp)
                    
                    interpolations.append(interp_images.cpu().numpy())
    
    return interpolations

def plot_images(images, saving_name, titles=None):
    """Enhanced image plotting with better layout"""
    if isinstance(images, list) and len(images) == 2:
        # Reconstruction format: [original, reconstructed]
        actual, reconstructed = images
        rows = 2
        cols = len(actual)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for j in range(cols):
            # Original images
            if actual[j].shape[0] == 3:  # RGB
                img = np.transpose(actual[j], (1, 2, 0))
            else:  # Grayscale
                img = actual[j].squeeze()
            axes[0, j].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[0, j].set_title('Original' if j == cols//2 else '')
            axes[0, j].axis('off')
            
            # Reconstructed images
            if reconstructed[j].shape[0] == 3:  # RGB
                img = np.transpose(reconstructed[j], (1, 2, 0))
            else:  # Grayscale
                img = reconstructed[j].squeeze()
            axes[1, j].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[1, j].set_title('Reconstructed' if j == cols//2 else '')
            axes[1, j].axis('off')
            
    else:
        # Generation format or other
        if isinstance(images, list):
            images = images[0] if len(images) == 1 else np.concatenate(images, axis=0)
        
        rows = len(images) if hasattr(images, '__len__') and not isinstance(images[0], np.ndarray) else len(images)
        cols = len(images[0]) if hasattr(images[0], '__len__') else 1
        
        if isinstance(images[0], np.ndarray) and len(images[0].shape) >= 2:
            # Single row of images
            rows = 2
            cols = len(images) // 2
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            if cols == 1:
                axes = axes.reshape(-1, 1)
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(images):
                        if images[idx].shape[0] == 3:  # RGB
                            img = np.transpose(images[idx], (1, 2, 0))
                        else:  # Grayscale
                            img = images[idx].squeeze()
                        axes[i, j].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                        if titles and idx < len(titles):
                            axes[i, j].set_title(titles[idx])
                        axes[i, j].axis('off')
        else:
            # Original format
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            
            for i in range(rows):
                for j in range(cols):
                    if rows == 1:
                        ax = axes[j] if cols > 1 else axes
                    elif cols == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i, j]
                    
                    if images[i][j].shape[0] == 3:  # RGB
                        img = np.transpose(images[i][j], (1, 2, 0))
                    else:  # Grayscale
                        img = images[i][j].squeeze()
                    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                    ax.axis('off')

    plt.tight_layout()
    plt.savefig(saving_name, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model_metrics(model, dataloader, device='cuda'):
    """Enhanced model evaluation with additional metrics"""
    model.to(device)
    model.eval()

    # Initialize metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Initialize FID only if available
    fid = None
    if FID_AVAILABLE:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Additional metrics
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    num_samples = 0
    
    # Track latent space utilization
    latent_means = []
    latent_vars = []

    with torch.no_grad():
        for inputs, y in dataloader:
            inputs = inputs.to(device)
            y = y.to(device)
            c = torch.nn.functional.one_hot(y, num_classes=len(DATASET)).float().to(device)

            if len(DATASET) == 1:
                outputs, mean, log_var = model(inputs)
            else:
                outputs, mean, log_var = model(inputs, c)

            # Calculate losses
            recon_loss = F.mse_loss(outputs, inputs, reduction='sum').item()
            kl_loss = (-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())).item()
            
            total_reconstruction_loss += recon_loss
            total_kl_loss += kl_loss
            num_samples += inputs.size(0)
            
            # Track latent statistics
            latent_means.append(mean.cpu())
            latent_vars.append(torch.exp(log_var).cpu())

            # Update traditional metrics
            psnr.update(outputs, inputs)
            ssim.update(outputs, inputs)

            # Update FID if available
            if fid is not None:
                # Resize images for FID if needed
                if inputs.shape[2] < 75 or inputs.shape[3] < 75:
                    inputs_resized = F.interpolate(inputs, size=(75, 75), mode='bilinear', align_corners=False)
                    outputs_resized = F.interpolate(outputs, size=(75, 75), mode='bilinear', align_corners=False)
                else:
                    inputs_resized = inputs
                    outputs_resized = outputs
                
                fid.update(inputs_resized, real=True)
                fid.update(outputs_resized, real=False)

    # Compute final scores
    metrics = {
        'PSNR': psnr.compute().item(),
        'SSIM': ssim.compute().item(),
        'Reconstruction_Loss': total_reconstruction_loss / num_samples,
        'KL_Divergence': total_kl_loss / num_samples,
        'Total_Loss': (total_reconstruction_loss + total_kl_loss) / num_samples
    }
    
    # Add FID if available
    if fid is not None:
        metrics['FID'] = fid.compute().item()
    
    # Latent space analysis
    all_means = torch.cat(latent_means, dim=0)
    all_vars = torch.cat(latent_vars, dim=0)
    
    metrics['Latent_Mean_Norm'] = torch.norm(all_means, dim=1).mean().item()
    metrics['Latent_Var_Mean'] = all_vars.mean().item()
    metrics['Active_Units'] = (all_vars.mean(dim=0) > 0.01).sum().item()  # Units with significant variance
    
    return metrics

def plot_latent_analysis(model, dataloader, device, save_name):
    """Analyze and visualize latent space properties"""
    model.eval()
    
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for inputs, y in dataloader:
            inputs = inputs.to(device)
            y = y.to(device)
            
            if len(DATASET) == 1:
                mean, log_var = model.encode(inputs)
            else:
                c = torch.nn.functional.one_hot(y, num_classes=len(DATASET)).float().to(device)
                mean, log_var = model.encode(inputs, c)
            
            z = model.reparameterization(mean, log_var)
            latent_codes.append(z.cpu())
            labels.append(y.cpu())
    
    latent_codes = torch.cat(latent_codes, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # Plot latent space visualization (2D projection using PCA if needed)
    if latent_codes.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_codes)
    else:
        latent_2d = latent_codes
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (PCA projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()