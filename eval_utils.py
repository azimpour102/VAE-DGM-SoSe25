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

def read_eval_config(device):
    print("Loading the model ...")
    checkpoint = {}
    
    model = MODEL(device)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH)['model_state_dict'])
    model.eval()

    return model

def plot_losses(train_losses, val_losses, data_flag):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.title("Model Loss During the Training Process")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(data_flag + "_performance_analysis.png")
    # plt.show()

def reconstruct_images(num, model, dataset, device):
    actual_images = []
    reconstructed_images = []

    indices = np.random.choice(range(len(dataset)), num)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num, shuffle=False)
    for actual_images, _ in loader:
        actual_images = actual_images.to(device)
        reconstructed_images = model(actual_images)[0]

    return actual_images.detach().numpy(), reconstructed_images.detach().numpy()

def generate_images(num, model, device):
    z_sample = torch.randn(num, 128).to(device)
    x_decoded = model.decode(z_sample)
    
    return x_decoded.detach().numpy()

def plot_images(images, saving_name):
    rows = len(images)
    cols = len(images[0])

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(images[i][j].T)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(saving_name)
    # plt.show()

def evaluate_model_metrics(model, dataloader, device='cuda'):
    # Move model to device
    model.to(device)
    model.eval()

    # Initialize metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # Initialize FID only if available
    fid = None
    if FID_AVAILABLE:
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)[0]

            # Update metrics
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
        'SSIM': ssim.compute().item()
    }
    
    # Add FID if available
    if fid is not None:
        metrics['FID'] = fid.compute().item()
    
    return metrics