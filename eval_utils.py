import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

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
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        # for batch in dataloader:
        for inputs, _ in dataloader:

            inputs = inputs.to(device)
            outputs = model(inputs)[0]

            # Update metrics
            psnr.update(outputs, inputs)
            ssim.update(outputs, inputs)

            # if inputs.shape[2] < 75 or inputs.shape[3] < 75:
            #     inputs = F.interpolate(inputs, size=(75, 75), mode='bilinear', align_corners=False)
            #     outputs = F.interpolate(outputs, size=(75, 75), mode='bilinear', align_corners=False)
            # fid.update(inputs, real=True)
            # fid.update(outputs, real=False)

    # Compute final scores
    return {
        # 'FID': fid.compute().item(),
        'PSNR': psnr.compute().item(),
        'SSIM': ssim.compute().item()
    }