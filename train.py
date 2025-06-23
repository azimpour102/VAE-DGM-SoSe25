import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_utils import *
from eval_utils import *
from train_utils import *



def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Training script for VAE.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=True,
        help="Device to use for training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="octmnist",
        required=True,
        help="Dataset to use for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        required=True,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        required=True,
        help="Number of epochs for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        required=True,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="FullyConnectedVAE",
        required=True,
        help="Model to use for training."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="FullyConnectedVAE",
        required=True,
        help="Model to use for training."
    )
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        required=True,
        help="Load model from path."
    )

    return parser.parse_args()

args = parse_args()


for dataset in DATASETS:
    train_dataset, test_dataset, val_dataset = get_datasets(dataset, batch_size=BATCH_SIZE, size=28, download=True)
    train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=BATCH_SIZE)

    train_datasets = train_dataset, train_loader
    val_datasets = val_dataset, val_loader
    
    model, checkpoint = read_config(device)

    train_losses, val_losses = train(model, train_datasets, val_datasets, 
                                     checkpoint, TRAINING_EPOCHS, device)

    plot_losses(train_losses, val_losses, dataset)

    torch.save({
        'epoch': checkpoint['start_epoch'] + TRAINING_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': checkpoint['optimizer'].state_dict(),
        'loss': checkpoint['loss']
    }, dataset + MODEL_NAME + '.pth')

    reconstructed_images = reconstruct_images(5, model, test_dataset, device)
    plot_images(reconstructed_images, dataset + "_reconstruction.png")

    generated_images = generate_images(10, model, device)
    plot_images([generated_images[:5], generated_images[5:]], dataset + "_generation.png")
