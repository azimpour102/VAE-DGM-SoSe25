import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse


import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt
import numpy as np

from data_utils import *
from eval_utils import *
from train_utils import *
from vae_base import *

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Training script for VAE.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="octmnist",
        help="Dataset to use for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs for training."
    )

    return parser.parse_args()

args = parse_args()

data_flag = args.dataset
BATCH_SIZE = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset, test_dataset, val_dataset = get_datasets(args.dataset, batch_size=args.batch_size, size=28, download=True)
train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=args.batch_size)

model = FullyConnectedVAE(device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Create checkpoint dictionary to match train function signature
checkpoint = {
    'optimizer': optimizer,
    'start_epoch': 0,
    'loss': None  # Will be set by train function
}

train_datasets = (train_dataset, train_loader)
val_datasets = (val_dataset, val_loader)

train_losses, val_losses = train(model, train_datasets, val_datasets, checkpoint, args.epochs, device)

# optimizer.param_groups[0]['lr'] = 0.0005

plot_losses(train_losses, val_losses, data_flag)
