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



data_flag = args.dataset
BATCH_SIZE = args.batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset, test_dataset, val_dataset = get_datasets(args.dataset, batch_size=args.batch_size, size=28, download=True)
train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=args.batch_size)

model = FullyConnectedVAE(device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

train_losses, val_losses = train(model, train_dataset, val_dataset, train_loader, val_loader, optimizer, 5, device)

# optimizer.param_groups[0]['lr'] = 0.0005

plot_losses(train_losses, val_losses, data_flag)
