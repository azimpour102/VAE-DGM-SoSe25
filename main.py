import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt
import numpy as np

from data_utils import *
from eval_utils import *
from train_utils import *
from vae_base import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_flag = 'octmnist'
BATCH_SIZE = 128

train_dataset, test_dataset, val_dataset = get_datasets(data_flag, batch_size=BATCH_SIZE, size=28, download=True)
train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=BATCH_SIZE)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = train(model, optimizer, 5, device)

# optimizer.param_groups[0]['lr'] = 0.0005

plot_losses(train_losses, val_losses, data_flag)
