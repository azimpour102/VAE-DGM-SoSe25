import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt
import numpy as np

from config import *
from data_utils import *
from eval_utils import *
from train_utils import *
from vae_base import *

train_dataset, test_dataset, val_dataset = get_datasets(DATASET, batch_size=BATCH_SIZE, size=28, download=True, labels=range(len(DATASET)))
train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=BATCH_SIZE)

train_datasets = train_dataset, train_loader
val_datasets = val_dataset, val_loader

model, checkpoint = read_config(device)

if TRAINING_EPOCHS > 0:
    train_losses, val_losses = train(model, train_datasets, val_datasets, 
                                        checkpoint, TRAINING_EPOCHS, device)

    plot_losses(train_losses, val_losses, '_'.join(DATASET) + '_' + MODEL_NAME )

    torch.save({
        'epoch': checkpoint['start_epoch'] + TRAINING_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': checkpoint['optimizer'].state_dict(),
        'loss': checkpoint['loss']
    }, '_'.join(DATASET) + '_' + MODEL_NAME + '.pth')

if len(DATASET) == 1:
    reconstructed_images = reconstruct_images(5, model, test_dataset, device)
    plot_images(reconstructed_images, '_'.join(DATASET) + '_' + MODEL_NAME + "_reconstruction.png")
else:
    for target in range(len(DATASET)):
        indices = [i for i, (_, label) in enumerate(test_dataset) if label == target]
        reconstructed_images = reconstruct_images(5, model, Subset(test_dataset, indices), device, target)
        plot_images(reconstructed_images, DATASET[target] + '_' + MODEL_NAME + "_reconstruction.png")

if len(DATASET) == 1:
    generated_images = generate_images(10, model, device)
    plot_images([generated_images[:5], generated_images[5:]], '_'.join(DATASET) + '_' + MODEL_NAME + "_generation.png")
else:
    generated_images = []
    for target in range(len(DATASET)):
        generated_images = generate_images(10, model, device, target)
        plot_images([generated_images[:5], generated_images[5:]], DATASET[target] + '_' + MODEL_NAME + "_generation.png")

metrics = evaluate_model_metrics(model, test_loader, device)
print(metrics)
