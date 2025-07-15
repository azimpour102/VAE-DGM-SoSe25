import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset

from config import *
from data_utils import *
from eval_utils import *
from train_utils import *
from vae_base import *

print(f"Using device: {device}")
print(f"Training configuration:")
print(f"- Datasets: {DATASET}")
print(f"- Model: {MODEL_NAME}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Epochs: {TRAINING_EPOCHS}")
print(f"- Learning rate: {LR}")

# Load datasets
train_dataset, test_dataset, val_dataset = get_datasets(DATASET, batch_size=BATCH_SIZE, size=28, download=True, labels=range(len(DATASET)))
train_loader, test_loader, val_loader = get_dataloaders(train_dataset, test_dataset, val_dataset, batch_size=BATCH_SIZE)

train_datasets = train_dataset, train_loader
val_datasets = val_dataset, val_loader

# Initialize model and training setup
model, checkpoint = read_config(device)
model = model.to(device)

print(f"Model architecture:")
print(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Training
if TRAINING_EPOCHS > 0:
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    train_losses, val_losses, reconstruction_losses, kl_losses = train(
        model, train_datasets, val_datasets, checkpoint, TRAINING_EPOCHS, device
    )

    # Enhanced plotting with detailed loss analysis
    plot_losses(
        train_losses, val_losses, 
        '_'.join(DATASET) + '_' + MODEL_NAME,
        reconstruction_losses, kl_losses
    )

    # Save final model
    final_model_path = '_'.join(DATASET) + '_' + MODEL_NAME + '.pth'
    torch.save({
        'epoch': checkpoint['start_epoch'] + TRAINING_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': checkpoint['optimizer'].state_dict(),
        'scheduler_state_dict': checkpoint['scheduler'].state_dict(),
        'loss': checkpoint['loss'],
        'train_losses': train_losses,
        'val_losses': val_losses,
        'reconstruction_losses': reconstruction_losses,
        'kl_losses': kl_losses,
        'best_val_loss': checkpoint['best_val_loss']
    }, final_model_path)
    
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: best_{final_model_path}")

print("\n" + "="*50)
print("EVALUATION AND VISUALIZATION")
print("="*50)

# Comprehensive model evaluation
print("\nEvaluating model performance...")
metrics = evaluate_model_metrics(model, test_loader, device)
print("Model Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

# Enhanced reconstruction visualization
print("\nGenerating reconstruction examples...")
if len(DATASET) == 1:
    actual, reconstructed = reconstruct_images(8, model, test_dataset, device)
    plot_images([actual, reconstructed], '_'.join(DATASET) + '_' + MODEL_NAME + "_reconstruction.png")
else:
    for target in range(len(DATASET)):
        indices = [i for i, (_, label) in enumerate(test_dataset) if label == target]
        if len(indices) > 0:
            subset_indices = np.random.choice(indices, min(5, len(indices)), replace=False)
            subset = Subset(test_dataset, subset_indices)
            actual, reconstructed = reconstruct_images(len(subset_indices), model, subset, device, target)
            plot_images([actual, reconstructed], DATASET[target] + '_' + MODEL_NAME + "_reconstruction.png")

# Enhanced generation with temperature control
print("\nGenerating new samples...")
if len(DATASET) == 1:
    # Generate with different temperatures
    for temp in [0.8, 1.0, 1.2]:
        generated_images = generate_images(10, model, device, temperature=temp)
        plot_images([generated_images[:5], generated_images[5:]], 
                   f'{DATASET[0]}_{MODEL_NAME}_generation_temp{temp}.png')
else:
    for target in range(len(DATASET)):
        # Generate with different temperatures
        for temp in [0.8, 1.0, 1.2]:
            generated_images = generate_images(10, model, device, target, temperature=temp)
            plot_images([generated_images[:5], generated_images[5:]], 
                       f'{DATASET[target]}_{MODEL_NAME}_generation_temp{temp}.png')

# Latent space interpolation
print("\nGenerating latent space interpolations...")
try:
    interpolations = interpolate_in_latent_space(model, test_dataset, device, num_steps=8, num_pairs=3)
    for i, interp in enumerate(interpolations):
        plot_images([interp], f'{DATASET[0] if len(DATASET) == 1 else "mixed"}_{MODEL_NAME}_interpolation_{i}.png')
    print(f"Generated {len(interpolations)} interpolation sequences")
except Exception as e:
    print(f"Could not generate interpolations: {e}")

# Latent space analysis
print("\nAnalyzing latent space...")
try:
    plot_latent_analysis(model, test_loader, device, 
                        f'{DATASET[0] if len(DATASET) == 1 else "mixed"}_{MODEL_NAME}_latent_analysis.png')
    print("Latent space analysis completed")
except Exception as e:
    print(f"Could not perform latent space analysis: {e}")

# Additional analysis for conditional models
if len(DATASET) > 1:
    print("\nConditional generation analysis...")
    
    # Generate samples for each class
    for class_idx in range(len(DATASET)):
        try:
            # Generate multiple samples for this class
            samples = generate_images(16, model, device, label=class_idx, temperature=1.0)
            
            # Arrange in 4x4 grid
            rows = []
            for i in range(0, 16, 4):
                rows.append(samples[i:i+4])
            
            plot_images(rows, f'{DATASET[class_idx]}_{MODEL_NAME}_class_samples.png')
            print(f"Generated samples for class {class_idx} ({DATASET[class_idx]})")
        except Exception as e:
            print(f"Could not generate samples for class {class_idx}: {e}")

print("\n" + "="*50)
print("TRAINING AND EVALUATION COMPLETE")
print("="*50)

print(f"\nGenerated files:")
print(f"- Performance plots: {DATASET[0] if len(DATASET) == 1 else 'mixed'}_{MODEL_NAME}_performance_analysis.png")
print(f"- Reconstruction examples: *_reconstruction.png")
print(f"- Generated samples: *_generation_*.png")
print(f"- Interpolations: *_interpolation_*.png")
print(f"- Latent analysis: *_latent_analysis.png")
if len(DATASET) > 1:
    print(f"- Class-specific samples: *_class_samples.png")

print(f"\nBest validation loss achieved: {checkpoint['best_val_loss']:.4f}")
print(f"Final model path: {final_model_path}")
print(f"Best model path: best_{final_model_path}")

print("\nTraining summary:")
if TRAINING_EPOCHS > 0:
    print(f"- Total epochs trained: {len(train_losses)}")
    print(f"- Final training loss: {train_losses[-1]:.4f}")
    print(f"- Final validation loss: {val_losses[-1]:.4f}")
    print(f"- Final reconstruction loss: {reconstruction_losses[-1]:.4f}")
    print(f"- Final KL loss: {kl_losses[-1]:.4f}")
else:
    print("- No training performed (TRAINING_EPOCHS = 0)")
