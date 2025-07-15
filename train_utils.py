import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from config import *
from vae_base import improved_loss_function

def loss_function(x, x_hat, mean, log_var, beta=1.0, use_perceptual=False):
    return improved_loss_function(x, x_hat, mean, log_var, beta, use_perceptual)

torch.serialization.add_safe_globals([loss_function])

def read_config(device):
    print("Loading / Initiating the model ...")
    checkpoint = {}
    
    if len(DATASET) == 1:
        model = MODEL_TYPE(device)
    else:
        model = MODEL_TYPE(device, num_classes=len(DATASET))
    
    # Use AdamW optimizer with weight decay for better regularization
    checkpoint['optimizer'] = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Add learning rate scheduler
    checkpoint['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
        checkpoint['optimizer'], mode='min', factor=0.5, patience=5, verbose=True
    )
    
    checkpoint['start_epoch'] = 0
    checkpoint['loss'] = loss_function
    checkpoint['best_val_loss'] = float('inf')
    checkpoint['patience_counter'] = 0
    checkpoint['beta'] = 0.1  # Start with small beta for β-VAE

    if LOAD_MODEL:
        _checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(_checkpoint['model_state_dict'])
        checkpoint['optimizer'].load_state_dict(_checkpoint['optimizer_state_dict'])
        checkpoint['start_epoch'] = _checkpoint['epoch']
        checkpoint['loss'] = _checkpoint['loss']
        if 'best_val_loss' in _checkpoint:
            checkpoint['best_val_loss'] = _checkpoint['best_val_loss']
    
    return model, checkpoint

def train(model, train_datasets, val_datasets, checkpoint, epochs, device):
    train_dataset, train_loader = train_datasets
    val_dataset, val_loader = val_datasets

    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    loss_fn = checkpoint['loss']
    start_epoch = checkpoint['start_epoch']
    best_val_loss = checkpoint['best_val_loss']
    patience_counter = checkpoint['patience_counter']
    beta = checkpoint['beta']
    
    print("Training", epochs, "epochs, starting from epoch", start_epoch, "...")
    print(f"Starting with β = {beta}")

    model.train()
    train_losses = []
    val_losses = []
    reconstruction_losses = []
    kl_losses = []
    
    for epoch in range(epochs):
        # β-VAE warmup: gradually increase beta
        if epoch < epochs // 3:
            current_beta = beta * (epoch + 1) / (epochs // 3)
        else:
            current_beta = min(1.0, beta * 1.5)  # Cap at 1.5 for stability
        
        model.train()
        overall_train_loss = 0
        overall_recon_loss = 0
        overall_kl_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 10 == 0:  # Print less frequently
                print(f"Training epoch {epoch+1}, batch {batch_idx}")
            
            x = x.to(device)
            y = y.to(device)
            c = torch.nn.functional.one_hot(y, num_classes=len(DATASET)).float().to(device)

            optimizer.zero_grad()

            if len(DATASET) == 1:
                x_hat, mean, log_var = model(x)
            else:
                x_hat, mean, log_var = model(x, c)
            
            # Calculate loss components separately for monitoring
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = recon_loss + current_beta * kl_loss

            overall_train_loss += loss.item()
            overall_recon_loss += recon_loss.item()
            overall_kl_loss += kl_loss.item()

            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

        # Validation phase
        model.eval()
        overall_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                if batch_idx % 10 == 0:
                    print(f"Validation epoch {epoch+1}, batch {batch_idx}")
                
                x = x.to(device)
                y = y.to(device)
                c = torch.nn.functional.one_hot(y, num_classes=len(DATASET)).float().to(device)

                if len(DATASET) == 1:
                    x_hat, mean, log_var = model(x)
                else:
                    x_hat, mean, log_var = model(x, c)
                
                recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = recon_loss + current_beta * kl_loss

                overall_val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = overall_train_loss / len(train_dataset)
        avg_val_loss = overall_val_loss / len(val_dataset)
        avg_recon_loss = overall_recon_loss / len(train_dataset)
        avg_kl_loss = overall_kl_loss / len(train_dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        reconstruction_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)

        print(f"\tEpoch {epoch + 1}")
        print(f"\t\tAverage Training Loss: {avg_train_loss:.4f}")
        print(f"\t\tAverage Validation Loss: {avg_val_loss:.4f}")
        print(f"\t\tReconstruction Loss: {avg_recon_loss:.4f}")
        print(f"\t\tKL Loss: {avg_kl_loss:.4f}")
        print(f"\t\tCurrent β: {current_beta:.4f}")
        print(f"\t\tLearning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_fn,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'reconstruction_losses': reconstruction_losses,
                'kl_losses': kl_losses
            }, 'best_' + '_'.join(DATASET) + '_' + MODEL_NAME + '.pth')
            print(f"\t\tNew best validation loss! Model saved.")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= 10:  # Stop if no improvement for 10 epochs
            print(f"\t\tEarly stopping after {patience_counter} epochs without improvement")
            break
    
    # Update checkpoint with training history
    checkpoint['best_val_loss'] = best_val_loss
    checkpoint['patience_counter'] = patience_counter
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    checkpoint['reconstruction_losses'] = reconstruction_losses
    checkpoint['kl_losses'] = kl_losses
    
    return train_losses, val_losses, reconstruction_losses, kl_losses