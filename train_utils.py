import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from config import *

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
torch.serialization.add_safe_globals([loss_function])

def read_config(device):
    print("Loading / Initiaiting the model ...")
    checkpoint = {}
    
    model = MODEL(device)
    checkpoint['optimizer'] = optim.Adam(model.parameters(), lr=0.001)
    checkpoint['start_epoch'] = 0
    checkpoint['loss'] = loss_function

    if LOAD_MODEL:
        _checkpoint = torch.load(SAVED_MODEL_PATH)
        model.load_state_dict(_checkpoint['model_state_dict'])
        checkpoint['optimizer'].load_state_dict(_checkpoint['optimizer_state_dict'])
        checkpoint['start_epoch'] = _checkpoint['epoch']
        checkpoint['loss'] = _checkpoint['loss']
    
    return model, checkpoint

def train(model, train_datasets, val_datasets, checkpoint, epochs, device):
    train_dataset, train_loader = train_datasets
    val_dataset, val_loader = val_datasets

    optimizer = checkpoint['optimizer']
    loss = checkpoint['loss']
    start_epoch = checkpoint['start_epoch']
    print("Training", epochs, "epochs, starting from epoch", start_epoch, "...")

    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        overall_train_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            # x = x.view(-1, 784).to(device)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        overall_val_loss = 0
        for batch_idx, (x, _) in enumerate(val_loader):
            # x = x.view(-1, 784).to(device)
            x = x.to(device)

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_val_loss += loss.item()

        train_losses.append(overall_train_loss/len(train_dataset))
        val_losses.append(overall_val_loss/len(val_dataset))

        print("\tEpoch", epoch + 1, "\tAverage Training Loss: ", train_losses[-1], "\tAverage Validation Loss: ", val_losses[-1])
    
    return train_losses, val_losses