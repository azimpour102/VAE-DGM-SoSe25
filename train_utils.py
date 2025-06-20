import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, train_dataset, val_dataset, train_loader, val_loader, optimizer, epochs, device):
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        overall_train_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        overall_val_loss = 0
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.view(-1, 784).to(device)

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_val_loss += loss.item()

        train_losses.append(overall_train_loss/len(train_dataset))
        val_losses.append(overall_val_loss/len(val_dataset))

        print("\tEpoch", epoch + 1, "\tAverage Training Loss: ", train_losses[-1], "\tAverage Validation Loss: ", val_losses[-1])

    return train_losses, val_losses