import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

class FullyConnectedVAE(VAE):
    def __init__(self, input_dim=784, latent_dim=128, device=device):
        super(FullyConnectedVAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.Sigmoid(),
            nn.Linear(800, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid()
            )

        # latent mean and variance
        self.mean_layer = nn.Linear(512, latent_dim)
        self.logvar_layer = nn.Linear(512, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 800),
            nn.Sigmoid(),
            nn.Linear(800, input_dim),
            nn.Sigmoid()
            )