import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.device = device

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
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
    def __init__(self, device, input_dim=784, latent_dim=128):
        super(FullyConnectedVAE, self).__init__(device)

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

class ConvolutionalVAE(VAE):
    def __init__(self, device, latent_dim=128):
        super(ConvolutionalVAE, self).__init__(device)

        # Encoder: Input shape (3, 28, 28)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.flatten_dim = 256 * 7 * 7
        self.mean_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # (3, 28, 28)
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 7, 7)
        x = self.decoder(x)
        return x