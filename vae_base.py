import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import math

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

# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Self-Attention for improved feature learning
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        proj_value = self.value(x).view(batch_size, -1, H * W)

        attention = self.softmax(torch.bmm(proj_query, proj_key))
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + x

# Residual Block with improved design
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.swish = Swish()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = self.swish(out)
        
        return out

class ImprovedFullyConnectedVAE(VAE):
    def __init__(self, device, input_dim=784, latent_dim=128, hidden_dims=[512, 256, 128]):
        super(ImprovedFullyConnectedVAE, self).__init__(device)
        
        # Build encoder layers
        encoder_layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                Swish(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mean_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers  
        decoder_layers = []
        in_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                Swish(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
            
        decoder_layers.extend([
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)

class ImprovedConvolutionalVAE(VAE):
    def __init__(self, device, latent_dim=128, input_channels=3):
        super(ImprovedConvolutionalVAE, self).__init__(device)
        
        # Encoder with residual blocks and attention
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            
            # Residual blocks with progressive downsampling
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),  # 14x14
            ResidualBlock(128, 128, stride=1),
            SelfAttention(128),  # Add attention at mid-level
            ResidualBlock(128, 256, stride=2),  # 7x7
            ResidualBlock(256, 256, stride=1),
            
            # Final processing
            nn.AdaptiveAvgPool2d((4, 4)),  # Ensure consistent size
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.BatchNorm2d(512),
            Swish()
        )
        
        self.flatten_dim = 512
        self.mean_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.Dropout(0.1)
        )
        self.logvar_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.Dropout(0.1)
        )
        
        # Decoder with upsampling and residual connections
        self.decoder_input = nn.Linear(latent_dim, 512)
        
        self.decoder = nn.Sequential(
            # Start from 1x1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),  # 4x4
            nn.BatchNorm2d(256),
            Swish(),
            
            # Progressive upsampling with residual blocks
            ResidualBlock(256, 256, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> interpolate to 14x14
            nn.BatchNorm2d(128),
            Swish(),
            
            ResidualBlock(128, 128, stride=1),
            SelfAttention(128),  # Attention for better detail reconstruction
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.BatchNorm2d(64),
            Swish(),
            
            ResidualBlock(64, 64, stride=1),
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1),
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
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)
        # Ensure output is exactly 28x28
        if x.size(-1) != 28:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return x

class ImprovedConditionalConvolutionalVAE(VAE):
    def __init__(self, device, latent_dim=128, num_classes=3, input_channels=3):
        super(ImprovedConditionalConvolutionalVAE, self).__init__(device)
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Improved label embedding with learnable spatial conditioning
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()  # Normalize the embedding
        )
        
        # Encoder with conditioning
        self.encoder = nn.Sequential(
            # Process concatenated input (image + label map)
            nn.Conv2d(input_channels + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            Swish(),
            
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),  # 14x14
            ResidualBlock(128, 128, stride=1),
            SelfAttention(128),
            ResidualBlock(128, 256, stride=2),  # 7x7
            ResidualBlock(256, 256, stride=1),
            
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.BatchNorm2d(512),
            Swish()
        )
        
        self.flatten_dim = 512
        self.mean_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)
        
        # Improved conditional decoder
        self.condition_projection = nn.Sequential(
            nn.Linear(num_classes, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.decoder_input = nn.Linear(latent_dim * 2, 512)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),  # 4x4
            nn.BatchNorm2d(256),
            Swish(),
            
            ResidualBlock(256, 256, stride=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            Swish(),
            
            ResidualBlock(128, 128, stride=1),
            SelfAttention(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 16x16
            nn.BatchNorm2d(64),
            Swish(),
            
            ResidualBlock(64, 64, stride=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 32x32
            nn.BatchNorm2d(32),
            Swish(),
            
            nn.Conv2d(32, input_channels, kernel_size=5, padding=2),  # Keep 32x32
            nn.Sigmoid()
        )
    
    def encode(self, x, c):
        batch_size = x.size(0)
        
        # Ensure c has the correct shape (batch_size, num_classes)
        if c.dim() > 2:
            c = c.view(batch_size, -1)
        elif c.dim() == 1:
            c = c.unsqueeze(0)
        
        # Create spatial conditioning map
        c_embed = self.label_embedding(c).view(batch_size, 1, 28, 28)
        x_conditioned = torch.cat([x, c_embed], dim=1)
        
        x = self.encoder(x_conditioned)
        x = x.view(x.size(0), -1)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        
        return mean, logvar
    
    def decode(self, z, c):
        batch_size = z.size(0)
        
        # Ensure c has the correct shape (batch_size, num_classes)
        if c.dim() > 2:
            c = c.view(batch_size, -1)
        elif c.dim() == 1:
            c = c.unsqueeze(0)
        
        # Project condition to latent space
        c_proj = self.condition_projection(c)
        z_conditioned = torch.cat([z, c_proj], dim=1)
        
        x = self.decoder_input(z_conditioned)
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)
        
        # Resize to match target size if needed
        if x.size(-1) != 28:
            x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        
        return x

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decode(z, c)
        return x_hat, mean, log_var

# Improved loss function with β-VAE and perceptual loss option
def improved_loss_function(x, x_hat, mean, log_var, beta=1.0, use_perceptual=False):
    """
    Improved VAE loss function with β-VAE weighting and optional perceptual loss
    
    Args:
        x: Original input
        x_hat: Reconstructed input
        mean: Latent mean
        log_var: Latent log variance
        beta: Weight for KL divergence term (β-VAE)
        use_perceptual: Whether to use perceptual loss instead of MSE
    """
    
    if use_perceptual:
        # Simple perceptual loss using feature differences
        # You could replace this with a pre-trained VGG loss for better results
        reproduction_loss = F.l1_loss(x_hat, x, reduction='sum')
    else:
        reproduction_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    return reproduction_loss + beta * KLD

# Keep backward compatibility
FullyConnectedVAE = ImprovedFullyConnectedVAE
ConvolutionalVAE = ImprovedConvolutionalVAE  
ConditionalConvolutionalVAE = ImprovedConditionalConvolutionalVAE
loss_function = improved_loss_function