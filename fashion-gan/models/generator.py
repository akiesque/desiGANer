"""
DCGAN Generator for 28x28 fashion silhouettes.
Maps a latent noise vector (size 100) to a (1, 28, 28) image via FC -> reshape -> ConvTranspose2d upsampling.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator: z (latent 100) -> FC -> reshape -> ConvTranspose2d blocks (BatchNorm + ReLU) -> Tanh -> (1, 28, 28).
    """

    def __init__(self, latent_dim: int = 100, ngf: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # Project latent vector to spatial feature map: 100 -> 128*7*7, then reshape to (128, 7, 7)
        self.fc = nn.Linear(latent_dim, ngf * 7 * 7)
        self.reshape_size = (ngf, 7, 7)

        # Upsample 7x7 -> 14x14 -> 28x28 using transposed convolutions
        self.conv_blocks = nn.Sequential(
            # (batch, 128, 7, 7) -> (batch, 64, 14, 14)
            nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            # (batch, 64, 14, 14) -> (batch, 1, 28, 28)
            nn.ConvTranspose2d(ngf // 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Output in [-1, 1] to match normalized images
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        x = self.fc(z)
        x = x.view(-1, *self.reshape_size)
        return self.conv_blocks(x)
