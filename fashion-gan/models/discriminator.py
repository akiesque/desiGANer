"""
DCGAN Discriminator for 28x28 images.
Maps (1, 28, 28) to a single probability (real vs fake) using conv layers, LeakyReLU, optional Dropout, Sigmoid.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator: (1, 28, 28) -> Conv layers (stride) -> LeakyReLU / Dropout -> Flatten -> Linear -> Sigmoid -> scalar.
    """

    def __init__(self, ndf: int = 64, dropout: float = 0.3):
        super().__init__()
        # 28 -> 14 -> 7 -> 3 (with kernel 4, stride 2, padding 1)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
        )
        # Flatten: (ndf*4) * 3 * 3 -> 1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = self.conv_blocks(x)
        return self.fc(x)
