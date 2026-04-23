"""
DCGAN Discriminator for 64x64 images.
Maps (1, 64, 64) to a single logit (real vs fake) using spectrally-normalised conv layers and LeakyReLU.
No BatchNorm (incompatible with spectral norm). No Sigmoid (hinge loss requires raw logits).
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator: (1, 64, 64) -> SN-Conv layers -> LeakyReLU / Dropout -> Flatten -> Linear -> raw logit.
    """

    def __init__(self, ndf: int = 64, dropout: float = 0.2):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4 (kernel 4, stride 2, padding 1)
        self.conv_blocks = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        # Flatten: (ndf*8) * 4 * 4 -> 1 (logit)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ndf * 8 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        return self.fc(x)
