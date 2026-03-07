"""
Utilities for DCGAN: loading checkpoint, generating a single silhouette, and path helpers.
"""

import os
from pathlib import Path

import torch

from models import Generator


# Default paths relative to this file (fashion-gan/utils.py -> fashion-gan/ is parent)
def _project_root() -> Path:
    return Path(__file__).resolve().parent


def get_checkpoints_dir() -> Path:
    """Directory where model checkpoints are saved."""
    return _project_root() / "checkpoints"


def get_samples_dir() -> Path:
    """Directory where sample image grids are saved."""
    return _project_root() / "samples"


def generate_silhouette(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    latent_dim: int = 100,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Load a trained generator and produce one generated image from random noise.

    Args:
        checkpoint_path: Path to generator checkpoint (e.g. checkpoints/generator_epoch_50.pt).
                         If None, uses the latest checkpoint in checkpoints/ if available.
        device: Device to run on (cuda/cpu). Auto-detected if None.
        latent_dim: Latent dimension (must match the trained model, default 100).
        seed: Optional random seed for reproducibility.

    Returns:
        Single image tensor of shape (1, 1, 28, 28) in range [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim=latent_dim)

    if checkpoint_path is None:
        ckpt_dir = get_checkpoints_dir()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"No checkpoints directory at {ckpt_dir}. Train the model first with: python train.py"
            )
        # Prefer generator checkpoint; fallback to any .pt file
        candidates = list(ckpt_dir.glob("generator*.pt")) or list(ckpt_dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint files found in {ckpt_dir}. Train the model first."
            )
        checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Support both full checkpoint dict and raw state_dict saved by train.py
    if isinstance(state, dict) and "model_state_dict" in state:
        generator.load_state_dict(state["model_state_dict"])
    else:
        generator.load_state_dict(state)
    generator = generator.to(device)
    generator.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        img = generator(z)

    return img
