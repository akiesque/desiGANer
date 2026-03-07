"""
Utilities for DCGAN: loading checkpoint, generating silhouettes, and latent-space exploration.
All exploration is unsupervised (no labels); latent interpolation and scaling only.
"""

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


# ---------------------------------------------------------------------------
# Latent-space exploration (unsupervised: no labels, no conditional GAN)
# ---------------------------------------------------------------------------


def interpolate_latent(
    z1: torch.Tensor, z2: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Linear interpolation between two latent vectors (morphing).
    alpha=0 -> z1, alpha=1 -> z2; values in between produce smooth transitions.

    Args:
        z1: First latent, shape (1, latent_dim).
        z2: Second latent, shape (1, latent_dim).
        alpha: Blend factor in [0, 1].

    Returns:
        z_interp = (1 - alpha) * z1 + alpha * z2, same shape as z1/z2.
    """
    return (1.0 - alpha) * z1 + alpha * z2


def scale_latent(z: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale latent vector to control silhouette complexity (variation intensity).
    Smaller scale -> simpler shapes; larger scale -> more exaggerated.

    Typical range: 0.5 (simpler) .. 1.0 (normal) .. 2.0 (more exaggerated).

    Args:
        z: Latent vector, shape (batch, latent_dim).
        scale: Multiplier; z_scaled = scale * z.

    Returns:
        Scaled latent, same shape as z.
    """
    return scale * z


def latent_from_seed(
    seed: int,
    latent_dim: int = 100,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Produce a deterministic latent vector from a seed (for reproducibility).
    Returns shape (1, latent_dim).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    return torch.randn(1, latent_dim, device=device)


def _load_generator(
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    latent_dim: int = 100,
) -> Generator:
    """Load trained generator from checkpoint; used by generate_from_latent and generate_silhouette."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim=latent_dim)
    if checkpoint_path is None:
        ckpt_dir = get_checkpoints_dir()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"No checkpoints directory at {ckpt_dir}. Train the model first with: python train.py"
            )
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
    if isinstance(state, dict) and "model_state_dict" in state:
        generator.load_state_dict(state["model_state_dict"])
    else:
        generator.load_state_dict(state)
    generator = generator.to(device).eval()
    return generator


def generate_from_latent(
    z: torch.Tensor,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    latent_dim: int = 100,
) -> torch.Tensor:
    """
    Load the trained generator and produce an image from the given latent vector.
    No randomness; output is fully determined by z.

    Args:
        z: Latent vector, shape (1, latent_dim) or (batch, latent_dim).
        checkpoint_path: Generator checkpoint path; None = latest in checkpoints/.
        device: Device; None = auto.
        latent_dim: Must match trained model.

    Returns:
        Generated image tensor, shape (batch, 1, 28, 28), values in [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = _load_generator(
        checkpoint_path=checkpoint_path, device=device, latent_dim=latent_dim
    )
    with torch.no_grad():
        z = z.to(device)
        return generator(z)


def generate_silhouette(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    latent_dim: int = 100,
    seed: int | None = None,
    seed_b: int | None = None,
    alpha: float | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate one silhouette, with optional latent interpolation and complexity scaling.

    Pipeline (unsupervised, no labels):
    1. If alpha is provided: z1 from seed, z2 from seed_b, then z = interpolate_latent(z1, z2, alpha).
       Else: z = latent from seed (or random if seed is None).
    2. z = scale_latent(z, scale).
    3. Run z through the trained generator.

    Args:
        checkpoint_path: Generator checkpoint; None = latest.
        device: Device; None = auto.
        latent_dim: Latent dimension (default 100).
        seed: Random seed for first latent (None = random).
        seed_b: Second seed for interpolation; used only when alpha is not None.
        alpha: Interpolation factor in [0, 1]; if None, no interpolation.
        scale: Latent scale for complexity (default 1.0).

    Returns:
        Image tensor (1, 1, 28, 28), range [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if alpha is not None:
        # Two latents for morphing: z1 from seed, z2 from seed_b (fallback: seed+1 if seed_b missing)
        z1 = latent_from_seed(
            seed if seed is not None else 0, latent_dim=latent_dim, device=device
        )
        z2 = latent_from_seed(
            seed_b
            if seed_b is not None
            else (seed if seed is not None else 0) + 1,
            latent_dim=latent_dim,
            device=device,
        )
        z = interpolate_latent(z1, z2, alpha)
    else:
        if seed is not None:
            z = latent_from_seed(seed, latent_dim=latent_dim, device=device)
        else:
            torch.manual_seed(torch.randint(0, 2**31, (1,)).item())
            z = torch.randn(1, latent_dim, device=device)
    z = scale_latent(z, scale)
    return generate_from_latent(
        z, checkpoint_path=checkpoint_path, device=device, latent_dim=latent_dim
    )
