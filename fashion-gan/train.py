"""
Train a DCGAN on Fashion-MNIST for generating fashion silhouettes.
Run from the fashion-gan directory: python train.py
Tracks G/D loss, saves training_loss.png, and optionally runs FID/diversity evaluation.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from models import Generator, Discriminator
from utils import get_checkpoints_dir, get_samples_dir

# ---------------------------------------------------------------------------
# Configuration (batch 128, 28x28, BCE, Adam lr=0.0002, betas=(0.5, 0.999))
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
IMAGE_SIZE = 64
LATENT_DIM = 100
LR_G = 0.0002
LR_D = 0.00003
BETAS = (0.5, 0.999)
NUM_EPOCHS = 200
# Legacy combined-loss patience (unused). Early stop uses G loss only: see train loop.
EARLY_STOPPING_PATIENCE = None

# Paths relative to this script (fashion-gan/train.py)
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = SCRIPT_DIR / "samples"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

print(f"Using device: {DEVICE}")

def get_dataloader():
    """
    Load Fashion-MNIST with torchvision (auto-download), normalize to [-1, 1], batch size 128.
    Images are 28x28 grayscale; ToTensor() gives [0,1], we map to [-1,1] to match Generator Tanh output.
    """
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    dataset = datasets.FashionMNIST(
        root=str(SCRIPT_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 avoids multiprocessing issues on Windows; increase on Linux
        pin_memory=True if torch.cuda.is_available() else False,
    )


def train():
    # Create output directories
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader()

    # Build models
    generator = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)

    # Fixed noise for consistent sample grids every epoch
    fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

    # Store loss per epoch for plotting (training stability)
    g_losses = []
    d_losses = []
    best_g_loss = float("inf")
    best_generator_state = None
    g_loss_high_streak = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0

        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(DEVICE)
            batch_len = real_imgs.size(0)

            d_real_labels = torch.full((batch_len, 1), 0.9, device=DEVICE)
            d_fake_labels = torch.zeros(batch_len, 1, device=DEVICE)
            g_fake_labels = torch.ones(batch_len, 1, device=DEVICE)

            # --- Step 1: Train Discriminator on real images (maximize log D(real)) ---
            opt_d.zero_grad()
            pred_real = discriminator(real_imgs)
            loss_d_real = criterion(pred_real, d_real_labels)
            loss_d_real.backward()

            # --- Step 2 & 3: Generate fakes and train Discriminator on fakes (maximize log(1 - D(G(z)))) ---
            z = torch.randn(batch_len, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z).detach()  # no grad through G
            pred_fake = discriminator(fake_imgs)
            loss_d_fake = criterion(pred_fake, d_fake_labels)
            loss_d_fake.backward()
            opt_d.step()

            d_loss = loss_d_real.item() + loss_d_fake.item()
            d_loss_epoch += d_loss
            num_batches += 1

            # --- Step 4: Train Generator to fool Discriminator (minimize log(1 - D(G(z))) or maximize log D(G(z))) ---
            opt_g.zero_grad()
            z = torch.randn(batch_len, LATENT_DIM, device=DEVICE)
            fake_imgs = generator(z)
            pred_fake = discriminator(fake_imgs)
            loss_g = criterion(pred_fake, g_fake_labels)
            loss_g.backward()
            opt_g.step()

            g_loss_epoch += loss_g.item()

        g_loss_epoch /= num_batches
        d_loss_epoch /= num_batches
        g_losses.append(g_loss_epoch)
        d_losses.append(d_loss_epoch)

        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch
            best_generator_state = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()}

        if g_loss_epoch > 1.5:
            g_loss_high_streak += 1
        else:
            g_loss_high_streak = 0

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]  D_loss: {d_loss_epoch:.4f}  G_loss: {g_loss_epoch:.4f}")

        # Save generated sample grid every epoch
        generator.eval()
        with torch.no_grad():
            fake = generator(fixed_noise)
        generator.train()
        grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
        samples_path = SAMPLES_DIR / f"epoch_{epoch:04d}.png"
        utils.save_image(grid, samples_path)
        print(f"  Saved samples -> {samples_path}")

        # Save model checkpoints every epoch
        ckpt_g = CHECKPOINTS_DIR / f"generator_epoch_{epoch:04d}.pt"
        ckpt_d = CHECKPOINTS_DIR / f"discriminator_epoch_{epoch:04d}.pt"
        torch.save(generator.state_dict(), ckpt_g)
        torch.save(discriminator.state_dict(), ckpt_d)
        print(f"  Saved checkpoints -> {ckpt_g.name}, {ckpt_d.name}")

        if g_loss_high_streak >= 20:
            print("  Early stopping: G loss > 1.5 for 20 consecutive epochs.")
            break

    if best_generator_state is not None:
        generator.load_state_dict(best_generator_state)
        print(f"  Restored generator to best G loss ({best_g_loss:.4f}).")

    # Save one generator for deployment (not gitignored; commit this so the app works live)
    best_path = CHECKPOINTS_DIR / "generator_best.pt"
    torch.save(generator.state_dict(), best_path)
    print(f"  Saved deployment model -> {best_path.name}")

    # Plot and save training loss (epoch vs loss)
    loss_plot_path = SCRIPT_DIR / "training_loss.png"
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.legend()
    plt.title("DCGAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training loss plot -> {loss_plot_path.name}")

    # Optional: run FID and diversity evaluation (saves real_images/ and generated_images/)
    fid, div = None, None
    try:
        from utils import (
            ensure_real_images_fashion_mnist,
            get_generated_images_dir,
            get_real_images_dir,
            save_generated_for_fid,
            compute_fid,
            compute_diversity,
        )
        real_dir = ensure_real_images_fashion_mnist(get_real_images_dir(), num_samples=2000)
        generator.eval()
        with torch.no_grad():
            z_eval = torch.randn(2000, LATENT_DIM, device=DEVICE)
            fake_eval = generator(z_eval)
        save_generated_for_fid(fake_eval, get_generated_images_dir())
        fid = compute_fid(get_generated_images_dir(), real_dir, cuda=torch.cuda.is_available())
        div = compute_diversity(fake_eval.cpu().numpy())
        print(f"  FID Score: {fid:.2f}  Diversity Score: {div:.4f}")
    except Exception as e:
        print(f"  Evaluation skipped: {e}")

    # Write metrics to a .txt file
    metrics_path = SCRIPT_DIR / "training_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("DCGAN Training Metrics\n")
        f.write("=====================\n")
        f.write(f"Epochs: {len(g_losses)}\n")
        f.write(f"Final Generator Loss: {g_losses[-1]:.4f}\n")
        f.write(f"Final Discriminator Loss: {d_losses[-1]:.4f}\n")
        if fid is not None:
            f.write(f"FID Score: {fid:.4f}\n")
        if div is not None:
            f.write(f"Diversity Score: {div:.4f}\n")
    print(f"  Saved metrics -> {metrics_path.name}")

    print("Training complete.")


if __name__ == "__main__":
    train()
