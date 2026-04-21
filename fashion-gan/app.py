"""
Gradio app for fashion silhouette generation with latent-space exploration.
Shows generated silhouettes, training loss plot, FID score, and diversity score.
Run from the fashion-gan directory: python app.py
"""

from pathlib import Path

import cv2
import numpy as np

import gradio as gr

from utils import (
    compute_diversity,
    compute_fid,
    ensure_real_images_fashion_mnist,
    generate_silhouette,
    get_generated_images_dir,
    get_real_images_dir,
    save_numpy_images_for_fid,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_LOSS_PLOT = SCRIPT_DIR / "training_loss.png"
CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
BEST_GENERATOR_CKPT = CHECKPOINTS_DIR / "generator_best.pt"
FID_REAL_SAMPLES = 1000
FID_EVAL_SAMPLES = 1000
FID_MODE_SETTINGS: dict[str, tuple[int, int]] = {
    "Fast": (500, 200),
    "Accurate": (1000, 1000),
}


def generate_image(
    seed: int | None = None,
    seed_b: int | None = None,
    alpha: float | None = None,
    scale: float = 1.0,
    checkpoint_path: str | Path | None = None,
) -> np.ndarray:
    """
    Generate one silhouette and return it as 64x64 numpy for display.
    """
    img_tensor = generate_silhouette(
        checkpoint_path=checkpoint_path,
        seed=seed,
        seed_b=seed_b,
        alpha=alpha,
        scale=scale,
    )
    if hasattr(img_tensor, "cpu"):
        img_tensor = img_tensor.squeeze().cpu().numpy()
    img = (img_tensor * 0.5 + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    return img


def run_generate_with_metrics(
    seed: int | float | None,
    morph: float,
    complexity: float,
    num_images: int | float,
    fid_mode: str = "Fast",
) -> tuple[list[np.ndarray], str | None, float | None, float]:
    """
    Generate silhouettes and compute FID + diversity. Returns (images, loss_plot_path, fid_score, diversity_score).
    """
    import torch

    seed = int(seed) if seed is not None else 0
    num_images = int(num_images)
    num_images = max(1, min(num_images, 16))
    checkpoint_path = BEST_GENERATOR_CKPT if BEST_GENERATOR_CKPT.exists() else None
    real_samples, eval_samples = FID_MODE_SETTINGS.get(fid_mode, FID_MODE_SETTINGS["Fast"])

    images = []
    for i in range(num_images):
        current_seed = seed + i
        seed_b = current_seed + 1
        img = generate_image(
            checkpoint_path=checkpoint_path,
            seed=current_seed,
            seed_b=seed_b,
            alpha=morph,
            scale=complexity,
        )
        images.append(img)

    diversity_score = compute_diversity(np.array(images))

    fid_score = None
    try:
        ensure_real_images_fashion_mnist(get_real_images_dir(), num_samples=real_samples)
        gen_dir = get_generated_images_dir()
        fid_images = []
        for i in range(eval_samples):
            current_seed = seed + 100_000 + i
            fid_images.append(
                generate_image(
                    checkpoint_path=checkpoint_path,
                    seed=current_seed,
                    seed_b=current_seed + 1,
                    alpha=morph,
                    scale=complexity,
                )
            )
        save_numpy_images_for_fid(fid_images, gen_dir)
        fid_score = compute_fid(
            gen_dir,
            get_real_images_dir(),
            cuda=torch.cuda.is_available(),
        )
    except Exception:
        fid_score = None

    loss_plot_path = str(TRAINING_LOSS_PLOT) if TRAINING_LOSS_PLOT.exists() else None

    return (images, loss_plot_path, fid_score, diversity_score)


with gr.Blocks(title="Fashion Silhouette Generator") as demo:
    gr.Markdown("# Fashion Silhouette Generator")
    gr.Markdown(
        "Explore the DCGAN latent space: **Seed**, **Design Morph**, **Silhouette Complexity**. "
        "Generate 1–16 silhouettes and see FID and diversity metrics."
    )

    with gr.Row():
        seed_input = gr.Slider(
            minimum=0,
            maximum=9999,
            value=0,
            step=1,
            label="Seed",
        )
        num_images = gr.Number(
            value=8,
            label="Number of Silhouettes",
            precision=0,
            minimum=1,
            maximum=16,
        )
    with gr.Row():
        complexity_slider = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Silhouette Complexity",
        )
        morph_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.01,
            label="Design Morph (Interpolation)",
        )
        fid_mode_input = gr.Radio(
            choices=["Fast", "Accurate"],
            value="Fast",
            label="FID Evaluation Mode",
            info="Fast: quicker but noisier. Accurate: slower but more stable.",
        )

    gen_btn = gr.Button("Generate", variant="primary")

    output_gallery = gr.Gallery(
        label="Generated Silhouettes",
        columns=4,
        height=600,
    )
    loss_plot = gr.Image(
        label="Training Loss Plot",
        type="filepath",
        height=300,
    )
    fid_number = gr.Number(
        label="FID Score",
        value=None,
        precision=2,
    )
    diversity_number = gr.Number(
        label="Diversity Score",
        value=None,
        precision=4,
    )

    def run_and_fill_metrics(seed, morph, complexity, num_images, fid_mode):
        images, loss_path, fid, div = run_generate_with_metrics(
            seed, morph, complexity, num_images, fid_mode
        )
        return (
            images,
            loss_path,
            fid if fid is not None else 0.0,
            div,
        )

    gen_btn.click(
        fn=run_and_fill_metrics,
        inputs=[seed_input, morph_slider, complexity_slider, num_images, fid_mode_input],
        outputs=[output_gallery, loss_plot, fid_number, diversity_number],
    )

    def on_load():
        images, loss_path, fid, div = run_generate_with_metrics(0, 0.0, 1.0, 8, "Fast")
        return (
            images,
            loss_path,
            fid if fid is not None else 0.0,
            div,
        )

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[output_gallery, loss_plot, fid_number, diversity_number],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
