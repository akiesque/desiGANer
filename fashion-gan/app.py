"""
Gradio app for fashion silhouette generation with latent-space exploration.
Generated silhouettes are shown in the UI; FID and diversity are printed to the terminal only.
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
# Wider network (ngf=256); architecture matches current Generator but is not the same as ngf=128.
BEST_GENERATOR_NSN_CKPT = CHECKPOINTS_DIR / "generator_best_nsn.pt"


def _model_radio_choices() -> list[tuple[str, str]]:
    """
    (Human label, absolute checkpoint path) for gr.Radio.
    Value passed to the callback is the path string, so renames to the label cannot break routing.
    """
    c: list[tuple[str, str]] = []
    if BEST_GENERATOR_CKPT.is_file():
        c.append(("Healthier Training (ngf=128)", str(BEST_GENERATOR_CKPT.resolve())))
    if BEST_GENERATOR_NSN_CKPT.is_file():
        c.append(("Higher FID score (ngf=256)", str(BEST_GENERATOR_NSN_CKPT.resolve())))
    if not c:
        c.append(
            (f"Add generator_*.pt to {CHECKPOINTS_DIR} and restart", "")
        )
    return c


def _resolve_checkpoint_path(radio_value: str) -> Path:
    """Turn Radio value (path string) into a verified Path."""
    s = (radio_value or "").strip()
    if not s:
        raise FileNotFoundError(
            f"No generator checkpoint. Add a .pt file under {CHECKPOINTS_DIR} and restart the app."
        )
    p = Path(s)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {p}")
    return p

FID_MODE_SETTINGS: dict[str, tuple[int, int]] = {
    "Fast": (500, 200),
    "Accurate": (1_000, 1_000),
}

# Snapshot at import (restart the app after adding a new .pt on disk).
_DEFAULT_MODEL_CHOICES = _model_radio_choices()
_DEFAULT_CHECKPOINT_STR = _DEFAULT_MODEL_CHOICES[0][1] if _DEFAULT_MODEL_CHOICES else ""

# Optional preview-only CV postprocessing (safe to disable/remove).
ENABLE_PREVIEW_CV_POSTPROCESS = True


def _generate_raw_image_uint8(
    seed: int | None = None,
    seed_b: int | None = None,
    alpha: float | None = None,
    scale: float = 1.0,
    checkpoint_path: str | Path | None = None,
) -> np.ndarray:
    """Generate one silhouette and return raw uint8 grayscale image."""
    effective_alpha = alpha if (alpha is not None and alpha > 0.0) else None
    img_tensor = generate_silhouette(
        checkpoint_path=checkpoint_path,
        seed=seed,
        seed_b=seed_b,
        alpha=effective_alpha,
        scale=scale,
    )
    if hasattr(img_tensor, "cpu"):
        img_tensor = img_tensor.squeeze().cpu().numpy()
    img = (img_tensor * 0.5 + 0.5) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_graphic_silhouette_cv(image: np.ndarray) -> np.ndarray:
    """
    Preview-only CV pipeline for an inverted, edge-sharpened silhouette look.
    This does not affect FID images or model outputs used for evaluation.
    """
    inverted = cv2.bitwise_not(image)
    denoised = cv2.bilateralFilter(inverted, d=5, sigmaColor=25, sigmaSpace=25)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 0.5)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
    edges = cv2.Laplacian(sharpened, cv2.CV_16S, ksize=1)
    edges = cv2.convertScaleAbs(edges)
    return cv2.addWeighted(sharpened, 1.0, edges, 0.25, 0)


def generate_image(
    seed: int | None = None,
    seed_b: int | None = None,
    alpha: float | None = None,
    scale: float = 1.0,
    checkpoint_path: str | Path | None = None,
    preview_size: int = 256,
) -> np.ndarray:
    """Generate one silhouette and return resized preview for gallery."""
    raw_img = _generate_raw_image_uint8(
        seed=seed,
        seed_b=seed_b,
        alpha=alpha,
        scale=scale,
        checkpoint_path=checkpoint_path,
    )
    preview = cv2.resize(
        raw_img,
        (preview_size, preview_size),
        interpolation=cv2.INTER_CUBIC,
    )
    if ENABLE_PREVIEW_CV_POSTPROCESS:
        preview = _apply_graphic_silhouette_cv(preview)
    return preview


def run_generate_with_metrics(
    seed: int | float | None,
    morph: float,
    complexity: float,
    num_images: int | float,
    model_checkpoint: str = "",
    fid_mode: str = "Fast",
) -> list[np.ndarray]:
    """Generate silhouettes; FID and diversity are logged to the terminal (not the UI)."""
    import torch

    seed = int(seed) if seed is not None else 0
    num_images = max(1, min(int(num_images), 30))
    path_guess = (model_checkpoint or _DEFAULT_CHECKPOINT_STR or "").strip() or (
        _DEFAULT_MODEL_CHOICES[0][1] if _DEFAULT_MODEL_CHOICES else ""
    )
    checkpoint_path = _resolve_checkpoint_path(path_guess)
    print(
        f"[fashion-gan] Generator: {checkpoint_path.name} — "
        "separate `nn.Module` and cache entry per file (no shared weights with other models)."
    )
    real_samples, eval_samples = FID_MODE_SETTINGS.get(fid_mode, FID_MODE_SETTINGS["Fast"])

    images: list[np.ndarray] = []
    for i in range(num_images):
        current_seed = seed + i
        images.append(
            generate_image(
                checkpoint_path=checkpoint_path,
                seed=current_seed,
                seed_b=current_seed + 1,
                alpha=morph,
                scale=complexity,
            )
        )

    diversity_score = compute_diversity(np.array(images))
    print(f"[fashion-gan] diversity: {diversity_score:.4f} (higher = more spread in latent space)")

    fid_score: float | None = None
    try:
        ensure_real_images_fashion_mnist(get_real_images_dir(), num_samples=real_samples)
        gen_dir = get_generated_images_dir()
        fid_images = []
        for i in range(eval_samples):
            current_seed = seed + 100_000 + i
            fid_images.append(
                _generate_raw_image_uint8(
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
        print(f"[fashion-gan] FID: {fid_score:.4f} (lower = closer to real distribution)")
    except Exception as exc:
        print(f"[fashion-gan] FID computation failed: {exc!r}")

    return images


with gr.Blocks(title="Fashion Silhouette Generator") as demo:
    gr.Markdown("# Fashion Silhouette Generator")
    gr.Markdown(
        "Explore the DCGAN latent space: **Seed**, **Design Morph**, "
        "**Silhouette Complexity**. (FID and diversity are printed in the server terminal.)"
    )

    with gr.Row():
        output_gallery = gr.Gallery(
            label="Generated Silhouettes",
            height=690,
            columns=4,
        )
        with gr.Column():
            seed_input = gr.Slider(
                minimum=0,
                maximum=9999,
                value=0,
                step=1,
                label="Style",
                info="Random seed for the style of the generated silhouette.",
            )
            num_images = gr.Number(
                value=8,
                label="Number of Silhouettes",
                precision=0,
                minimum=1,
                maximum=30,
                info="Number of silhouettes to generate.",
            )
            complexity_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.5,
                step=0.1,
                label="Silhouette Complexity",
                info="Lower makes simpler shapes; higher can make more chaotic shapes.",
            )
            morph_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                step=0.01,
                label="Design Morph (Interpolation)",
                info="Interpolation factor for generated silhouette style.",
            )
            # Dev-only: expose Fast vs Accurate FID sample sizes (see FID_MODE_SETTINGS).
            # fid_mode_input = gr.Radio(
            #     choices=["Fast", "Accurate"],
            #     value="Fast",
            #     label="FID Evaluation Mode",
            #     info="Fast: quicker but noisier. Accurate: slower but more stable.",
            # )
            gen_btn = gr.Button("Generate", variant="primary")
            with gr.Row():
                with gr.Column(min_width=220):
                    model_input = gr.Radio(
                        choices=_DEFAULT_MODEL_CHOICES,
                        value=_DEFAULT_CHECKPOINT_STR,
                        label="Generator checkpoint",
                        info=(
                            "Each option is a distinct checkpoint/model file. Default is ngf=128; the wider net is ngf=256. "
                            "This function was only utilized for testing purposes only."
                        ),
                    )
            # with gr.Row():
            #     with gr.Column(scale=1, min_width=220):
            #         fid_number = gr.Number(
            #             label="FID Score",
            #             value=None,
            #             precision=2,
            #         )
            #     with gr.Column(scale=1, min_width=220):
            #         diversity_number = gr.Number(
            #             label="Diversity Score",
            #             value=None,
            #             precision=4,
            #         )

    # loss_plot = gr.Image(
    #     label="Training Loss Plot",
    #     type="filepath",
    #     height=300,
    # )

    def run_and_fill_metrics(seed, morph, complexity, num_images, model_checkpoint):
        # FID sample budget: FID_MODE_SETTINGS / "Fast" (see commented fid_mode_input for Accurate).
        return run_generate_with_metrics(
            seed, morph, complexity, num_images, model_checkpoint=model_checkpoint, fid_mode="Fast"
        )

    gen_btn.click(
        fn=run_and_fill_metrics,
        inputs=[seed_input, morph_slider, complexity_slider, num_images, model_input],
        outputs=[output_gallery],
    )

    def on_load():
        return run_generate_with_metrics(0, 0.0, 1.0, 8, model_checkpoint=_DEFAULT_CHECKPOINT_STR)

    demo.load(
        fn=on_load,
        inputs=[],
        outputs=[output_gallery],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())
