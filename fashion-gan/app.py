"""
Gradio app for fashion silhouette generation with latent-space exploration.
Supports morphing, complexity control, and generating 1–16 silhouettes in a grid.
Run from the fashion-gan directory: python app.py
"""

import cv2
import numpy as np

import gradio as gr

from utils import generate_silhouette


def generate_image(
    seed: int | None = None,
    seed_b: int | None = None,
    alpha: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate one silhouette and return it as 256x256 numpy for display.
    After converting from tensor to [0,255], resize with INTER_NEAREST so silhouettes stay sharp.
    """
    img_tensor = generate_silhouette(
        seed=seed,
        seed_b=seed_b,
        alpha=alpha,
        scale=scale,
    )
    if hasattr(img_tensor, "cpu"):
        img_tensor = img_tensor.squeeze().cpu().numpy()
    img = (img_tensor * 0.5 + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    return img


def run_generate(
    seed: int | float | None,
    morph: float,
    complexity: float,
    num_images: int | float,
) -> list[np.ndarray]:
    """
    Generate num_images (1–16) silhouettes. Each image i uses seed_a=seed+i, seed_b=seed+i+1
    with the same morph and complexity. Returns list of 256x256 images for the gallery.
    """
    seed = int(seed) if seed is not None else 0
    num_images = int(num_images)
    num_images = max(1, min(num_images, 16))

    images = []
    for i in range(num_images):
        current_seed = seed + i
        seed_b = current_seed + 1
        img = generate_image(
            seed=current_seed,
            seed_b=seed_b,
            alpha=morph,
            scale=complexity,
        )
        images.append(img)
    return images


with gr.Blocks(title="Fashion Silhouette Generator") as demo:
    gr.Markdown("# Fashion Silhouette Generator")
    gr.Markdown(
        "Explore the DCGAN latent space: **Seed** defines the starting design; "
        "**Design Morph** interpolates between consecutive seeds; **Silhouette Complexity** scales the latent. "
        "Generate 1–16 silhouettes at once."
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

    gen_btn = gr.Button("Generate", variant="primary")

    output_gallery = gr.Gallery(
        label="Generated Silhouettes",
        columns=4,
        height=600,
    )

    gen_btn.click(
        fn=run_generate,
        inputs=[seed_input, morph_slider, complexity_slider, num_images],
        outputs=output_gallery,
    )

    # Show a few images on load
    demo.load(
        fn=lambda: run_generate(0, 0.0, 1.0, 8),
        inputs=[],
        outputs=output_gallery,
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
