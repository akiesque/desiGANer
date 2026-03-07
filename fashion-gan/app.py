"""
Simple Gradio app to generate fashion silhouettes using the trained DCGAN generator.
Run from the fashion-gan directory: python app.py
"""

import numpy as np

import gradio as gr

from utils import generate_silhouette


def generate_image(seed: int | None) -> np.ndarray:
    """
    Generate one silhouette from the trained generator.
    Returns a 28x28 grayscale image as numpy array in [0, 255] for display.
    """
    img = generate_silhouette(seed=seed)
    # (1, 1, 28, 28), range [-1, 1] -> (28, 28), range [0, 255]
    img = img.squeeze().cpu().numpy()
    img = (img * 0.5 + 0.5) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


with gr.Blocks(title="Fashion Silhouette Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Fashion Silhouette Generator")
    gr.Markdown("Generate new fashion silhouettes using the trained DCGAN. Click **Generate** for a random sample, or set a seed for reproducibility.")

    with gr.Row():
        seed_input = gr.Number(
            value=None,
            label="Seed (optional)",
            placeholder="Leave empty for random",
            precision=0,
        )
        gen_btn = gr.Button("Generate", variant="primary")

    output_image = gr.Image(
        label="Generated silhouette",
        type="numpy",
        height=280,
        image_mode="L",
    )

    def run_generate(seed):
        if seed is not None:
            seed = int(seed)
        return generate_image(seed=seed)

    gen_btn.click(fn=run_generate, inputs=seed_input, outputs=output_image)

    # Generate one on load so the UI isn't empty
    demo.load(fn=lambda: run_generate(None), inputs=[], outputs=output_image)


if __name__ == "__main__":
    demo.launch()
