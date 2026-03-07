# Fashion DCGAN

DCGAN for generating fashion silhouettes from Fashion-MNIST.

## Run training

From the `fashion-gan` directory:

```bash
python train.py
```

This will:

- Download Fashion-MNIST under `fashion-gan/data/`
- Train the generator and discriminator (default 50 epochs)
- Save sample grids every epoch to `samples/`
- Save generator/discriminator checkpoints every epoch to `checkpoints/`

## Run the Gradio app

After training, launch a simple UI to generate silhouettes:

```bash
python app.py
```

Then open the URL shown in the terminal (e.g. http://127.0.0.1:7860). Use **Generate** for a new random silhouette, or set a **Seed** for reproducibility.

## Generate a single silhouette (Python)

After training, load a checkpoint and generate one image:

```python
from utils import generate_silhouette

# Uses latest checkpoint in checkpoints/ if path is None
img = generate_silhouette(checkpoint_path=None, seed=42)
# img shape: (1, 1, 28, 28), values in [-1, 1]
```

To save as PNG (e.g. for a mini-app), denormalize to [0, 1] and use `torchvision.utils.save_image(img, "out.png", normalize=True, value_range=(-1, 1)`.
