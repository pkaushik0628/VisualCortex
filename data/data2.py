"""
data.py
=======
LGN preprocessing pipeline for Network2D training.

This implementation follows the preprocessing described in:

- Van Hateren natural image pipeline
- Divisively normalised Difference-of-Gaussians (DoG)
- ON/OFF LGN rectification
- Patch-based sampling (16×16)
- Rate-coded neural inputs (no spike generation)

Output format
-------------
R_X : FloatTensor (2*N_X, N_b)
    ON  channel: max(F, 0)
    OFF channel: max(-F, 0)

Values are scaled to approximate LGN firing rates (~20 Hz mean,
bounded implicitly by downstream models if needed).
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Divisively normalised DoG (LGN model)
# ─────────────────────────────────────────────────────────────

def dog_filter(
        img: np.ndarray,
        sigma_c: float = 1.0,
        sigma_s: float = 1.5,
        sigma_d: float = 1.5,
        eps: float = 1e-6,
) -> np.ndarray:
    """
    Divisively normalised Difference-of-Gaussians:

        F(x,y) = (F_c - F_s) / (F_d + eps)

    Parameters
    ----------
    img : (H, W) float32, zero-mean unit-std image
    """

    F_c = gaussian_filter(img, sigma=sigma_c)
    F_s = gaussian_filter(img, sigma=sigma_s)
    F_d = gaussian_filter(img, sigma=sigma_d)

    return ((F_c - F_s) / (F_d + eps)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Image preprocessing → LGN response
# ─────────────────────────────────────────────────────────────

def preprocess_image(
        pil_img: Image.Image,
        size: int = 64,
        patch_size: int = 16,
        scale: float = 70.0,
        sigma_c: float = 1.0,
        sigma_s: float = 1.5,
        sigma_d: float = 1.5,
) -> np.ndarray:
    """
    Convert image → LGN ON/OFF vector.

    Steps
    -----
    1. Grayscale + resize
    2. Zero-mean, unit-std normalization
    3. Divisively normalized DoG filter
    4. Extract random 16×16 patch
    5. Scale responses
    6. ON/OFF rectification

    Returns
    -------
    (2 * patch_size^2,) float32 vector
    """

    # grayscale + resize
    img = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # normalize (Van Hateren assumption)
    std = arr.std()
    if std > 1e-8:
        arr = (arr - arr.mean()) / std

    # LGN filter
    filt = dog_filter(arr, sigma_c, sigma_s, sigma_d)

    # sample random patch (16×16 default)
    H = size - patch_size
    x = np.random.randint(0, H + 1)
    y = np.random.randint(0, H + 1)

    patch = filt[x:x + patch_size, y:y + patch_size].reshape(-1)

    # scale to firing-rate regime (~20 Hz target mean)
    r_x = patch * scale

    # ON / OFF split (LGN populations)
    pos = r_x > 0
    on  = r_x * pos.astype(np.float32)
    off = -r_x * (~pos).astype(np.float32)

    return np.concatenate([on, off], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Dataset loader (ImageNet / TinyImageNet stream)
# ─────────────────────────────────────────────────────────────

class ImageNetStreamer:
    """
    Streams images from HuggingFace TinyImageNet.

    Produces LGN-processed ON/OFF patches.
    """

    def __init__(
            self,
            n_images: int = 10_000,
            img_size: int = 64,
            patch_size: int = 16,
            split: str = "train",
            cache_dir: str | None = None,
    ):

        self.n_images = n_images
        self.img_size = img_size
        self.patch_size = patch_size
        self.N_X = patch_size * patch_size

        print(f"Loading {n_images} images (TinyImageNet, split={split})...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets required")

        ds = load_dataset(
            "zh-plus/tiny-imagenet",
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )

        self._cache = np.zeros((n_images, 2 * self.N_X), dtype=np.float32)

        loaded = 0
        for sample in ds:
            if loaded >= n_images:
                break

            try:
                img = sample.get("image") or sample.get("jpg")
                if img is None:
                    continue

                vec = preprocess_image(
                    img,
                    size=img_size,
                    patch_size=patch_size,
                )

                self._cache[loaded] = vec
                loaded += 1

                if loaded % 1000 == 0:
                    print(f"{loaded}/{n_images} loaded")

            except Exception:
                continue

        self.n_images = loaded
        self._cache = self._cache[:loaded]

        print(f"Done. Dataset shape: {self._cache.shape}")

    # ─────────────────────────────────────────────────────────

    def sample_batch(
            self,
            N_b: int,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Returns LGN batch:
            (2*N_X, N_b)
        """

        idx = np.random.randint(0, self.n_images, size=N_b)
        batch = self._cache[idx].T  # (2*N_X, N_b)

        return torch.from_numpy(batch).to(device)
