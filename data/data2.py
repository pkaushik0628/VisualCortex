"""
data.py
=======
LGN preprocessing pipeline matching Van Hateren + Zylberberg-style model.

Pipeline:
1. Load natural image (0–1)
2. Global normalization (single dataset-consistent scale only)
3. Divisively normalised DoG filtering (FULL image)
4. Random 16×16 patch extraction (AFTER filtering)
5. ON/OFF rectification
6. Scaling to firing-rate regime (~20 Hz mean)

No spike generation included.
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image


# ─────────────────────────────────────────────────────────────
# LGN FILTER (Divisive DoG)
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
    """

    F_c = gaussian_filter(img, sigma=sigma_c)
    F_s = gaussian_filter(img, sigma=sigma_s)
    F_d = gaussian_filter(img, sigma=sigma_d)

    return (F_c - F_s) / (F_d + eps)


# ─────────────────────────────────────────────────────────────
# IMAGE → LGN PATCH
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
    Full paper-consistent LGN preprocessing.

    Returns
    -------
    (2 * patch_size^2,) ON/OFF vector
    """

    # ─────────────────────────────────────────────
    # 1. Load + grayscale + resize
    # ─────────────────────────────────────────────
    img = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # ─────────────────────────────────────────────
    # 2. Global normalization (ONLY ONCE)
    #    (paper: std = 1 across dataset scale)
    # ─────────────────────────────────────────────
    arr = arr / (arr.std() + 1e-8)

    # ─────────────────────────────────────────────
    # 3. LGN filtering (FULL IMAGE)
    # ─────────────────────────────────────────────
    filt = dog_filter(arr, sigma_c, sigma_s, sigma_d).astype(np.float32)

    # ─────────────────────────────────────────────
    # 4. Random patch extraction (AFTER filtering)
    # ─────────────────────────────────────────────
    H = size - patch_size
    x = np.random.randint(0, H + 1)
    y = np.random.randint(0, H + 1)

    patch = filt[x:x + patch_size, y:y + patch_size].reshape(-1)

    # ─────────────────────────────────────────────
    # 5. Scaling to firing-rate regime
    # ─────────────────────────────────────────────
    r_x = patch * scale

    # ─────────────────────────────────────────────
    # 6. ON / OFF rectification
    # ─────────────────────────────────────────────
    pos = r_x > 0
    on  = r_x * pos.astype(np.float32)
    off = -r_x * (~pos).astype(np.float32)

    return np.concatenate([on, off], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# DATASET STREAMER
# ─────────────────────────────────────────────────────────────

class ImageNetStreamer:
    """
    Streams TinyImageNet images and produces LGN-coded inputs.
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

        print(f"Loading {n_images} images (LGN pipeline)...")

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

        print(f"Done. Shape: {self._cache.shape}")

    # ─────────────────────────────────────────────

    def sample_batch(
            self,
            N_b: int,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            R_X: (2*N_X, N_b)
        """

        idx = np.random.randint(0, self.n_images, size=N_b)
        batch = self._cache[idx].T

        return torch.from_numpy(batch).to(device)
