"""
data.py
=======
ImageNet data loader for the Network2D training pipeline.

Streams 10,000 images from HuggingFace imagenet-1k, converts to
grayscale, resizes to 64×64, applies a DoG filter, then returns
ON/OFF split firing rate tensors matching the Van Hateren format.

DoG filter
----------
DoG(x) = G(x, sigma1) - G(x, sigma2)
where G is a 2D Gaussian and sigma1=1.0, sigma2=2.0.
This approximates retinal ganglion cell centre-surround responses
and matches the preprocessing applied to the Van Hateren dataset.

Output format
-------------
R_X : FloatTensor (2*N_X, N_b)
  First  N_X rows: ON  channel (DoG > 0, scaled by 70)
  Second N_X rows: OFF channel (|DoG < 0|, scaled by 70)
  Clamped to [0, 100].
"""

from __future__ import annotations
from typing import Iterator
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image


# ── DoG filter ───────────────────────────────────────────────────────────────

def dog_filter(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0) -> np.ndarray:
    """
    Apply Difference of Gaussians to a 2D float32 image.

    Parameters
    ----------
    img    : (H, W) float32 array, values in any range
    sigma1 : centre Gaussian sigma  (narrow, excitatory)
    sigma2 : surround Gaussian sigma (wide, inhibitory)

    Returns
    -------
    (H, W) float32 DoG response
    """
    g1 = gaussian_filter(img, sigma=sigma1)
    g2 = gaussian_filter(img, sigma=sigma2)
    return (g1 - g2).astype(np.float32)


def preprocess_image(
        pil_img: Image.Image,
        size: int = 64,
        scale: float = 70.0,
        dog_sigma1: float = 1.0,
        dog_sigma2: float = 2.0,
) -> np.ndarray:
    """
    Convert a PIL image to a (2*N_X,) ON/OFF firing rate vector.

    Steps
    -----
    1. Convert to grayscale
    2. Resize to (size, size)
    3. Normalise to zero mean, unit std
    4. Apply DoG filter
    5. Scale by 70
    6. Split into ON (positive) and OFF (|negative|) channels
    7. Clamp to [0, 100]

    Returns
    -------
    (2 * size * size,) float32 array
    """
    # Grayscale + resize
    img = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normalise: zero mean, unit std (matches Van Hateren DoG preprocessing)
    std = arr.std()
    if std > 1e-8:
        arr = (arr - arr.mean()) / std

    # DoG
    dog = dog_filter(arr, sigma1=dog_sigma1, sigma2=dog_sigma2)

    # Scale
    r_x = dog.reshape(-1) * scale   # (N_X,)

    # ON / OFF split
    pos = r_x > 0
    on  =  r_x *  pos.astype(np.float32)
    off = -r_x * (~pos).astype(np.float32)

    out = np.concatenate([on, off], axis=0).clip(0, 100)   # (2*N_X,)
    return out


# ── HuggingFace ImageNet streamer ────────────────────────────────────────────

class ImageNetStreamer:
    """
    Streams images from HuggingFace zh-plus/tiny-imagenet (no auth required).

    Parameters
    ----------
    n_images  : total images to cache (default 10,000; max ~100,000)
    img_size  : resize target (default 64 → N_X = 4096)
    split     : 'train' (100k images) or 'valid' (10k images)
    cache_dir : local directory for HF cache (optional)
    """

    def __init__(
            self,
            n_images:  int = 10_000,
            img_size:  int = 64,
            split:     str = "train",
            cache_dir: str | None = None,
    ) -> None:
        self.img_size  = img_size
        self.n_images  = n_images
        self.N_X       = img_size * img_size

        print(f"Loading {n_images} ImageNet images from HuggingFace "
              f"(split='{split}', size={img_size}x{img_size})…")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Run: pip install datasets"
            )

        # zh-plus/tiny-imagenet: 100k images, 200 classes, no auth required
        ds = load_dataset(
            "zh-plus/tiny-imagenet",
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )

        # Pre-process and cache all images as a (n_images, 2*N_X) array
        self._cache = np.zeros((n_images, 2 * self.N_X), dtype=np.float32)
        loaded = 0
        for sample in ds:
            if loaded >= n_images:
                break
            try:
                pil_img = sample.get("image") or sample.get("jpg")
                if pil_img is None:
                    continue
                vec = preprocess_image(pil_img, size=img_size)
                self._cache[loaded] = vec
                loaded += 1
                if loaded % 1000 == 0:
                    print(f"  {loaded}/{n_images} images loaded…", flush=True)
            except Exception:
                continue

        if loaded < n_images:
            print(f"  Warning: only loaded {loaded}/{n_images} images")
            self._cache = self._cache[:loaded]
            self.n_images = loaded

        print(f"  Done. Cache shape: {self._cache.shape}")

    def sample_batch(
            self,
            N_b: int,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Sample N_b random images from the cache.

        Returns
        -------
        R_X : FloatTensor (2*N_X, N_b) on `device`
        """
        idx  = np.random.randint(0, self.n_images, size=N_b)
        batch = self._cache[idx].T   # (2*N_X, N_b)
        return torch.from_numpy(batch).to(device)
