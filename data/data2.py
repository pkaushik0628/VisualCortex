"""
data.py
=======
ImageNet data loader for the Network2D training pipeline.

Matches the Van Hateren pipeline exactly:
  - Stream images from HuggingFace zh-plus/tiny-imagenet
  - Convert to grayscale
  - Apply DoG filter (sigma1=1.0, sigma2=2.0)
  - Randomly sample 16x16 patches (with boundary buffer)
  - Random 90 degree rotation
  - ON/OFF split, scale by 70, clamp to [0, 100]

N_X = 16 * 16 = 256  (same as Van Hateren)
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image


# ── Divisively normalised DoG filter ─────────────────────────────────────────

def dog_filter(
        img: np.ndarray,
        sigma_c: float = 1.0,
        sigma_s: float = 1.5,
        sigma_d: float = 1.5,
        eps:     float = 1e-6,
) -> np.ndarray:
    """
    Divisively normalised Difference of Gaussians, matching equation (1)
    from Zylberberg et al.:

        F(x,y) = (F_c(x,y) - F_s(x,y)) / F_d(x,y)

    where F_c, F_s, F_d are isotropic Gaussian filters with standard
    deviations sigma_c=1.0, sigma_s=1.5, sigma_d=1.5 respectively.

    Parameters
    ----------
    img     : (H, W) float32, zero-mean unit-std normalised image
    sigma_c : centre Gaussian sigma
    sigma_s : surround Gaussian sigma
    sigma_d : divisive normalisation Gaussian sigma
    eps     : small constant to prevent division by zero

    Returns
    -------
    (H, W) float32 filtered image, then re-normalised to unit std
    """
    F_c = gaussian_filter(img, sigma=sigma_c)
    F_s = gaussian_filter(img, sigma=sigma_s)
    F_d = gaussian_filter(img, sigma=sigma_d)

    # Divisive normalisation — F_d is used as the denominator
    # Add eps to avoid division by zero where local contrast is flat
    filtered = (F_c - F_s) / (np.abs(F_d) + eps)

    # Normalise to unit std (as stated in the paper)
    std = filtered.std()
    if std > eps:
        filtered /= std

    return filtered.astype(np.float32)


# ── HuggingFace ImageNet streamer ────────────────────────────────────────────

class ImageNetStreamer:
    """
    Streams images from HuggingFace zh-plus/tiny-imagenet (no auth required),
    pre-processes them to grayscale + DoG, and caches them as large arrays
    from which 16x16 patches are sampled at training time — exactly matching
    the Van Hateren sample_patches pipeline.

    Parameters
    ----------
    n_images  : number of images to cache (default 10,000; max ~100,000)
    split     : 'train' or 'valid'
    cache_dir : local HF cache directory (optional)
    """

    PATCH_SIZE = 64      # patch edge length in pixels
    BUFF       = 20      # boundary buffer (same as Van Hateren)
    SCALE      = 70.0    # firing rate scale factor
    MIN_SIZE   = PATCH_SIZE + 2 * BUFF + 1   # minimum image dimension

    def __init__(
            self,
            n_images:  int = 10_000,
            split:     str = "train",
            cache_dir: str | None = None,
    ) -> None:
        self.n_images = n_images
        self.N_X      = self.PATCH_SIZE ** 2   # 256

        print(f"Loading {n_images} ImageNet images from HuggingFace "
              f"(split='{split}')…")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Run: pip install datasets"
            )

        ds = load_dataset(
            "zh-plus/tiny-imagenet",
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )

        # Cache preprocessed images as list of (H, W) float32 DoG arrays.
        # We keep variable-size images so patch sampling has spatial variety.
        self._images: list[np.ndarray] = []
        loaded = 0

        for sample in ds:
            if loaded >= n_images:
                break
            try:
                pil_img = sample.get("image") or sample.get("jpg")
                if pil_img is None:
                    continue

                # Grayscale
                gray = np.array(
                    pil_img.convert("L"), dtype=np.float32
                )

                # Skip images too small to sample a patch with buffer
                if gray.shape[0] < self.MIN_SIZE or gray.shape[1] < self.MIN_SIZE:
                    continue

                # Normalise to zero mean, unit std first (paper: "images were
                # normalised to have a standard deviation of 1" before filtering)
                std = gray.std()
                if std < 1e-8:
                    continue
                gray = (gray - gray.mean()) / std

                # Divisively normalised DoG (equation 1 from Zylberberg et al.)
                dog = dog_filter(gray)

                self._images.append(dog)
                loaded += 1

                if loaded % 1000 == 0:
                    print(f"  {loaded}/{n_images} images loaded…", flush=True)

            except Exception:
                continue

        self.n_images = len(self._images)
        print(f"  Done. {self.n_images} images cached.")

    def sample_batch(
            self,
            N_b: int,
            device: torch.device,
    ) -> torch.Tensor:
        """
        Sample N_b random 16x16 patches — identical logic to Van Hateren
        sample_patches: buffer exclusion, random 90 degree rotation,
        ON/OFF split, scale by 70, clamp to [0, 100].

        Returns
        -------
        R_X : FloatTensor (2*N_X, N_b) = (512, N_b) on `device`
        """
        ps  = self.PATCH_SIZE
        N_X = self.N_X
        R_X = np.zeros((2 * N_X, N_b), dtype=np.float32)

        for i in range(N_b):
            while True:
                # Pick a random image
                img = self._images[np.random.randint(self.n_images)]
                H, W = img.shape

                row_max = H - ps - 2 * self.BUFF
                col_max = W - ps - 2 * self.BUFF
                if row_max < 1 or col_max < 1:
                    continue   # image too small, try another

                # Random patch location (matching MATLAB randi convention)
                r = self.BUFF + np.random.randint(row_max)
                c = self.BUFF + np.random.randint(col_max)

                patch = img[r:r+ps, c:c+ps].copy()

                if np.isnan(patch).any():
                    continue

                # Random 90 degree rotation
                k = np.random.randint(4)
                patch = np.rot90(patch, k)

                # Scale
                r_x = patch.reshape(-1) * self.SCALE   # (N_X,)

                # ON / OFF split
                pos = r_x > 0
                R_X[:N_X, i] =  r_x *  pos.astype(np.float32)
                R_X[N_X:, i] = -r_x * (~pos).astype(np.float32)
                break

        R_X = R_X.clip(0, 100)
        return torch.from_numpy(R_X).to(device)
