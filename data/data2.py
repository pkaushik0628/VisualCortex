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
            cache_dir: Optional[str] = None,
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
