"""
train.py — AlexNet-scale V1 spiking model trainer
==================================================

Usage
-----
    # Small network (sanity check, matches working model)
    python train.py --N_E 400 --N_X 256 --N_n 4000

    # AlexNet scale
    python train.py --N_E 290400 --N_X 256 --N_n 4000

    # Resume from checkpoint
    python train.py --checkpoint v1_retinotopic.pt
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
from scipy.io import loadmat

from src.Network3D import Network, N_E_ALEXNET, N_I_ALEXNET
from src.HParams import HParams


def load_images(path: str | Path, device: torch.device) -> torch.Tensor:
    mat = loadmat(str(path))
    return torch.from_numpy(mat["IMAGES_DoG"].astype("float32")).to(device)


def _best_rect(n: int) -> tuple[int, int]:
    h = int(n ** 0.5)
    while h > 1 and n % h != 0:
        h -= 1
    return h, n // h


def sample_patches(
    IMAGES: torch.Tensor,
    N_X: int,
    N_b: int,
    buff: int = 20,
    scale: float = 70.0,
) -> torch.Tensor:
    ph, pw   = _best_rect(N_X)
    H, W, nz = IMAGES.shape
    device   = IMAGES.device
    row_max  = H - ph - 2 * buff
    col_max  = W - pw - 2 * buff
    R_X      = torch.zeros(2 * N_X, N_b, device=device)

    for i in range(N_b):
        while True:
            r = buff + int(torch.randint(row_max, (1,)).item())
            c = buff + int(torch.randint(col_max, (1,)).item())
            z = int(torch.randint(nz, (1,)).item())
            patch = IMAGES[r:r+ph, c:c+pw, z].clone()
            if torch.isnan(patch).any():
                continue
            if ph == pw:
                k     = int(torch.randint(4, (1,)).item())
                patch = torch.rot90(patch, k, dims=[0, 1])
            r_x = patch.reshape(-1) * scale
            pos = r_x > 0
            R_X[:N_X, i] =  r_x *  pos.float()
            R_X[N_X:, i] = -r_x * (~pos).float()
            break

    R_X.clamp_(max=100.0)
    return R_X


def train(
    hp: HParams,
    device: torch.device,
    image_path: str = "VanHateren_DoG_small.mat",
    checkpoint_every: int = 100,
    resume_from: str | None = None,
) -> Network:
    t0     = time.time()
    ph, pw = _best_rect(hp.N_X)

    net    = Network(hp, device)
    IMAGES = load_images(image_path, device)

    start_iter = 1
    if resume_from:
        net.load(resume_from)
        # Infer iteration from filename if possible (e.g. ckpt_iter0500.pt)
        stem = Path(resume_from).stem
        if "iter" in stem:
            try:
                start_iter = int(stem.split("iter")[-1]) + 1
                print(f"Resuming from iteration {start_iter}")
            except ValueError:
                pass

    print(f"\nNetwork: N_E={hp.N_E:,}  N_I={hp.N_I:,}  N_X={hp.N_X} ({ph}x{pw})")
    print(f"rho_E={hp.rho_E:.3f}  A_d={hp.A_d:.4f}  L1_EX={hp.L1_EX}")
    print(f"Built in {time.time()-t0:.1f}s")

    # ── Phase 1: threshold warm-up ─────────────────────────────────────────
    warmup_end = hp.N_q
    if start_iter <= warmup_end:
        print(f"\nPhase 1 — threshold warm-up ({warmup_end} iters)")
        for i_T in range(start_iter, warmup_end + 1):
            R_X = sample_patches(IMAGES, hp.N_X, hp.N_b)
            net.run_batch(R_X, learn=False)
            net.apply_threshold_updates()
            if i_T % 10 == 0:
                th_e = net.E.theta.mean().item()
                e_r  = net.E.R.mean().item() / hp.N_t
                print(f"  [{i_T:>4}] theta_E={th_e:.3f}  E_rate={e_r:.4f}", flush=True)
        print(f"Phase 1 complete ({time.time()-t0:.1f}s)")
        start_iter = warmup_end + 1

    # ── Phase 2: main learning ─────────────────────────────────────────────
    t1 = time.time()
    print(f"\nPhase 2 — main learning ({hp.N_n - hp.N_q} iters)")

    for i_T in range(max(start_iter, hp.N_q + 1), hp.N_n + 1):
        R_X      = sample_patches(IMAGES, hp.N_X, hp.N_b)
        dW, _, _ = net.run_batch(R_X, learn=True)
        net.apply_weight_updates(dW)
        net.apply_threshold_updates()

        if i_T in (400, 800):
            net.halve_learning_rates()
            print(f"\n  [iter {i_T}] learning rates halved")

        # Progress every 10 iters, detailed every 100
        if i_T % 100 == 0:
            th_e    = net.E.theta.mean().item()
            e_r     = net.E.R.mean().item() / hp.N_t
            elapsed = time.time() - t1
            iters_done = i_T - hp.N_q
            iters_left = hp.N_n - i_T
            eta_s  = elapsed / iters_done * iters_left if iters_done > 0 else 0
            print(f"\n  [{i_T:>5}] theta_E={th_e:.3f}  E_rate={e_r:.5f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_s/60:.1f}min", flush=True)

            # Periodic checkpoint
            if i_T % checkpoint_every == 0:
                ckpt_path = f"ckpt_iter{i_T:05d}.pt"
                net.save(ckpt_path)
        else:
            print(f"{i_T} ", end="" if i_T % 10 else "\n", flush=True)

    print(f"\nPhase 2 complete ({time.time()-t1:.1f}s)")
    net.save("v1_retinotopic.pt")
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 spiking trainer (AlexNet scale)")
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--images",     default="/data/gpfs/projects/punim2907/paddy/code3/data/VanHateren_DoG_small.mat")
    parser.add_argument("--N_n",        type=int, default=4000)
    parser.add_argument("--N_q",        type=int, default=100)
    parser.add_argument("--N_b",        type=int, default=100)
    parser.add_argument("--N_E",        type=int, default=290400)
    parser.add_argument("--N_X",        type=int, default=None,
                        help="Override N_X patch size")
    parser.add_argument("--checkpoint_every", type=int, default=1000,
                        help="Save checkpoint every N learning iters")
    parser.add_argument("--resume",     default=None,
                        help=".pt checkpoint to resume from")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon MPS backend")

    # HParams
    hp = HParams(N_n=args.N_n, N_q=args.N_q, N_b=args.N_b)
    if args.N_E is not None:
        hp.N_E = args.N_E
        hp.N_I = int(hp.N_E * hp.N_I_frac)
    if args.N_X is not None:
        hp.N_X = args.N_X

    train(hp, device,
          image_path=args.images,
          checkpoint_every=args.checkpoint_every,
          resume_from=args.resume)


if __name__ == "__main__":
    main()
