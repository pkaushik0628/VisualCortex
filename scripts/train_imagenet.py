"""
train.py — Network2D trainer with ImageNet input
=================================================

Usage
-----
    python train.py
    python train.py --resume ckpt_iter01000.pt
"""
from __future__ import annotations
import time
from pathlib import Path

import torch

from Network3D import Network, N_E_ALEXNET, N_I_ALEXNET
from HParams import HParams
from data.data2 import ImageNetStreamer


# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE        = 16                    # resize ImageNet to 64×64 grayscale
N_IMAGENET        = 10_000               # images to cache from HuggingFace
CHECKPOINT_EVERY  = 1000                 # save .pt every N learning iters
RESUME_FROM       = None                 # set to "ckpt_iter01000.pt" to resume
HF_CACHE_DIR      = None                 # set to local path to cache HF data
                                         # e.g. "/data/gpfs/.../hf_cache"
# ─────────────────────────────────────────────────────────────────────────────


def train(
    hp: HParams,
    device: torch.device,
    resume_from: str | None = RESUME_FROM,
) -> Network:
    t0 = time.time()

    # ── Data ──────────────────────────────────────────────────────────────
    streamer = ImageNetStreamer(
        n_images  = N_IMAGENET,
        #img_size  = IMAGE_SIZE,
        cache_dir = HF_CACHE_DIR,
    )

    # ── Network ───────────────────────────────────────────────────────────
    net = Network(hp, device)

    start_iter = 1
    if resume_from:
        net.load(resume_from)
        stem = Path(resume_from).stem
        if "iter" in stem:
            try:
                start_iter = int(stem.split("iter")[-1]) + 1
                print(f"Resuming from iteration {start_iter}")
            except ValueError:
                pass

    print(f"\nNetwork: N_E={hp.N_E:,}  N_I={hp.N_I:,}  N_X={hp.N_X:,}")
    print(f"rho_E={hp.rho_E:.3f}  A_d={hp.A_d:.4f}  L1_EX={hp.L1_EX}")
    print(f"Ready in {time.time()-t0:.1f}s")

    # ── Phase 1: threshold warm-up ─────────────────────────────────────────
    if start_iter <= hp.N_q:
        print(f"\nPhase 1 — threshold warm-up ({hp.N_q} iters)")
        for i_T in range(start_iter, hp.N_q + 1):
            R_X = streamer.sample_batch(hp.N_b, device)
            net.run_batch(R_X, learn=False)
            net.apply_threshold_updates()
            if i_T % 10 == 0:
                th_e = net.E.theta.mean().item()
                e_r  = net.E.R.mean().item() / hp.N_t
                print(f"  [{i_T:>4}] theta_E={th_e:.3f}  E_rate={e_r:.5f}",
                      flush=True)
        print(f"Phase 1 complete ({time.time()-t0:.1f}s)")
        start_iter = hp.N_q + 1

    # ── Phase 2: main learning ─────────────────────────────────────────────
    t1 = time.time()
    print(f"\nPhase 2 — main learning ({hp.N_n - hp.N_q} iters)")

    for i_T in range(max(start_iter, hp.N_q + 1), hp.N_n + 1):
        R_X      = streamer.sample_batch(hp.N_b, device)
        dW, _, _ = net.run_batch(R_X, learn=True)
        net.apply_weight_updates(dW)
        net.apply_threshold_updates()

        if i_T in (400, 800):
            net.halve_learning_rates()
            print(f"\n  [iter {i_T}] learning rates halved")

        if i_T % 100 == 0:
            th_e    = net.E.theta.mean().item()
            e_r     = net.E.R.mean().item() / hp.N_t
            elapsed = time.time() - t1
            iters_done = i_T - hp.N_q
            eta_s  = (elapsed / iters_done) * (hp.N_n - i_T) if iters_done > 0 else 0
            print(f"\n  [{i_T:>5}] theta_E={th_e:.3f}  E_rate={e_r:.5f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_s/60:.1f}min", flush=True)

            if i_T % CHECKPOINT_EVERY == 0:
                net.save(f"ckpt_iter{i_T:05d}.pt")
        else:
            print(f"{i_T} ", end="" if i_T % 10 else "\n", flush=True)

    print(f"\nPhase 2 complete ({time.time()-t1:.1f}s)")
    net.save("v1_weights2D.pt")
    return net


def main() -> None:
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB")
    elif device.type == "mps":
        print("  Apple Silicon MPS backend")

    # HParams — N_X must match IMAGE_SIZE^2
    hp     = HParams()
    hp.N_X = IMAGE_SIZE * IMAGE_SIZE   # 64*64 = 4096

    train(hp, device)


if __name__ == "__main__":
    main()
