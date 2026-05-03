"""
Network2D.py  —  AlexNet-scale E/I spiking network
====================================================

Populations
-----------
X  : LGN input         (2·N_X,  on+off)
E  : excitatory output (N_E = 55·55·96 = 290,400)
I  : inhibitory output (N_I = 72,600  ≈ N_E / 4)

Weight storage strategy
-----------------------
W_EX  (N_E, 2·N_X)   DENSE   ~595 MB   — feedforward, same as working model
W_IX  (N_I, 2·N_X)   DENSE   ~149 MB   — feedforward, same as working model
W_EE  (N_E, N_E)     SPARSE CSR         — 338 GB dense → ~240 MB sparse
W_EI  (N_E, N_I)     SPARSE CSR         — feasible but still large dense
W_IE  (N_I, N_E)     SPARSE CSR
W_II  (N_I, N_I)     SPARSE CSR

Gaussian connectivity (all weights)
------------------------------------
All populations share a 55×55 spatial grid.
Neuron i → (row, col) = divmod(i % GRID_N, GRID_W).
Connection probability: p_sparse · exp(−d² / 2σ²),  σ = 0.10 · GRID_W = 5.5
Applied once at init; zero entries stay zero (non-negative constraint).

Learning
--------
Feedforward dW (EX, IX): dense matrices, exactly as working model.
Recurrent  dW (EE, EI, IE, II): accumulated ONLY at the sparse positions
  that exist in W.  Stored as 1-D value vectors (nnz,) — no dense alloc.

Normalisation
-------------
Feedforward: original two-step dense normalise (unchanged from working model).
Recurrent:   same two-step logic but applied to sparse value vectors.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
from NeuronGroup import NeuronGroup
from HParams import HParams


# ---------------------------------------------------------------------------
# Grid constants
# ---------------------------------------------------------------------------
GRID_W = 55
GRID_H = 55
GRID_N = GRID_W * GRID_H          # 3,025 spatial positions
SIGMA  = 0.10 * GRID_W            # σ = 5.5 grid units

# AlexNet scale defaults (overrideable via HParams)
N_E_ALEXNET = 55 * 55 * 96        # 290,400
N_I_ALEXNET = N_E_ALEXNET // 4    # 72,600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_positions(n: int, device: torch.device) -> torch.Tensor:
    """Return (n, 2) float32 (row, col) positions on the 55×55 grid."""
    idx  = torch.arange(n, device=device) % GRID_N
    rows = (idx // GRID_W).float()
    cols = (idx %  GRID_W).float()
    return torch.stack([rows, cols], dim=1)


def _gaussian_sparse_indices(
        n_post: int,
        n_pre: int,
        sigma: float,
        p_sparse: float,
        diag: bool,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample sparse connectivity indices using Gaussian distance kernel.

    Returns (row_idx, col_idx) LongTensors of the sampled connections.
    Connection probability for pair (i,j): p_sparse * exp(-d^2/2sigma^2).
    Chunk size is computed automatically so each slice uses at most ~32 MB,
    preventing OOM on MPS (Apple Silicon) and other constrained devices.
    """
    # 3 tensors of shape (chunk, n_pre) floats in the hot loop
    bytes_per_row = n_pre * 4 * 3
    chunk = max(1, (32 * 1024 * 1024) // bytes_per_row)

    pos_post = _grid_positions(n_post, device)
    pos_pre  = _grid_positions(n_pre,  device)
    two_s2   = 2.0 * sigma * sigma

    rows_out, cols_out = [], []

    for start in range(0, n_post, chunk):
        end  = min(start + chunk, n_post)
        diff = pos_post[start:end].unsqueeze(1) - pos_pre.unsqueeze(0)
        d2   = (diff ** 2).sum(2)                          # (chunk, n_pre)
        prob = p_sparse * torch.exp(-d2 / two_s2)          # (chunk, n_pre)
        mask = torch.rand_like(prob) < prob                 # stochastic sample

        if diag and n_post == n_pre:
            local = torch.arange(end - start, device=device)
            mask[local, start + local] = False              # no self-connections

        r, c = mask.nonzero(as_tuple=True)
        rows_out.append(r + start)
        cols_out.append(c)

    return torch.cat(rows_out), torch.cat(cols_out)


def _init_sparse_w(
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        shape: tuple[int, int],
        L1: float,
        device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialise sparse weight values at the given (row, col) positions.

    Returns (values, row_idx, col_idx) — values are Normal(1,0.5) clamped
    ≥0 and row-L1-normalised.
    """
    nnz  = row_idx.shape[0]
    vals = torch.empty(nnz, device=device).normal_(1.0, 0.5).clamp_(min=0.0)
    vals = _normalise_sparse_vals(vals, row_idx, shape[0], L1)
    return vals, row_idx, col_idx


def _normalise_sparse_vals(
        vals: torch.Tensor,
        row_idx: torch.Tensor,
        n_post: int,
        L1: float,
        diag_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Two-step soft L1 row-normalisation on sparse values.
    Mirrors the dense _normalise exactly.

    vals      : (nnz,) float
    row_idx   : (nnz,) long — which row each value belongs to
    n_post    : number of rows
    L1        : target row L1 norm
    diag_mask : (nnz,) bool — True for diagonal entries (zeroed first)
    """
    if diag_mask is not None:
        vals = vals.clone()
        vals[diag_mask] = 0.0

    vals = vals.clamp(min=0.0)
    divisor = float(vals.shape[0]) / max(n_post, 1)   # avg nnz per row

    # Step 1: additive
    row_sums = torch.zeros(n_post, device=vals.device)
    row_sums.scatter_add_(0, row_idx, vals)
    correction = (-row_sums + row_sums.clamp(max=L1)) / (divisor + 1e-12)
    vals = vals + correction[row_idx]
    vals = vals.clamp(min=0.0)

    # Step 2: multiplicative
    row_sums2 = torch.zeros(n_post, device=vals.device)
    row_sums2.scatter_add_(0, row_idx, vals)
    row_sums2.clamp_(min=1e-12)
    scale = row_sums2.clamp(max=L1) / row_sums2
    vals  = vals * scale[row_idx]

    return vals


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """
    AlexNet-scale E/I spiking network.

    W_EX, W_IX  — dense  (feedforward, identical to working small model)
    W_EE, W_EI, W_IE, W_II — sparse CSR  (recurrent, Gaussian connectivity)

    All learning rules and normalisation are identical to the working model.
    The only difference is that recurrent matmuls use torch.sparse.mm and
    recurrent dW is accumulated at sparse positions only.
    """

    def __init__(self, hp: HParams, device: torch.device) -> None:
        self.hp  = hp
        self.dev = device

        # ── Populations ───────────────────────────────────────────────────
        self.E = NeuronGroup(hp.N_E, hp.N_b, hp.theta_E0,
                             has_slow_trace=True,  device=device)
        self.I = NeuronGroup(hp.N_I, hp.N_b, hp.theta_I0,
                             has_slow_trace=False, device=device)
        self.U_Xi = torch.zeros(2 * hp.N_X, hp.N_b, device=device)

        # ── Dense feedforward weights (same init as working model) ────────
        print("Initialising dense feedforward weights (W_EX, W_IX)…")
        self.W_EX = self._init_dense_w((hp.N_E, 2*hp.N_X), hp.L1_EX)
        self.W_IX = self._init_dense_w((hp.N_I, 2*hp.N_X), hp.L1_IX)

        # ── Sparse recurrent weights ──────────────────────────────────────
        print("Building sparse recurrent connectivity (W_EE, W_EI, W_IE, W_II)…")
        print("  Sampling W_EE indices…")
        ee_r, ee_c = _gaussian_sparse_indices(
            hp.N_E, hp.N_E, SIGMA, hp.p_sparse, diag=True, device=device)
        print(f"  W_EE: {ee_r.shape[0]:,} connections "
              f"({ee_r.shape[0]/hp.N_E:.1f} avg/neuron)")

        print("  Sampling W_EI indices…")
        ei_r, ei_c = _gaussian_sparse_indices(
            hp.N_E, hp.N_I, SIGMA, hp.p_sparse, diag=False, device=device)

        print("  Sampling W_IE indices…")
        ie_r, ie_c = _gaussian_sparse_indices(
            hp.N_I, hp.N_E, SIGMA, hp.p_sparse, diag=False, device=device)

        print("  Sampling W_II indices…")
        ii_r, ii_c = _gaussian_sparse_indices(
            hp.N_I, hp.N_I, SIGMA, hp.p_sparse, diag=True, device=device)

        # Store indices permanently — sparsity pattern never changes
        self._ee_r, self._ee_c = ee_r, ee_c
        self._ei_r, self._ei_c = ei_r, ei_c
        self._ie_r, self._ie_c = ie_r, ie_c
        self._ii_r, self._ii_c = ii_r, ii_c

        # Diagonal masks for EE and II (for normalisation)
        self._ee_diag = (ee_r == ee_c)
        self._ii_diag = (ii_r == ii_c)

        # Initialise values and build CSR tensors
        print("  Initialising sparse weight values…")
        self.W_EE_v, _, _ = _init_sparse_w(ee_r, ee_c, (hp.N_E, hp.N_E), hp.L1_EE, device)
        self.W_EI_v, _, _ = _init_sparse_w(ei_r, ei_c, (hp.N_E, hp.N_I), hp.L1_EI, device)
        self.W_IE_v, _, _ = _init_sparse_w(ie_r, ie_c, (hp.N_I, hp.N_E), hp.L1_IE, device)
        self.W_II_v, _, _ = _init_sparse_w(ii_r, ii_c, (hp.N_I, hp.N_I), hp.L1_II, device)

        # Build CSR tensors (used for matmul)
        self._rebuild_sparse()
        print("Weights initialised.")

        # Save initial values for comparison
        self.W0 = {
            "W_EX": self.W_EX.clone(),
            "W_IX": self.W_IX.clone(),
            "W_EE": self.W_EE_v.clone(),
            "W_EI": self.W_EI_v.clone(),
            "W_IE": self.W_IE_v.clone(),
            "W_II": self.W_II_v.clone(),
        }

        # ── Learning rates ────────────────────────────────────────────────
        self.eta = dict(
            EX=hp.eta_EX, EE=hp.eta_EE, EI=hp.eta_EI,
            IX=hp.eta_IX, IE=hp.eta_IE, II=hp.eta_II,
            tE=hp.eta_tE, tI=hp.eta_tI,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_dense_w(self, shape: tuple[int, int], L1: float) -> torch.Tensor:
        """Dense weight init with Gaussian mask — identical to working model."""
        hp = self.hp
        W  = torch.empty(shape, device=self.dev).normal_(1.0, 0.5)
        rand_mask = torch.rand(shape, device=self.dev) > hp.p_sparse
        W[rand_mask] *= 0.01
        W.clamp_(min=0.0)

        # Gaussian spatial envelope — computed in adaptive chunks (~32 MB each)
        pos_post = _grid_positions(shape[0], self.dev)
        pos_pre  = _grid_positions(shape[1], self.dev)
        two_s2   = 2.0 * SIGMA * SIGMA
        chunk    = max(1, (32 * 1024 * 1024) // (shape[1] * 4 * 2))
        for start in range(0, shape[0], chunk):
            end  = min(start + chunk, shape[0])
            diff = pos_post[start:end].unsqueeze(1) - pos_pre.unsqueeze(0)
            d2   = (diff ** 2).sum(2)
            W[start:end].mul_(torch.exp(-d2 / two_s2))

        self._normalise_dense(W, L1, shape[1])
        return W

    @staticmethod
    def _normalise_dense(W: torch.Tensor, L1: float, divisor: int,
                         diag: bool = False) -> torch.Tensor:
        """Original two-step dense normalisation — unchanged from working model."""
        if diag:
            W.fill_diagonal_(0.0)
        W.clamp_(min=0.0)
        tmp = W.sum(dim=1, keepdim=True)
        W.add_((-tmp + tmp.clamp(max=L1)) / divisor)
        W.clamp_(min=0.0)
        tmp = W.sum(dim=1, keepdim=True)
        tmp.clamp_(min=1e-12)
        W.mul_(tmp.clamp(max=L1) / tmp)
        return W

    def _rebuild_sparse(self) -> None:
        """
        Rebuild CSR tensors from current value vectors and stored indices.
        Called after init and after each weight update.
        MPS note: torch.sparse_csr_tensor is CPU-only; we use COO and
        convert to CSR only on CUDA. On MPS/CPU we use COO directly since
        torch.sparse.mm supports COO on CPU.
        """
        hp = self.hp
        dev = self.dev

        def _coo(r, c, v, shape):
            idx = torch.stack([r, c])
            return torch.sparse_coo_tensor(idx, v, shape, device=dev).coalesce()

        self.W_EE = _coo(self._ee_r, self._ee_c, self.W_EE_v, (hp.N_E, hp.N_E))
        self.W_EI = _coo(self._ei_r, self._ei_c, self.W_EI_v, (hp.N_E, hp.N_I))
        self.W_IE = _coo(self._ie_r, self._ie_c, self.W_IE_v, (hp.N_I, hp.N_E))
        self.W_II = _coo(self._ii_r, self._ii_c, self.W_II_v, (hp.N_I, hp.N_I))

    # ──────────────────────────────────────────────────────────────────────
    # Forward step
    # ──────────────────────────────────────────────────────────────────────

    def _step(
            self,
            S_X: torch.Tensor,
            learn: bool,
            dW: Optional[dict],
    ) -> None:
        hp = self.hp

        S_E = self.E.check_spikes()
        S_I = self.I.check_spikes()
        self.E.R.add_(S_E.float())
        self.I.R.add_(S_I.float())

        noise_E = torch.randn_like(self.E.V)
        noise_I = torch.randn_like(self.I.V)

        # E membrane: dense feedforward + sparse recurrent
        self.E.V.mul_(hp.exp_m)
        self.E.V.addmm_(self.W_EX, S_X.float())                        # dense
        self.E.V.add_(torch.sparse.mm(self.W_EE,  S_E.float()))        # sparse
        self.E.V.add_(torch.sparse.mm(self.W_EI, -S_I.float()))        # sparse
        self.E.V.add_(noise_E).clamp_(min=hp.V_min)

        # I membrane: dense feedforward + sparse recurrent
        self.I.V.mul_(hp.exp_m)
        self.I.V.addmm_(self.W_IX, S_X.float())                        # dense
        self.I.V.add_(torch.sparse.mm(self.W_IE,  S_E.float()))        # sparse
        self.I.V.add_(torch.sparse.mm(self.W_II, -S_I.float()))        # sparse
        self.I.V.add_(noise_I).clamp_(min=hp.V_min)

        if not learn:
            self.E.decay_traces(hp.exp_i, hp.exp_l)
            self.I.decay_traces(hp.exp_i)
            self.U_Xi.mul_(hp.exp_i)
            self.E.update_traces(hp.invtau_i, hp.invtau_l)
            self.I.update_traces(hp.invtau_i)
            self.U_Xi[S_X] += hp.invtau_i
            return

        self.E.decay_traces(hp.exp_i, hp.exp_l)
        self.I.decay_traces(hp.exp_i)
        self.U_Xi.mul_(hp.exp_i)

        S_Xf = S_X.float()
        S_Ef = S_E.float()
        S_If = S_I.float()
        UEi  = self.E.U_i
        UEl  = self.E.U_l
        UIi  = self.I.U_i
        UXi  = self.U_Xi

        # ── Dense feedforward dW (identical to working model) ─────────────
        # X→E triplet STDP
        dW["EX"].addmm_(S_Ef * UEl, UXi.T, alpha=hp.A_p)
        dW["EX"].addmm_(UEi,        S_Xf.T, alpha=-hp.A_d)
        # X→I symmetric STDP
        dW["IX"].addmm_(S_If, UXi.T, alpha=hp.A_i)
        dW["IX"].addmm_(UIi,  S_Xf.T, alpha=hp.A_i)
        dW["IX"].addmm_(S_If, S_Xf.T, alpha=hp.A_i)

        # ── Sparse recurrent dW: accumulate only at existing positions ────
        # For each sparse position (i,j), contribution = sum_b A[i,b]*B[j,b]

        def _accum(A: torch.Tensor, B: torch.Tensor,
                   r: torch.Tensor, c: torch.Tensor,
                   dv: torch.Tensor, alpha: float) -> None:
            """dv[k] += alpha * sum_b A[r[k],b] * B[c[k],b]  for all k."""
            dv.add_((A[r] * B[c]).sum(dim=1), alpha=alpha)

        # E→E triplet STDP
        _accum(S_Ef * UEl, UEi, self._ee_r, self._ee_c, dW["EE"], hp.A_p)
        _accum(UEi,        S_Ef, self._ee_r, self._ee_c, dW["EE"], -hp.A_d)

        # I→E symmetric STDP
        _accum(S_Ef, UIi, self._ei_r, self._ei_c, dW["EI"], hp.A_i)
        _accum(UEi,  S_If, self._ei_r, self._ei_c, dW["EI"], hp.A_i)
        _accum(S_Ef, S_If, self._ei_r, self._ei_c, dW["EI"], hp.A_i)

        # E→I symmetric STDP
        _accum(S_If, UEi, self._ie_r, self._ie_c, dW["IE"], hp.A_i)
        _accum(UIi,  S_Ef, self._ie_r, self._ie_c, dW["IE"], hp.A_i)
        _accum(S_If, S_Ef, self._ie_r, self._ie_c, dW["IE"], hp.A_i)

        # I→I symmetric STDP
        _accum(UIi,  S_If, self._ii_r, self._ii_c, dW["II"], hp.A_i)
        _accum(S_If, UIi,  self._ii_r, self._ii_c, dW["II"], hp.A_i)
        _accum(S_If, S_If, self._ii_r, self._ii_c, dW["II"], hp.A_i)

        self.E.update_traces(hp.invtau_i, hp.invtau_l)
        self.I.update_traces(hp.invtau_i)
        self.U_Xi[S_X] += hp.invtau_i

    # ──────────────────────────────────────────────────────────────────────
    # Batch
    # ──────────────────────────────────────────────────────────────────────

    def run_batch(
            self,
            R_X: torch.Tensor,
            learn: bool,
            collect_raster: bool = False,
    ) -> tuple[Optional[dict], list, list]:
        hp  = self.hp
        self.E.reset_batch()
        self.I.reset_batch()
        self.U_Xi.zero_()

        dW: Optional[dict] = None
        if learn:
            dW = {
                # Dense feedforward
                "EX": torch.zeros_like(self.W_EX),
                "IX": torch.zeros_like(self.W_IX),
                # Sparse recurrent — value vectors only (nnz,)
                "EE": torch.zeros(self._ee_r.shape[0], device=self.dev),
                "EI": torch.zeros(self._ei_r.shape[0], device=self.dev),
                "IE": torch.zeros(self._ie_r.shape[0], device=self.dev),
                "II": torch.zeros(self._ii_r.shape[0], device=self.dev),
            }

        spikes_e: list = []
        spikes_i: list = []
        R_XDT = R_X * hp.dt

        for i_t in range(1, hp.N_t + 1):
            S_X = torch.rand_like(R_XDT) < R_XDT
            self._step(S_X, learn=learn, dW=dW)
            if collect_raster:
                spikes_e.extend(self.E.collect_raster(i_t))
                spikes_i.extend(self.I.collect_raster(i_t))

        return dW, spikes_e, spikes_i

    # ──────────────────────────────────────────────────────────────────────
    # Weight update
    # ──────────────────────────────────────────────────────────────────────

    def apply_weight_updates(self, dW: dict) -> None:
        hp  = self.hp
        N_b = hp.N_b

        # Dense feedforward — identical to working model
        self.W_EX.add_(dW["EX"], alpha=self.eta["EX"] / N_b)
        self._normalise_dense(self.W_EX, hp.L1_EX, 2 * hp.N_X)

        self.W_IX.add_(dW["IX"], alpha=self.eta["IX"] / N_b)
        self._normalise_dense(self.W_IX, hp.L1_IX, 2 * hp.N_X)

        # Sparse recurrent — update value vectors, then rebuild COO
        def _upd_sparse(v, dv, eta, L1, r, n_post, diag_mask=None):
            v.add_(dv, alpha=eta / N_b)
            return _normalise_sparse_vals(v, r, n_post, L1, diag_mask)

        self.W_EE_v = _upd_sparse(self.W_EE_v, dW["EE"], self.eta["EE"],
                                   hp.L1_EE, self._ee_r, hp.N_E, self._ee_diag)
        self.W_EI_v = _upd_sparse(self.W_EI_v, dW["EI"], self.eta["EI"],
                                   hp.L1_EI, self._ei_r, hp.N_E)
        self.W_IE_v = _upd_sparse(self.W_IE_v, dW["IE"], self.eta["IE"],
                                   hp.L1_IE, self._ie_r, hp.N_I)
        self.W_II_v = _upd_sparse(self.W_II_v, dW["II"], self.eta["II"],
                                   hp.L1_II, self._ii_r, hp.N_I, self._ii_diag)

        # Rebuild COO tensors for next forward pass
        self._rebuild_sparse()

    def apply_threshold_updates(self) -> None:
        hp = self.hp
        self.E.update_threshold(hp.rho_E, self.eta["tE"], hp.rho_E * self.eta["tE"])
        self.I.update_threshold(hp.rho_I, self.eta["tI"], hp.rho_I * self.eta["tI"])

    def halve_learning_rates(self) -> None:
        for k in self.eta:
            self.eta[k] *= 0.5

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save({
            "W_EX":   self.W_EX,
            "W_IX":   self.W_IX,
            "W_EE_v": self.W_EE_v, "ee_r": self._ee_r, "ee_c": self._ee_c,
            "W_EI_v": self.W_EI_v, "ei_r": self._ei_r, "ei_c": self._ei_c,
            "W_IE_v": self.W_IE_v, "ie_r": self._ie_r, "ie_c": self._ie_c,
            "W_II_v": self.W_II_v, "ii_r": self._ii_r, "ii_c": self._ii_c,
            "theta_E": self.E.theta,
            "theta_I": self.I.theta,
            "N_E": self.hp.N_E, "N_I": self.hp.N_I, "N_X": self.hp.N_X,
        }, path)
        print(f"Weights saved to {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.dev, weights_only=False)
        self.W_EX.copy_(ckpt["W_EX"])
        self.W_IX.copy_(ckpt["W_IX"])
        self.W_EE_v = ckpt["W_EE_v"].to(self.dev)
        self.W_EI_v = ckpt["W_EI_v"].to(self.dev)
        self.W_IE_v = ckpt["W_IE_v"].to(self.dev)
        self.W_II_v = ckpt["W_II_v"].to(self.dev)
        self._ee_r, self._ee_c = ckpt["ee_r"].to(self.dev), ckpt["ee_c"].to(self.dev)
        self._ei_r, self._ei_c = ckpt["ei_r"].to(self.dev), ckpt["ei_c"].to(self.dev)
        self._ie_r, self._ie_c = ckpt["ie_r"].to(self.dev), ckpt["ie_c"].to(self.dev)
        self._ii_r, self._ii_c = ckpt["ii_r"].to(self.dev), ckpt["ii_c"].to(self.dev)
        self._ee_diag = (self._ee_r == self._ee_c)
        self._ii_diag = (self._ii_r == self._ii_c)
        self.E.theta.copy_(ckpt["theta_E"])
        self.I.theta.copy_(ckpt["theta_I"])
        self._rebuild_sparse()
        print(f"Weights loaded from {path}")