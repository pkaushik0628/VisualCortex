from dataclasses import dataclass, field
import torch



@dataclass
class HParams:
    """All simulation hyperparameters in one place."""

    # Architecture
    N_X: int = 64 ** 2  # input (LGN) neurons
    N_E: int = 10000  # excitatory output neurons (AlexNet layer-1 scale)
    N_I_frac: float = 0.25  # fraction of N_E → inhibitory count

    # Training schedule
    N_n: int = 1200  # total batches (including threshold warm-up)
    N_q: int = 100  # threshold warm-up batches (weights frozen)
    N_b: int = 100  # batch size (images per batch)
    N_t: int = 400  # timesteps per image (ms)

    # Dynamics
    dt: float = 1e-3  # (s) timestep
    tau_m: float = 10e-3  # (s) membrane time constant
    tau_i: float = 20e-3  # (s) fast STDP trace time constant
    tau_l: float = 50e-3  # (s) slow STDP trace time constant

    # L1 weight norms
    L1_EX: float = 100.0
    L1_IX: float = 80.0
    L1_EE: float = 10.0
    L1_IE: float = 240.0
    L1_EI: float = 120.0
    L1_II: float = 120.0

    # Target firing rates (Hz)
    rho_E_hz: float = 2.0
    rho_I_hz: float = 4.0

    # Membrane floor
    V_min: float = -10.0

    # Learning rates
    eta_EX: float = 2e-4
    eta_EE: float = 1e-4
    eta_EI: float = 9e-2
    eta_tE: float = 1e0
    eta_IX: float = 3e-3
    eta_IE: float = 4e-2
    eta_II: float = 6e-2
    eta_tI: float = 1e0

    # Initial thresholds
    theta_E0: float = 10.0
    theta_I0: float = 10.0

    # STDP coefficients
    A_i: float = 0.5  # symmetric STDP coefficient
    A_p: float = 1.0  # triplet potentiation coefficient

    # Sparse init connection probability
    p_sparse: float = 0.2

    # Visualisation
    fig_show: int = 10
    raster_show: int = 10

    # Derived (filled in __post_init__)
    N_I: int = field(init=False)
    T: float = field(init=False)
    rho_E: float = field(init=False)  # spikes/image
    rho_I: float = field(init=False)
    exp_m: float = field(init=False)
    exp_i: float = field(init=False)
    exp_l: float = field(init=False)
    A_d: float = field(init=False)
    invtau_i: float = field(init=False)
    invtau_l: float = field(init=False)

    def __post_init__(self) -> None:
        self.N_I = int(self.N_E * self.N_I_frac)
        self.T = self.N_t * self.dt
        self.rho_E = self.rho_E_hz * self.T
        self.rho_I = self.rho_I_hz * self.T
        self.exp_m = torch.tensor(self.dt / self.tau_m).neg().exp().item()
        self.exp_i = torch.tensor(self.dt / self.tau_i).neg().exp().item()
        self.exp_l = torch.tensor(self.dt / self.tau_l).neg().exp().item()
        self.A_d = self.rho_E  # depression tied to target rate
        self.invtau_i = 1.0 / self.tau_i
        self.invtau_l = 1.0 / self.tau_l
