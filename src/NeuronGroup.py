import torch
from typing import Optional

class NeuronGroup:
    """
    State container for one population of leaky integrate-and-fire neurons.

    Holds membrane potentials, spiking thresholds, synaptic traces, and
    accumulated spike counts for an entire batch simultaneously.

    All tensors live on `device`.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in this population.
    batch_size : int
        Number of simultaneously simulated input images.
    theta0 : float
        Initial spiking threshold (same for all neurons).
    has_slow_trace : bool
        If True, maintain a second (slow) STDP trace (needed for excitatory
        populations using the triplet rule).
    device : torch.device
    """

    def __init__(
            self,
            n_neurons: int,
            batch_size: int,
            theta0: float, #apaptive spiking threshold
            has_slow_trace: bool, # true for X-E, and E-E recurrent connections.
            device: torch.device,
    ) -> None:
        self.n = n_neurons
        self.N_b = batch_size
        self.dev = device

        # Membrane potentials  (n, N_b)
        self.V = torch.zeros(n_neurons, batch_size, device=device)
        # Spiking thresholds   (n, 1) – broadcasts over batch
        self.theta = torch.full((n_neurons, 1), theta0, device=device)
        # Fast STDP trace      (n, N_b)
        self.U_i = torch.zeros(n_neurons, batch_size, device=device)
        # Slow STDP trace (triplet, excitatory only)
        self.U_l: Optional[torch.Tensor] = (
            torch.zeros(n_neurons, batch_size, device=device)
            if has_slow_trace else None
        )
        # Spike accumulator (reset each batch)     (n, N_b)
        self.R = torch.zeros(n_neurons, batch_size, device=device)
        # Current spike mask      (n, N_b)
        self.S = torch.zeros(n_neurons, batch_size, dtype=torch.bool, device=device) #This lists the neurons that fire at a given time

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def reset_batch(self) -> None:
        """Zero all per-batch accumulators at the start of each iteration."""
        self.V.zero_()
        self.U_i.zero_()
        self.R.zero_()
        if self.U_l is not None:
            self.U_l.zero_()

    def check_spikes(self) -> torch.Tensor:
        """
        Fire neurons whose membrane potential exceeds threshold, then reset.

        Returns
        -------
        S : BoolTensor  (n, N_b)
        """
        self.S = self.V > self.theta  #for all neurons whose membrane potential is greater than adaptive thresholds, bool true
        self.V[self.S] = 0.0 #for neurons that fired, set membrane potential to 0
        return self.S #return the firing matrix

    def decay_traces(self, exp_i: float, exp_l: Optional[float] = None) -> None:
        """Multiply traces by their per-step decay factors."""
        self.U_i.mul_(exp_i) #Compute fast trace
        if self.U_l is not None and exp_l is not None: #Compute slow trace if available
            self.U_l.mul_(exp_l)

    def update_traces(self, invtau_i: float, invtau_l: Optional[float] = None) -> None:
        """Increment traces at spike sites."""
        self.U_i[self.S] += invtau_i  #If a spike occured, update the trace at the spike site
        if self.U_l is not None and invtau_l is not None: #If slow trace is available, do the same
            self.U_l[self.S] += invtau_l

    def mean_rate(self) -> torch.Tensor:
        """Mean spike count across the batch, shape (n,)."""
        return self.R.mean(dim=1)

    def update_threshold(
            self,
            rho: float,
            eta_t: float,
            cap: float,
    ) -> None:
        """
        Homeostatic threshold adaptation.

            Δθ = min( (R̄ − ρ) · η_t,  ρ · η_t )

        Parameters
        ----------
        rho   : target spike count per image
        eta_t : threshold learning rate
        cap   : maximum allowed threshold increase per step (= ρ · η_t)
        """
        delta = (self.mean_rate() - rho) * eta_t  # (n,)
        delta = torch.clamp(delta, max=cap)
        self.theta.squeeze_(1).add_(delta).unsqueeze_(1)

    def collect_raster(self, timestep: int) -> list[tuple[int, int]]:
        """Return (timestep, neuron_idx) pairs for the first batch sample."""
        return [(timestep, i.item()) for i in self.S[:, 0].nonzero(as_tuple=False).squeeze(1)]