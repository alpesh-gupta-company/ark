import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import math


class WaveKernel(nn.Module):
    """
    Selective Damped Wave Kernel with input-dependent gating.

    Training:  O(D * L log L) via FFT convolution + content-aware gate.
    Inference: O(D) per-token via SSM recurrence with input-dependent decay.

    Key innovation over static wave kernels:
      - Output gate: sigmoid(W_gate @ x) controls what information passes through
      - Input-dependent delta (inference): content modulates the decay rate,
        allowing the model to selectively remember/forget based on input
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # --- Base wave parameters (learned, static) ---
        self.alpha = nn.Parameter(torch.exp(torch.linspace(0.0, 1.0, d_model)))
        self.omega = nn.Parameter(torch.exp(torch.linspace(0, math.log(2 * math.pi), d_model)))
        self.phi = nn.Parameter(torch.zeros(d_model))

        # --- Input-dependent selection (Mamba-inspired) ---
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)  # content-dependent decay
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)   # output gate

    def forward(self, x):
        x_float = x.float()
        B, L, D = x_float.shape
        device = x.device

        # Input-dependent output gate: decides what to pass through
        gate = torch.sigmoid(self.gate_proj(x_float))

        # Static damped wave kernel for FFT convolution
        t = torch.arange(L, device=device, dtype=torch.float32)
        alpha_pos = self.alpha.abs()  # ensure positive damping
        kernel = torch.exp(-alpha_pos * t[:, None]) * \
                 torch.cos(self.omega * t[:, None] + self.phi)

        # FFT convolution: O(D * L log L)
        n_fft = 2 ** math.ceil(math.log2(2 * L))
        X_f = fft.rfft(x_float, n=n_fft, dim=1)
        K_f = fft.rfft(kernel, n=n_fft, dim=0)
        X_f.mul_(K_f.unsqueeze(0))
        y = fft.irfft(X_f, n=n_fft, dim=1)

        # Gated output
        return (gate * y[:, :L, :]).to(x.dtype)

    def _get_ssm_params(self, x_t):
        """
        Compute SSM parameters with input-dependent decay modulation.
        Unlike static SSMs, the decay rate adapts to the input content,
        allowing selective memory retention (like Mamba's delta).
        """
        # Input-dependent modulation of decay rate
        delta = F.softplus(self.delta_proj(x_t))  # (B, D), always positive

        alpha_pos = self.alpha.abs()
        # Modulated decay: base_decay * content_modulation
        effective_decay = torch.exp(-alpha_pos * delta)

        lam = torch.complex(
            effective_decay * torch.cos(self.omega),
            effective_decay * torch.sin(self.omega)
        )
        phase = torch.complex(torch.cos(self.phi), torch.sin(self.phi))
        return lam, phase

    def step(self, x_t, state=None):
        """
        O(D) single-step State Space recurrence with selective gating:
            delta_t = softplus(W_delta @ x_t)      -- input-dependent decay
            h_t = lambda(delta_t) * h_{t-1} + x_t  -- selective state update
            y_t = gate(x_t) * Re(phase * h_t)       -- gated output
        """
        B, D = x_t.shape

        # Content-dependent SSM parameters
        lam, phase = self._get_ssm_params(x_t)

        # Output gate
        gate = torch.sigmoid(self.gate_proj(x_t))

        if state is None:
            state = torch.zeros(B, D, dtype=torch.cfloat, device=x_t.device)

        next_state = lam * state + x_t.to(torch.cfloat)
        y_t = (phase * next_state).real.to(x_t.dtype)

        return gate * y_t, next_state

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.d_model, dtype=torch.cfloat, device=device)