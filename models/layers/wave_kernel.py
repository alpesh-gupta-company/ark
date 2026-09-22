import torch
import torch.nn as nn
import torch.fft as fft
import math

class WaveKernel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # --- STABILITY UPDATES ---
        # Initialize alpha between exp(0)=1.0 and exp(1)=2.7. 
        # This ensures waves decay fast enough to prevent signal explosion.
        self.alpha = nn.Parameter(torch.exp(torch.linspace(0.0, 1.0, d_model)))
        
        # Omega (frequency) remains the same, but we initialize phi more quietly.
        self.omega = nn.Parameter(torch.exp(torch.linspace(0, math.log(2 * math.pi), d_model)))
        self.phi = nn.Parameter(torch.zeros(d_model)) # Start with zero phase for stability

    def forward(self, x):
        # Ensure input is float32 for FFT stability on GTX 1650
        x = x.float()
        B, L, D = x.shape
        device = x.device
        
        # Create time vector
        t = torch.arange(L, device=device, dtype=torch.float32)
        
        # Physical wave kernel: k(t) = exp(-alpha*t) * cos(omega*t + phi)
        # The negative alpha ensures exponential decay (damping)
        kernel = torch.exp(-self.alpha * t[:, None]) * \
                 torch.cos(self.omega * t[:, None] + self.phi)
        
        # FFT padding to power of 2 for speed and memory efficiency
        n_fft = 2**math.ceil(math.log2(2 * L)) 
        
        # Compute FFTs
        X_f = fft.rfft(x, n=n_fft, dim=1)
        K_f = fft.rfft(kernel, n=n_fft, dim=0)
        
        # In-place multiplication saves VRAM
        # We unsqueeze to align (B, N_FFT, D) with (N_FFT, D)
        X_f.mul_(K_f.unsqueeze(0)) 
        
        # Inverse FFT to return to time domain
        y = fft.irfft(X_f, n=n_fft, dim=1)
        
        # Slice back to original length and restore original dtype (e.g., float16)
        return y[:, :L, :].to(x.dtype)

    def _get_ssm_params(self, device):
        decay = torch.exp(-self.alpha)
        lam = torch.complex(decay * torch.cos(self.omega), decay * torch.sin(self.omega))
        phase = torch.complex(torch.cos(self.phi), torch.sin(self.phi))
        return lam.to(device), phase.to(device)

    def step(self, x_t, state=None):
        """
        O(D) single-step State Space recurrence for constant-time inference:
            h_t = lambda * h_{t-1} + x_t
            y_t = Re(phase * h_t)
        """
        B, D = x_t.shape
        lam, phase = self._get_ssm_params(x_t.device)
        if state is None:
            state = torch.zeros(B, D, dtype=torch.cfloat, device=x_t.device)
        next_state = lam * state + x_t.to(torch.cfloat)
        y_t = (phase * next_state).real.to(x_t.dtype)
        return y_t, next_state

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.d_model, dtype=torch.cfloat, device=device)