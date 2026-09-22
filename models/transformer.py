import torch
import torch.nn as nn
from .layers.wave_kernel import WaveKernel
from .layers.delta_memory import DeltaMemory
from .layers.norm import RMSNorm

class WaveDeltaBlock(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.wave = WaveKernel(d_model)
        self.norm2 = RMSNorm(d_model)
        self.delta = DeltaMemory(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )
        self.residual_scale = 1.0 / (n_layers ** 0.5)

    def forward(self, x):
        x = x + self.residual_scale * self.wave(self.norm1(x))
        x = x + self.residual_scale * self.delta(self.norm2(x))
        x = x + self.residual_scale * self.mlp(x)
        return x

class WaveDeltaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([WaveDeltaBlock(d_model, n_layers) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying: Shared logic for input and output
        self.head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x))