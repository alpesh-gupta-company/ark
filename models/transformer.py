import torch
import torch.nn as nn
from .layers.wave_kernel import WaveKernel
from .layers.delta_memory import DeltaMemory
from .layers.norm import RMSNorm

class WaveDeltaBlock(nn.Module):
    def __init__(self, d_model, n_layers, chunk_size=32):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.wave = WaveKernel(d_model)
        self.norm2 = RMSNorm(d_model)
        self.delta = DeltaMemory(d_model, chunk_size=chunk_size)
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

    def step(self, x_t, state=None):
        """
        O(1) single-step autoregressive generation.
        x_t: (B, D) float tensor
        state: tuple of (wave_state, delta_state)
        """
        wave_s, delta_s = state if state is not None else (None, None)

        # 1. Wave kernel SSM step
        x_norm1 = self.norm1(x_t)
        wave_out, next_wave_s = self.wave.step(x_norm1, wave_s)
        x_t = x_t + self.residual_scale * wave_out

        # 2. Delta memory associative step
        x_norm2 = self.norm2(x_t)
        delta_out, next_delta_s = self.delta.step(x_norm2, delta_s)
        x_t = x_t + self.residual_scale * delta_out

        # 3. Feed-forward MLP
        mlp_out = self.mlp(x_t)
        x_t = x_t + self.residual_scale * mlp_out

        return x_t, (next_wave_s, next_delta_s)

    def init_state(self, batch_size, device, dtype=torch.float32):
        return (
            self.wave.init_state(batch_size, device),
            self.delta.init_state(batch_size, device, dtype=dtype)
        )

class WaveDeltaTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, seq_len=128, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            WaveDeltaBlock(d_model, n_layers, chunk_size=chunk_size) for _ in range(n_layers)
        ])
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

    def init_cache(self, batch_size, device, dtype=torch.float32):
        """Initializes empty state cache for all layers."""
        return [layer.init_state(batch_size, device, dtype=dtype) for layer in self.layers]

    def prefill(self, tokens):
        """
        Prefill prompt tokens [x_0, ..., x_P], returning logits for the final token
        and an exact state cache ready for O(1) step generation.
        """
        B, L = tokens.shape
        device = tokens.device
        pos = torch.arange(0, L, device=device).unsqueeze(0)
        h = self.embedding(tokens) + self.pos_embedding(pos)
        cache = self.init_cache(B, device, dtype=h.dtype)

        for t in range(L):
            xt = h[:, t]
            new_cache = []
            for i, layer in enumerate(self.layers):
                xt, s = layer.step(xt, cache[i])
                new_cache.append(s)
            cache = new_cache

        logits = self.head(self.final_norm(xt))
        return logits, cache

    def step(self, token_t, pos_idx, cache):
        """
        O(1) inference step for generating a single token:
            token_t: (B,) or (B, 1) token id
            pos_idx: integer index for position embedding
            cache: list of layer state tuples [(wave_s, delta_s), ...]
        Returns:
            next_logits: (B, vocab_size)
            next_cache: list of updated layer states
        """
        if token_t.dim() == 2:
            token_t = token_t.squeeze(1)
        device = token_t.device
        pos = torch.tensor([pos_idx % self.seq_len], device=device)
        xt = self.embedding(token_t) + self.pos_embedding(pos)

        next_cache = []
        for i, layer in enumerate(self.layers):
            xt, layer_state = layer.step(xt, cache[i])
            next_cache.append(layer_state)

        logits = self.head(self.final_norm(xt))
        return logits, next_cache