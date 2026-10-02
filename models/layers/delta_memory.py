import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def precompute_rope_freqs(d_head: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute the rotary position embedding frequency table.
    Returns a (max_seq_len, d_head // 2) tensor of angles.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_seq_len, d_head // 2)
    return angles


def apply_rope(x, cos_cache, sin_cache):
    """
    Apply Rotary Position Embeddings to a tensor.
    x: (..., d_head) where d_head is even
    cos_cache, sin_cache: (L, d_head // 2) or (1, d_head // 2) for single step
    Returns: rotated x with same shape
    """
    d = x.shape[-1]
    half_d = d // 2
    x1 = x[..., :half_d]
    x2 = x[..., half_d:]

    # Broadcast cos/sin to match x dimensions
    # x1, x2: (..., half_d), cos/sin: (L, half_d) or (1, half_d)
    out1 = x1 * cos_cache - x2 * sin_cache
    out2 = x1 * sin_cache + x2 * cos_cache
    return torch.cat([out1, out2], dim=-1)


class DeltaMemory(nn.Module):
    """
    Multi-head causal associative memory with chunked linear attention and RoPE.

    Improvements over standard linear attention:
      - Rotary Position Embeddings (RoPE) on Q, K for relative position encoding
      - Multi-head for richer attention patterns
      - Chunked associative scan for sub-quadratic training

    Training: O(L * C * D + (L/C) * D^2) via chunked associative scan.
    Inference: O(D^2) per-token via recurrent state update.
    """
    def __init__(self, d_model, n_heads=4, chunk_size=32, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE frequencies and register as buffer (not a parameter)
        angles = precompute_rope_freqs(self.d_head, max_seq_len)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def _apply_rope_to_qk(self, q, k, seq_offset=0):
        """
        Apply RoPE to Q and K tensors.
        q, k: (B, H, L, d_head)
        seq_offset: starting position index (for cached inference)
        """
        L = q.shape[2]
        cos = self.rope_cos[seq_offset: seq_offset + L].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d/2)
        sin = self.rope_sin[seq_offset: seq_offset + L].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        return q, k

    def forward(self, x):
        B, L, D = x.shape
        H, d = self.n_heads, self.d_head
        C = self.chunk_size

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head: (B, H, L, d)
        q = q.view(B, L, H, d).transpose(1, 2)
        k = k.view(B, L, H, d).transpose(1, 2)
        v = v.view(B, L, H, d).transpose(1, 2)

        # Apply RoPE to Q, K
        q, k = self._apply_rope_to_qk(q, k)

        # Normalize keys along feature dimension
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if L <= C:
            # Direct causal attention for short sequences
            attn = torch.matmul(q, k.transpose(-1, -2))  # (B, H, L, L)
            mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, 0.0)
            out = torch.matmul(attn, v)
        else:
            # Chunked Linear Associative Scan
            pad_len = (C - (L % C)) % C
            if pad_len > 0:
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))

            L_pad = L + pad_len
            num_chunks = L_pad // C

            # (B, H, num_chunks, C, d)
            q_c = q.view(B, H, num_chunks, C, d)
            k_c = k.view(B, H, num_chunks, C, d)
            v_c = v.view(B, H, num_chunks, C, d)

            # Intra-chunk causal attention: O(num_chunks * C^2 * d)
            intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2))  # (B, H, nc, C, C)
            mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
            intra_attn = intra_attn.masked_fill(~mask, 0.0)
            intra_out = torch.matmul(intra_attn, v_c)

            # Inter-chunk state propagation: O(num_chunks * d^2)
            kv_c = torch.matmul(k_c.transpose(-1, -2), v_c)  # (B, H, nc, d, d)
            state_cumsum = torch.cumsum(kv_c, dim=2)
            state_shifted = torch.cat([
                torch.zeros(B, H, 1, d, d, device=x.device, dtype=x.dtype),
                state_cumsum[:, :, :-1]
            ], dim=2)

            inter_out = torch.matmul(q_c, state_shifted)  # (B, H, nc, C, d)
            out = (intra_out + inter_out).view(B, H, L_pad, d)
            out = out[:, :, :L]

        # Reshape back: (B, H, L, d) -> (B, L, D) and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

    def step(self, x_t, state=None, pos_idx=0):
        """
        O(D^2) single-step associative recall with RoPE for constant-time inference.
        State shape: (B, H, d_head, d_head) — 4x smaller than single-head.
        """
        B, D = x_t.shape
        H, d = self.n_heads, self.d_head

        qkv = self.qkv(x_t)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, H, d)
        k = k.view(B, H, d)
        v = v.view(B, H, d)

        # Apply RoPE at the current position
        cos = self.rope_cos[pos_idx % self.rope_cos.shape[0]].unsqueeze(0).unsqueeze(0)  # (1, 1, d/2)
        sin = self.rope_sin[pos_idx % self.rope_sin.shape[0]].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Normalize keys
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if state is None:
            state = torch.zeros(B, H, d, d, device=x_t.device, dtype=x_t.dtype)

        # S_t = S_{t-1} + k_t^T v_t  (per head)
        next_state = state + torch.einsum('bhd, bhe -> bhde', k, v)
        # y_t = q_t S_t
        y_t = torch.einsum('bhd, bhde -> bhe', q, next_state)
        y_t = y_t.reshape(B, D)
        return self.out_proj(y_t), next_state

    def init_state(self, batch_size, device, dtype=torch.float32):
        return torch.zeros(batch_size, self.n_heads, self.d_head, self.d_head,
                           device=device, dtype=dtype)