import torch
import torch.nn as nn

class DeltaMemory(nn.Module):
    def __init__(self, d_model, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # 1. Direct causal kernel if sequence is shorter than or equal to chunk_size
        if L <= self.chunk_size:
            attn = torch.matmul(q, k.transpose(-1, -2))
            mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, 0.0)
            return torch.matmul(attn, v)

        # 2. Chunked Linear Associative Scan: O(L * D * (chunk_size + D)) -> strictly O(L) linear time
        C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            q = torch.cat([q, torch.zeros(B, pad_len, D, device=x.device, dtype=q.dtype)], dim=1)
            k = torch.cat([k, torch.zeros(B, pad_len, D, device=x.device, dtype=k.dtype)], dim=1)
            v = torch.cat([v, torch.zeros(B, pad_len, D, device=x.device, dtype=v.dtype)], dim=1)

        L_pad = L + pad_len
        num_chunks = L_pad // C

        q_c = q.view(B, num_chunks, C, D)
        k_c = k.view(B, num_chunks, C, D)
        v_c = v.view(B, num_chunks, C, D)

        # Intra-chunk causal attention: O(num_chunks * C^2 * D) = O(L * C * D)
        intra_attn = torch.matmul(q_c, k_c.transpose(-1, -2))
        mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        intra_attn = intra_attn.masked_fill(~mask, 0.0)
        intra_out = torch.matmul(intra_attn, v_c)

        # Inter-chunk state propagation: O(num_chunks * D^2) = O((L/C) * D^2)
        kv_c = torch.matmul(k_c.transpose(-1, -2), v_c)
        state_cumsum = torch.cumsum(kv_c, dim=1)
        state_shifted = torch.cat([
            torch.zeros(B, 1, D, D, device=x.device, dtype=x.dtype),
            state_cumsum[:, :-1]
        ], dim=1)

        inter_out = torch.matmul(q_c, state_shifted)
        out = (intra_out + inter_out).view(B, L_pad, D)
        return out[:, :L]

    def step(self, x_t, state=None):
        """
        O(D^2) single-step associative recall for constant-time O(1) inference:
            S_t = S_{t-1} + k_t^T v_t
            y_t = q_t S_t
        """
        B, D = x_t.shape
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if state is None:
            state = torch.zeros(B, D, D, device=x_t.device, dtype=x_t.dtype)

        next_state = state + torch.einsum('bd, be -> bde', k, v)
        y_t = torch.einsum('bd, bde -> be', q, next_state)
        return y_t, next_state

    def init_state(self, batch_size, device, dtype=torch.float32):
        return torch.zeros(batch_size, self.d_model, self.d_model, device=device, dtype=dtype)