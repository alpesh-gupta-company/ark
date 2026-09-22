import torch
import torch.nn as nn

class DeltaMemory(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Ultra-Low Memory Causal Linear Attention (L x L formulation)
        # Because Sequence Length (L=128) is smaller than D_model (D=256),
        # the DxD state matrix (65,536 elements) is much larger than an LxL attention matrix (16,384 elements).
        # By doing (Q @ K^T) first and masking, we avoid storing the massive DxD tensor entirely
        # and eliminate the loop that causes PyTorch's autograd to explode in memory.
        
        # Calculate Q @ K^T -> Shape: (B, L, L)
        attn = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply causal mask so tokens only see the past (lower triangular matrix)
        mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, 0.0)
        
        # Multiply by V -> Shape: (B, L, D)
        context = torch.matmul(attn, v)
        
        return context