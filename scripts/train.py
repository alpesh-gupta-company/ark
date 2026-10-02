import os
import sys
import math

# Prevent CUDA memory fragmentation and allow maximum allocation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add project root to python path so it can find 'data' and 'models' modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import pandas as pd
from data.tokenizer import BPETokenizer
from models.transformer import WaveDeltaTransformer

class ShakespeareBPEDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        # STRIDE FIX: We jump by seq_len instead of 1
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate the actual start point for this block
        start_idx = idx * self.seq_len
        chunk = self.tokens[start_idx : start_idx + self.seq_len + 1]
        
        # Safety padding if the last chunk is too short
        if len(chunk) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
            
        return chunk[:-1], chunk[1:]


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer_path = 'data/bpe_state.pt'
    tokens_path = 'data/encoded_tokens.pt'
    requested_vocab_size = config['model']['vocab_size']
    tokenizer = BPETokenizer(vocab_size=requested_vocab_size)

    cache_valid = False
    if os.path.exists(tokenizer_path) and os.path.exists(tokens_path):
        try:
            state = torch.load(tokenizer_path, map_location='cpu')
            if hasattr(tokenizer, 'load_state'):
                tokenizer.load_state(state)
            else:
                tokenizer.merges = state['merges']
                tokenizer.vocab = state['vocab']
            if len(tokenizer.vocab) == requested_vocab_size:
                tokens = torch.load(tokens_path)
                cache_valid = True
                print(f"📂 [CACHE] Loaded valid BPE state (vocab_size: {len(tokenizer.vocab)})...")
            else:
                print(f"🔄 [CACHE MISMATCH] Cached vocab {len(tokenizer.vocab)} != config {requested_vocab_size}. Rebuilding...")
        except Exception as e:
            print(f"🔄 [CACHE ERROR] {e}. Rebuilding...")

    if not cache_valid:
        print(f"🔄 [SETUP] Training BPE tokenizer (vocab_size={requested_vocab_size})...")
        input_path = 'data/input.txt'
        tokenizer.train(input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("🔄 [SETUP] Encoding dataset tokens...")
        tokens = tokenizer.encode(text)
        save_state = tokenizer.get_state() if hasattr(tokenizer, 'get_state') else {'merges': tokenizer.merges, 'vocab': tokenizer.vocab}
        torch.save(save_state, tokenizer_path)
        torch.save(tokens, tokens_path)
        print(f"✅ Tokenizer saved ({len(tokenizer.vocab)} tokens), dataset cached ({len(tokens)} tokens).")

    actual_vocab_size = len(tokenizer.vocab)
    config['model']['vocab_size'] = actual_vocab_size

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    seq_len = config['training']['seq_len']
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training'].get('grad_accum_steps', 1)
    warmup_steps = config['training'].get('warmup_steps', 500)
    epochs = config['training']['epochs']

    num_workers = min(4, os.cpu_count() or 1)
    dataset = ShakespeareBPEDataset(tokens, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    model = WaveDeltaTransformer(**config['model']).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=0.01)
    
    total_steps = (len(loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n🚀 Training Started | Batches/Epoch: {len(loader)} | Effective batch: {batch_size * grad_accum_steps}")
    print(f"   LR Schedule: cosine warmup ({warmup_steps} steps) → decay over {total_steps} steps")
    print("="*60)

    history = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits.view(-1, actual_vocab_size), y.view(-1))
                loss = loss / grad_accum_steps  # Normalize for accumulation
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            total_loss += loss.item() * grad_accum_steps

            # --- REAL-TIME FEEDBACK ---
            if batch_idx % 20 == 0:
                progress = (batch_idx / len(loader)) * 100
                current_lr = scheduler.get_last_lr()[0]
                sys.stdout.write(f"\rEpoch {epoch+1:02d} | {progress:2.0f}% | Loss: {loss.item() * grad_accum_steps:.4f} | LR: {current_lr:.2e} ")
                sys.stdout.flush()

        avg_loss = total_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoints
        torch.save(model.state_dict(), 'wave_delta_model.pt')
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'lr': scheduler.get_last_lr()[0]})
        pd.DataFrame(history).to_csv('training_log.csv', index=False)

if __name__ == "__main__":
    train()