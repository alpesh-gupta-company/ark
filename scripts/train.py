import os
import sys
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
    
def train():
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Hardware: {torch.cuda.get_device_name(0)}")

    tokenizer_path = 'data/bpe_state.pt'
    tokens_path = 'data/encoded_tokens.pt'
    tokenizer = BPETokenizer(vocab_size=config['model']['vocab_size'])

    if os.path.exists(tokenizer_path) and os.path.exists(tokens_path):
        print("📂 [CACHE] Loading BPE state...")
        state = torch.load(tokenizer_path)
        tokenizer.merges = state['merges']
        tokenizer.vocab = state['vocab']
        tokens = torch.load(tokens_path)
    else:
        print("🔄 [SETUP] Encoding text (This takes a few minutes)...")
        with open('data/input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.train(text)
        tokens = tokenizer.encode(text)
        torch.save({'merges': tokenizer.merges, 'vocab': tokenizer.vocab}, tokenizer_path)
        torch.save(tokens, tokens_path)

    actual_vocab_size = len(tokenizer.vocab)
    config['model']['vocab_size'] = actual_vocab_size

    dataset = ShakespeareBPEDataset(tokens, config['training']['seq_len'])
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, pin_memory=True, num_workers=0)

    model = WaveDeltaTransformer(**config['model']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n🚀 Training Started | Batches per Epoch: {len(loader)}")
    print("="*50)

    history = []
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits.view(-1, actual_vocab_size), y.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            # --- REAL-TIME FEEDBACK ---
            if batch_idx % 20 == 0:
                # Use sys.stdout.write and flush to force terminal update
                progress = (batch_idx / len(loader)) * 100
                sys.stdout.write(f"\rEpoch {epoch+1:02d} | Progress: {progress:2.1f}% | Loss: {loss.item():.4f} ")
                sys.stdout.flush()

        avg_loss = total_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        torch.save(model.state_dict(), 'wave_delta_model.pt')
        history.append({'epoch': epoch+1, 'loss': avg_loss})
        pd.DataFrame(history).to_csv('training_log.csv', index=False)

if __name__ == "__main__":
    train()