#!/usr/bin/env python3
"""
============================================================
  Ark-Chat Colab Training Script
  Target: Google Colab T4 GPU (16GB VRAM)
  Model:  ~55M parameter Wave-Delta Transformer
============================================================

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab (colab.research.google.com)
  2. Go to Runtime → Change runtime type → Select T4 GPU
  3. Upload your ark-chat project (or clone from git)
  4. Run this script: !python scripts/colab_train.py --all

This script handles:
  - Installing dependencies
  - Downloading datasets (FineWeb-Edu + OpenOrca)
  - Training BPE tokenizer (32K vocab)
  - Base pretraining (~500M tokens, 3 epochs)
  - Chat fine-tuning (50K instruction examples, 5 epochs)
  - Saving final model weights for download
"""

import argparse
import math
import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def install_deps():
    """Install required packages."""
    print("📦 Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -q torch>=2.0 pyyaml pandas tokenizers datasets")


def step1_download_data(pretrain_tokens_m=500, chat_examples=50000):
    """Download pretraining and chat data."""
    from scripts.prepare_colab_data import prepare_pretraining_data, prepare_chat_data

    if not os.path.exists("data/colab_pretrain.txt"):
        prepare_pretraining_data(target_tokens_m=pretrain_tokens_m)
    else:
        size_gb = os.path.getsize("data/colab_pretrain.txt") / 1e9
        print(f"📂 Pretraining data already exists ({size_gb:.2f} GB)")

    if not os.path.exists("data/colab_chat_train.jsonl"):
        prepare_chat_data(max_examples=chat_examples)
    else:
        with open("data/colab_chat_train.jsonl") as f:
            count = sum(1 for _ in f)
        print(f"📂 Chat data already exists ({count} examples)")


def step2_train_tokenizer(config):
    """Train BPE tokenizer on the pretraining corpus."""
    import torch
    from data.tokenizer import BPETokenizer

    tokenizer_path = "data/colab_bpe_state.pt"
    tokens_path = "data/colab_encoded_tokens.pt"

    if os.path.exists(tokenizer_path) and os.path.exists(tokens_path):
        print("📂 Tokenizer cache found, loading...")
        return

    print("🔤 Training BPE tokenizer (32K vocab)...")
    tokenizer = BPETokenizer(vocab_size=config["model"]["vocab_size"])
    tokenizer.train("data/colab_pretrain.txt")

    print("🔤 Encoding pretraining corpus (chunked to save RAM)...")
    tokens = []
    import sys
    with open("data/colab_pretrain.txt", "r", encoding="utf-8") as f:
        # Read in chunks of lines to avoid OOM killer
        chunk_lines = []
        for i, line in enumerate(f):
            if line.strip():
                chunk_lines.append(line)
            if len(chunk_lines) >= 100000:
                # Encode chunk
                for l in chunk_lines:
                    tokens.extend(tokenizer.encode(l))
                chunk_lines = []
                sys.stdout.write(f"\r  Encoded {i} lines... ({len(tokens)/1e6:.1f}M tokens)")
                sys.stdout.flush()
        # Encode remaining
        for l in chunk_lines:
            tokens.extend(tokenizer.encode(l))
            
    print(f"\n  Finished encoding {len(tokens)/1e6:.1f}M tokens.")

    save_state = tokenizer.get_state()
    torch.save(save_state, tokenizer_path)
    torch.save(tokens, tokens_path)
    print(f"✅ Tokenizer saved (vocab: {len(tokenizer.vocab)}), tokens cached ({len(tokens):,})")


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return __import__("torch").optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def step3_base_pretrain(config):
    """Run base pretraining on the encoded corpus."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import pandas as pd
    from data.tokenizer import BPETokenizer
    from models.transformer import WaveDeltaTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config["model"]["vocab_size"])
    state = torch.load("data/colab_bpe_state.pt", map_location="cpu")
    tokenizer.load_state(state)
    config["model"]["vocab_size"] = len(tokenizer.vocab)

    # Load tokens
    tokens = torch.load("data/colab_encoded_tokens.pt")
    print(f"📊 Corpus: {len(tokens):,} tokens")

    # Dataset
    class PretrainDataset(Dataset):
        def __init__(self, tokens, seq_len):
            self.tokens = torch.tensor(tokens, dtype=torch.long) if not isinstance(tokens, torch.Tensor) else tokens
            self.seq_len = seq_len
            self.num_samples = (len(self.tokens) - 1) // seq_len

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            start = idx * self.seq_len
            chunk = self.tokens[start: start + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                padding = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
                chunk = torch.cat([chunk, padding])
            return chunk[:-1], chunk[1:]

    seq_len = config["training"]["seq_len"]
    batch_size = config["training"]["batch_size"]
    grad_accum = config["training"].get("grad_accum_steps", 1)
    warmup = config["training"].get("warmup_steps", 500)
    epochs = config["training"]["epochs"]
    lr = float(config["training"]["learning_rate"])

    dataset = PretrainDataset(tokens, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=True, num_workers=2)

    model = WaveDeltaTransformer(**config["model"]).to(device)
    
    # ⚡ T4 Optimization: Compile the model for massive speedup (PyTorch 2.0+)
    print("⚡ Compiling model for T4 Tensor Cores (this takes a minute...)")
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"⚠️ torch.compile failed (falling back to eager mode): {e}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model: {param_count:,} parameters ({param_count/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(loader) // grad_accum) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    vocab_size = config["model"]["vocab_size"]
    print(f"\n🚀 Base Pretraining | {len(loader)} batches/epoch | {epochs} epochs")
    print(f"   Effective batch: {batch_size * grad_accum} | LR: {lr} with cosine warmup")
    print("=" * 60)

    history = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += loss.item() * grad_accum

            if batch_idx % 100 == 0:
                progress = (batch_idx / len(loader)) * 100
                cur_lr = scheduler.get_last_lr()[0]
                sys.stdout.write(
                    f"\rEpoch {epoch+1:02d} | {progress:2.0f}% | Loss: {loss.item()*grad_accum:.4f} | LR: {cur_lr:.2e} "
                )
                sys.stdout.flush()

        avg_loss = total_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), "wave_delta_model_colab.pt")
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        pd.DataFrame(history).to_csv("colab_pretrain_log.csv", index=False)

    print("✅ Base pretraining complete! Saved: wave_delta_model_colab.pt")


def step4_chat_finetune(config):
    """Run chat fine-tuning on instruction data."""
    # Reuse the existing train_chat.py with Colab-specific args
    cmd = (
        f"{sys.executable} scripts/train_chat.py "
        f"--data data/colab_chat_train.jsonl "
        f"--config configs/colab_config.yaml "
        f"--tokenizer data/colab_bpe_state.pt "
        f"--base-model wave_delta_model_colab.pt "
        f"--output wave_delta_chat_model_colab.pt "
        f"--epochs 5 "
        f"--batch-size 16 "
        f"--learning-rate 5e-5 "
        f"--grad-accum-steps 4 "
        f"--patience 3 "
        f"--save-mode val_loss"
    )
    print(f"\n🚀 Starting chat fine-tuning...")
    print(f"   Command: {cmd}")
    os.system(cmd)
    print("✅ Chat fine-tuning complete!")


def step5_package_for_download():
    """List all files needed for local inference."""
    files = [
        "wave_delta_chat_model_colab.pt",
        "wave_delta_chat_model_colab_best_val.pt",
        "wave_delta_model_colab.pt",
        "data/colab_bpe_state.pt",
        "configs/colab_config.yaml",
    ]
    print("\n📦 Files to download for local inference:")
    for f in files:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / 1e6
            print(f"  ✅ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {f} (not found)")

    print("\n💡 To run locally after downloading:")
    print("   python scripts/chat_interface.py \\")
    print("     --config configs/colab_config.yaml \\")
    print("     --tokenizer data/colab_bpe_state.pt \\")
    print("     --model-path wave_delta_chat_model_colab.pt")


def main():
    parser = argparse.ArgumentParser(description="Ark-Chat Colab Training Pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--data", action="store_true", help="Download datasets")
    parser.add_argument("--tokenizer", action="store_true", help="Train tokenizer")
    parser.add_argument("--pretrain", action="store_true", help="Run base pretraining")
    parser.add_argument("--finetune", action="store_true", help="Run chat fine-tuning")
    parser.add_argument("--package", action="store_true", help="List files for download")
    parser.add_argument("--pretrain-tokens-m", type=int, default=500,
                        help="Pretraining data size in millions of tokens")
    args = parser.parse_args()

    import yaml
    with open("configs/colab_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    steps = {
        "install": lambda: install_deps(),
        "data": lambda: step1_download_data(args.pretrain_tokens_m),
        "tokenizer": lambda: step2_train_tokenizer(config),
        "pretrain": lambda: step3_base_pretrain(config),
        "finetune": lambda: step4_chat_finetune(config),
        "package": lambda: step5_package_for_download(),
    }

    if args.all:
        for name, fn in steps.items():
            print(f"\n{'='*60}")
            print(f"  STEP: {name.upper()}")
            print(f"{'='*60}")
            fn()
    else:
        ran_any = False
        for name, fn in steps.items():
            if getattr(args, name, False):
                fn()
                ran_any = True
        if not ran_any:
            parser.print_help()


if __name__ == "__main__":
    main()
