import argparse
import json
import math
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from data.tokenizer import BPETokenizer, EOT_STRING
from models.transformer import WaveDeltaTransformer


def normalize_conversation(record):
    if "messages" in record:
        messages = record["messages"]
    elif "conversations" in record:
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
        messages = [
            {
                "role": role_map.get(str(message.get("from", "")).lower(), ""),
                "content": message.get("value", ""),
            }
            for message in record["conversations"]
        ]
    elif "instruction" in record and "output" in record:
        prompt = str(record["instruction"])
        extra_input = str(record.get("input", "")).strip()
        if extra_input:
            prompt = f"{prompt}\n{extra_input}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": record["output"]},
        ]
    elif "user" in record and "assistant" in record:
        messages = [
            {"role": "user", "content": record["user"]},
            {"role": "assistant", "content": record["assistant"]},
        ]
    else:
        raise ValueError("Each record needs messages or user and assistant fields")

    normalized = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})

    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("Each conversation needs at least one assistant message")
    return {"messages": normalized}


def load_conversations(path):
    if not path or not os.path.exists(path):
        raise ValueError(f"Chat dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        records = (
            [json.loads(line) for line in file if line.strip()]
            if path.endswith(".jsonl")
            else json.load(file)
        )

    if not isinstance(records, list) or not records:
        raise ValueError(f"Chat dataset must contain a non-empty list: {path}")
    return [normalize_conversation(record) for record in records]


class ConversationalDataset(Dataset):
    def __init__(self, conversations, tokenizer, seq_len):
        self.samples = []
        eot_tokens = tokenizer.encode(EOT_STRING) if tokenizer.eot_id is None else [tokenizer.eot_id]

        for conversation in conversations:
            tokens = []
            assistant_positions = []
            for message in conversation["messages"]:
                prefix = tokenizer.encode(f"{message['role'].capitalize()}: ")
                content = tokenizer.encode(message["content"])

                if message["role"] == "assistant":
                    suffix = eot_tokens + tokenizer.encode("\n")
                else:
                    suffix = tokenizer.encode("\n")

                message_tokens = prefix + content + suffix
                tokens.extend(message_tokens)
                assistant_positions.extend(
                    [message["role"] == "assistant"] * len(message_tokens)
                )

            for start in range(0, max(1, len(tokens) - 1), seq_len):
                chunk_tokens = tokens[start : start + seq_len + 1]
                chunk_assistant = assistant_positions[start : start + seq_len + 1]
                padding = seq_len + 1 - len(chunk_tokens)
                if padding > 0:
                    chunk_tokens += [0] * padding
                    chunk_assistant += [False] * padding

                inputs = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
                labels = torch.tensor(
                    [
                        token if is_assistant else -100
                        for token, is_assistant in zip(
                            chunk_tokens[1:], chunk_assistant[1:]
                        )
                    ],
                    dtype=torch.long,
                )
                if (labels != -100).any():
                    self.samples.append((inputs, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            with torch.amp.autocast('cuda'):
                logits = model(x.to(device))
                loss = criterion(logits.view(-1, vocab_size), y.to(device).view(-1))
            total_loss += loss.item()
            batches += 1
    return total_loss / max(1, batches)


def train_chat(args):
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conversations = load_conversations(args.data)
    print(f"📊 Loaded {len(conversations)} conversations from {args.data}")

    tokenizer = BPETokenizer(vocab_size=config["model"]["vocab_size"])
    state = torch.load(args.tokenizer, map_location="cpu")
    if hasattr(tokenizer, "load_state"):
        tokenizer.load_state(state)
    else:
        tokenizer.merges = state["merges"]
        tokenizer.vocab = state["vocab"]
    config["model"]["vocab_size"] = len(tokenizer.vocab)
    print(f"📦 Tokenizer loaded: vocab_size={config['model']['vocab_size']}, EOT id={tokenizer.eot_id}")

    dataset = ConversationalDataset(
        conversations, tokenizer, config["training"]["seq_len"]
    )
    print(f"📊 Created {len(dataset)} training samples (seq_len={config['training']['seq_len']})")

    if len(dataset) < 2:
        raise ValueError("The chat dataset produced fewer than two usable samples")

    validation_size = max(1, int(len(dataset) * args.validation_fraction))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)

    model = WaveDeltaTransformer(**config["model"]).to(device)
    if os.path.exists(args.base_model):
        print(f"📦 Loading base checkpoint: {args.base_model}")
        model.load_state_dict(
            torch.load(args.base_model, map_location=device), strict=False
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda')

    # Cosine LR schedule with warmup
    grad_accum = args.grad_accum_steps
    total_steps = (len(train_loader) // grad_accum) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print(f"\n🚀 Fine-tuning on {len(train_set)} samples; validating on {len(validation_set)}")
    print(f"   Batch: {args.batch_size} x {grad_accum} accum = {args.batch_size * grad_accum} effective")
    print(f"   LR: {args.learning_rate} with cosine warmup ({warmup_steps} steps)")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(train_loader):
            with torch.amp.autocast('cuda'):
                logits = model(x.to(device))
                loss = criterion(
                    logits.view(-1, config["model"]["vocab_size"]), y.to(device).view(-1)
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum

        train_loss = total_loss / max(1, len(train_loader))
        validation_loss = evaluate(
            model, validation_loader, criterion, device, config["model"]["vocab_size"],
        )
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:02d} | Train: {train_loss:.4f} | Val: {validation_loss:.4f} | LR: {current_lr:.2e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            val_checkpoint = os.path.splitext(args.output)[0] + "_best_val.pt"
            torch.save(model.state_dict(), val_checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save strategy
        if args.save_mode == "final" and epoch == args.epochs - 1:
            torch.save(model.state_dict(), args.output)
            print(f"Saved final conversational checkpoint to {args.output}")
        elif args.save_mode == "train_loss":
            torch.save(model.state_dict(), args.output)
        elif args.save_mode == "val_loss" and validation_loss == best_validation_loss:
            torch.save(model.state_dict(), args.output)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement")
            break

    if args.save_mode != "val_loss":
        torch.save(model.state_dict(), args.output)
        print(f"Saved trained conversational model to {args.output}")
    else:
        print(f"Saved best validation model to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction-tune the Wave-Delta chat model"
    )
    parser.add_argument(
        "--data", default=os.environ.get("CHAT_DATA_PATH", "data/chat_train.jsonl")
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--tokenizer", default="data/bpe_state.pt")
    parser.add_argument("--base-model", default="wave_delta_model.pt")
    parser.add_argument("--output", default="wave_delta_chat_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--save-mode",
        choices=["final", "train_loss", "val_loss"],
        default="val_loss",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_chat(parse_args())
