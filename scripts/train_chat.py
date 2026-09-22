import argparse
import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from data.tokenizer import BPETokenizer
from models.transformer import WaveDeltaTransformer


FALLBACK_CONVERSATIONS = [
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the wave equation?"},
            {
                "role": "assistant",
                "content": "The wave equation models how disturbances propagate through space and time.",
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who are you?"},
            {
                "role": "assistant",
                "content": "I am a small conversational model built with the Wave-Delta architecture.",
            },
        ]
    },
]


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
        print("Warning: no chat dataset supplied; using the tiny fallback dataset.")
        return FALLBACK_CONVERSATIONS

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

        for conversation in conversations:
            tokens = []
            assistant_positions = []
            for message in conversation["messages"]:
                prefix = tokenizer.encode(f"{message['role'].capitalize()}: ")
                content = tokenizer.encode(message["content"])
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


def evaluate(model, loader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
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

    tokenizer = BPETokenizer(vocab_size=config["model"]["vocab_size"])
    state = torch.load(args.tokenizer, map_location="cpu")
    if hasattr(tokenizer, "load_state"):
        tokenizer.load_state(state)
    else:
        tokenizer.merges = state["merges"]
        tokenizer.vocab = state["vocab"]
    config["model"]["vocab_size"] = len(tokenizer.vocab)

    dataset = ConversationalDataset(
        conversations, tokenizer, config["training"]["seq_len"]
    )
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
        print(f"Loading base checkpoint: {args.base_model}")
        model.load_state_dict(
            torch.load(args.base_model, map_location=device), strict=False
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print(
        f"Fine-tuning on {len(train_set)} samples; "
        f"validating on {len(validation_set)}"
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = criterion(
                logits.view(-1, config["model"]["vocab_size"]), y.to(device).view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(1, len(train_loader))
        validation_loss = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            config["model"]["vocab_size"],
        )
        print(
            f"Epoch {epoch + 1:02d} | Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            val_checkpoint = os.path.splitext(args.output)[0] + "_best_val.pt"
            torch.save(model.state_dict(), val_checkpoint)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # In small-scale instruction fine-tuning, save each progressing epoch
        # or final epoch to ensure fine-tuning weights are preserved
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--save-mode",
        choices=["final", "train_loss", "val_loss"],
        default="val_loss",
        help="Checkpoint save strategy: 'final' saves at end, 'val_loss' deploys the best validation checkpoint",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train_chat(parse_args())
