import argparse
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from data.tokenizer import BPETokenizer, EOT_STRING
from models.transformer import WaveDeltaTransformer
from utils.web_reader import extract_urls, build_web_context


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Penalize tokens that have already been generated to reduce loops."""
    if not generated_ids:
        return logits
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if token_id < logits.size(-1):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def chat(args):
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Target Device: {device}")

    # Load Tokenizer
    tokenizer_path = args.tokenizer
    if not os.path.exists(tokenizer_path):
        print("❌ Tokenizer state not found. Please train the model first.")
        return

    tokenizer = BPETokenizer(vocab_size=config['model']['vocab_size'])
    state = torch.load(tokenizer_path, map_location='cpu')
    if hasattr(tokenizer, 'load_state'):
        tokenizer.load_state(state)
    else:
        tokenizer.merges = state['merges']
        tokenizer.vocab = state['vocab']
    config['model']['vocab_size'] = len(tokenizer.vocab)

    eot_id = tokenizer.eot_id
    print(f"📝 Tokenizer: vocab_size={config['model']['vocab_size']}, EOT id={eot_id}")

    # Load Model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"⚠️ {model_path} not found, falling back to base model...")
        model_path = 'wave_delta_model.pt'

    if not os.path.exists(model_path):
        print("❌ Model weights not found.")
        return

    print(f"📦 Loaded model weights from: {model_path}")
    model = WaveDeltaTransformer(**config['model']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    max_seq_len = config.get('training', {}).get('seq_len', config['model'].get('seq_len', 512))

    print("=" * 60)
    print("🤖 Ark-Chat Interface Initialized!")
    print(f"📊 Model: {param_count/1e6:.1f}M parameters | Context: {max_seq_len} tokens")
    sampling_desc = 'greedy' if args.greedy or args.temperature <= 0.05 else f'temp={args.temperature}, top_k={args.top_k}'
    print(f"🎲 Sampling: {sampling_desc} | rep_penalty={args.rep_penalty}")
    print(f"🌐 Web reading: enabled (paste URLs in your messages)")
    print("💬 Type 'quit' to stop | 'clear' to reset history")
    print("=" * 60)

    # Maintain conversation history context
    history = ""

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input.lower() == 'clear':
                history = ""
                print("🗑️ History cleared.")
                continue

            # --- Web reading: detect URLs and fetch content ---
            web_context, cleaned_input = build_web_context(user_input)

            if web_context:
                # Inject web content into the prompt
                prompt_addition = f"User: [The user shared web content for reference]\n{web_context}\n\nUser: {cleaned_input}\nAssistant: "
            else:
                prompt_addition = f"User: {user_input}\nAssistant: "

            history += prompt_addition

            # Encode history
            tokens = tokenizer.encode(history)

            # Smart truncation: keep recent context within window
            if len(tokens) > max_seq_len - args.max_tokens:
                tokens = tokens[-(max_seq_len - args.max_tokens):]

            x = torch.tensor([tokens], dtype=torch.long).to(device)
            generated_tokens = []

            sys.stdout.write("Assistant: ")
            sys.stdout.flush()

            with torch.no_grad():
                # 1. Prefill prompt tokens and initialize O(1) recurrent state cache
                last_logits, cache = model.prefill(x)
                last_logits = last_logits[0]
                cur_pos = len(tokens)

                for _ in range(args.max_tokens):
                    # Apply repetition penalty
                    working_logits = apply_repetition_penalty(
                        last_logits.clone(), generated_tokens, args.rep_penalty
                    )

                    if args.greedy or args.temperature <= 0.05:
                        next_token = torch.argmax(working_logits).item()
                    else:
                        # Temperature scaling + Top-K filtering
                        scaled_logits = working_logits / args.temperature
                        top_k = min(args.top_k, scaled_logits.size(-1))
                        indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                        scaled_logits[indices_to_remove] = -float('Inf')
                        probs = torch.softmax(scaled_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()

                    # Stop on EOT token
                    if eot_id is not None and next_token == eot_id:
                        break

                    generated_tokens.append(next_token)

                    # Live streaming output
                    chunk = tokenizer.decode([next_token])
                    
                    # Color formatting for <think> tags
                    if "<think>" in chunk:
                        chunk = chunk.replace("<think>", "\033[90m<think>")
                    if "</think>" in chunk:
                        chunk = chunk.replace("</think>", "</think>\033[0m")
                        
                    sys.stdout.write(chunk)
                    sys.stdout.flush()

                    # Stop if model starts a new User turn
                    current_text = tokenizer.decode(generated_tokens)
                    if "User:" in current_text:
                        generated_tokens = generated_tokens[:current_text.index("User:")]
                        break

                    # 2. O(1) state space step
                    next_token_tensor = torch.tensor([next_token], device=device)
                    logits_step, cache = model.step(next_token_tensor, pos_idx=cur_pos, cache=cache)
                    last_logits = logits_step[0]
                    cur_pos += 1

            print()  # Newline after response

            # Append generated text to history
            final_generation = tokenizer.decode(generated_tokens).split("User:")[0].strip()
            history += f"{final_generation}\n"

        except KeyboardInterrupt:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Ark-Chat Interactive Interface")
    parser.add_argument("--model-path", default="wave_delta_chat_model.pt", help="Path to trained model weights")
    parser.add_argument("--config", default="configs/base_config.yaml", help="Model config file")
    parser.add_argument("--tokenizer", default="data/bpe_state.pt", help="Tokenizer state file")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K filtering limit")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (argmax)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--rep-penalty", type=float, default=1.2, help="Repetition penalty factor")
    return parser.parse_args()


if __name__ == "__main__":
    chat(parse_args())
