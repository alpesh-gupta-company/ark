import argparse
import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from data.tokenizer import BPETokenizer
from models.transformer import WaveDeltaTransformer

def chat(args):
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Target Device: {device}")

    # Load Tokenizer
    tokenizer_path = 'data/bpe_state.pt'
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

    print("="*50)
    print("🤖 Wave-Delta Chat Interface Initialized!")
    print(f"Sampling: {'greedy' if args.greedy or args.temperature <= 0.05 else f'temperature={args.temperature}, top_k={args.top_k}'}")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)

    # Maintain conversation history context
    history = ""

    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                break
                
            # Append to history matching the training format ("User: ...\nAssistant: ")
            history += f"User: {user_input}\nAssistant: "
            
            # Encode history
            tokens = tokenizer.encode(history)
            
            # Truncate to max sequence length to prevent index errors
            max_seq_len = config.get('training', {}).get('seq_len', 128)
            if len(tokens) > max_seq_len - 20: 
                # Keep the recent context if it gets too long
                tokens = tokens[-(max_seq_len - 20):]
                
            x = torch.tensor([tokens], dtype=torch.long).to(device)
            generated_tokens = []
            
            sys.stdout.write("Assistant: ")
            sys.stdout.flush()
            
            with torch.no_grad():
                for _ in range(100):  # Max new tokens
                    # Forward pass. We MUST truncate x to max_seq_len 
                    # otherwise the positional embedding lookup goes out of bounds!
                    logits = model(x[:, -max_seq_len:])
                    last_logits = logits[0, -1, :]
                    
                    if args.greedy or args.temperature <= 0.05:
                        next_token = torch.argmax(last_logits).item()
                    else:
                        # Temperature scaling + Top-K filtering
                        scaled_logits = last_logits / args.temperature
                        top_k = min(args.top_k, scaled_logits.size(-1))
                        indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                        scaled_logits[indices_to_remove] = -float('Inf')
                        probs = torch.softmax(scaled_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    generated_tokens.append(next_token)
                    x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
                    
                    # Live streaming output
                    chunk = tokenizer.decode([next_token])
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    
                    # Stop condition: stop if model starts a new User turn or completes response
                    current_text = tokenizer.decode(generated_tokens)
                    if "User:" in current_text:
                        break
                    if "\n" in chunk and len(generated_tokens) >= 5:
                        break
            
            # Append generated text to history
            final_generation = tokenizer.decode(generated_tokens).split("User:")[0].strip()
            history += f"{final_generation}\n"
            
        except KeyboardInterrupt:
            break

def parse_args():
    parser = argparse.ArgumentParser(description="Wave-Delta Interactive Chat Interface")
    parser.add_argument("--model-path", default="wave_delta_chat_model.pt", help="Path to trained model weights")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K filtering limit")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (argmax)")
    return parser.parse_args()

if __name__ == "__main__":
    chat(parse_args())
