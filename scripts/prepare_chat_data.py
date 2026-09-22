import json
import os
import urllib.request

def download_instruction_dataset():
    """
    Downloads a curated instruction dataset and converts it to data/chat_train.jsonl.
    """
    output_path = "data/chat_train.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # Alpaca sample source or Hugging Face raw mirror
    sources = [
        ("Alpaca Data (GitHub Raw)", "https://raw.githubusercontent.com/tatsu-lab/alpaca/main/alpaca_data.json")
    ]
    
    for name, url in sources:
        print(f"🌍 Attempting to download {name}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_data = json.loads(response.read().decode('utf-8'))
                
            formatted = []
            for item in raw_data[:1000]:  # Take first 1000 examples for clean light fine-tuning
                instruction = item.get("instruction", "").strip()
                extra_input = item.get("input", "").strip()
                output = item.get("output", "").strip()
                
                if not instruction or not output:
                    continue
                
                user_msg = f"{instruction}\n{extra_input}".strip() if extra_input else instruction
                formatted.append({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": output}
                    ]
                })
                
            with open(output_path, "w", encoding="utf-8") as f:
                for record in formatted:
                    f.write(json.dumps(record) + "\n")
                    
            print(f"✅ Successfully saved {len(formatted)} instruction samples to {output_path}")
            return
        except Exception as e:
            print(f"⚠️ Failed from {name}: {e}")
            
    print("ℹ️ Keeping current local data/chat_train.jsonl.")

if __name__ == "__main__":
    download_instruction_dataset()
