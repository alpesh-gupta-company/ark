import json
import os
import urllib.request

def download_instruction_dataset():
    """
    Downloads a curated instruction dataset and converts it to data/chat_train.jsonl.
    Tries multiple mirrors for the Alpaca dataset.
    """
    output_path = "data/chat_train.jsonl"
    os.makedirs("data", exist_ok=True)

    sources = [
        ("Stanford Alpaca (GitHub)", "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"),
        ("Alpaca Cleaned (GitHub)", "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"),
        ("HuggingFace Alpaca", "https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"),
    ]

    for name, url in sources:
        print(f"🌍 Attempting to download {name}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as response:
                raw_bytes = response.read()

            if url.endswith('.parquet'):
                # For parquet, fall through to next source
                print(f"⚠️ Parquet format not supported without pandas, trying next source...")
                continue

            raw_data = json.loads(raw_bytes.decode('utf-8'))

            formatted = []
            skipped = 0
            for item in raw_data:
                instruction = item.get("instruction", "").strip()
                extra_input = item.get("input", "").strip()
                output = item.get("output", "").strip()

                if not instruction or not output:
                    skipped += 1
                    continue
                # Quality filters
                if len(output.split()) < 3:
                    skipped += 1
                    continue
                if len(instruction) < 5:
                    skipped += 1
                    continue

                user_msg = f"{instruction}\n{extra_input}".strip() if extra_input else instruction
                formatted.append({
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": output}
                    ]
                })

                if len(formatted) >= 15000:
                    break

            with open(output_path, "w", encoding="utf-8") as f:
                for record in formatted:
                    f.write(json.dumps(record) + "\n")

            print(f"✅ Saved {len(formatted)} instruction samples to {output_path} (skipped {skipped})")
            return
        except Exception as e:
            print(f"⚠️ Failed from {name}: {e}")

    print("❌ Could not download from any source.")
    print("ℹ️ Keeping current local data/chat_train.jsonl.")

if __name__ == "__main__":
    download_instruction_dataset()
