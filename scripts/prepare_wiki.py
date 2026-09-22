import urllib.request
import os

def download_wikitext():
    print("🌍 Downloading Wikitext-2 (Wikipedia subset)...")
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    output_path = "data/input.txt"
    
    os.makedirs("data", exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        
        # Clean the dataset slightly (remove empty lines and headers)
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        cleaned_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('=')]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(cleaned_lines))
            
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Success! Saved to {output_path} ({size_mb:.2f} MB)")
        print(f"Total lines of text: {len(cleaned_lines)}")
        print("\nYou can now run: python scripts/train.py to pre-train your model!")
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")

if __name__ == "__main__":
    download_wikitext()
