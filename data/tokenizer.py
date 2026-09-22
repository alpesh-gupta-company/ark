import os
import torch
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text):
        print(f"🔄 [BPE] Analyzing {len(text)} characters...")
        tokens = list(text.encode('utf-8'))
        ids = list(tokens)
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if (i + 1) % 100 == 0 or i == num_merges - 1:
                print(f"  > 🛠️ Merge {i+1}/{num_merges} complete (Vocab: {256+i+1})")

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx); i += 2
            else:
                newids.append(ids[i]); i += 1
        return newids

    def encode(self, s):
        tokens = list(s.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break
            tokens = self._merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode('utf-8', errors='replace')