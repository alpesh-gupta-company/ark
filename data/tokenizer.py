import os
import torch

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class BPETokenizer:
    def __init__(self, vocab_size=2048):
        self.vocab_size = vocab_size
        self._tokenizer = None
        self._vocab = {}
        self._merges = {}

    def train(self, text_or_path):
        if HAS_TOKENIZERS:
            tok = ByteLevelBPETokenizer()
            if isinstance(text_or_path, str) and os.path.isfile(text_or_path):
                print(f"🔄 [BPE] Training ByteLevelBPETokenizer on file {text_or_path} (target vocab: {self.vocab_size})...")
                tok.train(files=[text_or_path], vocab_size=self.vocab_size, min_frequency=2)
            else:
                corpus = [text_or_path] if isinstance(text_or_path, str) else list(text_or_path)
                print(f"🔄 [BPE] Training ByteLevelBPETokenizer on text iterator (target vocab: {self.vocab_size})...")
                tok.train_from_iterator(corpus, vocab_size=self.vocab_size, min_frequency=1)
            self._tokenizer = tok
            self._vocab = tok.get_vocab()
            self.vocab_size = tok.get_vocab_size()
            print(f"✅ [BPE] Training complete. Vocabulary size: {self.vocab_size}")
        else:
            self._train_pure_python(text_or_path)

    def _train_pure_python(self, text):
        if isinstance(text, str) and os.path.isfile(text):
            with open(text, 'r', encoding='utf-8') as f:
                text = f.read()
        print(f"🔄 [BPE fallback] Analyzing {len(text)} characters...")
        tokens = list(text.encode('utf-8'))
        ids = list(tokens)
        num_merges = max(0, self.vocab_size - 256)
        self._vocab = {i: bytes([i]) for i in range(256)}
        self._merges = {}
        for i in range(num_merges):
            stats = {}
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            newids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == pair[0] and ids[j + 1] == pair[1]:
                    newids.append(idx)
                    j += 2
                else:
                    newids.append(ids[j])
                    j += 1
            ids = newids
            self._merges[pair] = idx
            self._vocab[idx] = self._vocab[pair[0]] + self._vocab[pair[1]]

    def get_state(self):
        return {
            'tokenizer_json': self._tokenizer.to_str() if self._tokenizer else '',
            'vocab_size': len(self.vocab),
            'vocab': self.vocab,
            'merges': self._merges
        }

    def load_state(self, state):
        if isinstance(state, dict) and state.get('tokenizer_json'):
            if HAS_TOKENIZERS:
                self._tokenizer = Tokenizer.from_str(state['tokenizer_json'])
                self._vocab = self._tokenizer.get_vocab()
                self._merges = state.get('merges', {})
                self.vocab_size = self._tokenizer.get_vocab_size()
                return
        if isinstance(state, dict):
            self._vocab = state.get('vocab', {})
            self._merges = state.get('merges', {})
            self.vocab_size = len(self._vocab) if self._vocab else self.vocab_size

    @property
    def vocab(self):
        if self._tokenizer:
            return self._tokenizer.get_vocab()
        return self._vocab

    @vocab.setter
    def vocab(self, val):
        self._vocab = val

    @property
    def merges(self):
        return self._merges

    @merges.setter
    def merges(self, val):
        self._merges = val

    def encode(self, s):
        if self._tokenizer:
            return self._tokenizer.encode(s).ids
        tokens = list(s.encode('utf-8'))
        while len(tokens) >= 2:
            counts = {}
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            pair = min(counts.keys(), key=lambda p: self._merges.get(p, float('inf')))
            if pair not in self._merges:
                break
            newids = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    newids.append(self._merges[pair])
                    i += 2
                else:
                    newids.append(tokens[i])
                    i += 1
            tokens = newids
        return tokens

    def decode(self, ids):
        if self._tokenizer:
            return self._tokenizer.decode(ids)
        if self._vocab and all(isinstance(v, bytes) for v in self._vocab.values()):
            tokens = b"".join(self._vocab[idx] for idx in ids if idx in self._vocab)
            return tokens.decode('utf-8', errors='replace')
        return ""