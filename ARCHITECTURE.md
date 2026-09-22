# Current Wave-Delta Chat Model Architecture

This document describes the implementation currently in `ark-chat`. It is based on the code copied from `wave_delta_transformer` and the design discussion in `chat/Building A Conservation-Based LLM.md`.

## System diagram

```mermaid
flowchart TD
    A[Wikipedia text or conversation records] --> B[Byte-level BPE tokenizer\n2048-token vocabulary]
    B --> C[Token IDs\nB x L]
    C --> D[Token embedding\nB x L x D]
    P[Learned absolute position embedding\nL x D] --> E
    D --> E[Add token and position embeddings\nB x L x D]

    E --> F1[Wave-Delta Block 1]
    F1 --> F2[Wave-Delta Block 2]
    F2 --> F3[Wave-Delta Block 3]
    F3 --> F4[Wave-Delta Block 4]
    F4 --> F5[Wave-Delta Block 5]
    F5 --> F6[Wave-Delta Block 6]

    subgraph BLOCK[Each Wave-Delta block]
        X[Input x] --> N1[RMSNorm]
        N1 --> W[WaveKernel\nDamped depthwise FFT convolution]
        W --> R1[Residual add\nx + scale * wave]
        R1 --> N2[RMSNorm]
        N2 --> QKV[Linear projection\nQ, K, V]
        QKV --> CM[Normalize K\ncausal lower-triangular mask]
        CM --> DM[DeltaMemory\nQK^T V]
        DM --> R2[Residual add\nx + scale * delta]
        R2 --> MLP[MLP\nD -> 4D -> D with GELU]
        MLP --> R3[Residual add\nx + scale * MLP]
    end

    F6 --> G[Final RMSNorm]
    G --> H[Tied linear output head\nD -> vocabulary]
    H --> I[Logits\nB x L x V]
    I --> J[Cross-entropy next-token loss]
    J --> K[AdamW + mixed precision\nbase pretraining or chat fine-tuning]

    I --> L[Last-position logits]
    L --> M[Temperature 0.7 + top-k 10]
    M --> N[Sample next token]
    N --> O[Append token to context]
    O --> P2[Keep rolling context window\nmax 128 tokens]
    P2 --> I
```

## Dimensions and configuration

The current configuration is:

| Symbol | Meaning | Current value |
|---|---|---:|
| $B$ | Batch size during base training | 64 |
| $L$ | Maximum sequence length | 128 |
| $D$ | Model width (`d_model`) | 256 |
| $N$ | Number of Wave-Delta blocks | 6 |
| $V$ | Vocabulary size | 256 in the config; the tokenizer state determines the runtime size |
| $r$ | MLP expansion ratio | 4 |

The model input is an integer tensor with shape $B \times L$. After embedding, the hidden state has shape $B \times L \times D$. The output head produces $B \times L \times V$ logits.

## Data and training path

### Tokenization

`data/tokenizer.py` starts with all 256 byte values and learns up to `vocab_size - 256` merges from UTF-8 text. The tokenizer stores its merge table and vocabulary in `data/bpe_state.pt`; encoded pretraining tokens are cached in `data/encoded_tokens.pt`.

This is a simple Python implementation intended for experimentation. It is not equivalent to a modern optimized tokenizer and does not use a separate end-of-turn token.

### Base pretraining

`scripts/train.py` reads `data/input.txt`, creates non-overlapping chunks of $L+1$ tokens, and trains next-token prediction:

$$
\mathcal{L} = -\frac{1}{BL}\sum_{b=1}^{B}\sum_{t=1}^{L}
\log p(x_{b,t}\mid x_{b,<t}).
$$

The base checkpoint is saved as `wave_delta_model.pt` after each epoch.

### Chat fine-tuning

`scripts/train_chat.py` accepts either a JSON array or JSONL file. Each record may use the compact form:

```json
{"user": "question", "assistant": "answer"}
```

or a multi-turn form:

```text
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"}
]}
```

It initializes from `wave_delta_model.pt` when available and saves the best validation checkpoint as `wave_delta_chat_model.pt`. Prompt and padding positions are masked with `-100`, so the loss is calculated only on assistant tokens. The default is three epochs at a low learning rate with gradient clipping and a deterministic validation split. Use `data/chat_train.example.jsonl` as the schema template and provide a real dataset with `--data` or `CHAT_DATA_PATH`.

## WaveKernel: conserved/damped signal path

For each feature channel, the kernel is parameterized as:

$$
k_d(t) = e^{-\alpha_d t}\cos(\omega_d t + \phi_d).
$$

The code computes the input and kernel FFTs, multiplies them channel-wise, and applies an inverse FFT. It uses zero-padding to:

$$
n_{fft}=2^{\lceil\log_2(2L)\rceil}.
$$

The damping parameter $\alpha$ makes the learned response decay over time. The implementation is a depthwise channel-wise convolution; it does not enforce a formal physical conservation law such as an exactly conserved energy or norm.

## DeltaMemory: current causal retrieval path

The module projects the normalized hidden state into $Q$, $K$, and $V$:

$$
[Q,K,V] = XW_{qkv}.
$$

Keys are normalized, then the module computes:

$$
A = QK^T,
\qquad
A_{ij}=0\ \text{for}\ j>i,
\qquad
Y=AV.
$$

## DeltaMemory: Chunked Linear Attention & Associative Scan

The module projects the normalized hidden state into $Q$, $K$, and $V$:

$$
[Q,K,V] = XW_{qkv}.
$$

Keys are normalized along the feature dimension. In parallel training mode, the module computes a **Chunked Linear Associative Scan** with chunk size $C=32$:

1. **Intra-chunk causal attention**:
   $$A_{\text{intra}} = \text{tril}(Q_c K_c^T) V_c \in \mathbb{R}^{B \times C \times D}$$
2. **Inter-chunk state accumulation**:
   $$S_c = S_{c-1} + K_c^T V_c \in \mathbb{R}^{B \times D \times D}$$
3. **Inter-chunk query context**:
   $$Y_{\text{inter}} = Q_c S_{c-1} \in \mathbb{R}^{B \times C \times D}$$

Total output for each chunk is:
$$Y_c = A_{\text{intra}} + Y_{\text{inter}}$$

This eliminates the quadratic $L \times L$ attention matrix, achieving strictly **$O(L)$ linear-time training complexity** ($O(L \cdot C \cdot D + \frac{L}{C} D^2)$).

During token generation, DeltaMemory runs an exact single-step associative update in **$O(D^2)$ constant time ($O(1)$ with respect to sequence length)**:

$$S_t = S_{t-1} + k_t^T v_t, \quad y_t = q_t S_t$$

## Complexity Analysis

Let $N$ be the number of blocks, $L$ sequence length, $D$ model dimension, $C$ chunk size (32), and $V$ vocabulary size (2,048).

| Component | Training / Prefill Time | Inference Time (per token) | State Memory |
|---|---:|---:|---:|
| RMSNorm | $O(L \cdot D)$ | $O(D)$ | None |
| WaveKernel (SSM Dual) | $O(D \cdot L \log L)$ | $O(D)$ | $2D$ floats (cfloat state $h_t$) |
| DeltaMemory (Chunked) | $O(L \cdot C \cdot D + \frac{L}{C} D^2)$ | $O(D^2)$ | $D^2$ floats (matrix state $S_t$) |
| MLP | $O(L \cdot D^2)$ | $O(D^2)$ | None |
| Output Projection | $O(L \cdot D \cdot V)$ | $O(D \cdot V)$ | None |

### Full Sequence Training Complexity

$$
O\left(N\left(D \cdot L \log L + L \cdot C \cdot D + \frac{L}{C} D^2 + L \cdot D^2\right) + L \cdot D \cdot V\right) = \mathcal{O}(L \log L)
$$

The architecture is sub-quadratic during training, scaling near-linearly with sequence length $L$.

### Generation Complexity ($O(1)$ per Token)

During autoregressive generation, Ark-Chat utilizes the **recurrent state-space cache** rather than recomputing past sequence context:

$$
\text{Time per generated token} = O(N \cdot D^2 + D \cdot V) = \mathcal{O}(1) \text{ with respect to sequence length } L.
$$

Generating $T$ tokens costs strictly $O(T \cdot (N D^2 + D V))$, eliminating the $O(T \cdot L^2)$ bottleneck of un-cached architectures and outperforming the $O(T \cdot L \cdot D)$ KV-cache latency of standard Transformers.

## Inference Controls

The interactive chat interface (`scripts/chat_interface.py`):

1. Loads conversational weights (`wave_delta_chat_model.pt`).
2. Runs `model.prefill(prompt)` once to initialize the state-space cache across all 6 layers.
3. Calls `model.step(token_t, pos_idx, cache)` token-by-token in constant $O(1)$ time.
4. Supports greedy decoding (`--greedy`) or temperature/top-k creative sampling (`--temperature`, `--top-k`).
5. Halts when the model produces an end-of-turn delimiter or completes 100 new tokens.

## Current limitations shown by the architecture

- The fallback chat dataset is intentionally tiny; meaningful quality requires replacing it with a diverse dataset such as the supplied JSONL template expanded to many examples.
- The tokenizer is byte-based and small, which produces longer sequences than modern subword tokenizers.
- Absolute positional embeddings cap the supported context at 128 tokens.
- DeltaMemory is causal but currently quadratic in sequence length because it materializes $QK^T$.
- Autoregressive generation recomputes the full context on every step.
- No validation split, perplexity evaluation, held-out chat set, or benchmark harness is included.
- The wave layer is an efficient learned signal filter, but the code does not yet prove an exact conservation invariant.

## What this model is

The current system is a small decoder-only language model with six repeated blocks. Each block combines:

1. A damped frequency-domain signal-mixing path.
2. A causal masked content-retrieval path.
3. A conventional feed-forward expansion.
4. Residual connections and RMS normalization.

Its research claim should currently be evaluated as a hybrid wave-plus-causal-retrieval architecture, not as a complete $O(L\log L)$ replacement for transformer attention.