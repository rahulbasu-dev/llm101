# Building Your Own LLM From Scratch — Complete Guide
## From RNNs to Transformers to a Working NanoLLM on RTX 4080 (12GB)

**Goal:** Webinar-ready understanding + working code in 7 days  
**Hardware:** NVIDIA RTX 4080 12GB VRAM, Windows + WSL2, Ollama already running  
**Toolchain:** Claude Code (WSL2 launch), PyTorch 2.x, CUDA 12.x

---

## PART 1: THE EVOLUTIONARY CHAIN (Days 1–3)

Understanding *why* each architecture was invented matters more than memorizing equations. Each solved a specific failure of its predecessor.

---

### 1.1 Recurrent Neural Networks (RNNs)

**The Problem They Solve:** Feedforward networks treat every input independently — they have no concept of *sequence*. Language is inherently sequential: "bank" means different things after "river" vs. "investment".

**Core Mechanism:**

An RNN maintains a *hidden state* `h_t` that gets updated at every timestep:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `x_t` = input at timestep t (e.g., embedding of word t)
- `h_{t-1}` = hidden state from previous timestep (the "memory")
- `W_hh` = hidden-to-hidden weight matrix (how past influences present)
- `W_xh` = input-to-hidden weight matrix
- `W_hy` = hidden-to-output weight matrix

**What actually happens:** The hidden state `h_t` is a compressed representation of everything the network has seen so far. At each step, it blends the new input with the previous state through a non-linear transformation.

**Training: Backpropagation Through Time (BPTT)**

The network is "unrolled" across timesteps and gradients flow backward through every step. This creates the fundamental problem:

**The Vanishing Gradient Problem:**

When you chain multiplications through `tanh` (which squashes to [-1,1]), gradients shrink exponentially:

```
∂L/∂h_0 = ∂L/∂h_T · ∏(t=1 to T) ∂h_t/∂h_{t-1}
```

Each Jacobian `∂h_t/∂h_{t-1}` has eigenvalues < 1 (due to tanh saturation), so after ~10-20 steps the gradient is effectively zero. The network *cannot learn long-range dependencies*.

**Exploding gradients** (eigenvalues > 1) are the opposite problem — solved trivially by gradient clipping. Vanishing gradients required an architectural solution.

**PyTorch Implementation (from scratch):**

```python
import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev=None):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            h_prev = torch.tanh(
                self.W_xh(x[:, t, :]) + self.W_hh(h_prev) + self.b_h
            )
            outputs.append(self.W_hy(h_prev))

        return torch.stack(outputs, dim=1), h_prev
```

**Key Limitation:** Effective memory window ~10-20 tokens. Useless for real language.

---

### 1.2 Long Short-Term Memory (LSTM)

**The Problem It Solves:** Vanishing gradients in vanilla RNNs. Invented by Hochreiter & Schmidhuber (1997).

**Core Insight:** Instead of forcing information through a multiplicative bottleneck at every step, create a *separate cell state* `C_t` that flows through time with *additive* updates (addition doesn't cause vanishing gradients). Control information flow with *learnable gates*.

**The Three Gates + Cell State:**

```
Forget Gate:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:     i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:      C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell Update:    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:   h_t = o_t ⊙ tanh(C_t)
```

Where `σ` = sigmoid (outputs 0-1, acts as a "valve"), `⊙` = element-wise multiplication.

**What each gate does (intuitively):**
- **Forget gate (f_t):** "What percentage of each dimension of the old cell state should I keep?" A value of 0.0 = completely forget, 1.0 = completely remember.
- **Input gate (i_t):** "What percentage of the new candidate information should I write into the cell state?"
- **Candidate (C̃_t):** The actual new information computed from current input + previous hidden state.
- **Cell update:** This is the critical line — it's *additive*. The gradient flows through `f_t ⊙ C_{t-1}` which is just multiplication by a gate value, not a squashed nonlinearity. If `f_t ≈ 1`, gradient flows unimpeded for hundreds of steps.
- **Output gate (o_t):** "What part of the cell state should I expose as the hidden state for this timestep?"

**Why it works:** The cell state `C_t` acts as a "conveyor belt" — information can flow unchanged across many timesteps as long as the forget gate stays open. The network *learns* when to forget vs. remember.

**Parameter count:** For hidden size `h` and input size `x`, an LSTM has `4 × (h² + h·x + h)` parameters (4× a vanilla RNN because of 4 linear transformations: f, i, C̃, o).

---

### 1.3 Gated Recurrent Unit (GRU)

**The Simplification:** Cho et al. (2014) showed you can merge the forget and input gates into a single "update gate" and eliminate the separate cell state:

```
Update Gate:   z_t = σ(W_z · [h_{t-1}, x_t])
Reset Gate:    r_t = σ(W_r · [h_{t-1}, x_t])
Candidate:     h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])
Output:        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Key insight:** The update gate `z_t` simultaneously controls forgetting (`1 - z_t`) and inputting (`z_t`). Fewer parameters than LSTM (~75%), competitive performance on most tasks.

**When to use which:** GRU trains faster on smaller datasets. LSTM has more representational capacity for complex long-range patterns. In practice, both are now superseded by Transformers for language tasks.

---

### 1.4 The Encoder-Decoder (Seq2Seq) Architecture

**The Problem:** RNN/LSTM process sequences, but what about *sequence-to-sequence* tasks where input and output have different lengths? Translation: "Je suis étudiant" (3 tokens) → "I am a student" (4 tokens).

**Sutskever et al. (2014) Solution:**

```
Encoder: reads input sequence, compresses into fixed-size context vector
Decoder: generates output sequence conditioned on context vector

[x1, x2, x3] → Encoder LSTM → context vector c → Decoder LSTM → [y1, y2, y3, y4]
```

The context vector `c` = final hidden state of the encoder = `h_T^enc`.

**The Bottleneck Problem:** ALL information about the source sentence must squeeze through a single fixed-size vector. For long sentences (30+ words), this causes catastrophic information loss.

**Teacher Forcing:** During training, feed the decoder the *ground truth* previous token instead of its own prediction. Speeds up convergence but causes *exposure bias* (train/inference distribution mismatch).

---

### 1.5 The Attention Mechanism

**The Breakthrough:** Bahdanau et al. (2015) — Instead of compressing the entire source into one vector, let the decoder *look back at all encoder hidden states* at every decoding step.

**Bahdanau (Additive) Attention:**

```
For each decoder timestep t:
1. Score each encoder hidden state:
   e_{t,i} = v^T · tanh(W_1 · s_{t-1} + W_2 · h_i)
   where s_{t-1} = decoder state, h_i = encoder hidden state i

2. Normalize scores to get attention weights:
   α_{t,i} = softmax(e_{t,i})  (sums to 1 across all source positions)

3. Compute context vector as weighted sum:
   c_t = Σ_i α_{t,i} · h_i

4. Use context in decoder:
   s_t = LSTM(s_{t-1}, [y_{t-1}; c_t])
```

**What this means:** At each output step, the model learns a *soft alignment* — which source words are relevant for generating the current target word. When translating "student" it attends strongly to "étudiant".

**Luong (Multiplicative) Attention (2015):**

Simpler scoring: `e_{t,i} = s_t^T · W · h_i` (dot product after linear transform). Faster to compute, works equally well.

**Why attention changed everything:**
1. Solves the bottleneck — decoder accesses ALL encoder states
2. Provides interpretability — attention weights show alignment
3. Enables gradient shortcuts — gradients flow directly from decoder to relevant encoder states
4. Removed the effective sequence length limit of LSTMs

---

## PART 2: THE TRANSFORMER (Days 3–4)

### 2.1 "Attention Is All You Need" — Vaswani et al. (2017)

**The Radical Claim:** Discard recurrence entirely. Use *only* attention to process sequences. This enables:
- **Parallelism:** RNNs must process tokens sequentially (h_t depends on h_{t-1}). Attention computes all positions simultaneously.
- **Constant path length:** Information between any two positions travels through O(1) layers, not O(n) recurrent steps.

**The Transformer Block (the fundamental unit):**

```
Input → [Multi-Head Self-Attention] → Add & LayerNorm → [Feed-Forward Network] → Add & LayerNorm → Output
```

Every modern LLM is a stack of these blocks.

### 2.2 Self-Attention (Scaled Dot-Product Attention)

This is the single most important mechanism in modern AI. Understand this deeply.

**Setup:** Given a sequence of `n` token embeddings, each of dimension `d_model`, we want each token to "attend to" every other token.

**Step 1: Create Q, K, V projections**

For each token embedding `x_i`, create three vectors:
```
q_i = W_Q · x_i    (Query: "What am I looking for?")
k_i = W_K · x_i    (Key: "What do I contain?")
v_i = W_V · x_i    (Value: "What information do I provide?")
```

`W_Q, W_K, W_V` are learned projection matrices of shape `(d_model, d_k)`.

**Database analogy:** Think of a key-value store. The Query says "I need information about X", Keys say "I have information about Y", and when Q and K match (high dot product), the corresponding V is retrieved.

**Step 2: Compute attention scores**

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Broken down:
1. `Q · K^T` = matrix of all pairwise dot products. Shape: `(n, n)`. Entry (i,j) = how much token i should attend to token j.
2. `/ √d_k` = scaling factor. Without this, dot products grow proportionally to `d_k`, pushing softmax into saturation (near-zero gradients). This is the "scaled" part.
3. `softmax` = normalize each row to sum to 1. Now each row is a probability distribution over source positions.
4. `· V` = weighted combination of value vectors. Each token's output is a blend of all tokens' values, weighted by relevance.

**Concrete example with 4 tokens ["The", "cat", "sat", "down"]:**

The attention matrix (after softmax) might look like:
```
         The   cat   sat  down
The    [0.6   0.2   0.1  0.1]   ← "The" attends mostly to itself
cat    [0.1   0.5   0.3  0.1]   ← "cat" attends to itself and "sat"
sat    [0.05  0.4   0.4  0.15]  ← "sat" attends to "cat" (subject) and itself
down   [0.05  0.1   0.5  0.35]  ← "down" attends to "sat" (what went down?)
```

### 2.3 Multi-Head Attention

**Problem:** A single attention function learns one type of relationship. But language has many simultaneous relationships (syntactic, semantic, positional, coreference).

**Solution:** Run `h` parallel attention functions with different learned projections:

```
head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)
MultiHead(X) = Concat(head_1, ..., head_h) · W_O
```

Typically: `d_model = 768`, `h = 12` heads, so each head operates on `d_k = 64` dimensions.

**What different heads learn (empirically observed):**
- Head 3 might learn syntactic dependency (subject→verb)
- Head 7 might learn positional (attend to previous token)
- Head 11 might learn coreference ("it" → "the cat")

`W_O` projects the concatenated heads back to `d_model` dimensions.

### 2.4 Positional Encoding

**The Problem:** Self-attention is *permutation-equivariant* — it has no concept of token order. "Dog bites man" and "Man bites dog" produce identical attention patterns without positional information.

**Vaswani's Solution — Sinusoidal Encoding:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique vector. The frequencies span from `2π` to `20000π`, creating a multi-scale encoding. Nearby positions have similar encodings (enabling generalization), and the model can learn relative positions because `PE(pos+k)` can be expressed as a linear function of `PE(pos)`.

**Modern alternative:** Rotary Position Embeddings (RoPE) — used in LLaMA, Qwen, Mistral. Encodes position directly into Q/K via rotation matrices, enabling better length generalization.

### 2.5 Feed-Forward Network (FFN)

Each transformer block contains a position-wise FFN:

```
FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2
```

Where `W_1`: (d_model, d_ff), typically `d_ff = 4 × d_model`.

**What it does:** Self-attention is essentially a *routing* mechanism — it decides what information to combine. The FFN is where actual *computation* happens. Think of attention as "gathering ingredients" and FFN as "cooking."

**Recent insight (Geva et al., 2021):** FFN layers act as key-value memories. The first matrix `W_1` stores "patterns" (keys), and `W_2` stores corresponding "facts" (values). This is where the LLM stores factual knowledge.

### 2.6 Layer Normalization and Residual Connections

**Residual connections:** `output = LayerNorm(x + Sublayer(x))`

Without residuals, gradients must flow through every transformation, causing degradation in deep networks. Residuals create "gradient highways" — the identity path `x` always provides gradient flow.

**Layer Normalization (vs Batch Norm):**

```
LayerNorm(x) = γ ⊙ (x - μ) / (σ + ε) + β
```

Normalizes across the feature dimension *per example* (not across the batch). This is essential because:
1. Sequence lengths vary (batch norm across positions makes no sense)
2. Works with batch size 1 (inference)
3. Stabilizes training of deep transformers

**Pre-Norm vs Post-Norm:** Original Transformer used Post-Norm. Modern LLMs (GPT-2+, LLaMA) use Pre-Norm (normalize before attention/FFN). Pre-Norm is more stable during training and doesn't require careful learning rate warmup.

### 2.7 The Full Transformer Architecture

**Encoder (BERT-style):**
- Stack of N blocks, each: Multi-Head Self-Attention → FFN
- Bidirectional: every token attends to every other token
- Used for: classification, NER, embedding, understanding tasks

**Decoder (GPT-style):**
- Stack of N blocks, each: *Masked* Multi-Head Self-Attention → FFN
- Causal mask: token at position i can only attend to positions ≤ i
- Used for: text generation, completion, autoregressive tasks

**Encoder-Decoder (Original Transformer, T5, BART):**
- Encoder processes source sequence bidirectionally
- Decoder has masked self-attention + cross-attention to encoder outputs
- Cross-attention: decoder queries attend to encoder keys/values
- Used for: translation, summarization

**The Causal Mask (critical for decoder-only models):**

```
Mask = [[1, 0, 0, 0],    ← token 0 sees only itself
        [1, 1, 0, 0],    ← token 1 sees tokens 0-1
        [1, 1, 1, 0],    ← token 2 sees tokens 0-2
        [1, 1, 1, 1]]    ← token 3 sees tokens 0-3
```

Positions with 0 are set to -∞ before softmax, making them contribute zero attention weight. This ensures the model cannot "cheat" by looking at future tokens during training.

---

## PART 3: PRE-TRAINING PARADIGMS (Day 4)

### 3.1 BERT — Bidirectional Encoder Representations from Transformers

**Architecture:** Encoder-only Transformer (12 layers, 768 hidden, 12 heads = 110M params for BERT-base).

**Pre-training Objectives:**

**1. Masked Language Modeling (MLM):**
- Randomly mask 15% of input tokens
- Of those: 80% replaced with [MASK], 10% random token, 10% unchanged
- Model predicts the original token at masked positions
- Bidirectional context: uses both left and right context

```
Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat   sat on the  mat"
```

**Why the 80/10/10 split:** If always [MASK], the model never sees [MASK] at inference time (distribution mismatch). Random replacement forces robustness. Unchanged forces the model to learn that even unmasked tokens might need to be "predicted."

**2. Next Sentence Prediction (NSP):**
- 50% of the time: sentence B follows sentence A (label: IsNext)
- 50%: sentence B is random (label: NotNext)
- Later shown to be unnecessary (RoBERTa removes it with no performance loss)

**BERT's limitation for generation:** Because BERT is bidirectional, it cannot generate text autoregressively. It's an *understanding* model, not a *generation* model.

### 3.2 GPT — Generative Pre-trained Transformer

**Architecture:** Decoder-only Transformer with causal masking.

**Pre-training Objective: Causal Language Modeling (CLM)**

```
Given: "The cat sat on"
Predict next token at each position:
  Position 0 ("The") → predict "cat"
  Position 1 ("The cat") → predict "sat"
  Position 2 ("The cat sat") → predict "on"
  Position 3 ("The cat sat on") → predict "the"
```

**Loss function:** Cross-entropy averaged over all positions:

```
L = -1/T × Σ_{t=1}^{T} log P(x_t | x_{<t})
```

**Why decoder-only won (for LLMs):**
1. **Simplicity:** One objective, one architecture, one training loop
2. **Emergent abilities:** Scaling causal LMs revealed in-context learning, chain-of-thought, tool use
3. **Unification:** Every NLP task can be framed as text generation
4. **Efficiency:** No separate encoder needed

### 3.3 Key Variants to Know

| Model | Type | Key Innovation |
|-------|------|---------------|
| GPT-2 (2019) | Decoder | Showed zero-shot transfer; 1.5B params |
| T5 (2020) | Enc-Dec | "Text-to-text" — every task is sequence-to-sequence |
| GPT-3 (2020) | Decoder | 175B params; in-context learning; few-shot prompting |
| RoBERTa | Encoder | BERT without NSP, more data, dynamic masking |
| LLaMA (2023) | Decoder | RMSNorm, RoPE, SwiGLU, GQA — efficient open-source |
| Mistral (2023) | Decoder | Sliding window attention, GQA |

---

## PART 4: BUILD YOUR NANOLLM (Days 5–6)

### 4.1 Architecture Decisions for NanoLLM

```
Type:             Decoder-only (GPT-style)
Parameters:       ~15M (fits comfortably in RTX 4080 12GB)
Layers:           6
d_model:          384
Heads:            6
d_ff:             1536 (4 × d_model)
Context length:   256 tokens
Vocab size:       ~10,000 (BPE)
Position encoding: Learned (simplest) or RoPE
Norm:             RMSNorm (Pre-Norm)
Activation:       GELU
Dropout:          0.1
```

**Why these numbers:**
- 6 layers × 384 hidden = enough capacity to learn real language patterns
- 256 context = short but sufficient for demonstrating attention
- ~15M params trains in minutes on RTX 4080, fits entirely in VRAM
- BPE tokenizer handles subword units properly (real LLM behavior)

### 4.2 Project Structure

```
nanollm/
├── tokenizer.py        # BPE tokenizer (train + encode/decode)
├── model.py            # Transformer decoder from scratch
├── dataset.py          # Data loading and batching
├── train.py            # Training loop with mixed precision
├── generate.py         # Text generation with sampling strategies
├── config.py           # All hyperparameters
├── utils.py            # Logging, checkpointing
└── data/
    └── corpus.txt      # Training data
```

### 4.3 Complete Implementation

**config.py:**

```python
from dataclasses import dataclass

@dataclass
class NanoLLMConfig:
    # Model architecture
    vocab_size: int = 10000
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536          # 4 * d_model
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads
```

**model.py — The Full NanoLLM (every component from scratch):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    Used in LLaMA, Mistral, Qwen instead of LayerNorm.
    Simpler: no mean subtraction, no bias. Just scale by RMS."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (Su et al., 2021)
    Encodes position by rotating Q and K vectors.
    Key property: dot product of rotated Q_m and K_n depends only on (m-n),
    giving the model relative position awareness."""

    def __init__(self, d_head: int, max_seq_len: int = 2048):
        super().__init__()
        # Compute frequency bands: theta_i = 10000^(-2i/d)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2).float() / d_head))
        # Precompute position × frequency table
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)  # (max_seq_len, d_head/2)
        self.register_buffer('cos_cached', angles.cos())  # (max_seq_len, d_head/2)
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, x, seq_len):
        # x: (batch, n_heads, seq_len, d_head)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Split x into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Apply rotation: [x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)  # Interleave back


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention with causal mask and RoPE."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Combined QKV projection (more efficient than separate)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len)

        # Causal mask — precomputed lower triangular matrix
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, d_model

        # Project to Q, K, V simultaneously
        qkv = self.qkv_proj(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, T, d_model)

        # Reshape for multi-head: (B, T, d_model) → (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (not V — position info enters only through Q·K scores)
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Compute attention scores
        # (B, nh, T, dh) @ (B, nh, dh, T) → (B, nh, T, T)
        scale = 1.0 / math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: set future positions to -inf
        attn_weights = attn_weights.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )

        # Softmax → attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # (B, nh, T, T) @ (B, nh, T, dh) → (B, nh, T, dh)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (B, nh, T, dh) → (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(attn_output))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with SwiGLU activation.
    SwiGLU (Shazeer, 2020) used in LLaMA/Mistral instead of ReLU/GELU.
    FFN_SwiGLU(x) = (Swish(x·W_gate) ⊙ (x·W_up)) · W_down"""

    def __init__(self, config):
        super().__init__()
        # SwiGLU uses 2/3 of the FFN dim for gate and up, to keep param count equal
        hidden_dim = int(2 * config.d_ff / 3)
        # Round to nearest multiple of 8 for GPU efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: swish(gate) * up, then project down
        gate = F.silu(self.gate_proj(x))  # silu = swish = x * sigmoid(x)
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with Pre-Norm."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Pre-Norm: normalize BEFORE the sublayer, add residual AFTER
        x = x + self.attn(self.norm1(x))    # Residual + attention
        x = x + self.ffn(self.norm2(x))     # Residual + FFN
        return x


class NanoLLM(nn.Module):
    """Complete GPT-style language model."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding: vocab_size → d_model
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm and output projection
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token_emb and lm_head weights
        # (Press & Wolf, 2017) — reduces params and improves performance
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoLLM: {n_params / 1e6:.2f}M parameters")

    def _init_weights(self, module):
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (batch, seq_len) — token indices
        targets: (batch, seq_len) — target token indices (shifted by 1)
        Returns: logits, loss
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence {T} > max {self.config.max_seq_len}"

        # Token embeddings (no separate positional embedding — RoPE handles it)
        x = self.dropout(self.token_emb(idx))  # (B, T, d_model)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm_f(x)

        if targets is not None:
            # Training: compute logits and loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Padding token
            )
            return logits, loss
        else:
            # Inference: only compute logits for last position (efficient)
            logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.9):
        """Autoregressive generation with top-k and nucleus (top-p) sampling."""
        for _ in range(max_new_tokens):
            # Crop to max context length
            idx_crop = idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_crop)  # (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering: keep only top k logits
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above top_p
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                # Scatter back
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx
```

**train.py — Training Loop with Mixed Precision:**

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os

from config import NanoLLMConfig
from model import NanoLLM
from dataset import TextDataset, create_dataloader
from tokenizer import BPETokenizer

def train(config: NanoLLMConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 1. Build tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    if not os.path.exists('tokenizer.model'):
        print("Training tokenizer...")
        tokenizer.train('data/corpus.txt')
        tokenizer.save('tokenizer.model')
    else:
        tokenizer.load('tokenizer.model')

    # 2. Create dataset
    dataset = TextDataset('data/corpus.txt', tokenizer, config.max_seq_len)
    dataloader = create_dataloader(dataset, config.batch_size)
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # 3. Create model
    model = NanoLLM(config).to(device)

    # 4. Optimizer: AdamW with weight decay
    # Separate params: apply weight decay only to weight matrices, not biases/norms
    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2]  # Weight matrices
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2]  # Biases, LayerNorm

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))

    # 5. Learning rate scheduler with warmup
    total_steps = len(dataloader) * config.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 6. Mixed precision training (bf16 on RTX 4080)
    scaler = GradScaler()
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using {'bf16' if use_bf16 else 'fp16'} mixed precision")

    # 7. Training loop
    model.train()
    global_step = 0

    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=amp_dtype):
                logits, loss = model(input_ids, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # LR warmup
            if global_step < config.warmup_steps:
                lr = config.learning_rate * (global_step + 1) / config.warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            else:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                tokens_per_sec = (batch_idx + 1) * config.batch_size * config.max_seq_len / (time.time() - t0)
                print(f"  Epoch {epoch+1} | Step {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                      f"{tokens_per_sec:.0f} tok/s")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{config.max_epochs} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        # Generate sample
        model.eval()
        prompt = tokenizer.encode("The meaning of life is")
        prompt_tensor = torch.tensor([prompt], device=device)
        output = model.generate(prompt_tensor, max_new_tokens=50)
        generated_text = tokenizer.decode(output[0].tolist())
        print(f"  Sample: {generated_text}")
        model.train()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config,
        }, f'checkpoint_epoch{epoch+1}.pt')

    print("Training complete!")
    return model


if __name__ == '__main__':
    config = NanoLLMConfig()
    train(config)
```

**tokenizer.py — Simple BPE Tokenizer:**

```python
"""Minimal Byte-Pair Encoding tokenizer.
For production, use SentencePiece or tiktoken. This is for understanding BPE."""

import re
from collections import Counter
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[int, int]] = []
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}

        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def train(self, filepath: str):
        """Train BPE tokenizer on a text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Start with byte-level vocabulary (256 base tokens + 4 special)
        tokens = list(text.encode('utf-8'))
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Add special tokens
        self.vocab[256] = b'<PAD>'
        self.vocab[257] = b'<BOS>'
        self.vocab[258] = b'<EOS>'
        self.vocab[259] = b'<UNK>'

        num_merges = self.vocab_size - 260  # Remaining vocab slots for merges

        for i in range(num_merges):
            # Count all adjacent pairs
            pairs = Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j + 1])] += 1

            if not pairs:
                break

            # Find most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            new_token_id = 260 + i

            # Create new token by concatenating the pair
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges.append(best_pair)

            # Replace all occurrences of the pair with the new token
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j+1] == best_pair[1]:
                    new_tokens.append(new_token_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            if (i + 1) % 500 == 0:
                print(f"  BPE merge {i+1}/{num_merges}: "
                      f"'{self.vocab[best_pair[0]]}' + '{self.vocab[best_pair[1]]}' "
                      f"→ '{self.vocab[new_token_id]}' (freq: {pairs[best_pair]})")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer trained: {len(self.vocab)} tokens")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = list(text.encode('utf-8'))

        # Apply merges in order
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    # Replace pair with merged token
                    merged_id = 260 + self.merges.index(pair)
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.bos_token_id] + tokens + [self.eos_token_id]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_sequence = b''
        for tid in token_ids:
            if tid in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            if tid in self.vocab:
                byte_sequence += self.vocab[tid]
        return byte_sequence.decode('utf-8', errors='replace')

    def save(self, path: str):
        import json
        data = {
            'vocab_size': self.vocab_size,
            'merges': self.merges,
            'vocab': {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data['merges']]
        self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
```

**dataset.py:**

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """Converts text corpus into overlapping sequences for causal LM training."""

    def __init__(self, filepath, tokenizer, max_seq_len, stride=None):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokens = tokenizer.encode(text)
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2  # 50% overlap by default

        # Create sliding window samples
        self.samples = []
        for i in range(0, len(self.tokens) - max_seq_len - 1, self.stride):
            input_ids = self.tokens[i : i + max_seq_len]
            targets = self.tokens[i + 1 : i + max_seq_len + 1]
            self.samples.append((input_ids, targets))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, targets = self.samples[idx]
        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(targets, dtype=torch.long)


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
```

---

## PART 5: WEBINAR NARRATIVE ARC (Day 7)

### Recommended Talk Structure (45–60 min)

**1. "The Sequence Problem" (5 min)**
- Feedforward nets cannot handle sequences
- Demo: why word order matters

**2. "Memory: RNN → LSTM → GRU" (10 min)**
- Show the vanishing gradient with a graph
- LSTM gates as a solution
- Key limitation: sequential processing bottleneck

**3. "Attention: The Breakthrough" (10 min)**
- Encoder-decoder bottleneck problem
- Bahdanau attention visualisation (alignment matrix)
- "What if attention is ALL we need?"

**4. "The Transformer" (15 min)**
- Self-attention walkthrough with 4-word example
- Multi-head attention intuition
- Positional encoding necessity
- The full block: attention → FFN → residual → norm
- Why it scales: parallelism + constant path length

**5. "BERT vs GPT: Two Philosophies" (5 min)**
- Bidirectional understanding vs autoregressive generation
- Why decoder-only won for LLMs

**6. "Live Demo: NanoLLM" (10 min)**
- Show the code structure
- Train on a small corpus LIVE (takes 2-3 minutes on RTX 4080)
- Show generation improving across epochs
- Show attention patterns

**7. "From Nano to GPT-4" (5 min)**
- What changes at scale: more layers, more data, RLHF, MoE
- What stays the same: the transformer block

### Key Visualisations to Prepare

1. **RNN unrolled diagram** showing gradient path
2. **LSTM gate diagram** with data flow
3. **Attention heatmap** (4×4 matrix with colors)
4. **Transformer block diagram** (the canonical one)
5. **Training loss curve** from your NanoLLM run
6. **Generated text samples** at epoch 1 vs epoch 10

---

## PART 6: GETTING STARTED RIGHT NOW

### Step 1: Set up the environment (Claude Code via WSL2)

```bash
# In WSL2 (your preferred launch path)
mkdir -p ~/nanollm/data
cd ~/nanollm

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm matplotlib

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 2: Get training data

For a nano model, use a small but high-quality corpus:

```bash
# Option A: TinyShakespeare (~1MB, classic demo)
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/corpus.txt

# Option B: WikiText-2 (~10MB, real language)
# Download from HuggingFace datasets

# Option C: Your own content (lectures, papers, etc.)
```

### Step 3: Build incrementally with Claude Code

```
claude "Read model.py. First implement just the RMSNorm and test it.
       Then add CausalSelfAttention and verify shapes.
       Then add the full NanoLLM and run a forward pass with random data."
```

### Step 4: Train and iterate

```bash
python train.py
# Expected on RTX 4080: ~50,000 tokens/sec, full training in 5-15 min
# Watch loss drop from ~9.2 (log(vocab_size)) to ~3-4
```

### Step 5: Analyse what your model learned

```python
# Visualise attention patterns
import matplotlib.pyplot as plt

def plot_attention(model, tokenizer, text, layer=0, head=0):
    tokens = tokenizer.encode(text)
    x = torch.tensor([tokens]).cuda()
    
    # Hook to capture attention weights
    attn_weights = []
    def hook_fn(module, input, output):
        # Capture the attention weight matrix
        pass  # Implementation depends on returning attn from forward()
    
    # Plot as heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(weights, cmap='viridis')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels([tokenizer.decode([t]) for t in tokens], rotation=45)
    ax.set_yticklabels([tokenizer.decode([t]) for t in tokens])
    plt.title(f"Layer {layer}, Head {head}")
    plt.savefig("attention_map.png", dpi=150, bbox_inches='tight')
```

---

## APPENDIX: CONCEPT QUICK REFERENCE

| Concept | One-Line Explanation |
|---------|---------------------|
| **Embedding** | Lookup table mapping discrete token IDs to dense vectors |
| **Self-Attention** | Every token computes a weighted sum of all tokens' representations |
| **Causal Mask** | Prevents attending to future tokens (enables autoregressive generation) |
| **Multi-Head** | Parallel attention functions learning different relationship types |
| **RoPE** | Encodes position via rotation; enables relative position awareness |
| **RMSNorm** | Normalises by root-mean-square; simpler than LayerNorm |
| **SwiGLU** | Gated activation in FFN; Swish(xW₁) ⊙ xW₂ |
| **Residual Connection** | x + f(x); creates gradient highways for deep networks |
| **Weight Tying** | Share embedding and output projection matrices |
| **BPE Tokenization** | Iteratively merge most frequent byte pairs into subword tokens |
| **Cross-Entropy Loss** | -log(probability of correct next token); standard LM loss |
| **AdamW** | Adam with decoupled weight decay; standard LLM optimizer |
| **Mixed Precision** | Use bf16/fp16 for forward/backward, fp32 for optimizer state |
| **Gradient Clipping** | Cap gradient norm to prevent training instability |
| **Cosine LR Schedule** | Decay learning rate following a cosine curve |
| **Temperature** | Scales logits before softmax; lower = more deterministic |
| **Top-k Sampling** | Only sample from the k highest-probability tokens |
| **Top-p (Nucleus)** | Only sample from the smallest set of tokens with cumulative prob ≥ p |

---

## APPENDIX: SCALING FROM NANO TO REAL

| Dimension | NanoLLM (Yours) | LLaMA-2 7B | GPT-4 (estimated) |
|-----------|----------------|------------|-------------------|
| Parameters | 15M | 7B | ~1.8T (MoE) |
| Layers | 6 | 32 | 120+ |
| d_model | 384 | 4096 | 12288+ |
| Heads | 6 | 32 | 96+ |
| Context | 256 | 4096 | 128K |
| Training tokens | ~1M | 2T | 13T+ |
| Training compute | 1 GPU-minute | 184K GPU-hours | ~25K A100-years |
| Key additions | — | GQA, RoPE | MoE, RLHF, tool use |

**What changes at scale:**
- Grouped Query Attention (GQA): share K/V across groups of heads → less memory
- Mixture of Experts (MoE): activate only a subset of FFN params per token
- RLHF/RLAIF: align model behavior with human preferences
- Context extension: ALiBi, YaRN, Dynamic NTK for longer sequences
- Infrastructure: FSDP, tensor/pipeline parallelism, ZeRO optimiser

**What stays exactly the same:**
- The transformer block structure
- Self-attention mechanism
- Pre-Norm + residual connections
- Autoregressive training objective
- BPE tokenization family
