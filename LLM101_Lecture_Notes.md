# LLM101 — Speaker Lecture Notes

**Read-as-is verbatim notes for presenting the Jupyter notebook to a class of students who are encountering these concepts for the first time.**

Each section below corresponds to a cell in `LLM101_From_Scratch.ipynb`. Read the speaker notes aloud, then run the cell. Pause for questions at the marked points.

---

## Cell 0 — Title (Markdown)

> Welcome everyone. Today we are going to build a language model from scratch. Not use one — build one. Every single piece, from the tokenizer to the attention mechanism to the training loop. By the end, you will understand exactly how models like ChatGPT, LLaMA, and Claude work inside.
>
> Our model is about 15 million parameters. That sounds big, but GPT-4 is rumored to be over a trillion. Ours is tiny by comparison — small enough to train on a single GPU in a few minutes, but large enough to actually learn to write Shakespeare.
>
> The architecture we are building is the same one used in production. RMSNorm, Rotary Position Embeddings, SwiGLU activation, causal self-attention. These are not toy simplifications — they are what LLaMA and Mistral actually use, just scaled down.

---

## Cell 1 — Section 0 Header (Markdown)

> Before we write any model code, we need to know what hardware we are running on. Deep learning is computationally expensive. A GPU can be 10 to 100 times faster than a CPU for the matrix multiplications that dominate neural networks.
>
> This cell will detect whether we have an NVIDIA GPU available. If we do, great — training will be fast. If not, we will switch to CPU mode. Everything still works, it is just slower.

---

## Cell 2 — Setup & GPU Detection (Code)

> *Run the cell.*
>
> Look at the output. If you see "GPU detected" with a name like "Tesla T4" or "L4", you are good to go. The VRAM number tells you how much GPU memory you have — our 15-million-parameter model needs less than 2 GB, so even a free T4 with 16 GB is plenty.
>
> If you see "No CUDA GPU detected," that is okay. We set an environment variable called NANOLLM_ALLOW_CPU that tells our code to run on CPU instead of crashing.
>
> Notice we also print the PyTorch version and working directory. These are basic sanity checks — if something goes wrong later, these are the first things to verify.

---

## Cell 3 — Imports (Code)

> *Run the cell.*
>
> This cell imports everything we need from the LLM101 project. The key thing to notice is that we are importing real, complete implementations — BPETokenizer, NanoLLM, ForwardCapture — not simplified teaching stubs. The code you see running in this notebook is the same code that trains and runs the model from the command line.
>
> If this cell fails with an import error, the project files are not on the Python path. Go back and re-run the setup cell.

---

## Cell 4 — Section 1 Header (Markdown)

> Let me explain something important about how real machine learning projects are organized. Every model has dozens of numbers that control its behavior — how wide the layers are, how many layers, how fast it learns, how long it trains. These are called hyperparameters.
>
> In a well-organized project, ALL of these live in one place. Our NanoLLMConfig dataclass is that one place. Every other file in the project imports from it. That means if you want to change the model size or learning rate, you change it in one spot and everything else adapts.

---

## Cell 5 — Configuration Table (Code)

> *Run the cell.*
>
> Look at this table carefully. These numbers define our entire model.
>
> **d_model equals 384.** This is the hidden dimension — every token in our model is represented as a vector of 384 numbers. Think of it as 384 different features that describe each word's meaning and position in context.
>
> **n_layers equals 6.** We stack 6 transformer blocks on top of each other. Each one refines the representation. Layer 0 might learn basic patterns like "a noun usually follows 'the'." Layer 5 might learn deeper patterns like "this sentence is a question."
>
> **n_heads equals 6.** Inside each layer, we have 6 attention heads working in parallel. Each one learns to focus on different relationships between words. One head might track subject-verb agreement, another might attend to nearby words.
>
> **d_head equals 64.** That is 384 divided by 6. Each head gets its own 64-dimensional subspace to work in. 64 is actually the standard head size used all the way from GPT-2 to LLaMA-3.
>
> **max_seq_len equals 256.** This is our context window — the model can only see 256 tokens at a time. ChatGPT's is 128,000. Ours is small, but sufficient for learning patterns in Shakespeare.
>
> *Pause for questions.*

---

## Cell 6 — Section 2 Header (Markdown)

> Now here is a fundamental question: how do you turn text into numbers? Computers do not understand words. They understand numbers. So we need a system to convert "To be or not to be" into a sequence of integers that the model can process.
>
> The answer is a tokenizer, and we are using Byte Pair Encoding — BPE for short. This is the exact same algorithm used by GPT-2, GPT-3, and LLaMA. Here is how it works.
>
> We start with individual bytes. Every character in every language can be represented as one or more bytes. The letter 'A' is byte 65. The space character is byte 32. That gives us a base vocabulary of 256 tokens — one per possible byte value.
>
> Then we look at the training text and ask: which pair of adjacent tokens appears most often? Maybe 't' and 'h' appear together thousands of times. So we merge them into a new token 'th'. Now we ask again: what is the most frequent pair? Maybe 'th' and 'e' — so we get 'the'. We keep merging until we reach our target vocabulary size of 4,096.
>
> The beauty of byte-level BPE is that it can handle ANY text. Chinese characters, emoji, code, binary data — there is never an "unknown token" problem because every possible byte is already in the vocabulary.

---

## Cell 7 — Download Corpus (Code)

> *Run the cell.*
>
> We are using TinyShakespeare — about 1.1 million characters of Shakespeare's plays. It is a classic dataset for small language model experiments. You can see the first 200 characters — it starts with "First Citizen" from Coriolanus.
>
> In the real world, models like LLaMA train on trillions of tokens from the internet. We are using 1 million characters, which will compress to about 280,000 tokens. Small, but enough to learn the style and vocabulary of Shakespeare.

---

## Cell 8 — Train Tokenizer (Code)

> *Run the cell.*
>
> Watch the output. If the tokenizer was already trained and saved, it loads instantly. If not, you will see it training — it prints the first few merges so you can see what it is learning. The most common byte pairs in Shakespeare get merged first.
>
> The final vocab size should be around 4,096 — that is 4 special tokens, plus 256 byte tokens, plus about 3,836 learned merges.

---

## Cell 9 — Encode/Decode Demo (Code)

> *Run the cell.*
>
> This is the key test: can we convert text to numbers and back without losing anything? Look at the output. The original text goes in, we get a list of token IDs, and when we decode those IDs back to text, we get the exact same string. That round-trip match is True.
>
> Now look at the token breakdown. Each token is labeled as BYTE or MERGE. The MERGE tokens represent common sequences that BPE learned — things like "the" or "ing" or " to". Notice how common words become single tokens, while rare character combinations stay as individual bytes.
>
> This is compression in action. Instead of one token per character, common patterns get their own tokens, so the model sees fewer, more meaningful units.

---

## Cell 10 — Vocabulary Visualization (Code)

> *Run the cell.*
>
> Two charts here. On the left, the vocabulary composition — the 4 special tokens are a tiny sliver, the 256 byte tokens are a small bar, and the bulk is the 3,836 learned merges. That is where the intelligence of the tokenizer lives.
>
> On the right, compression ratios. Shakespeare text compresses about 3.5 to 4 times — meaning 4 bytes of text become roughly 1 token. Random character sequences like "abcdefghijklmnop" barely compress at all because BPE never saw those patterns together.
>
> This compression ratio matters for efficiency. Fewer tokens means fewer computations, which means faster training and inference.
>
> *Pause for questions about tokenization.*

---

## Cell 11 — Section 3 Header (Markdown)

> Now that we can convert text to numbers, we need to create training examples. For a language model, the training task is simple: given some tokens, predict the next one.
>
> We do this by creating overlapping sliding windows across the tokenized corpus. Each window gives us an input sequence and a target sequence, where the target is just the input shifted right by one position.
>
> Look at the diagram in the markdown. Window 1 takes tokens A, B, C, D as input and B, C, D, E as targets. At position 0, the model sees A and should predict B. At position 1, it sees A and B and should predict C. And so on.
>
> The windows overlap by 50 percent. This means the model sees every token boundary during training, which helps it learn transitions between windows.

---

## Cell 12 — Tokenize & Split (Code)

> *Run the cell.*
>
> We tokenize the entire corpus into one long list of integers, then split it 90/10 for training and validation. The split is sequential, not random — we take the first 90 percent for training and the last 10 percent for validation. Why sequential? Because if we shuffled, overlapping windows would leak information between train and val sets.
>
> Notice the compression ratio printed here — that tells you how efficiently our tokenizer compressed the Shakespeare text.

---

## Cell 13 — Sliding Window Visualization (Code)

> *Run the cell.*
>
> Two visualizations here. The top one shows the input-target shift for one training sample. Blue boxes are input tokens, red boxes are targets, and the arrows show: at each position, the model's job is to predict the token one step to the right.
>
> The bottom one shows how the sliding windows overlap. Each colored bar is one training sample. They overlap by half their length, so every token transition gets covered.

---

## Cell 14 — Section 4 Header (Markdown)

> Now we get to the heart of the matter — the model architecture. We are going to build it from the bottom up, piece by piece, starting with the smallest components and assembling them into the full transformer.
>
> Look at the tree structure in the markdown. At the top is NanoLLM, our complete model. It contains an embedding layer, 6 transformer blocks, a final normalization, and an output head. Each transformer block contains attention and a feed-forward network.
>
> We will build each piece, run dummy data through it to see the shapes, and then assemble them. By the end you will have a complete mental model of how data flows through a transformer.

---

## Cell 15 — RMSNorm (Code)

> *Run the cell.*
>
> RMSNorm is our first building block. Normalization is critical in deep networks — without it, the numbers flowing through the layers can grow huge or shrink to zero, making training unstable.
>
> The formula is simple: divide each vector by its root-mean-square, then multiply by a learnable scale parameter. Compare this to LayerNorm, which also subtracts the mean and adds a bias. RMSNorm drops both of those — simpler, faster, and works just as well.
>
> Notice the parameter count: RMSNorm has 384 parameters (one scale value per dimension), while LayerNorm has 768 (scale plus bias). Every production LLM since LLaMA-1 uses RMSNorm.

---

## Cell 16 — RoPE Header (Markdown)

> Position encoding is one of the most interesting problems in transformer design. The attention mechanism, by itself, has no concept of word order. "The cat sat on the mat" and "mat the on sat cat the" would look identical to it.
>
> We need to inject position information somehow. The original transformer used fixed sinusoidal patterns added to the embeddings. GPT-2 used learned position embeddings. But modern models use something more elegant: Rotary Position Embeddings, or RoPE.
>
> RoPE works by rotating the query and key vectors based on their position. The key insight is that when you take the dot product of two rotated vectors, the result depends on the difference in their rotation angles — which is the relative distance between the two tokens. So we encode absolute position but the attention naturally captures relative position. Brilliant design.

---

## Cell 17 — RoPE Visualization (Code)

> *Run the cell.*
>
> Three plots. The left two show the cosine and sine tables that RoPE uses. Each row is a position (0 to 63), each column is a frequency band. Notice how the leftmost columns change slowly — those are the low-frequency bands that capture long-range relationships. The rightmost columns oscillate rapidly — those capture local, position-by-position patterns.
>
> The rightmost plot is my favorite. It shows a unit vector being rotated to different positions. Position 0 points right. Position 2 has rotated slightly. Position 32 has rotated much further. This IS the position encoding — it is literally rotation.

---

## Cell 18 — Attention Header (Markdown)

> Self-attention is the core mechanism that makes transformers work. Let me walk through it step by step.
>
> Each token generates three vectors: a Query (what am I looking for?), a Key (what do I contain?), and a Value (what information do I carry?). The Query of one token is compared against the Keys of all other tokens using a dot product. High dot product means "these tokens are relevant to each other."
>
> Those dot products become attention weights after softmax — a probability distribution over which tokens to pay attention to. Then we take a weighted sum of the Value vectors. The result: each token becomes a mixture of information from all the tokens it attended to.
>
> The "causal" part means each token can only attend to tokens that came before it. Token 5 can see tokens 0 through 5, but not token 6. This is what makes it a language model — it cannot cheat by looking at the future.

---

## Cell 19 — Attention Visualization (Code)

> *Run the cell.*
>
> Look at the causal mask. It is a lower triangular matrix of ones and zeros. Position 0 can only see itself. Position 1 can see positions 0 and 1. Position 11 can see all 12 positions. The upper triangle is all zeros — that is the future, which is forbidden.
>
> During the forward pass, we fill those zero positions with negative infinity. When softmax sees negative infinity, it outputs zero. So the model literally cannot attend to future tokens. This is how we enforce the autoregressive property.
>
> *Pause for questions about attention.*

---

## Cell 20 — SwiGLU Header (Markdown)

> After attention decides WHICH tokens to mix together, the feed-forward network computes ON the mixed result. If attention is the routing mechanism — deciding where information flows — the FFN is the computation engine where the model stores and retrieves factual knowledge.
>
> We use SwiGLU, which has a gating mechanism. Think of it as a learned filter. The gate projection decides WHAT information to let through. The up projection expands the representation to a higher dimension. They get multiplied together (element-wise), then the down projection compresses back to the original dimension.
>
> The SiLU activation — x times sigmoid of x — is smoother than ReLU. It does not have the "dead neuron" problem where negative values get permanently zeroed out.

---

## Cell 21 — SwiGLU Demo (Code)

> *Run the cell.*
>
> Notice the dimensions. Input is 384, it expands to 1,024 in the hidden layer, then back to 384. Why 1,024 and not 1,536? Because SwiGLU has three matrices instead of two. To keep the total parameter count the same as a standard two-matrix FFN, we use 2/3 of the nominal d_ff, then round up to a multiple of 8 for GPU efficiency.

---

## Cell 22 — TransformerBlock Header (Markdown)

> Now we assemble attention and FFN into a complete transformer block. The key design choice is Pre-Norm — we normalize BEFORE each sublayer, not after. And we add the input back via a residual connection AFTER the sublayer.
>
> This ordering matters enormously for training stability. In the original transformer (Post-Norm), gradients had to flow through the normalization layers during backpropagation, which could cause instability. Pre-Norm lets gradients flow directly through the residual path, making training much more stable for deep networks. Every modern LLM uses Pre-Norm.

---

## Cell 23 — TransformerBlock + Full Model (Code)

> *Run the cell.*
>
> First we see one block: input shape in, same shape out. That is important — the residual connection requires matching dimensions, so the block is a shape-preserving transformation.
>
> Then we build the full model. Look at the parameter breakdown. The embedding and lm_head share the same weight matrix — that is weight tying. The embedding matrix converts token IDs to vectors, and the same matrix (transposed) converts vectors back to token probabilities. This saves parameters and actually improves quality because the input and output representations are forced to be consistent.
>
> The total parameter count should be around 12 million. With the 4,096-token vocabulary, the tied embedding adds about 1.5 million, bringing us to roughly 15 million total.

---

## Cell 24 — Forward Pass Sanity Check (Code)

> *Run the cell.*
>
> We feed random token IDs through the model and get back logits and a loss. The loss should be close to ln(4096) which is about 8.3. Why? Because with random weights, the model assigns roughly equal probability to all 4,096 tokens. The cross-entropy loss of a uniform distribution over N items is ln(N). If the loss is much higher or lower than 8.3, something is wrong.
>
> This is a quick sanity check you should always run after building a model. If the initial loss matches the theoretical random value, the architecture is probably wired up correctly.

---

## Cell 25 — Section 5 Header (Markdown)

> Now for something really cool. We are going to trace a real prompt through the model and visualize what happens at every stage. Not the numbers — the patterns.
>
> We use a technique called forward hooks. PyTorch lets you attach callback functions to any layer that get called during the forward pass. Our ForwardCapture class hooks into the attention and FFN layers to capture the intermediate tensors — the Q, K, V matrices, the attention scores before and after masking, the softmax weights, the FFN input and output.
>
> The beauty of hooks is that we do not modify the model code at all. The model runs exactly as normal; we just observe.

---

## Cell 26 — Forward Capture Setup (Code)

> *Run the cell.*
>
> We feed the prompt "The cat sat on the" through the model with hooks attached to layer 0, head 0. Look at the captured tensors: embeddings, q, k, v, scores_raw, scores_masked, attn_weights, ffn_input, ffn_output. We have captured the complete internal state of the model for this prompt.

---

## Cell 27 — Embedding Visualization (Code)

> *Run the cell.*
>
> This heatmap shows the embedding vectors for each token. Each row is a token, each column is one of the 384 embedding dimensions. Red means positive, blue means negative.
>
> Notice that different tokens have different patterns. "The" and "the" (with and without capitalization) will have different embeddings. "cat" and "sat" have their own distinct patterns. These are the raw lookup vectors — the model has not processed them yet, it is just looking up each token's representation in a table.

---

## Cell 28 — Q, K, V Visualization (Code)

> *Run the cell.*
>
> Three heatmaps side by side: Query, Key, and Value for one attention head. These are the token representations AFTER they have been projected through the QKV linear layer and rotated by RoPE.
>
> The Query vectors represent "what am I looking for?" The Key vectors represent "what do I contain?" When we take the dot product of a Query with all Keys, we get the attention scores — measuring how relevant each token is to the querying token.
>
> The Value vectors are what actually gets mixed together. After attention decides the weights, it is the Values that carry the information forward.

---

## Cell 29 — Attention Scores Visualization (Code)

> *Run the cell.*
>
> This is the most important visualization in the entire notebook. Three heatmaps showing the attention mechanism step by step.
>
> **Left: Raw scores.** This is Q dot K-transpose divided by square root of 64. Notice it is a full matrix — every token has a score for every other token, including future tokens. Some scores are positive (these tokens are relevant), some are negative (irrelevant).
>
> **Middle: After causal mask.** The upper triangle is now grey — those are positions where we put negative infinity. The model cannot attend to future tokens. This is the fundamental constraint that makes autoregressive generation work.
>
> **Right: After softmax.** Now each row is a probability distribution that sums to 1. Bright cells mean strong attention. Look for patterns — does the last token attend strongly to any particular earlier token? Do certain tokens attract a lot of attention from everything after them?
>
> This is literally what attention looks like inside the model. Every time ChatGPT generates a token, it computes matrices just like this one.
>
> *Pause for questions. This is a key conceptual moment.*

---

## Cell 30 — FFN Visualization (Code)

> *Run the cell.*
>
> Three heatmaps showing what the FFN does. Left is the input (after attention), middle is the delta (what the FFN adds), right is the output.
>
> The delta — the middle panel — is the interesting one. It shows what new information the FFN injects at each position. Research suggests this is where factual knowledge lives. When a model "knows" that Paris is the capital of France, that knowledge is encoded in the FFN weight matrices.

---

## Cell 31 — Section 6 Header (Markdown)

> Time to train the model. We have the data, we have the architecture, now we need to optimize the weights so the model actually learns to predict Shakespeare.
>
> The training loop has several important components. AdamW is our optimizer — it is the standard for transformer training. The learning rate follows a schedule: it starts at zero, ramps up linearly during a warmup phase, then decays following a cosine curve down to 10 percent of the peak. This schedule prevents early instability and gives the model a gentle landing at the end.
>
> Mixed precision means we use 16-bit floating point for most computations instead of 32-bit. This halves memory usage and doubles throughput on modern GPUs, with negligible impact on quality.

---

## Cell 32 — LR Schedule Visualization (Code)

> *Run the cell.*
>
> This is the learning rate schedule visualized. See the linear ramp during warmup — the red dashed line marks where warmup ends. Then the smooth cosine decay. The peak is 3e-4, which is the standard for small transformers with AdamW.
>
> Why warmup? At the very beginning of training, the model's gradients are noisy and can be very large. If we start with a high learning rate, those large gradients cause huge parameter updates that destabilize training. By starting small and ramping up, we give the model time to find a reasonable region of parameter space before increasing the step size.

---

## Cell 33 — Training (Code)

> *Run the cell.*
>
> This will take a minute or two on GPU, longer on CPU. Watch the output as it trains.
>
> The loss starts around 8 — remember, that is the random baseline. As training progresses, you should see it drop. After one epoch, it might be around 6 or 7. After three epochs, hopefully under 5.
>
> The perplexity is just e raised to the power of the loss. A perplexity of 100 means the model is as confused as if it were choosing randomly between 100 equally likely tokens. We want this to go down.
>
> Also watch the generation samples that print after each epoch. In early epochs, the model outputs garbage. By epoch 3, it should start producing something that looks vaguely like English, maybe even vaguely like Shakespeare.

---

## Cell 34 — Loss Curve Plot (Code)

> *Run the cell.*
>
> Two charts. On the left, the training curve — the light blue line is the per-step loss (noisy) and the dark blue circles are the epoch averages. The red squares are the validation loss.
>
> Pay attention to the gap between train and val. If they are close, the model is generalizing well. If train loss keeps dropping but val loss starts rising, that is overfitting — the model is memorizing the training data instead of learning patterns. With only 3 epochs, we probably have not overfit yet, but with 15 or 20 epochs you would start to see that gap open up.
>
> On the right, perplexity per epoch. This is the same information in a more interpretable scale. A drop from perplexity 2000 to perplexity 100 means the model went from choosing between 2000 equally likely tokens to choosing between 100. That is real learning.

---

## Cell 35 — Section 7 Header (Markdown)

> Now the fun part — using the model we just trained to generate text. The model predicts one token at a time. We take its prediction, append it to the prompt, and feed the extended sequence back in. Repeat until we have enough text. This is called autoregressive generation.
>
> We have two generation methods. The standard one recomputes everything from scratch at each step. The fast one uses a KV cache — we will explore that in Section 9. For now, just know that the fast one gives the same output but much more efficiently.
>
> The sampling parameters control how we pick the next token from the model's probability distribution. Temperature scales the probabilities — low temperature makes the model very confident and repetitive, high temperature makes it random and creative. Top-k limits the choices to the k most likely tokens. Top-p, also called nucleus sampling, dynamically adjusts the cutoff.

---

## Cell 36 — Load Checkpoint (Code)

> *Run the cell.*
>
> We load the best checkpoint — the one saved at the epoch with the lowest validation loss. If no checkpoint exists (because you skipped training), we fall back to the model with random weights. Random weights will generate gibberish, which is actually instructive — it shows you what the model looks like before it has learned anything.

---

## Cell 37 — Generation Comparison (Code)

> *Run the cell.*
>
> Four prompts, each generated with both methods. Look at the speed comparison — generate_fast with the KV cache should be several times faster than the standard generate. The outputs will differ because sampling is random, but the quality should be similar.
>
> With only 3 epochs of training, the output will be somewhat coherent but still rough. If you train for 15 epochs, the Shakespeare-like quality improves dramatically. The model is learning from only 1 million characters, so it will never be perfect, but it captures the rhythm, vocabulary, and structure of the plays remarkably well.

---

## Cell 38 — Sampling Parameters Demo (Code)

> *Run the cell.*
>
> Same prompt, five different sampling settings. This is really instructive.
>
> **Greedy** — temperature near zero, top_k equals 1. The model always picks the single most likely token. The output is deterministic and often repetitive — it gets stuck in loops.
>
> **Conservative** — low temperature, limited top_k. More varied than greedy but still quite predictable.
>
> **Default** — our standard settings. A good balance between coherence and creativity.
>
> **Creative** — higher temperature, wider top_k. More surprising word choices, occasionally brilliant, occasionally nonsensical.
>
> **Wild** — temperature 2, no top_k filtering. Almost random. The model is considering even very unlikely tokens. The output is usually incoherent but can produce unexpected creative juxtapositions.
>
> This demonstrates a fundamental trade-off in language models: predictability versus creativity. There is no universally "best" setting — it depends on the application.
>
> *Pause for questions.*

---

## Cell 39 — Section 8 Header (Markdown)

> Let us go back inside the model and look at what all 36 attention heads — 6 layers times 6 heads — are doing. Up until now we only looked at one head in one layer. Now we see the full picture.
>
> Different heads learn different specializations. Some become "previous-token heads" that always attend to the immediately preceding token. Some become "positional heads" that always look at the first token in the sequence. Some learn semantic patterns — attending to tokens that are related in meaning regardless of how far apart they are.

---

## Cell 40 — Collect Visualization Data (Code)

> *Run the cell.*
>
> We run a special hooked forward pass that captures the attention weights from every single layer and head. This gives us 36 attention matrices (6 layers times 6 heads), plus the hidden state norms at each layer.

---

## Cell 41 — All-Heads Attention Grid (Code)

> *Run the cell.*
>
> This is the full 6-by-6 grid. Rows are layers (0 at top, 5 at bottom), columns are heads. Each small heatmap shows how attention is distributed for the prompt "To be or not to be."
>
> Take a moment to scan the grid. Look for patterns.
>
> In layer 0, you often see strong diagonal patterns — each token attending to itself or its immediate neighbor. These are the "local" heads.
>
> In deeper layers — layers 4 and 5 — the patterns become more diffuse. The attention is spread across multiple tokens, not just the nearest ones. These heads are capturing longer-range relationships.
>
> Some heads might show a strong column — one token attracting attention from everywhere. That is usually a function word like "to" or "be" that serves as an anchor in the sequence.
>
> This is how we study what neural networks learn. Not by reading the code, but by visualizing the internal representations.

---

## Cell 42 — Activation Norms (Code)

> *Run the cell.*
>
> This bar chart shows the L2 norm of the hidden state after each layer. Notice how it grows. That is the residual connections accumulating signal. Each layer adds something to the representation, and the residual path keeps all the previous contributions.
>
> If the norms grew exponentially, that would indicate instability. A gentle, roughly linear increase is healthy. RMSNorm and the Pre-Norm architecture keep this growth controlled.

---

## Cell 43 — Section 9 Header (Markdown)

> The KV cache is one of the most important optimizations in production LLM deployment. Every time you interact with ChatGPT and it generates a response token by token, it is using a KV cache behind the scenes.
>
> Here is the problem. Without a cache, generating the 100th token requires the model to recompute attention over all 100 tokens from scratch. The 101st token requires 101 computations. The 102nd requires 102. That is O of N squared total work to generate N tokens.
>
> With a cache, we store the Key and Value vectors from all previous tokens. When generating the new token, we only compute Q, K, V for that single new token, concatenate K and V with the cache, and compute attention. Each step is now O of N — total work to generate N tokens is O of N squared for the full batch, but O of N per step, which is a huge improvement.
>
> The critical correctness requirement is the RoPE offset. When the 50th token enters the model during decode, it needs the same RoPE rotation angle it would have gotten if all 50 tokens were processed together. That means we pass start_pos equals 49 (the 0-indexed position) to the RoPE module. If we accidentally pass start_pos equals 0, every decode token gets treated as position 0, and the model produces garbage.

---

## Cell 44 — KV Cache Equivalence Proof (Code)

> *Run the cell.*
>
> This is the correctness test. We process a 16-token sequence two ways: (1) all at once in a single forward pass, and (2) first 15 tokens as prefill, then the 16th token with the cache.
>
> The maximum difference in logits should be extremely small — on the order of 1e-5 to 1e-7. That is just floating-point rounding error. The two methods are mathematically identical; the tiny difference comes from the order of floating-point operations.
>
> If this test ever fails — if the max difference is larger than 1e-4 — it means the KV cache implementation has a bug, almost certainly in the RoPE start_pos.
>
> Also look at the cache structure: it stores K and V tensors per layer, each with shape (batch, n_heads, cached_tokens, d_head). This is the accumulated memory cost of the cache — it grows linearly with sequence length.

---

## Cell 45 — Multi-Step Equivalence (Code)

> *Run the cell.*
>
> Same idea, but more extreme. We feed 20 tokens one at a time through the cache, building up incrementally, and compare the final logits to a single full forward pass on all 20 tokens. This tests that the cache concatenation logic works correctly over many steps, not just one.
>
> Again, the difference should be tiny. This is the test that catches subtle bugs like off-by-one errors in the cache indexing.

---

## Cell 46 — KV Cache Timing Benchmark (Code)

> *Run the cell.*
>
> Now we measure the actual speedup. We generate 50 tokens from prompts of different lengths, timing both the cached and uncached methods.
>
> The bar chart shows the result. The red bars (uncached) should be consistently taller than the green bars (cached). The speedup factor printed above each pair tells you how much faster the cache is — typically 3 to 10 times, depending on sequence length.
>
> On a GPU, the speedup is dramatic because the cache avoids redundant matrix multiplications. On CPU, the speedup is still significant but less dramatic because CPU computation has more overhead per operation.
>
> This is why every production LLM uses KV caching. The algorithm is mathematically identical, it just avoids redundant work. Free performance.

---

## Cell 47 — Summary (Markdown)

> And that is it. We have built a complete language model from scratch.
>
> Starting from raw text, we built a tokenizer that converts characters to numbers. We created a dataset pipeline that generates training examples. We built every component of the transformer architecture — normalization, position encoding, attention, feed-forward networks, residual connections. We trained it on Shakespeare and watched the loss drop from random to coherent. We generated text and explored how sampling parameters affect creativity. We visualized all 36 attention heads to see what the model learned. And we proved that the KV cache optimization is mathematically correct while being dramatically faster.
>
> Every component we used — RMSNorm, RoPE, SwiGLU, Pre-Norm, weight tying — is exactly what production models like LLaMA, Mistral, and Qwen use. The only difference is scale. Our model has 15 million parameters; LLaMA-3 has 405 billion. But the architecture is the same.
>
> If you understand what we built today, you understand the foundation of every large language model in the world.
>
> *Questions?*
