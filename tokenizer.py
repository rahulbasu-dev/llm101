"""Minimal Byte-Pair Encoding tokenizer built from scratch.
For production use SentencePiece or tiktoken — this is for understanding BPE internals."""

import json
import os
from typing import List, Tuple, Dict


# Special token IDs (reserved at the start of vocabulary)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
NUM_SPECIAL = 4
BYTE_OFFSET = NUM_SPECIAL  # byte 0 → token 4, byte 255 → token 259
NUM_BASE = 256 + NUM_SPECIAL  # 260 base tokens before merges


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Vocabulary layout:
        0       = <PAD>
        1       = <BOS>
        2       = <EOS>
        3       = <UNK>
        4–259   = raw bytes 0x00–0xFF
        260+    = BPE merge tokens
    """

    def __init__(self, target_vocab_size: int = 4096):
        self.target_vocab_size = target_vocab_size
        self.merges: List[Tuple[int, int]] = []
        # vocab maps token_id → bytes representation
        self.vocab: Dict[int, bytes] = {}
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Initialise the 260 base tokens (4 special + 256 bytes)."""
        self.vocab = {
            PAD_ID: b"<PAD>",
            BOS_ID: b"<BOS>",
            EOS_ID: b"<EOS>",
            UNK_ID: b"<UNK>",
        }
        for b in range(256):
            self.vocab[BYTE_OFFSET + b] = bytes([b])

    # ── Training ────────────────────────────────────────────

    def train(self, text: str):
        """Train BPE on raw text string. Learns merge rules."""
        # Encode entire text as byte-level token IDs
        tokens = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        num_merges = self.target_vocab_size - NUM_BASE

        print(f"Training BPE: {len(tokens):,} bytes → {num_merges} merges")

        for i in range(num_merges):
            # Count adjacent pairs
            pair_counts: Dict[Tuple[int, int], int] = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                print(f"  No more pairs to merge at step {i}")
                break

            # Most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            new_id = NUM_BASE + i

            # Register merge
            self.merges.append(best_pair)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to token stream
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best_pair[0] and tokens[j + 1] == best_pair[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            if (i + 1) % 500 == 0 or i < 5:
                merged_repr = self.vocab[new_id]
                try:
                    label = merged_repr.decode("utf-8", errors="replace")
                except Exception:
                    label = str(merged_repr)
                print(f"  merge {i+1:>5}/{num_merges}: "
                      f"freq={best_count:>6} → id={new_id} '{label}'")

        actual_vocab = NUM_BASE + len(self.merges)
        print(f"Tokenizer ready: {actual_vocab} tokens "
              f"({NUM_SPECIAL} special + 256 bytes + {len(self.merges)} merges)")
        return actual_vocab

    # ── Encode / Decode ─────────────────────────────────────

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text → list of token IDs."""
        tokens = [BYTE_OFFSET + b for b in text.encode("utf-8")]

        # Apply merges in learned order
        for merge_idx, (a, b) in enumerate(self.merges):
            merged_id = NUM_BASE + merge_idx
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        if add_special:
            tokens = [BOS_ID] + tokens + [EOS_ID]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs → text string."""
        raw_bytes = b""
        for tid in token_ids:
            if tid in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                continue
            if tid in self.vocab:
                val = self.vocab[tid]
                # Skip special token byte-strings
                if val in (b"<PAD>", b"<BOS>", b"<EOS>", b"<UNK>"):
                    continue
                raw_bytes += val
        return raw_bytes.decode("utf-8", errors="replace")

    def decode_token(self, tid: int) -> str:
        """Decode a single token ID for display."""
        if tid == PAD_ID: return "<PAD>"
        if tid == BOS_ID: return "<BOS>"
        if tid == EOS_ID: return "<EOS>"
        if tid == UNK_ID: return "<UNK>"
        if tid in self.vocab:
            try:
                return self.vocab[tid].decode("utf-8", errors="replace")
            except Exception:
                return f"<{tid}>"
        return f"<{tid}>"

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "merges": self.merges,
            "vocab_size": len(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self._build_base_vocab()
        self.merges = [tuple(m) for m in data["merges"]]
        self.target_vocab_size = data["target_vocab_size"]
        # Rebuild merge vocab entries
        for i, (a, b) in enumerate(self.merges):
            new_id = NUM_BASE + i
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        print(f"Tokenizer loaded: {len(self.vocab)} tokens from {path}")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ── Quick test ──────────────────────────────────────────────

if __name__ == "__main__":
    tok = BPETokenizer(target_vocab_size=500)

    sample = "Hello world! The cat sat on the mat. The cat sat on the hat."
    actual_size = tok.train(sample)
    print(f"\nVocab size: {actual_size}")

    encoded = tok.encode("The cat sat")
    print(f"Encoded: {encoded}")
    print(f"Tokens:  {[tok.decode_token(t) for t in encoded]}")

    decoded = tok.decode(encoded)
    print(f"Decoded: '{decoded}'")
