"""Dataset for causal language modelling.

Creates overlapping sliding-window sequences from tokenised text.
For each window of length L:
  input  = tokens[i   : i+L]
  target = tokens[i+1 : i+L+1]   (shifted right by 1)

This is how GPT/LLaMA training works — predict the next token at every position.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class TextDataset(Dataset):
    """Sliding-window dataset for causal language modelling."""

    def __init__(
        self,
        tokens: List[int],
        seq_len: int,
        stride: int = None,
    ):
        """
        Args:
            tokens:  Full tokenised corpus as list of int IDs
            seq_len: Context window size (= max_seq_len from config)
            stride:  Step between windows (default: seq_len // 2 for 50% overlap)
        """
        self.seq_len = seq_len
        self.stride = stride or seq_len // 2

        # Pre-compute all (input, target) windows
        self.samples = []
        for i in range(0, len(tokens) - seq_len - 1, self.stride):
            input_ids = tokens[i : i + seq_len]
            target_ids = tokens[i + 1 : i + seq_len + 1]
            self.samples.append((input_ids, target_ids))

        print(f"Dataset: {len(tokens):,} tokens → {len(self.samples):,} samples "
              f"(seq_len={seq_len}, stride={self.stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Create a DataLoader with pinned memory for GPU transfer."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid incomplete last batch
    )
