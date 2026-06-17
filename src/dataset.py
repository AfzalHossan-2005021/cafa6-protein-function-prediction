"""
PyTorch Dataset classes for CAFA6 protein function prediction.

Provides datasets for:
  - ESM2 tokenized sequences
  - ProtT5 space-separated sequences
  - Raw embedding vectors (for baseline models)
"""

from __future__ import annotations

import re
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ProteinFunctionDataset(Dataset):
    """
    Base dataset for protein function prediction.

    Args:
        sequences: list of amino-acid strings.
        labels: float32 array of shape (n, n_terms); None for inference.
        protein_ids: optional list of string IDs for tracking.
    """

    def __init__(
        self,
        sequences: list[str],
        labels=None,
        protein_ids: list[str] | None = None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.protein_ids = protein_ids or [str(i) for i in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "sequence": self.sequences[idx],
            "protein_id": self.protein_ids[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class ESM2Dataset(Dataset):
    """
    Dataset for ESM2: tokenizes sequences using the ESM2 tokenizer.
    Sequences longer than max_length are truncated.
    """

    def __init__(
        self,
        sequences: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        labels=None,
        protein_ids: list[str] | None = None,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.protein_ids = protein_ids or [str(i) for i in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "protein_id": self.protein_ids[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class ProtT5Dataset(Dataset):
    """
    Dataset for ProtT5: sequences must be space-separated amino acids
    and uppercase (ProtT5 requirement).
    """

    def __init__(
        self,
        sequences: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        labels=None,
        protein_ids: list[str] | None = None,
    ):
        self.sequences = [self._format(s) for s in sequences]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.protein_ids = protein_ids or [str(i) for i in range(len(sequences))]

    @staticmethod
    def _format(seq: str) -> str:
        """Convert to uppercase, replace rare AAs with X, add spaces."""
        seq = seq.upper()
        seq = re.sub(r"[UZOB]", "X", seq)
        return " ".join(list(seq))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "protein_id": self.protein_ids[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class EmbeddingDataset(Dataset):
    """
    Dataset of pre-computed embeddings for the frozen-encoder baseline.
    """

    def __init__(self, embeddings, labels=None, protein_ids: list[str] | None = None):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = labels
        self.protein_ids = protein_ids or [str(i) for i in range(len(embeddings))]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "embedding": self.embeddings[idx],
            "protein_id": self.protein_ids[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_fn_pad(batch: list[dict]) -> dict:
    """Pad a batch of tokenized sequences to the same length."""
    out: dict = {}
    for key in batch[0]:
        if key in ("input_ids", "attention_mask"):
            out[key] = torch.stack([b[key] for b in batch])
        elif key == "labels":
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
