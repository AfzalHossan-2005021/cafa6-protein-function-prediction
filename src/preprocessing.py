"""
Data preprocessing for CAFA6 protein function prediction.

Handles FASTA parsing, GO annotation loading, label matrix construction,
train/val/test splitting, and ontology-specific subsetting.
"""

from __future__ import annotations

import os
import re
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
ONTOLOGY_MAP = {
    "GO:0008150": "BPO",  # biological_process
    "GO:0003674": "MFO",  # molecular_function
    "GO:0005575": "CCO",  # cellular_component
}


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Return {protein_id: sequence} from a FASTA file."""
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_seq: list[str] = []

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # CAFA IDs are the first token after ">"
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.upper())

    if current_id is not None:
        sequences[current_id] = "".join(current_seq)

    logger.info(f"Parsed {len(sequences):,} sequences from {fasta_path}")
    return sequences


# ---------------------------------------------------------------------------
# GO annotation loading
# ---------------------------------------------------------------------------

def parse_go_annotations(
    annotation_path: str | Path,
    evidence_codes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse a GAF or TSV annotation file.

    Expected TSV columns (CAFA format):
        protein_id  go_term  ontology  evidence_code

    Returns a DataFrame with columns [protein_id, go_term, ontology].
    """
    df = pd.read_csv(
        annotation_path,
        sep="\t",
        comment="!",
        header=None,
        names=["protein_id", "go_term", "ontology", "evidence_code"],
        usecols=[0, 1, 2, 3],
        dtype=str,
    )

    df = df.dropna(subset=["protein_id", "go_term"])
    df["protein_id"] = df["protein_id"].str.strip()
    df["go_term"] = df["go_term"].str.strip()

    if evidence_codes is not None:
        df = df[df["evidence_code"].isin(evidence_codes)]

    logger.info(f"Loaded {len(df):,} annotations covering {df['protein_id'].nunique():,} proteins")
    return df[["protein_id", "go_term", "ontology"]]


def load_test_sequences(test_fasta: str | Path) -> pd.DataFrame:
    """Parse CAFA test sequences into a DataFrame."""
    seqs = parse_fasta(test_fasta)
    return pd.DataFrame({"protein_id": list(seqs.keys()), "sequence": list(seqs.values())})


# ---------------------------------------------------------------------------
# Sequence filtering
# ---------------------------------------------------------------------------

def filter_sequences(
    sequences: dict[str, str],
    min_len: int = 40,
    max_len: int = 2048,
    max_ambiguous_frac: float = 0.1,
) -> dict[str, str]:
    """Remove sequences that are too short, too long, or too ambiguous."""
    filtered: dict[str, str] = {}
    for pid, seq in sequences.items():
        if len(seq) < min_len or len(seq) > max_len:
            continue
        ambiguous = sum(1 for aa in seq if aa not in VALID_AA)
        if ambiguous / len(seq) > max_ambiguous_frac:
            continue
        filtered[pid] = seq
    logger.info(f"After filtering: {len(filtered):,} sequences remain (from {len(sequences):,})")
    return filtered


# ---------------------------------------------------------------------------
# Label matrix construction
# ---------------------------------------------------------------------------

def build_label_matrix(
    protein_ids: list[str],
    annotations_df: pd.DataFrame,
    ontology: str | None = None,
    min_proteins_per_term: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a binary label matrix of shape (n_proteins, n_terms).

    Args:
        protein_ids: ordered list of protein IDs (defines row order).
        annotations_df: DataFrame from parse_go_annotations.
        ontology: filter to 'BPO', 'MFO', or 'CCO'; None = all.
        min_proteins_per_term: drop GO terms with fewer annotated proteins.

    Returns:
        (label_matrix, go_terms_list)
    """
    df = annotations_df.copy()
    if ontology is not None:
        df = df[df["ontology"] == ontology]

    # count proteins per term and filter rare terms
    term_counts = df.groupby("go_term")["protein_id"].nunique()
    valid_terms = term_counts[term_counts >= min_proteins_per_term].index.tolist()
    df = df[df["go_term"].isin(valid_terms)]

    # build mapping: protein_id -> set of GO terms
    prot_to_terms: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        prot_to_terms[row["protein_id"]].add(row["go_term"])

    # fit binarizer on the filtered term list
    mlb = MultiLabelBinarizer(classes=sorted(valid_terms))
    label_lists = [list(prot_to_terms.get(pid, set())) for pid in protein_ids]
    label_matrix = mlb.fit_transform(label_lists)

    logger.info(
        f"Label matrix: {label_matrix.shape} | "
        f"{int(label_matrix.sum()):,} positive entries | "
        f"sparsity={1 - label_matrix.mean():.3f}"
    )
    return label_matrix.astype(np.float32), mlb.classes_.tolist()


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    protein_ids: list[str],
    sequences: dict[str, str],
    label_matrix: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Stratified (on label presence) train/val/test split.

    Returns dict with keys 'train', 'val', 'test', each containing
    {protein_ids, sequences, labels}.
    """
    n = len(protein_ids)
    indices = np.arange(n)

    # stratify on whether each protein has ANY annotation
    has_label = (label_matrix.sum(axis=1) > 0).astype(int)

    idx_trainval, idx_test = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=has_label
    )
    has_label_trainval = has_label[idx_trainval]
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=seed, stratify=has_label_trainval
    )

    def subset(idx_arr):
        pids = [protein_ids[i] for i in idx_arr]
        return {
            "protein_ids": pids,
            "sequences": [sequences[p] for p in pids],
            "labels": label_matrix[idx_arr],
        }

    splits = {"train": subset(idx_train), "val": subset(idx_val), "test": subset(idx_test)}
    for name, s in splits.items():
        logger.info(f"{name}: {len(s['protein_ids']):,} proteins")
    return splits


# ---------------------------------------------------------------------------
# Disk I/O helpers
# ---------------------------------------------------------------------------

def save_processed(data: dict, output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved → {path}")


def load_processed(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------

def preprocess_cafa6(
    fasta_path: str | Path,
    annotation_path: str | Path,
    output_dir: str | Path,
    ontology: str | None = None,
    min_proteins_per_term: int = 10,
    min_len: int = 40,
    max_len: int = 2048,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    End-to-end preprocessing pipeline.

    Steps:
        1. Parse sequences
        2. Filter sequences
        3. Load annotations
        4. Build label matrix
        5. Split dataset
        6. Save to disk

    Returns the split dict.
    """
    sequences = parse_fasta(fasta_path)
    sequences = filter_sequences(sequences, min_len=min_len, max_len=max_len)

    annotations_df = parse_go_annotations(annotation_path)

    # Keep only proteins that appear in both sequences and annotations
    annotated_pids = set(annotations_df["protein_id"].unique())
    protein_ids = sorted(pid for pid in sequences if pid in annotated_pids)
    logger.info(f"Proteins with both sequence and annotation: {len(protein_ids):,}")

    label_matrix, go_terms = build_label_matrix(
        protein_ids, annotations_df, ontology=ontology,
        min_proteins_per_term=min_proteins_per_term,
    )

    splits = split_dataset(protein_ids, sequences, label_matrix, val_size, test_size, seed)
    splits["go_terms"] = go_terms
    splits["ontology"] = ontology

    prefix = ontology.lower() if ontology else "all"
    save_processed(splits, output_dir, f"cafa6_{prefix}")
    return splits
