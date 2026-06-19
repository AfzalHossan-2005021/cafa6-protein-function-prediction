"""
Inference script: predict GO terms for new protein sequences.

Usage:
    python src/inference.py \
        --model esm2 \
        --checkpoint outputs/esm2_lora/best_checkpoint \
        --fasta data/raw/test_sequences.fasta \
        --obo data/raw/go.obo \
        --output predictions/submission.tsv \
        --threshold 0.3 \
        --propagate
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import ssl
from pathlib import Path

try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    ssl._create_default_https_context = ssl.create_default_context
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ESM2Dataset, ProtT5Dataset, collate_fn_pad
from models.esm2_classifier import ESM2Classifier
from models.prott5_classifier import ProtT5Classifier
from ontology_postprocessing import postprocess_predictions, format_cafa_submission
from preprocessing import parse_fasta, filter_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type: str, checkpoint: str):
    if model_type == "esm2":
        return ESM2Classifier.from_pretrained(checkpoint)
    return ProtT5Classifier.from_pretrained(checkpoint)


@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    model.eval()
    all_logits, all_pids = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids, attention_mask)
        all_logits.append(out["logits"].cpu().float().numpy())
        all_pids.extend(batch["protein_id"])

    logits = np.concatenate(all_logits, axis=0)
    scores = torch.sigmoid(torch.tensor(logits)).numpy()
    return all_pids, scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Protein function prediction inference")
    p.add_argument("--model", choices=["esm2", "prott5"], required=True)
    p.add_argument("--checkpoint", required=True, help="Path to saved checkpoint directory")
    p.add_argument("--fasta", required=True, help="Input FASTA file with protein sequences")
    p.add_argument("--go_terms", default=None, help="Path to go_terms list (.pkl or .txt)")
    p.add_argument("--obo", default="data/raw/Train/go-basic.obo", help="Path to go-basic.obo for propagation")
    p.add_argument("--output", default="predictions/submission.tsv", help="Output TSV file")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--propagate", action="store_true", help="Apply GO hierarchy propagation")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    logger.info(f"Device: {device}")

    # Load model
    model = load_model(args.model, args.checkpoint)
    model = model.to(device)
    base_model_name = model.config.base_model_name
    go_terms = model.config.go_terms

    # Override go_terms from file if provided
    if args.go_terms:
        ext = Path(args.go_terms).suffix
        if ext == ".pkl":
            with open(args.go_terms, "rb") as fh:
                go_terms = pickle.load(fh)
        else:
            go_terms = Path(args.go_terms).read_text().strip().splitlines()

    if not go_terms:
        raise ValueError("GO terms list is empty. Provide via --go_terms or ensure config.go_terms is populated.")

    # Load sequences
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    sequences = parse_fasta(args.fasta)
    sequences = filter_sequences(sequences, min_len=1, max_len=args.max_length)
    protein_ids = list(sequences.keys())
    seqs = [sequences[p] for p in protein_ids]

    logger.info(f"Running inference on {len(protein_ids):,} sequences")

    if args.model == "esm2":
        dataset = ESM2Dataset(seqs, tokenizer, args.max_length, protein_ids=protein_ids)
    else:
        dataset = ProtT5Dataset(seqs, tokenizer, args.max_length, protein_ids=protein_ids)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
        num_workers=0,
    )

    pids, scores = run_inference(model, loader, device)

    # Post-process: propagation + thresholding
    propagated_scores, binary_preds = postprocess_predictions(
        raw_scores=scores,
        go_terms=go_terms,
        obo_path=args.obo,
        threshold=args.threshold,
        propagate=args.propagate,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    submission_str = format_cafa_submission(
        protein_ids=pids,
        go_terms=go_terms,
        scores=propagated_scores,
        threshold=args.threshold,
    )
    with open(args.output, "w") as fh:
        fh.write(submission_str)

    logger.info(f"Predictions saved → {args.output}")
    n_pred = int(binary_preds.sum())
    coverage = (binary_preds.sum(axis=1) > 0).mean()
    logger.info(f"Total predictions: {n_pred:,} | Coverage: {coverage:.3f}")


if __name__ == "__main__":
    main()
