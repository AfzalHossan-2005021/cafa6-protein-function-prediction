"""
Training script for ESM2 or ProtT5 LoRA classifiers.

Usage:
    python src/train.py --model esm2 --config config/esm2_lora.yaml \
        --data_path data/processed/cafa6_bpo.pkl --ontology BPO

    python src/train.py --model prott5 --config config/prott5_lora.yaml \
        --data_path data/processed/cafa6_mfo.pkl --ontology MFO
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import ssl
from pathlib import Path

# Fix broken SSL_CERT_FILE conda env variable (common on Windows)
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    ssl._create_default_https_context = ssl.create_default_context
except Exception:
    pass

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import ESM2Dataset, ProtT5Dataset, collate_fn_pad
from evaluate import compute_fmax, compute_aupr
from models.esm2_classifier import build_esm2_classifier
from models.prott5_classifier import build_prott5_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_split(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def build_dataloader(
    model_type: str,
    split: dict,
    tokenizer,
    cfg: dict,
    shuffle: bool,
    batch_size: int,
) -> DataLoader:
    seqs = split["sequences"]
    labels = split["labels"]
    pids = split["protein_ids"]
    max_len = cfg["model"]["max_length"]

    if model_type == "esm2":
        dataset = ESM2Dataset(seqs, tokenizer, max_len, labels, pids)
    else:
        dataset = ProtT5Dataset(seqs, tokenizer, max_len, labels, pids)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_pad,
        num_workers=cfg["training"].get("dataloader_num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            out = model(input_ids, attention_mask, labels)
            loss = out["loss"] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, attention_mask, labels)
        total_loss += out["loss"].item()
        all_logits.append(out["logits"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    scores = torch.sigmoid(torch.tensor(logits)).numpy()
    return y_true, scores, total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train protein function classifier")
    parser.add_argument("--model", choices=["esm2", "prott5"], required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data_path", required=True, help="Path to preprocessed .pkl split file")
    parser.add_argument("--ontology", default="BPO", help="Ontology name (BPO / MFO / CCO)")
    parser.add_argument("--output_dir", default=None, help="Override output directory from config")
    # CLI overrides for quick experimentation
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size from config")
    parser.add_argument("--base_model", default=None,
                        help="Override base model (e.g. facebook/esm2_t6_8M_UR50D for fast testing)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit train split to N samples (for smoke-testing)")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--hub_repo", default=None, help="HuggingFace repo ID to push to after training")
    parser.add_argument("--hub_token", default=None, help="HuggingFace API token")
    parser.add_argument("--merge_before_push", action="store_true",
                        help="Merge LoRA into base weights before pushing to Hub")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data = load_split(args.data_path)

    # Apply CLI overrides
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.base_model is not None:
        cfg["model"]["base_model"] = args.base_model

    # Optionally limit training data for smoke-testing
    if args.max_train_samples is not None:
        n = args.max_train_samples
        data["train"]["protein_ids"] = data["train"]["protein_ids"][:n]
        data["train"]["sequences"]   = data["train"]["sequences"][:n]
        data["train"]["labels"]      = data["train"]["labels"][:n]
        logger.info(f"Limiting train set to {n} samples (smoke-test mode)")

    output_dir = args.output_dir or cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Device: {device}")

    go_terms = data["go_terms"]
    num_labels = len(go_terms)
    logger.info(f"Number of GO terms ({args.ontology}): {num_labels:,}")

    tokenizer = build_tokenizer(cfg["model"]["base_model"])

    bs = cfg["training"]["batch_size"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]

    train_loader = build_dataloader(args.model, data["train"], tokenizer, cfg, shuffle=True, batch_size=bs)
    val_loader = build_dataloader(args.model, data["val"], tokenizer, cfg, shuffle=False, batch_size=bs * 2)

    if args.model == "esm2":
        model = build_esm2_classifier(cfg, num_labels, go_terms)
    else:
        model = build_prott5_classifier(cfg, num_labels, go_terms)

    model = model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]
    total_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_fp16 = cfg["training"].get("fp16", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    if WANDB_AVAILABLE and args.wandb_project:
        wandb.init(project=args.wandb_project, config=cfg, name=f"{args.model}_{args.ontology}")

    best_fmax = 0.0
    best_ckpt_path = os.path.join(output_dir, "best_checkpoint")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, grad_accum)
        y_true, scores, val_loss = evaluate(model, val_loader, device)
        fmax, best_t = compute_fmax(y_true, scores)
        aupr = compute_aupr(y_true, scores)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"Fmax={fmax:.4f} (t={best_t:.2f}) | AUPR={aupr:.4f}"
        )

        if WANDB_AVAILABLE and args.wandb_project:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                f"{args.ontology}/fmax": fmax,
                f"{args.ontology}/aupr": aupr,
            })

        if fmax > best_fmax:
            best_fmax = fmax
            model.save_pretrained(best_ckpt_path)
            logger.info(f"New best Fmax={fmax:.4f} → checkpoint saved")

    logger.info(f"Training complete. Best Fmax: {best_fmax:.4f}")

    if args.hub_repo:
        logger.info(f"Pushing to HuggingFace Hub: {args.hub_repo}")
        ckpt_model = (
            build_esm2_classifier(cfg, num_labels, go_terms)
            if args.model == "esm2"
            else build_prott5_classifier(cfg, num_labels, go_terms)
        )
        ckpt_model = type(ckpt_model).from_pretrained(best_ckpt_path)
        if args.merge_before_push:
            ckpt_model.merge_and_push(args.hub_repo, token=args.hub_token)
        else:
            ckpt_model.push_to_hub(args.hub_repo, token=args.hub_token)

    if WANDB_AVAILABLE and args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
