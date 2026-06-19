"""
End-to-end smoke test — runs in ~2 minutes on CPU using the 8M ESM2 model.

Tests the full pipeline: load data → build model → 1 training step → evaluate → save.
"""
import sys
import logging
import numpy as np
import torch
sys.path.insert(0, "src")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from preprocessing import load_processed
from dataset import ESM2Dataset, collate_fn_pad
from models.esm2_classifier import ESM2ClassifierConfig, ESM2Classifier
from evaluate import compute_fmax, compute_aupr, load_ia_weights
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

BASE_MODEL = "facebook/esm2_t6_8M_UR50D"  # tiny 8M — downloads fast
N_SAMPLES  = 200   # tiny subset so it finishes quickly
N_TERMS    = 50    # reduce label space for speed

log.info("Loading preprocessed BPO split...")
data = load_processed("data/processed/cafa6_bpo.pkl")
go_terms = data["go_terms"][:N_TERMS]

# Subset
def take(split, n):
    return {
        "protein_ids": split["protein_ids"][:n],
        "sequences":   split["sequences"][:n],
        "labels":      split["labels"][:n, :N_TERMS],
    }

train_split = take(data["train"], N_SAMPLES)
val_split   = take(data["val"],   50)

log.info(f"Train: {N_SAMPLES} | Val: 50 | GO terms: {N_TERMS}")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
train_ds = ESM2Dataset(train_split["sequences"], tokenizer, 256, train_split["labels"])
val_ds   = ESM2Dataset(val_split["sequences"],   tokenizer, 256, val_split["labels"])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  collate_fn=collate_fn_pad)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, collate_fn=collate_fn_pad)

# Build model
cfg = ESM2ClassifierConfig(
    base_model_name=BASE_MODEL,
    num_labels=N_TERMS,
    hidden_dims=[64, 32],
    dropout=0.1,
    lora_r=4,
    lora_alpha=8,
    lora_target_modules=["query", "key", "value"],
    go_terms=go_terms,
)
model = ESM2Classifier(cfg).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

# 3 training steps
model.train()
for step, batch in enumerate(train_loader):
    if step >= 3:
        break
    ids  = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    labs = batch["labels"].to(DEVICE)
    out  = model(ids, mask, labs)
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()
    log.info(f"  Train step {step+1} | loss={out['loss'].item():.4f}")

# Evaluate
model.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        out  = model(ids, mask)
        all_logits.append(out["logits"].cpu().float().numpy())
        all_labels.append(batch["labels"].numpy())

scores = torch.sigmoid(torch.tensor(np.concatenate(all_logits))).numpy()
y_true = np.concatenate(all_labels)
fmax, best_t = compute_fmax(y_true, scores)
aupr = compute_aupr(y_true, scores)
log.info(f"Val Fmax={fmax:.4f} (t={best_t:.2f}) | AUPR={aupr:.4f}")

# Save checkpoint
model.save_pretrained("outputs/smoke_test_checkpoint")
log.info("Checkpoint saved → outputs/smoke_test_checkpoint")

# Load back and verify
loaded = ESM2Classifier.from_pretrained("outputs/smoke_test_checkpoint")
log.info(f"Checkpoint loaded — num_labels={loaded.config.num_labels}, go_terms count={len(loaded.config.go_terms)}")

# IA weights
ia = load_ia_weights("data/raw/IA.tsv", go_terms)
log.info(f"IA weights: {(ia > 0).sum()} / {len(go_terms)} terms have IA > 0")

print("\n=== SMOKE TEST PASSED ===")
print(f"  Val Fmax : {fmax:.4f}")
print(f"  Val AUPR : {aupr:.4f}")
print("  Checkpoint save/load: OK")
print("  IA weights: OK")
print("\nReady to run full training with:")
print("  python src/train.py --model esm2 --config config/esm2_lora.yaml \\")
print("      --data_path data/processed/cafa6_bpo.pkl --ontology BPO \\")
print("      --base_model facebook/esm2_t6_8M_UR50D --num_epochs 1")
