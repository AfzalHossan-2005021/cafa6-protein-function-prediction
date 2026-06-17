# Protein Function Prediction using Fine-Tuned ESM2 and ProtT5 Foundation Models

CAFA6 challenge solution: multi-label Gene Ontology (GO) term prediction from protein sequences using LoRA-fine-tuned protein language models.

## Architecture Overview

```
Protein Sequence
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │  ESM2-650M or ProtT5-XL-UniRef50            │
 │  + LoRA adapters (r=16/8, ~1–2% params)     │
 └─────────────────────────────────────────────┘
      │  Mean Pool
      ▼
 MLP Head  →  Sigmoid  →  GO term probabilities
      │
      ▼
 GO Hierarchy Propagation (true-path rule)
      │
      ▼
 Binary Predictions (threshold tuned per ontology)
```

## Three-Level Strategy

| Level | Approach | When to Use |
|-------|----------|-------------|
| **1** | Frozen embeddings + XGBoost/LightGBM/MLP | Fast baseline, Kaggle CPU |
| **2** | ESM2 or ProtT5 + LoRA fine-tuning | Best accuracy, Kaggle GPU |
| **3** | + GO propagation + per-ontology threshold tuning | CAFA submission |

## Project Structure

```
cafa6-protein-function-prediction/
├── config/
│   ├── esm2_lora.yaml          # ESM2 LoRA hyperparameters
│   └── prott5_lora.yaml        # ProtT5 LoRA hyperparameters (QLoRA option)
├── data/
│   ├── raw/                    # FASTA + annotation TSV (download from Kaggle)
│   ├── processed/              # Pickled train/val/test splits
│   └── label_matrix/           # Pre-built label matrices
├── notebooks/
│   ├── 01_eda_cafa6.ipynb          # Exploratory data analysis
│   ├── 02_embedding_baseline.ipynb # Level 1: frozen embeddings + sklearn/xgb
│   ├── 03_esm2_lora_training.ipynb # Level 2: ESM2 + LoRA
│   └── 04_prott5_finetuning.ipynb  # Level 2: ProtT5 + LoRA / QLoRA
├── src/
│   ├── preprocessing.py        # FASTA parsing, label matrix, train/val/test split
│   ├── dataset.py              # ESM2Dataset, ProtT5Dataset, EmbeddingDataset
│   ├── models/
│   │   ├── esm2_classifier.py  # ESM2 + LoRA + MLP head
│   │   └── prott5_classifier.py# ProtT5 + LoRA + MLP head
│   ├── train.py                # Training script (CLI)
│   ├── evaluate.py             # Fmax, AUPR, Smin, Coverage
│   ├── ontology_postprocessing.py  # GO propagation, IC, CAFA submission format
│   └── inference.py            # Batch inference on new sequences
├── reports/
│   ├── results_table.md
│   ├── model_card.md
│   └── error_analysis.md
└── app/
    └── gradio_demo.py          # Interactive web demo
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download CAFA6 data from Kaggle and place in `data/raw/`:
- `train_sequences.fasta`
- `train_annotations.tsv`  (columns: protein_id, go_term, ontology, evidence_code)
- `test_sequences.fasta`
- `go.obo` (Gene Ontology OBO file)

## Preprocessing

```bash
python -c "
from src.preprocessing import preprocess_cafa6
for ont in ['BPO', 'MFO', 'CCO']:
    preprocess_cafa6(
        fasta_path='data/raw/train_sequences.fasta',
        annotation_path='data/raw/train_annotations.tsv',
        output_dir='data/processed',
        ontology=ont,
    )
"
```

## Training

**ESM2 + LoRA (BPO):**
```bash
python src/train.py \
  --model esm2 \
  --config config/esm2_lora.yaml \
  --data_path data/processed/cafa6_bpo.pkl \
  --ontology BPO \
  --hub_repo your-username/cafa6-esm2-lora-bpo \
  --merge_before_push
```

**ProtT5 + LoRA (MFO):**
```bash
python src/train.py \
  --model prott5 \
  --config config/prott5_lora.yaml \
  --data_path data/processed/cafa6_mfo.pkl \
  --ontology MFO \
  --hub_repo your-username/cafa6-prott5-lora-mfo
```

Use `--wandb_project cafa6` to enable Weights & Biases logging.

## Inference

```bash
python src/inference.py \
  --model esm2 \
  --checkpoint outputs/esm2_lora/best_checkpoint \
  --fasta data/raw/test_sequences.fasta \
  --obo data/raw/go.obo \
  --output predictions/submission.tsv \
  --threshold 0.3 \
  --propagate
```

## Push to HuggingFace Hub

```python
from src.models.esm2_classifier import ESM2Classifier

model = ESM2Classifier.from_pretrained("outputs/esm2_lora/best_checkpoint")

# Push LoRA adapters only (recommended — lightweight):
model.push_to_hub("your-username/cafa6-esm2-lora-bpo", token="hf_...")

# OR merge LoRA into base weights and push full model:
model.merge_and_push("your-username/cafa6-esm2-lora-bpo-merged", token="hf_...")
```

## Demo

```bash
python app/gradio_demo.py
```

Set `ESM2_CHECKPOINT` and `PROTT5_CHECKPOINT` environment variables to point to trained checkpoints.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Fmax** | Max protein-centric F-measure over all thresholds |
| **AUPR** | Area under precision-recall curve (micro-averaged) |
| **Smin** | Minimum semantic distance (IC-weighted FP/FN) |
| **Coverage** | Fraction of proteins with ≥1 predicted term |

Evaluated separately for BPO, MFO, and CCO.

## Kaggle Tips

- Use `esm2_t6_8M_UR50D` (8M params) for the frozen baseline — runs on CPU in minutes
- Use `esm2_t33_650M_UR50D` (650M) for LoRA fine-tuning — needs 1× T4 GPU
- Enable QLoRA (`qlora.enabled: true` in config) to run ProtT5-XL on a single T4
- Use gradient checkpointing (`gradient_checkpointing: true`) to halve VRAM usage
