# Model Card — CAFA6 Protein Function Prediction

## Model Summary

Two fine-tuned protein language models for multi-label GO term prediction,
trained on the [CAFA6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) dataset.

| Model | Base | Fine-tuning | Parameters (trainable) |
|-------|------|-------------|------------------------|
| `cafa6-esm2-lora` | facebook/esm2_t33_650M_UR50D | LoRA (r=16) | ~8M / 650M |
| `cafa6-prott5-lora` | Rostlab/prot_t5_xl_uniref50 | LoRA (r=8) | ~5M / 3B |

## Intended Use

- Computational protein function prediction (Gene Ontology terms)
- Compatible with CAFA evaluation (Fmax, AUPR, Smin)
- Supports BPO, MFO, and CCO sub-ontologies

## Architecture

```
Protein sequence
      │
      ▼
PLM Encoder (ESM2 or ProtT5) ← LoRA adapters applied to attention layers
      │
  Mean Pooling
      │
  MLP Head  [d_model → 512 → 256 → n_GO_terms]
      │
  Sigmoid per term
      │
GO term probabilities  →  GO hierarchy propagation  →  Binary predictions
```

## Training

- **Loss**: Binary Cross-Entropy with Logits (per GO term)
- **Optimizer**: AdamW, weight_decay=0.01
- **LR Schedule**: Cosine with warmup (10% of steps)
- **Mixed Precision**: FP16 (ESM2) / BF16 (ProtT5)
- **Data split**: 80% train / 10% val / 10% test (stratified)
- **GO term filter**: ≥10 annotated proteins per term

## Post-processing

1. Apply sigmoid to logits → per-term probabilities
2. GO hierarchy propagation (true-path rule): if a child term is predicted,
   all ancestor terms are assigned at least the same score
3. Threshold tuning: optimal threshold selected on validation set per ontology

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Fmax** | Maximum protein-centric F-measure over the PR curve |
| **AUPR** | Area under the protein-centric precision-recall curve |
| **Smin** | Minimum semantic distance (uses IC-weighted FP/FN) |
| **Coverage** | Fraction of proteins receiving ≥1 prediction |

## Limitations

- Trained only on experimentally annotated proteins (not transferred annotations)
- Long sequences (>1024 aa for ESM2, >512 aa for ProtT5) are truncated
- GO term coverage limited to terms with ≥10 annotated training proteins
- Does not use structural or evolutionary information (sequence-only)

## How to Use

```python
from src.models.esm2_classifier import ESM2Classifier
from transformers import AutoTokenizer
import torch

model = ESM2Classifier.from_pretrained("your-username/cafa6-esm2-lora-bpo")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

sequence = "MKTAYIAKQRQISFV..."
inputs = tokenizer(sequence, return_tensors="pt", max_length=1024, truncation=True)
with torch.no_grad():
    out = model(**inputs)
scores = torch.sigmoid(out["logits"]).squeeze().numpy()
```

## Citation

```bibtex
@software{cafa6_plm_2025,
  title  = {Protein Function Prediction with Fine-Tuned ESM2 and ProtT5},
  year   = {2025},
  url    = {https://github.com/your-username/cafa6-protein-function-prediction}
}
```
