# Results Table

All metrics are protein-centric, evaluated on the held-out **test set** per ontology.
Fmax and AUPR are the primary CAFA metrics. Smin is reported where IC weights are available.

## Biological Process Ontology (BPO)

| Model | Fmax | AUPR | Smin | Coverage | Threshold |
|-------|------|------|------|----------|-----------|
| LogReg (ESM2-8M frozen) | — | — | — | — | — |
| XGBoost (ESM2-8M frozen) | — | — | — | — | — |
| LightGBM (ESM2-8M frozen) | — | — | — | — | — |
| **ESM2-650M + LoRA** | — | — | — | — | 0.30 |
| **ProtT5-XL + LoRA** | — | — | — | — | 0.30 |

## Molecular Function Ontology (MFO)

| Model | Fmax | AUPR | Smin | Coverage | Threshold |
|-------|------|------|------|----------|-----------|
| LogReg (ESM2-8M frozen) | — | — | — | — | — |
| XGBoost (ESM2-8M frozen) | — | — | — | — | — |
| **ESM2-650M + LoRA** | — | — | — | — | 0.35 |
| **ProtT5-XL + LoRA** | — | — | — | — | 0.35 |

## Cellular Component Ontology (CCO)

| Model | Fmax | AUPR | Smin | Coverage | Threshold |
|-------|------|------|------|----------|-----------|
| LogReg (ESM2-8M frozen) | — | — | — | — | — |
| XGBoost (ESM2-8M frozen) | — | — | — | — | — |
| **ESM2-650M + LoRA** | — | — | — | — | 0.40 |
| **ProtT5-XL + LoRA** | — | — | — | — | 0.40 |

---

> Fill in the dashes after running the training notebooks and `src/evaluate.py`.
> All thresholds are tuned on the validation set using `evaluate.tune_thresholds()`.
