# Error Analysis

## Overview

This document guides systematic error analysis after training.
Run `src/evaluate.py` with `--error_analysis` flag to generate these plots.

## 1. False Positive Analysis

**What to look for:**
- GO terms that are consistently over-predicted
- Often ancestor terms that appear in training more than child terms
- Check if GO propagation is causing cascading false positives

**Remedy:** Raise the threshold per term or use IC-weighted calibration.

## 2. False Negative Analysis

**What to look for:**
- GO terms with very low recall despite non-trivial frequency
- Terms at the bottom of the GO hierarchy (very specific)
- Proteins with unusual sequence composition

**Remedy:** Lower threshold for rare terms; use label smoothing during training.

## 3. Per-Ontology Error Breakdown

| Ontology | Most Missed Terms | Most Over-predicted Terms |
|----------|------------------|--------------------------|
| BPO | (fill after training) | (fill after training) |
| MFO | (fill after training) | (fill after training) |
| CCO | (fill after training) | (fill after training) |

## 4. Sequence Length vs Performance

Expected pattern:
- Very short sequences (<100 aa) → lower Fmax (less context for PLM)
- Very long sequences (>1024 aa, truncated) → moderate drop in recall

## 5. GO Term Depth vs Performance

Shallow terms (depth 1-3): high precision, high recall (common, easy)
Deep terms (depth >6): low precision, low recall (rare, specific)

## 6. Known Failure Modes

- **Multi-domain proteins**: mean pooling loses domain-specific information
  → consider per-region predictions or segment-then-pool
- **Novel folds**: PLM embeddings may be less informative for sequences
  with no homologs in pre-training data
- **Redundant GO terms**: semantically similar terms may both be predicted
  or both missed; GO slim mapping can help for coarse evaluation
