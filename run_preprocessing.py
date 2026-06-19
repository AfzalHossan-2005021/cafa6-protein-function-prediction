"""
One-shot preprocessing script using the actual CAFA6 data layout:

  data/raw/Train/train_sequences.fasta
  data/raw/Train/train_terms.tsv
  data/raw/Train/go-basic.obo
  data/raw/Test/testsuperset.fasta

Run from the project root:
    python run_preprocessing.py
"""

import logging
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from preprocessing import preprocess_cafa6

RAW_DIR = Path("data/raw")
TRAIN_DIR = RAW_DIR / "Train"
PROCESSED_DIR = Path("data/processed")

FASTA_PATH  = TRAIN_DIR / "train_sequences.fasta"
ANNOT_PATH  = TRAIN_DIR / "train_terms.tsv"

# Preprocess each ontology separately so notebooks can load them individually
for ontology in ["BPO", "MFO", "CCO"]:
    print(f"\n{'='*50}")
    print(f"Preprocessing ontology: {ontology}")
    print(f"{'='*50}")
    preprocess_cafa6(
        fasta_path=FASTA_PATH,
        annotation_path=ANNOT_PATH,
        output_dir=PROCESSED_DIR,
        ontology=ontology,
        min_proteins_per_term=10,
        min_len=1,           # CAFA6 has some very short sequences — keep all
        max_len=2048,
        val_size=0.1,
        test_size=0.1,
        seed=42,
    )

print("\nDone! Processed files written to:", PROCESSED_DIR)
print("Files created:")
for f in sorted(PROCESSED_DIR.glob("*.pkl")):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name}  ({size_mb:.1f} MB)")
