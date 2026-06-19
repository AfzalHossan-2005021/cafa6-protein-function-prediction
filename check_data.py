"""Quick smoke-test to verify preprocessing works with the real CAFA6 data."""
import sys
import logging
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from preprocessing import parse_fasta, parse_go_annotations, filter_sequences

print("=== Parsing train sequences ===")
seqs = parse_fasta("data/raw/Train/train_sequences.fasta")
print(f"First 3 IDs : {list(seqs.keys())[:3]}")
print(f"Last  3 IDs : {list(seqs.keys())[-3:]}")

print("\n=== Filtering ===")
seqs_f = filter_sequences(seqs, min_len=1)

print("\n=== Parsing annotations ===")
ann = parse_go_annotations("data/raw/Train/train_terms.tsv")
print(ann.head())
print("\nOntology counts:")
print(ann["ontology"].value_counts())

overlap = set(seqs_f.keys()) & set(ann["protein_id"].unique())
print(f"\nProteins with both sequence and annotation: {len(overlap):,}")

print("\n=== Parsing test sequences ===")
test_seqs = parse_fasta("data/raw/Test/testsuperset.fasta")
print(f"First 3 test IDs: {list(test_seqs.keys())[:3]}")
print(f"Total test sequences: {len(test_seqs):,}")
