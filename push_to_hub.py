"""
Push a trained ESM2 or ProtT5 checkpoint to HuggingFace Hub.

Usage:
    python push_to_hub.py \
        --checkpoint outputs/esm2_8m_bpo/best_checkpoint \
        --repo your-username/cafa6-esm2-lora-bpo \
        --token hf_xxxxxxxxxxxxxxxxxxxx

    # Merge LoRA into base weights before pushing (larger upload, no PEFT needed at inference):
    python push_to_hub.py --checkpoint ... --repo ... --token ... --merge
"""

import os
import ssl
import sys

# Fix broken SSL_CERT_FILE in conda envs on Windows
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    ssl._create_default_https_context = ssl.create_default_context
except Exception:
    pass

import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to saved checkpoint directory")
    p.add_argument("--repo", required=True, help="HuggingFace repo ID, e.g. your-username/cafa6-esm2-lora-bpo")
    p.add_argument("--token", required=True, help="HuggingFace API token (write access)")
    p.add_argument("--merge", action="store_true",
                   help="Merge LoRA into base weights before pushing (standalone model, larger upload)")
    return p.parse_args()


def main():
    args = parse_args()

    # Detect model class from saved classifier_config.json
    cfg_path = Path(args.checkpoint) / "classifier_config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found. Is this a valid checkpoint?")
        sys.exit(1)

    with open(cfg_path) as fh:
        cfg = json.load(fh)
    model_class = cfg.get("model_class", "ESM2Classifier")

    print(f"Loading {model_class} from {args.checkpoint} ...")
    if model_class == "ESM2Classifier":
        from models.esm2_classifier import ESM2Classifier
        model = ESM2Classifier.from_pretrained(args.checkpoint)
    else:
        from models.prott5_classifier import ProtT5Classifier
        model = ProtT5Classifier.from_pretrained(args.checkpoint)

    num_labels = model.config.num_labels
    go_terms   = len(model.config.go_terms)
    base_model = model.config.base_model_name
    print(f"  Base model : {base_model}")
    print(f"  GO terms   : {num_labels:,} labels ({go_terms} in config)")

    if args.merge:
        print(f"\nMerging LoRA adapters into base weights and pushing to {args.repo} ...")
        print("(This uploads the full model weights — may be several GB)")
        model.merge_and_push(args.repo, token=args.token)
    else:
        print(f"\nPushing LoRA adapters to {args.repo} ...")
        print("(Base model weights stay on Hub — upload is small ~50 MB)")
        model.push_to_hub(args.repo, token=args.token)

    print(f"\nDone! View your model at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
