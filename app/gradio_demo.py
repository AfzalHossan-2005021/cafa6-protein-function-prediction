"""
Gradio demo for CAFA6 Protein Function Prediction.

Runs both ESM2 and ProtT5 models (if available) and shows
predicted GO terms with confidence scores.

Launch:
    python app/gradio_demo.py
"""

from __future__ import annotations

import os
import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.esm2_classifier import ESM2Classifier
from models.prott5_classifier import ProtT5Classifier
from dataset import ESM2Dataset, ProtT5Dataset, collate_fn_pad
from ontology_postprocessing import postprocess_predictions
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — edit these paths or HuggingFace repo IDs
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "ESM2 (LoRA fine-tuned)": {
        "type": "esm2",
        "checkpoint": os.getenv("ESM2_CHECKPOINT", "outputs/esm2_lora/best_checkpoint"),
    },
    "ProtT5 (LoRA fine-tuned)": {
        "type": "prott5",
        "checkpoint": os.getenv("PROTT5_CHECKPOINT", "outputs/prott5_lora/best_checkpoint"),
    },
}

_model_cache: dict[str, tuple] = {}


def load_model_cached(name: str):
    if name in _model_cache:
        return _model_cache[name]

    info = MODEL_REGISTRY[name]
    ckpt = info["checkpoint"]
    model_type = info["type"]

    if not Path(ckpt).exists():
        return None, None, None

    if model_type == "esm2":
        model = ESM2Classifier.from_pretrained(ckpt)
    else:
        model = ProtT5Classifier.from_pretrained(ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_name)
    go_terms = model.config.go_terms

    _model_cache[name] = (model, tokenizer, go_terms, device, model_type)
    return _model_cache[name]


@torch.no_grad()
def predict_function(
    sequence: str,
    model_name: str,
    threshold: float,
    propagate: bool,
    obo_path: str,
    top_k: int,
) -> tuple[str, dict]:
    """
    Main prediction function called by Gradio.
    Returns a formatted string and a bar chart data dict.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        return "Please enter a protein sequence.", {}

    cached = load_model_cached(model_name)
    if cached[0] is None:
        return f"Checkpoint not found for {model_name}. Please train and save the model first.", {}

    model, tokenizer, go_terms, device, model_type = cached

    max_len = model.config.max_length

    if model_type == "esm2":
        dataset = ESM2Dataset([sequence], tokenizer, max_len, protein_ids=["query"])
    else:
        dataset = ProtT5Dataset([sequence], tokenizer, max_len, protein_ids=["query"])

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_pad)
    batch = next(iter(loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    out = model(input_ids, attention_mask)
    logits = out["logits"].cpu().float()
    scores = torch.sigmoid(logits).numpy()

    propagated_scores, binary_preds = postprocess_predictions(
        raw_scores=scores,
        go_terms=go_terms,
        obo_path=obo_path if Path(obo_path).exists() else None,
        threshold=threshold,
        propagate=propagate and Path(obo_path).exists(),
    )

    protein_scores = propagated_scores[0]
    top_indices = np.argsort(protein_scores)[::-1][:top_k]

    lines = [f"**Model:** {model_name}\n", f"**Sequence length:** {len(sequence)} aa\n\n"]
    lines.append("| GO Term | Score | Predicted |\n|---|---|---|\n")

    chart_data = {"GO Term": [], "Score": []}
    for idx in top_indices:
        term = go_terms[idx]
        score = float(protein_scores[idx])
        predicted = "✓" if score >= threshold else ""
        lines.append(f"| {term} | {score:.3f} | {predicted} |\n")
        chart_data["GO Term"].append(term)
        chart_data["Score"].append(score)

    n_predicted = int(binary_preds[0].sum())
    lines.append(f"\n**Total predicted GO terms:** {n_predicted}")

    return "".join(lines), chart_data


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_SEQUENCES = [
    ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLSLQKAGIVSR",
     "ESM2 (LoRA fine-tuned)", 0.3, True, "data/raw/go.obo", 20],
]

with gr.Blocks(title="CAFA6 Protein Function Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CAFA6 Protein Function Prediction
        Predict Gene Ontology (GO) terms for protein sequences using fine-tuned ESM2 and ProtT5 models.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            seq_input = gr.Textbox(
                label="Protein Sequence (single-letter amino acid code)",
                placeholder="MKTAYIAKQRQISFV...",
                lines=6,
            )
            model_selector = gr.Dropdown(
                label="Model",
                choices=list(MODEL_REGISTRY.keys()),
                value=list(MODEL_REGISTRY.keys())[0],
            )
            with gr.Row():
                threshold_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Prediction Threshold")
                top_k_slider = gr.Slider(5, 50, value=20, step=5, label="Top-K Terms to Show")
            propagate_toggle = gr.Checkbox(label="Apply GO Hierarchy Propagation", value=True)
            obo_input = gr.Textbox(label="Path to go.obo (for propagation)", value="data/raw/go.obo")
            predict_btn = gr.Button("Predict GO Terms", variant="primary")

        with gr.Column(scale=3):
            output_table = gr.Markdown(label="Predictions")
            output_plot = gr.BarPlot(
                x="GO Term",
                y="Score",
                title="Top GO Term Scores",
                x_label_angle=-45,
            )

    predict_btn.click(
        fn=predict_function,
        inputs=[seq_input, model_selector, threshold_slider, propagate_toggle, obo_input, top_k_slider],
        outputs=[output_table, output_plot],
    )

    gr.Examples(examples=EXAMPLE_SEQUENCES, inputs=[seq_input, model_selector, threshold_slider, propagate_toggle, obo_input, top_k_slider])

    gr.Markdown(
        """
        ---
        **Models:** ESM2-650M and ProtT5-XL-UniRef50 fine-tuned with LoRA on CAFA6 annotations.
        **Metrics:** Fmax, AUPR evaluated per ontology (BPO / MFO / CCO).
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
