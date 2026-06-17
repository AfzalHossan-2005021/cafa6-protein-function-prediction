"""
ESM2 + LoRA multi-label protein function classifier.

Architecture:
  ESM2 backbone (LoRA fine-tuned)
  → mean pooling over residue embeddings (excluding padding)
  → MLP classification head
  → sigmoid per GO term

The model can be saved / loaded as a standard HuggingFace model and pushed
to the Hub with push_to_hub().
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ESM2ClassifierConfig(PretrainedConfig):
    model_type = "esm2_classifier"

    def __init__(
        self,
        base_model_name: str = "facebook/esm2_t33_650M_UR50D",
        num_labels: int = 1000,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        pooling: str = "mean",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] = None,
        go_terms: list[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        self.pooling = pooling
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["query", "key", "value", "dense"]
        self.go_terms = go_terms or []


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list[int], num_labels: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, num_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ESM2Classifier(nn.Module):
    """
    ESM2 backbone + LoRA + multi-label classification head.

    Call save_pretrained() / push_to_hub() to upload to HuggingFace Hub.
    """

    def __init__(self, config: ESM2ClassifierConfig):
        super().__init__()
        self.config = config

        # Load ESM2 backbone
        backbone = AutoModel.from_pretrained(config.base_model_name)

        # Wrap with LoRA
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(backbone, lora_cfg)
        self.encoder.print_trainable_parameters()

        # Infer embedding dimension from model config
        embed_dim = backbone.config.hidden_size

        self.head = ClassificationHead(
            in_features=embed_dim,
            hidden_dims=config.hidden_dims,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return summed / counts
        elif self.config.pooling == "cls":
            return last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        logits = self.head(pooled)

        result = {"logits": logits}
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            result["loss"] = loss_fct(logits, labels)
        return result

    def save_pretrained(self, save_dir: str) -> None:
        """Save LoRA adapters + head + config for HuggingFace Hub upload."""
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save_pretrained(save_dir)
        torch.save(self.head.state_dict(), os.path.join(save_dir, "classification_head.pt"))
        self.config.save_pretrained(save_dir)
        logger.info(f"ESM2Classifier saved → {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "ESM2Classifier":
        config = ESM2ClassifierConfig.from_pretrained(load_dir)
        model = cls(config)
        # Replace the PEFT model with the saved one
        backbone = AutoModel.from_pretrained(config.base_model_name)
        model.encoder = PeftModel.from_pretrained(backbone, load_dir)
        head_path = os.path.join(load_dir, "classification_head.pt")
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        logger.info(f"ESM2Classifier loaded from {load_dir}")
        return model

    def push_to_hub(self, repo_id: str, token: str | None = None) -> None:
        """Push the LoRA adapters, head, and config to HuggingFace Hub."""
        from huggingface_hub import HfApi
        import tempfile

        api = HfApi(token=token)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                repo_type="model",
            )
        logger.info(f"Pushed to Hub: https://huggingface.co/{repo_id}")

    def merge_and_push(self, repo_id: str, token: str | None = None) -> None:
        """
        Merge LoRA weights into the base model and push the full model to Hub.
        Useful for creating a standalone model that doesn't require PEFT at inference.
        """
        from huggingface_hub import HfApi
        import tempfile

        merged_encoder = self.encoder.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            merged_encoder.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)
            torch.save(self.head.state_dict(), os.path.join(tmpdir, "classification_head.pt"))
            self.config.save_pretrained(tmpdir)

            api = HfApi(token=token)
            api.upload_folder(folder_path=tmpdir, repo_id=repo_id, repo_type="model")
        logger.info(f"Merged model pushed to Hub: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_esm2_classifier(cfg: dict, num_labels: int, go_terms: list[str]) -> ESM2Classifier:
    config = ESM2ClassifierConfig(
        base_model_name=cfg["model"]["base_model"],
        num_labels=num_labels,
        hidden_dims=cfg["classifier"]["hidden_dims"],
        dropout=cfg["classifier"]["dropout"],
        pooling=cfg["model"]["pooling"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        lora_target_modules=cfg["lora"]["target_modules"],
        go_terms=go_terms,
    )
    return ESM2Classifier(config)
