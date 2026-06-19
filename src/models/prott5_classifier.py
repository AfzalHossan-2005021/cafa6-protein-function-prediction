"""
ProtT5 + LoRA multi-label protein function classifier.

Architecture:
  ProtT5-XL-UniRef50 encoder (LoRA fine-tuned, or QLoRA with 4-bit)
  → mean pooling over token embeddings
  → MLP classification head
  → sigmoid per GO term

The model is compatible with HuggingFace Hub and can be merged / pushed
after training.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    PretrainedConfig,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ProtT5ClassifierConfig(PretrainedConfig):
    model_type = "prott5_classifier"

    def __init__(
        self,
        base_model_name: str = "Rostlab/prot_t5_xl_uniref50",
        num_labels: int = 1000,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        pooling: str = "mean",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] = None,
        use_qlora: bool = False,
        go_terms: list[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.hidden_dims = hidden_dims or [1024, 512]
        self.dropout = dropout
        self.pooling = pooling
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
        self.use_qlora = use_qlora
        self.go_terms = go_terms or []


# ---------------------------------------------------------------------------
# MLP head (shared with ESM2 but duplicated for independence)
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

class ProtT5Classifier(nn.Module):
    """
    ProtT5 encoder-only backbone + LoRA + multi-label classification head.

    For QLoRA pass use_qlora=True in the config; requires bitsandbytes.
    """

    def __init__(self, config: ProtT5ClassifierConfig):
        super().__init__()
        self.config = config

        # Optional 4-bit quantization config
        bnb_config = None
        if config.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        backbone = T5EncoderModel.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
        )

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

        embed_dim = backbone.config.d_model
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

    _CONFIG_FIELDS = [
        "base_model_name", "num_labels", "hidden_dims", "dropout", "pooling",
        "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "use_qlora", "go_terms",
    ]

    def save_pretrained(self, save_dir: str) -> None:
        """Save LoRA adapters + head + config."""
        import json
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save_pretrained(save_dir)
        torch.save(self.head.state_dict(), os.path.join(save_dir, "classification_head.pt"))
        cfg_dict = {k: getattr(self.config, k) for k in self._CONFIG_FIELDS}
        cfg_dict["model_class"] = "ProtT5Classifier"
        with open(os.path.join(save_dir, "classifier_config.json"), "w") as fh:
            json.dump(cfg_dict, fh, indent=2)
        logger.info(f"ProtT5Classifier saved → {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "ProtT5Classifier":
        import json
        cfg_path = os.path.join(load_dir, "classifier_config.json")
        with open(cfg_path) as fh:
            cfg_dict = json.load(fh)
        config = ProtT5ClassifierConfig(
            base_model_name=cfg_dict["base_model_name"],
            num_labels=cfg_dict["num_labels"],
            hidden_dims=cfg_dict["hidden_dims"],
            dropout=cfg_dict["dropout"],
            pooling=cfg_dict["pooling"],
            lora_r=cfg_dict["lora_r"],
            lora_alpha=cfg_dict["lora_alpha"],
            lora_dropout=cfg_dict["lora_dropout"],
            lora_target_modules=cfg_dict["lora_target_modules"],
            use_qlora=cfg_dict.get("use_qlora", False),
            go_terms=cfg_dict.get("go_terms", []),
        )
        model = cls.__new__(cls)
        super(ProtT5Classifier, model).__init__()
        model.config = config
        backbone = T5EncoderModel.from_pretrained(config.base_model_name)
        model.encoder = PeftModel.from_pretrained(backbone, load_dir)
        embed_dim = backbone.config.d_model
        model.head = ClassificationHead(
            in_features=embed_dim,
            hidden_dims=config.hidden_dims,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )
        head_path = os.path.join(load_dir, "classification_head.pt")
        model.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        logger.info(f"ProtT5Classifier loaded from {load_dir}")
        return model

    def push_to_hub(self, repo_id: str, token: str | None = None) -> None:
        """Push LoRA adapters + head to HuggingFace Hub."""
        from huggingface_hub import HfApi
        import tempfile

        api = HfApi(token=token)
        with tempfile.TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(folder_path=tmpdir, repo_id=repo_id, repo_type="model")
        logger.info(f"Pushed to Hub: https://huggingface.co/{repo_id}")

    def merge_and_push(self, repo_id: str, token: str | None = None) -> None:
        """Merge LoRA into base weights and push full model to Hub."""
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

def build_prott5_classifier(cfg: dict, num_labels: int, go_terms: list[str]) -> ProtT5Classifier:
    use_qlora = cfg.get("qlora", {}).get("enabled", False)
    config = ProtT5ClassifierConfig(
        base_model_name=cfg["model"]["base_model"],
        num_labels=num_labels,
        hidden_dims=cfg["classifier"]["hidden_dims"],
        dropout=cfg["classifier"]["dropout"],
        pooling=cfg["model"]["pooling"],
        lora_r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        lora_target_modules=cfg["lora"]["target_modules"],
        use_qlora=use_qlora,
        go_terms=go_terms,
    )
    return ProtT5Classifier(config)
