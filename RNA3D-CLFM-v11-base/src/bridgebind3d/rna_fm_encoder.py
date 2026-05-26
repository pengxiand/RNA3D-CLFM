"""RNA-FM per-nucleotide embedder using multimolecule/rnafm (HuggingFace).

Replaces the original fm.pretrained.rna_fm_t12() API with
AutoModel.from_pretrained("multimolecule/rnafm").  No local .pth file needed —
weights are downloaded from HuggingFace Hub automatically into blue storage.

Strategy
────────
  - Freeze RNA-FM encoder layers 0 … (12 - n_unfrozen_layers - 1)
  - Unfreeze the last n_unfrozen_layers (default 2) with a small LR in the
    optimizer (set via cfg["train"]["lr_rna_fm"])
  - Project 640-dim RNA-FM output → embed_dim with LayerNorm

Weights
───────
  Downloaded on first use by huggingface_hub into:
    /blue/qsong1/liangjialu/.torch_hub/hf_cache/
  HF_HOME is set before any import so HiPerGator cannot interfere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

# ── Persistent HuggingFace cache in blue storage ──────────────────────────────
_BLUE_CACHE = Path("/blue/qsong1/liangjialu/.torch_hub")
_HF_CACHE   = str(_BLUE_CACHE / "hf_cache")

RNA_FM_DIM   = 640    # hidden dim of multimolecule/rnafm (RNA-FM T12)
_HF_MODEL_ID = "multimolecule/rnafm"


def _set_hf_cache() -> str | None:
    """Redirect HuggingFace downloads to blue storage before any model load.

    HiPerGator nodes cannot write to the default ~/.cache location when the
    job runs on a compute node without home-dir access.
    """
    cache = Path(_HF_CACHE)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME",            _HF_CACHE)
        os.environ.setdefault("TRANSFORMERS_CACHE",  _HF_CACHE)
        os.environ.setdefault("HF_DATASETS_CACHE",   _HF_CACHE)
        return _HF_CACHE
    except PermissionError:
        return None


def _get_encoder_layers(model):
    """Return the list of transformer encoder layers from the HF model.

    multimolecule/rnafm follows the ESM/BERT convention: model.encoder.layer
    """
    enc = getattr(model, "encoder", None)
    if enc is not None:
        layers = getattr(enc, "layer", None)
        if layers is not None:
            return list(layers)
    # Fallback: search for any attribute named 'layers'
    for attr in ("layers", "transformer", "bert"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            sublayers = getattr(candidate, "layer", None) or getattr(candidate, "layers", None)
            if sublayers is not None:
                return list(sublayers)
    raise AttributeError(
        f"Cannot locate encoder layers in {type(model).__name__}. "
        "Check model architecture with model.named_parameters()."
    )


# ──────────────────────────────────────────────────────────────────────────────

class RNAFMNodeEmbedder(nn.Module):
    """Per-nucleotide embeddings via multimolecule/rnafm (HuggingFace).

    Args
    ────
    out_dim            : output dimension after projection (= embed_dim)
    n_unfrozen_layers  : number of final encoder layers to keep trainable
    model_location     : HuggingFace model id or local path (default: multimolecule/rnafm)
    """

    def __init__(
        self,
        out_dim: int,
        n_unfrozen_layers: int = 2,
        model_location: str | None = None,
    ) -> None:
        super().__init__()

        cache_dir = _set_hf_cache()
        model_id  = model_location or _HF_MODEL_ID
        print(f"[RNA-FM] loading '{model_id}'  hf_cache={cache_dir}", flush=True)

        from multimolecule import RnaFmModel, RnaTokenizer

        hf_kw = {}
        if cache_dir:
            hf_kw["cache_dir"] = cache_dir

        self.tokenizer = RnaTokenizer.from_pretrained(model_id, **hf_kw)
        self.fm_model  = RnaFmModel.from_pretrained(model_id, **hf_kw)

        # ── Freeze all parameters ────────────────────────────────────────────
        for param in self.fm_model.parameters():
            param.requires_grad = False

        # ── Unfreeze last n encoder layers ───────────────────────────────────
        self.n_unfrozen = n_unfrozen_layers
        self._enc_layers = _get_encoder_layers(self.fm_model)
        if n_unfrozen_layers > 0:
            for layer in self._enc_layers[-n_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.fm_model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.fm_model.parameters())
        print(f"[RNA-FM] trainable backbone params: {trainable:,} / {total:,} "
              f"(last {n_unfrozen_layers} layers)", flush=True)

        # ── Projection 640 → out_dim ─────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(RNA_FM_DIM, out_dim),
            nn.LayerNorm(out_dim),
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _unfrozen_params(self):
        """Trainable parameters in the RNA-FM backbone (last n layers)."""
        if self.n_unfrozen > 0:
            for layer in self._enc_layers[-self.n_unfrozen:]:
                yield from layer.parameters()

    def _other_params(self):
        """Trainable parameters outside the RNA-FM backbone (projection)."""
        yield from self.proj.parameters()

    # ── forward ──────────────────────────────────────────────────────────────

    @torch.autocast("cuda", enabled=False)   # keep fp32 for LayerNorm stability
    def forward(
        self,
        sequences: List[str],
        seq_lengths: List[int],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Per-node RNA-FM embeddings.

        Args
        ────
        sequences   : RNA nt_code strings (one per graph in the batch)
        seq_lengths : number of nodes per graph
        device      : output device (defaults to projection layer's device)

        Returns
        ───────
        Tensor [total_N, out_dim]
        """
        if device is None:
            device = next(self.proj.parameters()).device

        # Truncate to 510 nt (tokenizer adds 2 special tokens → 512 total)
        seqs_trunc = [seq[:510] for seq in sequences]

        enc = self.tokenizer(
            seqs_trunc,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        fm_grad = self.training and self.n_unfrozen > 0 and any(
            p.requires_grad for p in self.fm_model.parameters()
        )
        with torch.set_grad_enabled(fm_grad):
            out = self.fm_model(**enc)

        reps = out.last_hidden_state.float()  # [B, L+2, 640]
        reps = reps[:, 1:-1, :]               # remove CLS/EOS → [B, L, 640]

        # Assemble flat [total_N, 640]
        parts: list[torch.Tensor] = []
        for i, n in enumerate(seq_lengths):
            n_avail = min(n, reps.shape[1])
            emb = reps[i, :n_avail]           # [n_avail, 640]
            if n_avail < n:
                pad = torch.zeros(n - n_avail, RNA_FM_DIM, device=device, dtype=reps.dtype)
                emb = torch.cat([emb, pad], dim=0)
            parts.append(emb)

        flat = torch.cat(parts, dim=0)        # [total_N, 640]
        return self.proj(flat.float())        # [total_N, out_dim]
