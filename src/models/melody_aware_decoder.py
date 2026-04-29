"""
MelodyAwareDecoderLayer: wrapper around IndicTransDecoderLayer that adds
per-step decoder-to-melody cross-attention.

Architecture:
    Original layer (self-attn → encoder-attn → FFN)
    → melody cross-attention sublayer (query=decoder output, key/value=melody)
    → residual + LayerNorm

The melody features (256-dim from HierarchicalMelodyEncoder) are projected to
the decoder hidden dimension (1024) via learned key/value projections inside
nn.MultiheadAttention.

Melody features are passed at runtime via the parent MCNST model's
`_current_melody_features` and `_current_melody_mask` attributes, which are
set before each seq2seq forward/generate call.
"""

import weakref

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MelodyAwareDecoderLayer(nn.Module):
    """Wraps an IndicTransDecoderLayer, appending melody cross-attention."""

    def __init__(
        self,
        original_layer: nn.Module,
        melody_dim: int = 256,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        parent_model: Optional[object] = None,
    ):
        super().__init__()
        self.original_layer = original_layer
        # Store as weakref to avoid registering the full MCNST as a submodule
        # (which would duplicate all parameters in named_parameters()).
        self._parent_ref = weakref.ref(parent_model) if parent_model is not None else None

        # Melody cross-attention: queries are decoder hidden (1024),
        # keys/values are melody features (256 → projected to 1024 internally
        # by nn.MultiheadAttention via kdim/vdim).
        self.melody_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=melody_dim,
            vdim=melody_dim,
            batch_first=True,
        )
        self.melody_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.melody_dropout = nn.Dropout(dropout)

        # Gate: learned scalar initialized to 0.01 (tanh(0.01) ≈ 0.01).
        # This gives ~1% melody contribution at init — small enough to
        # preserve pretrained decoder behavior, but non-zero so that
        # gradients flow through the gate to the attention parameters
        # from the very first training step.
        self.melody_gate = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        # 1. Run the original IndicTrans decoder layer unchanged
        outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=layer_head_mask,
            cross_attn_layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # outputs is a tuple: (hidden_states, [self_attn_weights, cross_attn_weights], [present_key_value])
        layer_output = outputs[0]

        # 2. Melody cross-attention sublayer (pre-norm style, matching IndicTrans)
        melody_features = None
        melody_mask = None
        parent = self._parent_ref() if self._parent_ref is not None else None
        if parent is not None:
            melody_features = getattr(parent, '_current_melody_features', None)
            melody_mask = getattr(parent, '_current_melody_mask', None)

        if melody_features is not None:
            residual = layer_output

            # Pre-norm (consistent with normalize_before=True in IndicTrans)
            normed = self.melody_attn_layer_norm(layer_output)

            # During beam search, the decoder batch is expanded by num_beams
            # (e.g., batch=1 → 5 beams), but melody features are stashed once
            # with the original batch size. Expand to match.
            bsz = normed.size(0)
            mel_bsz = melody_features.size(0)
            if bsz != mel_bsz and bsz % mel_bsz == 0:
                expand_factor = bsz // mel_bsz
                melody_features = melody_features.repeat_interleave(expand_factor, dim=0)
                if melody_mask is not None:
                    melody_mask = melody_mask.repeat_interleave(expand_factor, dim=0)

            # query: decoder hidden [B, T_dec, 1024]
            # key/value: melody features [B, N_notes, 256]
            # key_padding_mask: [B, N_notes] True where padded
            melody_out, _ = self.melody_attn(
                query=normed,
                key=melody_features,
                value=melody_features,
                key_padding_mask=melody_mask,
            )

            # Gated residual: tanh(0.01) ≈ 0.01 → ~1% initial contribution
            gate = torch.tanh(self.melody_gate)
            melody_out = self.melody_dropout(melody_out)
            layer_output = residual + gate * melody_out

        # Reconstruct the output tuple with the modified hidden states
        return (layer_output,) + outputs[1:]

    # Proxy common attributes so that code checking layer properties still works
    @property
    def normalize_before(self):
        return getattr(self.original_layer, 'normalize_before', True)

    @property
    def embed_dim(self):
        return getattr(self.original_layer, 'embed_dim', 1024)

    @property
    def self_attn(self):
        return self.original_layer.self_attn

    @property
    def encoder_attn(self):
        return self.original_layer.encoder_attn

    @property
    def self_attn_layer_norm(self):
        return self.original_layer.self_attn_layer_norm

    @property
    def encoder_attn_layer_norm(self):
        return self.original_layer.encoder_attn_layer_norm

    @property
    def final_layer_norm(self):
        return self.original_layer.final_layer_norm
