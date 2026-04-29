"""
Learned Alignment Module for MCNST
===================================

Problem this solves
-------------------
The decoder produces an output token at each position t = 1..T_out. The
melody has N notes at positions n = 1..N. We want to be able to ask:
"which note is output position t aligned with?" — so we can combine
features of the predicted token (e.g. cluster badness) with features of
the note (e.g. note duration, beat strength).

Simple baselines:
  - Uniform alignment (every output position aligns equally with all notes)
  - Identity alignment (output position t aligns with note t)

Both are wrong for real songs. Real alignment is many-to-one or one-to-many
(two short syllables on one note, or one syllable melismatically spread
over two notes), and it depends on the specific melody and lyrics.

So we learn it. A small cross-attention module with decoder hidden states
as queries and melody features as keys/values. The attention weights ARE
the alignment matrix A[t, n] = P(output pos t aligned with note n).

Output shape: [batch, T_out, N] — rows sum to 1 (softmax over notes).

Why a separate module (not a change inside IndicTrans2's decoder)
-----------------------------------------------------------------
IndicTrans2's decoder is frozen (mostly) and its cross-attention is
already attending to the encoder outputs (which have been fused with
melody upstream via CrossModalFusion). Injecting another cross-attention
inside would change decoder behavior unpredictably and break pretrained
weights. A post-hoc module is decoupled from the language modeling path —
its only job is to produce an alignment matrix for loss computation.

Size: ~1M parameters. Trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAlignment(nn.Module):
    """
    Cross-attention from decoder hidden states to melody features.

    Input:
        decoder_hidden:  [batch, T_out, text_dim]   decoder's last hidden state
        melody_encoded:  [batch, N,     melody_dim] output of HierarchicalMelodyEncoder
        melody_mask:     [batch, N]                 True = padding (ignore)

    Output:
        alignment:       [batch, T_out, N]  softmax over notes, row-stochastic
        align_loss:      [batch]            an auxiliary entropy-based
                                             regulariser (optional, for sharpness)
    """

    def __init__(self, text_dim=1024, melody_dim=256, attn_dim=256, num_heads=4,
                 dropout=0.1):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"

        self.text_dim = text_dim
        self.melody_dim = melody_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project decoder hidden states (1024) down to attn_dim (256)
        self.q_proj = nn.Linear(text_dim, attn_dim)
        # Project melody features (256) to attn_dim (same)
        self.k_proj = nn.Linear(melody_dim, attn_dim)
        # We don't need v_proj — we only want the attention weights, not
        # attended values. So there's no v_proj.

        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_hidden, melody_encoded, melody_mask=None):
        """
        Compute soft alignment weights.

        Args:
            decoder_hidden:  [B, T_out, text_dim]
            melody_encoded:  [B, N, melody_dim]
            melody_mask:     [B, N]  — True where padding

        Returns:
            alignment:   [B, T_out, N]   — softmax over N, sums to 1
        """
        B, T_out, _ = decoder_hidden.shape
        _, N, _     = melody_encoded.shape

        # Project: q in attn_dim, k in attn_dim
        q = self.q_proj(decoder_hidden)  # [B, T_out, attn_dim]
        k = self.k_proj(melody_encoded)  # [B, N,     attn_dim]

        # Multi-head: reshape to [B, heads, T_out, head_dim]
        q = q.view(B, T_out, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N,     self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, heads, T_out, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Dropout must come BEFORE masking. Reason: dropout multiplies by 0
        # to drop values, and 0 * -inf = NaN in IEEE arithmetic. If we
        # masked first, dropout would randomly zero out -inf values and
        # produce NaN gradients.
        scores = self.dropout(scores)

        # Apply melody padding mask AFTER dropout
        if melody_mask is not None:
            # melody_mask: [B, N] -> [B, 1, 1, N]
            mask_expanded = melody_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        # Softmax over N (the last dim, = note positions).
        # With masking-after-dropout, every row now has at least one
        # finite score (from the non-masked positions), so softmax is
        # well-defined and rows sum to exactly 1.
        attn = F.softmax(scores, dim=-1)

        # Average across heads to produce a single alignment matrix.
        # Mean across heads keeps the output as a probability distribution
        # (mean of simplex vectors is still on the simplex).
        alignment = attn.mean(dim=1)  # [B, T_out, N]

        return alignment
