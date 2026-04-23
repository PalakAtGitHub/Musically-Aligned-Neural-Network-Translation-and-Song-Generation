"""
Hierarchical Melody Encoder - Multi-level melody encoding following GTTM

Inspired by Lerdahl & Jackendoff's Generative Theory of Tonal Music (1983):
  Level 1 (Note):    CNN extracts local melodic patterns (2-3 note motifs)
  Level 2 (Phrase):  BiGRU captures phrase-level temporal evolution
  Level 3 (Section): Self-attention models long-range structure (verse/chorus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMelodyEncoder(nn.Module):
    """
    Multi-level melody encoding following GTTM.

    Output: [batch, num_notes, output_dim]
    """

    def __init__(self,
                 input_dim=5,
                 conv_channels=128,
                 gru_hidden=256,
                 num_heads=8,
                 output_dim=256,
                 dropout=0.2):
        super().__init__()

        # LEVEL 1: Note-level features (LOCAL PATTERNS)
        self.conv1 = nn.Conv1d(input_dim, conv_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels // 2, conv_channels, kernel_size=3, padding=1)
        self.conv_dropout = nn.Dropout(dropout)

        # LEVEL 2: Phrase-level dependencies (TEMPORAL)
        self.phrase_gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        gru_output_dim = gru_hidden * 2  # bidirectional

        # LEVEL 3: Section-level structure (GLOBAL)
        self.section_attention = nn.MultiheadAttention(
            embed_dim=gru_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.section_norm = nn.LayerNorm(gru_output_dim)

        # Output projection
        self.output_proj = nn.Linear(gru_output_dim, output_dim)
        self.output_dropout = nn.Dropout(dropout)

        self.hidden_size = output_dim

    def forward(self, melody_features):
        """
        Args:
            melody_features: [batch, num_notes, input_dim]
        Returns:
            hierarchical_embeddings: [batch, num_notes, output_dim]
        """
        # LEVEL 1: Local patterns (CNN)
        x = melody_features.transpose(1, 2)  # [batch, input_dim, num_notes]
        x = F.relu(self.conv1(x))
        x = self.conv_dropout(F.relu(self.conv2(x)))
        x = x.transpose(1, 2)  # [batch, num_notes, conv_channels]
        
        # Save CNN output to a text file
        import os
        os.makedirs("src/data/melody_vectors", exist_ok=True)
        with open("src/data/melody_vectors/cnn_output.txt", "w", encoding="utf-8") as f:
            f.write(f"--- CNN Output ---\nShape: {x.shape}\n\nTensor Values:\n{x.detach().cpu().numpy().tolist()}\n")

        # LEVEL 2: Phrase evolution (BiGRU)
        phrase_features, _ = self.phrase_gru(x)  # [batch, num_notes, gru_hidden*2]
        
        # Save GRU output to a text file
        with open("src/data/melody_vectors/gru_output.txt", "w", encoding="utf-8") as f:
            f.write(f"--- GRU Output ---\nShape: {phrase_features.shape}\n\nTensor Values:\n{phrase_features.detach().cpu().numpy().tolist()}\n")

        # LEVEL 3: Section structure (Self-attention + residual)
        section_features, _ = self.section_attention(
            phrase_features, phrase_features, phrase_features
        )

        # Residual connection + layer norm
        hierarchical = self.section_norm(phrase_features + section_features)

        # Project to output dimension
        output = self.output_dropout(self.output_proj(hierarchical))

        return output  # [batch, num_notes, output_dim]


# Test
if __name__ == "__main__":
    batch_size = 4
    num_notes = 12

    encoder = HierarchicalMelodyEncoder()
    test_melody = torch.randn(batch_size, num_notes, 5)

    encoded = encoder(test_melody)
    print(f"✓ Hierarchical Melody Encoder Test:")
    print(f"  Input:  {test_melody.shape}")
    print(f"  Output: {encoded.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
