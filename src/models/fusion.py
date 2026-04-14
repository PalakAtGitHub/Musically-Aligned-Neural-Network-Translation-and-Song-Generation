"""
Cross-Modal Fusion - Allows lyrics to attend to melody
"""

import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    """
    Multi-head attention for lyrics-melody fusion
    
    Allows each word embedding to attend to relevant melody segments
    """
    
    def __init__(self, 
                 text_dim=1024,     # Dimension from mBART encoder (1024 for mBART-large)
                 melody_dim=256,    # Dimension from MelodyEncoder
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        # Project melody to match text dimension
        self.melody_proj = nn.Linear(melody_dim, text_dim)
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(text_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_features, melody_features, text_mask=None, melody_mask=None):
        """
        Args:
            text_features: [batch, text_seq_len, text_dim] - from mBART encoder
            melody_features: [batch, num_notes, melody_dim] - from MelodyEncoder
            text_mask: [batch, text_seq_len] - padding mask
            melody_mask: [batch, num_notes] - padding mask
        
        Returns:
            fused_features: [batch, text_seq_len, text_dim]
        """
        # Project melody to text dimension
        melody_projected = self.melody_proj(melody_features)  # [batch, num_notes, text_dim]
        
        # Cross-attention: text queries melody
        attn_output, attn_weights = self.cross_attention(
            query=text_features,           # What we want to enhance
            key=melody_projected,          # What we attend to
            value=melody_projected,        # What we extract
            key_padding_mask=melody_mask,  # Ignore padded notes
            need_weights=True
        )
        
        # Residual connection + layer norm
        fused = self.layer_norm(text_features + self.dropout(attn_output))
        
        return fused, attn_weights


# Test
if __name__ == "__main__":
    fusion = CrossModalFusion()
    
    # Sample inputs
    text_features = torch.randn(4, 10, 512)    # 4 examples, 10 words
    melody_features = torch.randn(4, 12, 256)  # 4 examples, 12 notes
    
    fused, weights = fusion(text_features, melody_features)
    
    print(f"✓ Cross-Modal Fusion Test:")
    print(f"  Text input:     {text_features.shape}")
    print(f"  Melody input:   {melody_features.shape}")
    print(f"  Fused output:   {fused.shape}")
    print(f"  Attention weights: {weights.shape}")