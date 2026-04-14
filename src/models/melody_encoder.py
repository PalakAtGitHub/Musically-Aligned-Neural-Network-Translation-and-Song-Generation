"""
Melody Encoder - CNN-GRU hybrid for melody feature extraction
"""

import torch
import torch.nn as nn

class MelodyEncoder(nn.Module):
    """
    CNN-GRU hybrid for encoding MIDI melody features
    
    Architecture:
    - 3 Conv1D layers: Extract local melodic patterns (intervals, ornaments)
    - 2 Bidirectional GRU layers: Capture temporal dependencies
    - Output projection: Map to final hidden dimension
    
    Data flow:
    [batch, num_notes, 5] → Conv1D → [batch, num_notes, conv_channels] → GRU → projection
    """
    
    def __init__(self, 
                 input_dim=5,           # MIDI features per note
                 conv_channels=128,     # CNN hidden dimension
                 gru_hidden=256,        # GRU hidden dimension
                 num_conv_layers=3,
                 num_gru_layers=2,
                 dropout=0.3):
        super().__init__()
        
        # Convolutional layers (operate on feature channels)
        # Conv1d expects [batch, channels, seq_len]
        conv_layers = []
        in_channels = input_dim  # 5 MIDI features as input channels
        for i in range(num_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = conv_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # GRU layers — input from CNN output
        self.gru = nn.GRU(
            input_size=conv_channels,  # Fed by CNN output
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(gru_hidden * 2, gru_hidden)  # *2 for bidirectional
        
        self.hidden_size = gru_hidden
    
    def forward(self, melody_features):
        """
        Args:
            melody_features: [batch_size, num_notes, 5]
                Feature columns: [pitch, pitch_class, duration, duration_bin, beat_strength]
        
        Returns:
            encoded: [batch_size, num_notes, gru_hidden]
        """
        # Transpose for Conv1d: [batch, 5, num_notes]
        x = melody_features.transpose(1, 2)
        
        # CNN: extract local melodic patterns
        x = self.conv_layers(x)  # [batch, conv_channels, num_notes]
        
        # Transpose back for GRU: [batch, num_notes, conv_channels]
        x = x.transpose(1, 2)
        
        # GRU: capture temporal dependencies
        gru_out, _ = self.gru(x)  # [batch, num_notes, gru_hidden*2]
        
        # Project to final dimension
        output = self.output_proj(gru_out)  # [batch, num_notes, gru_hidden]
        
        return output


# Test
if __name__ == "__main__":
    # Test with sample data
    batch_size = 4
    num_notes = 12
    
    melody_encoder = MelodyEncoder()
    test_melody = torch.randn(batch_size, num_notes, 5)
    
    encoded = melody_encoder(test_melody)
    print(f"✓ Melody Encoder Test:")
    print(f"  Input:  {test_melody.shape}")
    print(f"  Output: {encoded.shape}")
    print(f"  Parameters: {sum(p.numel() for p in melody_encoder.parameters()):,}")