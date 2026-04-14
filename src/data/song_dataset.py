"""
PyTorch Dataset for song translation
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict

class SongTranslationDataset(Dataset):
    """
    Dataset for MCNST training
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to .pt file created by DatasetBuilder
        """
        self.data = torch.load(data_path)
        print(f"Loaded {len(self.data)} examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch with padding
        """
        # Find max lengths
        max_src_len = max(ex['src_ids'].size(0) for ex in batch)
        max_tgt_len = max(ex['tgt_ids'].size(0) for ex in batch)
        max_notes = max(ex['melody_features'].size(0) for ex in batch)
        
        # Pad sequences
        src_ids = []
        tgt_ids = []
        melody_features = []
        
        for ex in batch:
            # Pad source
            src_pad = torch.zeros(max_src_len, dtype=torch.long)
            src_pad[:ex['src_ids'].size(0)] = ex['src_ids']
            src_ids.append(src_pad)
            
            # Pad target
            tgt_pad = torch.zeros(max_tgt_len, dtype=torch.long)
            tgt_pad[:ex['tgt_ids'].size(0)] = ex['tgt_ids']
            tgt_ids.append(tgt_pad)
            
            # Pad melody
            melody_pad = torch.zeros(max_notes, 5, dtype=torch.float32)
            melody_pad[:ex['melody_features'].size(0)] = ex['melody_features']
            melody_features.append(melody_pad)
        
        return {
            'src_ids': torch.stack(src_ids),
            'tgt_ids': torch.stack(tgt_ids),
            'melody_features': torch.stack(melody_features),
            'num_notes': torch.tensor([ex['num_notes'] for ex in batch]),
            'src_syllables': torch.tensor([ex['src_syllables'] for ex in batch]),
            'tgt_syllables': torch.tensor([ex['tgt_syllables'] for ex in batch]),
            'song_names': [ex['song_name'] for ex in batch]
        }


# Test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Load dataset
    dataset = SongTranslationDataset("data/processed/training_data.pt")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Test batch
    batch = next(iter(dataloader))
    print(f"Batch shapes:")
    print(f"  src_ids: {batch['src_ids'].shape}")
    print(f"  tgt_ids: {batch['tgt_ids'].shape}")
    print(f"  melody: {batch['melody_features'].shape}")