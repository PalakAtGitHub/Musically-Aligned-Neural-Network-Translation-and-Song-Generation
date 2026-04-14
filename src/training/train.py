"""
Training script for MCNST (Two-Stage Architecture)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm

from data.song_dataset import SongTranslationDataset
from src.models.mcnst_model import MCNST


class Trainer:
    def __init__(self,
                 data_path: str,
                 save_dir: str = "checkpoints",
                 batch_size: int = 4,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 10,
                 use_cpu_mode: bool = True):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        full_dataset = SongTranslationDataset(data_path)
        
        # Split train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size]
        )
        
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=full_dataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=full_dataset.collate_fn
        )
        
        # Initialize model (Two-Stage MCNST)
        print("Initializing model...")
        self.model = MCNST(
            freeze_encoder=use_cpu_mode,
            freeze_decoder_layers=10 if use_cpu_mode else 0
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Device: {self.device}")
        
        # Optimizer — only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.num_epochs = num_epochs
        self.history = []
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            melody = batch['melody_features'].to(self.device)
            num_notes = batch['num_notes'].to(self.device)
            tgt_syllables = batch['tgt_syllables'].to(self.device)
            
            # Forward (English → fuse with melody → Hindi)
            loss, loss_dict = self.model(
                input_ids=src_ids,
                melody_features=melody,
                labels=tgt_ids,
                num_notes=num_notes,
                tgt_syllables=tgt_syllables
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'trans': f"{loss_dict['translation_loss']:.3f}",
                'syl': f"{loss_dict['syllable_loss']:.3f}"
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src_ids = batch['src_ids'].to(self.device)
                tgt_ids = batch['tgt_ids'].to(self.device)
                melody = batch['melody_features'].to(self.device)
                num_notes = batch['num_notes'].to(self.device)
                tgt_syllables = batch['tgt_syllables'].to(self.device)
                
                loss, _ = self.model(
                    input_ids=src_ids,
                    melody_features=melody,
                    labels=tgt_ids,
                    num_notes=num_notes,
                    tgt_syllables=tgt_syllables
                )
                
                total_loss += loss.item()
        
        return total_loss / max(len(self.val_loader), 1)
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            
            # Save history
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = self.save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, checkpoint_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save history
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)
    
    trainer = Trainer(
        data_path="data/processed/training_data.pt",
        batch_size=4,
        num_epochs=10
    )
    trainer.train()