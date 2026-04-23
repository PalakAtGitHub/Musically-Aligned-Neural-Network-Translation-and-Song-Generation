"""
Training script for MCNST (Two-Stage Architecture)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm

from src.data.song_dataset import SongTranslationDataset
from src.models.mcnst_model import MCNST


class Trainer:
    def __init__(self,
                 data_path: str,
                 save_dir: str = "checkpoints",
                 batch_size: int = 4,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 10,
                 use_cpu_mode: bool = True,
                 test_data_path: str = None):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        full_dataset = SongTranslationDataset(data_path)
        
        # Split train/val (90/10 of the training file)
        # The held-out test set is a separate .pt file (created by fma_data_builder).
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size]
        )
        
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size:   {len(self.val_dataset)}")

        # Optional held-out test set (separate file, 10% of songs held back at build time)
        self.test_loader = None
        if test_data_path:
            test_dataset = SongTranslationDataset(test_data_path)
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=test_dataset.collate_fn
            )
            print(f"Test size:  {len(test_dataset)}")
        
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
            freeze_decoder_layers=16 if use_cpu_mode else 0
        )
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            # Allow MPS to use all available memory
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
        print(f"Device: {self.device}")
        
        # Optimizer — only optimize trainable parameters
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )

        self.num_epochs = num_epochs
        self.history = []
    
    def _unpack_batch(self, batch):
        """Move batch tensors to device and return model kwargs."""
        src_ids = batch['src_ids'].to(self.device)
        tgt_ids = batch['tgt_ids'].to(self.device)
        melody = batch['melody_features'].to(self.device)
        num_notes = batch['num_notes'].to(self.device)
        tgt_syllables = batch['tgt_syllables'].to(self.device)

        stress_pattern = batch.get('stress_pattern')
        if stress_pattern is not None:
            stress_pattern = stress_pattern.to(self.device)
        beat_pattern = melody[:, :, 4]

        return dict(
            input_ids=src_ids,
            melody_features=melody,
            labels=tgt_ids,
            num_notes=num_notes,
            tgt_syllables=tgt_syllables,
            stress_pattern=stress_pattern,
            beat_pattern=beat_pattern,
        )

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            loss, loss_dict = self.model(**self._unpack_batch(batch))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            # Free MPS memory periodically
            if self.device.type == 'mps' and step % 50 == 0:
                torch.mps.empty_cache()

            postfix = {
                'loss': f"{loss.item():.4f}",
                'trans': f"{loss_dict['translation_loss']:.3f}",
                'syl': f"{loss_dict['syllable_loss']:.3f}",
            }
            if loss_dict.get('naturalness_loss', 0) > 0:
                postfix['nat'] = f"{loss_dict['naturalness_loss']:.3f}"
            if loss_dict.get('rhythm_loss', 0) > 0:
                postfix['rhy'] = f"{loss_dict['rhythm_loss']:.3f}"
            if loss_dict.get('rhyme_loss', 0) > 0:
                postfix['rhm'] = f"{loss_dict['rhyme_loss']:.3f}"
            pbar.set_postfix(postfix)

        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss, _ = self.model(**self._unpack_batch(batch))
                total_loss += loss.item()

        return total_loss / max(len(self.val_loader), 1)
    
    def evaluate_test_set(self):
        """Evaluate on held-out test set (called once after training)."""
        if self.test_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                loss, _ = self.model(**self._unpack_batch(batch))
                total_loss += loss.item()

        test_loss = total_loss / max(len(self.test_loader), 1)
        print(f"\n{'='*60}")
        print(f"Held-out Test Loss: {test_loss:.4f}")
        print(f"{'='*60}\n")
        return test_loss

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

        # Final evaluation on held-out test set
        self.evaluate_test_set()


if __name__ == "__main__":
    import os
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)

    trainer = Trainer(
        data_path="src/data/processed/fma_train_data.pt",
        test_data_path="src/data/processed/fma_test_data.pt",
        batch_size=4,
        num_epochs=10
    )
    trainer.train()