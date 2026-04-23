"""
RL Fine-Tuning via Minimum Risk Training (MRT)

This file implements Approach 2 of the MCNST project ablation:

  Approach 1 (supervised):   src.training.train.Trainer
    — cross-entropy + differentiable pentathlon loss only.
    — checkpoint:  checkpoints/best_model.pt

  Approach 2 (supervised + RL):   this file
    — warm-start from the supervised checkpoint, then MRT fine-tune
      directly against the PentathlonReward (non-differentiable
      singability criteria).
    — checkpoint:  checkpoints/best_model_rl.pt

Running both lets you report a clean side-by-side comparison in the
viva: "here is what supervised MLE achieves, here is what the same
model achieves after RL on discrete musical rewards."

=== The MRT math (1-example form) ===

For input x, sample K candidate translations y_1..y_K from the model.
Score each with the reward R_k. Minimise

    L = sum_k  (R_k / Z) * NLL(y_k | x, melody)
    Z = sum_k R_k

This is policy gradient with self-normalised importance sampling.
High-reward candidates pull NLL down (= they become more likely);
low-reward candidates get ignored. No value network, no KL penalty —
appropriate for bounded rewards and short sequences.
"""

import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.song_dataset import SongTranslationDataset
from src.models.mcnst_model import MCNST
from src.training.reward import PentathlonReward


class RLTrainer:
    """
    Two-phase trainer: supervised warm-start then RL fine-tuning.

    If `warm_start_from` is passed, the supervised phase is skipped and
    the existing checkpoint weights are loaded — this is how you run
    RL as a standalone second stage on top of an already-trained
    supervised model (the recommended flow for the two-approach
    ablation).
    """

    def __init__(self,
                 data_path: str,
                 test_data_path: str = None,
                 save_dir: str = "checkpoints",
                 batch_size: int = 4,
                 lr_supervised: float = 5e-5,
                 lr_rl: float = 1e-5,
                 supervised_epochs: int = 5,
                 rl_epochs: int = 10,
                 num_candidates: int = 5,
                 temperature: float = 0.8,
                 warm_start_from: str = None,
                 save_name: str = "best_model_rl.pt"):
        """
        Args:
            warm_start_from:  Optional path to a supervised checkpoint.
                              If provided, loads those weights BEFORE the
                              RL phase and SKIPS the supervised warmstart
                              loop. Enables the supervised-vs-RL ablation.
            save_name:        Filename for the RL checkpoint. Defaults to
                              'best_model_rl.pt' so it does not clobber
                              the supervised checkpoint at 'best_model.pt'.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Hyperparams
        self.batch_size = batch_size
        self.lr_supervised = lr_supervised
        self.lr_rl = lr_rl
        self.supervised_epochs = supervised_epochs
        self.rl_epochs = rl_epochs
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.warm_start_from = warm_start_from
        self.save_name = save_name

        # ---- Dataset ----
        print("Loading dataset...")
        full_dataset = SongTranslationDataset(data_path)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        print(f"Train: {train_size}  Val: {val_size}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=full_dataset.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=full_dataset.collate_fn,
        )

        self.test_loader = None
        if test_data_path:
            test_ds = SongTranslationDataset(test_data_path)
            self.test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                collate_fn=test_ds.collate_fn,
            )
            print(f"Test:  {len(test_ds)}")

        # ---- Model ----
        print("Initializing model...")
        self.model = MCNST(freeze_encoder=True, freeze_decoder_layers=10)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        print(f"Device: {self.device}")

        # ---- Warm-start from a supervised checkpoint, if provided ----
        if warm_start_from is not None:
            ws_path = Path(warm_start_from)
            if ws_path.exists():
                print(f"Loading supervised checkpoint: {ws_path}")
                ckpt = torch.load(ws_path, map_location=self.device,
                                   weights_only=False)
                state = ckpt['model_state_dict']
                # strict=False in case the supervised ckpt's loss has
                # different log_var parameters than the RL-time loss.
                self.model.load_state_dict(state, strict=False)
                print(f"  ✓ Warm-started from epoch "
                      f"{ckpt.get('epoch', '?')} "
                      f"(val_loss={ckpt.get('val_loss', '?')})")
            else:
                print(f"  ⚠ warm_start_from={ws_path} not found — "
                      f"RL will start from base IndicTrans2 weights")

        self.trainable_params = [p for p in self.model.parameters()
                                 if p.requires_grad]

        # Reward function (operates on decoded strings — no gradient)
        self.reward_fn = PentathlonReward()
        self.history = []

    # ==================================================================
    # Phase 1 — Supervised warm-start (skipped when warm_start_from set)
    # ==================================================================

    def _supervised_epoch(self, epoch, optimizer):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"[SL] Epoch {epoch}")
        for batch in pbar:
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            mel = batch['melody_features'].to(self.device)
            notes = batch['num_notes'].to(self.device)

            # New loss signature: num_notes drives the syllable target,
            # logits drive the differentiable expected-syllable estimate.
            # tgt_syllables is accepted-but-ignored in the new loss.
            loss, ld = self.model(
                input_ids=src, melody_features=mel, labels=tgt,
                num_notes=notes,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             trans=f"{ld['translation_loss']:.3f}")

        return total_loss / len(self.train_loader)

    # ==================================================================
    # Phase 2 — RL fine-tuning via Minimum Risk Training
    # ==================================================================

    def _rl_epoch(self, epoch, optimizer):
        """
        One MRT epoch.

        For each batch:
          1. Sample K candidates per example (nucleus sampling)
          2. Score each with PentathlonReward (non-differentiable)
          3. MRT loss = reward-weighted NLL of candidates
          4. Backprop — gradient flows through NLL only,
             reward is treated as a black-box scalar.
        """
        self.model.train()
        total_loss = 0
        total_reward = 0
        count = 0

        pbar = tqdm(self.train_loader, desc=f"[RL] Epoch {epoch}")
        for batch in pbar:
            src = batch['src_ids'].to(self.device)
            mel = batch['melody_features'].to(self.device)
            notes = batch['num_notes']   # kept on CPU for reward function
            B = src.size(0)
            K = self.num_candidates

            # --- 1. Sample K candidates ---
            self.model.eval()
            with torch.no_grad():
                encoder_outputs, attention_mask, _ = \
                    self.model._encode_and_fuse(src, mel)
                # generate() with num_return_sequences does an in-place
                # repeat_interleave on encoder_outputs; clone first so we
                # can recompute per-candidate NLL cleanly afterwards.
                enc_hidden_orig = encoder_outputs.last_hidden_state.clone()
                attn_mask_orig = attention_mask.clone()

                all_candidates = self.model.seq2seq.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    num_return_sequences=K,
                    max_length=60,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=2,
                )  # [B*K, seq_len]
            self.model.train()

            # --- 2. Score each candidate with the pentathlon reward ---
            rewards = []
            for i in range(B * K):
                text = self.model.tokenizer.decode(
                    all_candidates[i], skip_special_tokens=True
                )
                processed = self.model.postprocess_tgt(text)
                text = processed[0] if processed else text

                b_idx = i // K
                n_b = notes[b_idx]
                n = int(n_b.item()) if torch.is_tensor(n_b) else int(n_b)
                mel_np = mel[b_idx].detach().cpu().numpy()

                r = self.reward_fn.score(text, n, melody_features=mel_np)
                rewards.append(r['total'])

            rewards_t = torch.tensor(rewards, device=self.device).view(B, K)

            # Self-normalised importance weights (MRT). eps avoids div-by-zero
            # when all K candidates score zero.
            Z = rewards_t.sum(dim=1, keepdim=True).clamp(min=1e-6)
            weights = rewards_t / Z  # [B, K]

            # --- 3. Compute per-candidate NLL under the current policy ---
            from transformers.modeling_outputs import BaseModelOutput
            enc_expanded = enc_hidden_orig.repeat_interleave(K, dim=0)
            attn_expanded = attn_mask_orig.repeat_interleave(K, dim=0)
            enc_out_expanded = BaseModelOutput(last_hidden_state=enc_expanded)

            # Manual cross-entropy: the seq2seq's internal .loss hits a
            # view() error on non-contiguous tensors produced by
            # repeat_interleave. .contiguous() avoids it.
            decoder_input_ids = all_candidates[:, :-1].contiguous()
            target_ids = all_candidates[:, 1:].contiguous()

            outputs = self.model.seq2seq(
                encoder_outputs=enc_out_expanded,
                attention_mask=attn_expanded,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
            )
            logits = outputs.logits  # [B*K, seq_len-1, vocab]

            per_token_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=self.model.pad_token_id,
                reduction='none',
            ).reshape(B * K, -1)

            mask = (target_ids != self.model.pad_token_id).float()
            per_candidate_nll = (per_token_nll * mask).sum(dim=1) / \
                                 mask.sum(dim=1).clamp(min=1)
            per_candidate_nll = per_candidate_nll.view(B, K)

            # --- 4. MRT loss: reward-weighted NLL ---
            # detach(weights) so gradients flow through NLL only.
            mrt_loss = (weights.detach() * per_candidate_nll).sum(dim=1).mean()

            optimizer.zero_grad()
            mrt_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
            optimizer.step()

            total_loss += mrt_loss.item()
            mean_r = rewards_t.mean().item()
            total_reward += mean_r
            count += 1

            pbar.set_postfix(mrt=f"{mrt_loss.item():.3f}",
                             reward=f"{mean_r:.3f}")

        avg_loss = total_loss / max(count, 1)
        avg_reward = total_reward / max(count, 1)
        return avg_loss, avg_reward

    # ==================================================================
    # Validation (reused across both phases)
    # ==================================================================

    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                mel = batch['melody_features'].to(self.device)
                notes = batch['num_notes'].to(self.device)
                loss, _ = self.model(
                    input_ids=src, melody_features=mel, labels=tgt,
                    num_notes=notes,
                )
                total_loss += loss.item()
        return total_loss / max(len(self.val_loader), 1)

    # ==================================================================
    # Full training loop
    # ==================================================================

    def train(self):
        best_val = float('inf')

        # --- Phase 1: supervised warm-start (skipped on warm_start_from) ---
        if self.warm_start_from is None:
            print(f"\n{'='*60}")
            print("Phase 1: Supervised Warm-Start")
            print(f"{'='*60}\n")

            optimizer = optim.AdamW(self.trainable_params,
                                     lr=self.lr_supervised, weight_decay=0.01)

            for epoch in range(1, self.supervised_epochs + 1):
                train_loss = self._supervised_epoch(epoch, optimizer)
                val_loss = self._validate()
                print(f"  Epoch {epoch}: train={train_loss:.4f}  "
                      f"val={val_loss:.4f}")
                self.history.append({
                    'phase': 'supervised', 'epoch': epoch,
                    'train_loss': train_loss, 'val_loss': val_loss,
                })
                if val_loss < best_val:
                    best_val = val_loss
                    self._save_checkpoint(epoch, val_loss, tag="supervised")
        else:
            print(f"\n{'='*60}")
            print(f"Skipping supervised phase — warm-started from "
                  f"{self.warm_start_from}")
            print(f"{'='*60}\n")

        # --- Phase 2: RL fine-tuning (MRT) ---
        print(f"\n{'='*60}")
        print("Phase 2: RL Fine-Tuning (Minimum Risk Training)")
        print(f"{'='*60}\n")

        # Lower LR for RL phase — avoid catastrophic forgetting of the
        # supervised translation behaviour while nudging toward reward.
        optimizer = optim.AdamW(self.trainable_params,
                                 lr=self.lr_rl, weight_decay=0.01)

        best_reward = 0.0
        for epoch in range(1, self.rl_epochs + 1):
            mrt_loss, avg_reward = self._rl_epoch(epoch, optimizer)
            val_loss = self._validate()
            print(f"  Epoch {epoch}: mrt={mrt_loss:.4f}  "
                  f"reward={avg_reward:.3f}  val={val_loss:.4f}")
            self.history.append({
                'phase': 'rl', 'epoch': epoch,
                'mrt_loss': mrt_loss, 'avg_reward': avg_reward,
                'val_loss': val_loss,
            })
            if avg_reward > best_reward:
                best_reward = avg_reward
                self._save_checkpoint(epoch, val_loss, tag="rl_best")

        # Save training history (tagged so it does not overwrite supervised)
        history_path = self.save_dir / "training_history_rl.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"RL training complete!")
        print(f"  Best supervised val: {best_val:.4f}"
              if self.warm_start_from is None
              else "  Supervised phase skipped")
        print(f"  Best RL reward:      {best_reward:.3f}")
        print(f"  Checkpoint:          {self.save_dir / self.save_name}")
        print(f"  History:             {history_path}")
        print(f"{'='*60}\n")

        self._evaluate_test()

    def _save_checkpoint(self, epoch, val_loss, tag="best"):
        """Save to self.save_name so the RL checkpoint does not overwrite
        the supervised one (which lives at checkpoints/best_model.pt)."""
        path = self.save_dir / self.save_name
        torch.save({
            'epoch': epoch,
            'tag': tag,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
        }, path)
        print(f"  → Saved {tag} checkpoint to {path.name} (epoch {epoch})")

    def _evaluate_test(self):
        """Final evaluation on the held-out test set, if provided."""
        if self.test_loader is None:
            return

        self.model.eval()
        total_reward = 0
        total_syl = 0
        count = 0

        with torch.no_grad():
            for batch in self.test_loader:
                src = batch['src_ids'].to(self.device)
                mel = batch['melody_features'].to(self.device)
                notes = batch['num_notes']

                gen_ids = self.model.generate(
                    src, mel, max_length=60, num_beams=5,
                )

                for i in range(gen_ids.size(0)):
                    text = self.model.tokenizer.decode(
                        gen_ids[i], skip_special_tokens=True
                    )
                    processed = self.model.postprocess_tgt(text)
                    text = processed[0] if processed else text

                    n_b = notes[i]
                    n = int(n_b.item()) if torch.is_tensor(n_b) else int(n_b)
                    r = self.reward_fn.score(
                        text, n,
                        melody_features=mel[i].detach().cpu().numpy()
                    )
                    total_reward += r['total']
                    total_syl += r['syllable']
                    count += 1

        if count:
            print(f"\n  Test Set ({count} examples):")
            print(f"    Avg Reward:    {total_reward/count:.3f}")
            print(f"    Avg Syl Match: {total_syl/count:.3f}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="RL fine-tuning for MCNST via Minimum Risk Training"
    )
    parser.add_argument('--data', type=str,
                        default="src/data/processed/fma_train_data.pt")
    parser.add_argument('--test-data', type=str,
                        default="src/data/processed/fma_test_data.pt")
    parser.add_argument('--sl-epochs', type=int, default=5,
                        help='Supervised epochs (ignored if --warm-start set)')
    parser.add_argument('--rl-epochs', type=int, default=10)
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Path to supervised checkpoint to warm-start '
                             'from (skips the supervised phase)')
    parser.add_argument('--save-name', type=str,
                        default="best_model_rl.pt",
                        help='Filename for the RL checkpoint')
    args = parser.parse_args()

    trainer = RLTrainer(
        data_path=args.data,
        test_data_path=args.test_data,
        supervised_epochs=args.sl_epochs,
        rl_epochs=args.rl_epochs,
        warm_start_from=args.warm_start,
        save_name=args.save_name,
    )
    trainer.train()
