"""
MCNST - Musically Constrained Neural Song Translation

Two-Stage Architecture (Grounded in Cognitive Science):
  Stage 1: Semantic Translation   (Ventral pathway — meaning)
  Stage 2: Musical Adaptation     (Dorsal pathway — rhythm/motor constraints)

Training:
  English → mBART encoder (frozen) → fuse with melody → mBART decoder → Hindi
  Loss = PentathlonLoss (translation + syllable + alignment)

Inference:
  Option A: generate()              — standard melody-fused generation
  Option B: generate_two_stage()    — draft then re-decode with melody
  Option C: generate_constrained()  — syllable-constrained beam search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from src.models.hierarchical_melody_encoder import HierarchicalMelodyEncoder
from src.models.fusion import CrossModalFusion
from src.training.loss import PentathlonLoss
from src.utils.syllable_utils import count_hindi_syllables
from src.utils.phoneme_utils import rhyme_similarity


class MCNST(nn.Module):
    """
    Two-Stage Musically Constrained Neural Song Translation.
    
    Stage 1 (Semantic): Frozen mBART translates English → Hindi
    Stage 2 (Musical):  Melody features fuse into the representation,
                        constrained decoding enforces singability
    """
    
    def __init__(self,
                 pretrained_model="facebook/mbart-large-50-many-to-many-mmt",
                 melody_hidden=256,
                 freeze_encoder=True,
                 freeze_decoder_layers=10):
        super().__init__()
        
        print("Initializing MCNST (Two-Stage)...")
        
        # === SHARED mBART BACKBONE ===
        print(f"  Loading mBART from {pretrained_model}...")
        self.mbart = MBartForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained_model)
        
        self.text_dim = self.mbart.config.d_model  # 1024 for mBART-large
        self.pad_token_id = self.mbart.config.pad_token_id  # 1
        
        # === FREEZING STRATEGY ===
        if freeze_encoder:
            print("  🔒 Freezing mBART encoder...")
            for param in self.mbart.model.encoder.parameters():
                param.requires_grad = False
        
        if freeze_decoder_layers > 0:
            num_layers = len(self.mbart.model.decoder.layers)
            freeze_count = min(freeze_decoder_layers, num_layers)
            print(f"  🔒 Freezing bottom {freeze_count}/{num_layers} decoder layers...")
            for i in range(freeze_count):
                for param in self.mbart.model.decoder.layers[i].parameters():
                    param.requires_grad = False
        
        # === STAGE 2 COMPONENTS (always trainable) ===
        print("  ✓ Initializing hierarchical melody encoder...")
        self.melody_encoder = HierarchicalMelodyEncoder(
            input_dim=5,
            conv_channels=128,
            gru_hidden=melody_hidden,
            output_dim=melody_hidden
        )
        
        print("  ✓ Initializing cross-modal fusion...")
        self.fusion = CrossModalFusion(
            text_dim=self.text_dim,
            melody_dim=melody_hidden
        )
        
        # === LOSS ===
        self.loss_fn = PentathlonLoss()
        
        # === PARAMETER SUMMARY ===
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"✓ MCNST initialized (Two-Stage)")
        print(f"  Text dim:   {self.text_dim}")
        print(f"  Melody dim: {melody_hidden}")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"  Frozen params:    {frozen:,} ({100*frozen/total:.1f}%)")
    
    # ================================================================
    # CORE HELPERS
    # ================================================================
    
    def _encode_and_fuse(self, input_ids, melody_features):
        """
        Encode text with mBART, encode melody, fuse them.
        
        This is the SHARED logic used by forward(), generate(), and beam search.
        
        Args:
            input_ids:       [batch, seq_len] — tokenized text
            melody_features: [batch, num_notes, 5]
        
        Returns:
            encoder_outputs: BaseModelOutput with fused last_hidden_state
            attention_mask:  [batch, seq_len]
            attn_weights:    [batch, seq_len, num_notes] fusion attention map
        """
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 1. Encode text with mBART encoder (frozen)
        encoder_outputs = self.mbart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 2. Encode melody hierarchically
        melody_encoded = self.melody_encoder(melody_features)
        
        # 3. Create melody padding mask (True = ignore position)
        melody_mask = (melody_features.sum(dim=-1) == 0)
        
        # 4. Fuse: text attends to melody
        fused_features, attn_weights = self.fusion(
            text_features=encoder_outputs.last_hidden_state,
            melody_features=melody_encoded,
            melody_mask=melody_mask
        )
        
        # 5. Replace encoder output — decoder now sees melody-aware representations
        encoder_outputs.last_hidden_state = fused_features
        
        return encoder_outputs, attention_mask, attn_weights
    
    # ================================================================
    # TRAINING
    # ================================================================
    
    def forward(self, input_ids, melody_features, labels=None,
                tgt_syllables=None, num_notes=None, **kwargs):
        """
        Training forward pass.
        
        English → mBART encoder → fuse with melody → mBART decoder → Hindi
        
        Args:
            input_ids:       [batch, src_len] — tokenized English
            melody_features: [batch, num_notes, 5] — MIDI features
            labels:          [batch, tgt_len] — tokenized Hindi (training)
            tgt_syllables:   [batch] — ground truth Hindi syllable counts
            num_notes:       [batch] — melody note counts (syllable target)
        
        Returns:
            If labels: (total_loss, loss_dict)
            Else: model outputs with logits
        """
        # Encode English + fuse with melody
        encoder_outputs, attention_mask, attn_weights = self._encode_and_fuse(
            input_ids, melody_features
        )
        
        if labels is not None:
            # === TRAINING MODE ===
            outputs = self.mbart(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            translation_loss = outputs.loss
            logits = outputs.logits  # [batch, seq_len, vocab]

            # Pass all available signals to PentathlonLoss
            loss_kwargs = dict(
                translation_loss=translation_loss,
                logits=logits,
                labels=labels,
                melody_features=melody_features,
            )
            if num_notes is not None and tgt_syllables is not None:
                loss_kwargs['predicted_syllables'] = tgt_syllables.float()
                loss_kwargs['target_syllables'] = num_notes.float()

            total_loss, loss_dict = self.loss_fn(**loss_kwargs)

            return total_loss, loss_dict
        else:
            # === INFERENCE MODE (get logits) ===
            return self.mbart(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                return_dict=True
            )
    
    # ================================================================
    # INFERENCE — Option A: Standard Generation
    # ================================================================
    
    def generate(self, input_ids, melody_features,
                 use_diverse_beam=False, **generation_kwargs):
        """
        Standard melody-fused generation with quality defaults.

        English → encode → fuse with melody → decode Hindi

        Args:
            use_diverse_beam: If True, uses diverse beam search with
                              num_beam_groups and diversity_penalty.
        """
        # Quality defaults (caller can override)
        generation_kwargs.setdefault('repetition_penalty', 2.5)
        generation_kwargs.setdefault('no_repeat_ngram_size', 2)

        if use_diverse_beam:
            num_beams = generation_kwargs.get('num_beams', 6)
            # Ensure num_beams is divisible by num_beam_groups
            generation_kwargs.setdefault('num_beam_groups', min(3, num_beams))
            generation_kwargs.setdefault('diversity_penalty', 1.0)
            generation_kwargs['num_beams'] = num_beams

        encoder_outputs, attention_mask, _ = self._encode_and_fuse(
            input_ids, melody_features
        )

        generated = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        return generated

    def generate_best_candidate(self, input_ids, melody_features,
                                num_candidates=6, **generation_kwargs):
        """
        Generate diverse candidates and rerank by self-perplexity.

        Uses diverse beam search to produce num_candidates translations,
        then scores each with the decoder's own cross-entropy and returns
        the lowest-perplexity candidate.

        Returns:
            (best_ids, all_candidates_ids)
        """
        gen_kwargs = dict(generation_kwargs)
        gen_kwargs['num_beams'] = num_candidates
        gen_kwargs['num_return_sequences'] = num_candidates
        gen_kwargs.setdefault('repetition_penalty', 2.5)
        gen_kwargs.setdefault('no_repeat_ngram_size', 2)
        gen_kwargs.setdefault('num_beam_groups', min(3, num_candidates))
        gen_kwargs.setdefault('diversity_penalty', 1.0)

        encoder_outputs, attention_mask, _ = self._encode_and_fuse(
            input_ids, melody_features
        )

        all_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **gen_kwargs
        )  # [num_candidates, seq_len]

        # Score each candidate by decoder NLL
        best_score = float('inf')
        best_idx = 0

        # Expand encoder outputs to match candidates
        enc_hidden = encoder_outputs.last_hidden_state  # [1, src_len, dim]

        with torch.no_grad():
            for i in range(all_ids.size(0)):
                cand = all_ids[i].unsqueeze(0)  # [1, tgt_len]
                outputs = self.mbart(
                    encoder_outputs=type(encoder_outputs)(
                        last_hidden_state=enc_hidden
                    ),
                    attention_mask=attention_mask,
                    labels=cand,
                    return_dict=True
                )
                nll = outputs.loss.item()
                if nll < best_score:
                    best_score = nll
                    best_idx = i

        return all_ids[best_idx].unsqueeze(0), all_ids
    
    # ================================================================
    # INFERENCE — Option B: Two-Stage Generation
    # ================================================================
    
    def generate_two_stage(self, input_ids, melody_features, **generation_kwargs):
        """
        Two-stage generation (mirrors how professional translators work):
        
        Stage 1: Plain mBART translation (no melody, frozen)
                 → draft Hindi (semantically accurate, may not be singable)
        
        Stage 2: Re-encode draft Hindi, fuse with melody, re-decode
                 → singable Hindi (melody-adapted)
        """
        # STAGE 1: Semantic translation (melody-agnostic)
        with torch.no_grad():
            draft_ids = self.mbart.generate(
                input_ids=input_ids,
                max_length=generation_kwargs.get('max_length', 50),
                num_beams=generation_kwargs.get('num_beams', 5),
                forced_bos_token_id=generation_kwargs.get('forced_bos_token_id', None)
            )
        
        # STAGE 2: Musical adaptation
        # Re-encode the Hindi draft and fuse with melody
        encoder_outputs, attention_mask, _ = self._encode_and_fuse(
            draft_ids, melody_features
        )
        
        # Re-decode with melody-aware representations
        adapted_ids = self.mbart.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        
        return adapted_ids
    
    # ================================================================
    # INFERENCE — Option C: Constrained Generation
    # ================================================================
    
    def generate_constrained(self, input_ids, melody_features,
                              target_syllables, num_beams=5,
                              target_rhyme=None, rhyme_weight=2.0, **kwargs):
        """
        Generation with syllable constraints and optional rhyme guidance.
        
        Uses SyllableConstrainedBeamSearch to enforce that generated
        Hindi text has the right number of syllables for the melody.
        If target_rhyme is provided, boosts beams that rhyme with it.
        
        Args:
            input_ids:         [1, src_len] — tokenized English
            melody_features:   [1, num_notes, 5]
            target_syllables:  int — number of syllables required
            num_beams:         beam width
            target_rhyme:      str — target rhyme ending or text to rhyme with
            rhyme_weight:      float — score multiplier for rhyme
        
        Returns:
            (token_ids, syllable_count)
        """
        searcher = SyllableConstrainedBeamSearch(
            model=self,
            tokenizer=self.tokenizer
        )
        return searcher.generate(
            input_ids=input_ids,
            melody_features=melody_features,
            target_syllables=target_syllables,
            num_beams=num_beams,
            target_rhyme=target_rhyme,
            rhyme_weight=rhyme_weight
        )


# ====================================================================
# SYLLABLE-CONSTRAINED BEAM SEARCH
# ====================================================================

class SyllableConstrainedBeamSearch:
    """
    Beam search that enforces a syllable budget during generation.
    
    At each decoding step:
    1. Expand each beam with top-k candidate tokens
    2. Count syllables in expanded sequence
    3. PRUNE candidates that violate syllable constraint
    4. Only allow EOS if syllable count is near target
    """
    
    def __init__(self, model, tokenizer, max_syllable_deviation=2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_deviation = max_syllable_deviation
    
    def generate(self, input_ids, melody_features, target_syllables, num_beams=5, target_rhyme=None, rhyme_weight=2.0):
        """
        Args:
            input_ids:         [1, src_len] — tokenized English (batch=1)
            melody_features:   [1, num_notes, 5]
            target_syllables:  int — number of syllables required
            num_beams:         beam width
        
        Returns:
            (best_token_ids, syllable_count)
        """
        device = input_ids.device
        
        # Pre-compute encoder outputs ONCE (efficiency)
        with torch.no_grad():
            encoder_outputs, attention_mask, _ = self.model._encode_and_fuse(
                input_ids, melody_features
            )
        
        # Determine BOS token (Hindi language code for mBART)
        hindi_lang_id = self.tokenizer.lang_code_to_id.get("hi_IN", 2)
        eos_id = self.tokenizer.eos_token_id
        
        # Initialize beams
        beams = [
            {
                'tokens': [hindi_lang_id],
                'score': 0.0,
                'syllable_count': 0
            }
        ]
        
        max_length = target_syllables + 10  # allow some buffer
        
        for step in range(max_length):
            candidates = []
            
            for beam in beams:
                # Skip finished beams
                if beam['tokens'][-1] == eos_id:
                    candidates.append(beam)
                    continue
                
                # Build decoder input
                decoder_ids = torch.tensor(
                    [beam['tokens']], dtype=torch.long, device=device
                )
                
                # Get next-token logits from mBART decoder
                with torch.no_grad():
                    outputs = self.model.mbart(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_ids,
                        return_dict=True
                    )
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Get top-k candidates
                top_k = num_beams * 2
                top_probs, top_ids = torch.topk(
                    F.softmax(logits, dim=-1), k=top_k
                )
                
                for prob, token_id in zip(top_probs, top_ids):
                    token_id_int = token_id.item()
                    new_tokens = beam['tokens'] + [token_id_int]
                    
                    # Count syllables of the full sequence so far
                    decoded_text = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )
                    new_syl_count = count_hindi_syllables(decoded_text)
                    
                    # CONSTRAINT: Don't exceed target + deviation
                    if new_syl_count > target_syllables + self.max_deviation:
                        continue  # prune
                    
                    extra_score = 0.0
                    # CONSTRAINT: Only allow EOS if near target
                    if token_id_int == eos_id:
                        if abs(new_syl_count - target_syllables) > self.max_deviation:
                            continue  # don't end yet
                        
                        # RHYME CONSTRAINT: Boost score if it rhymes well
                        if target_rhyme is not None:
                            similarity = rhyme_similarity(decoded_text, target_rhyme)
                            extra_score = similarity * rhyme_weight
                    
                    candidates.append({
                        'tokens': new_tokens,
                        'score': beam['score'] + torch.log(prob).item() + extra_score,
                        'syllable_count': new_syl_count
                    })
            
            if not candidates:
                break  # no valid candidates
            
            # Select top beams by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = candidates[:num_beams]
            
            # Stop if all beams have ended
            if all(b['tokens'][-1] == eos_id for b in beams):
                break
        
        # Return best beam
        best = max(beams, key=lambda x: x['score'])
        return best['tokens'], best['syllable_count']