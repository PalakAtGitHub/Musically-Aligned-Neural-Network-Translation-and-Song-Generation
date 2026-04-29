"""
MCNST - Musically Constrained Neural Song Translation

Two-Stage Architecture (Grounded in Cognitive Science):
  Stage 1: Semantic Translation   (Ventral pathway — meaning)
  Stage 2: Musical Adaptation     (Dorsal pathway — rhythm/motor constraints)

Backbone: IndicTrans2 (AI4Bharat) — en→indic translation (1B params)

Training:
  English → IndicTrans2 encoder (frozen) → fuse with melody → decoder → Hindi
  Loss = PentathlonLoss (translation + syllable + alignment)

Inference:
  Option A: generate()              — standard melody-fused generation
  Option B: generate_two_stage()    — draft then re-decode with melody
  Option C: generate_constrained()  — syllable-constrained beam search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.models.hierarchical_melody_encoder import HierarchicalMelodyEncoder
from src.models.fusion import CrossModalFusion
from src.models.alignment import LearnedAlignment
from src.training.loss import PentathlonLoss
from src.utils.syllable_utils import count_hindi_syllables
from src.utils.phoneme_utils import rhyme_similarity


class MCNST(nn.Module):
    """
    Two-Stage Musically Constrained Neural Song Translation.

    Stage 1 (Semantic): Frozen IndicTrans2 translates English → Hindi
    Stage 2 (Musical):  Melody features fuse into the representation,
                        constrained decoding enforces singability
    """

    SRC_LANG = "eng_Latn"
    TGT_LANG = "hin_Deva"

    def __init__(self,
                 pretrained_model="ai4bharat/indictrans2-en-indic-1B",
                 melody_hidden=256,
                 freeze_encoder=True,
                 freeze_decoder_layers=10):
        super().__init__()

        print("Initializing MCNST (Two-Stage, IndicTrans2)...")

        # === INDICTRANS2 BACKBONE ===
        print(f"  Loading IndicTrans2 from {pretrained_model}...")
        self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model, trust_remote_code=True
        )
        # Fix: transformers 5.x corrupts non-persistent sinusoidal buffers during
        # init_weights. Rebuild them after from_pretrained.
        self._fix_sinusoidal_buffers()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, trust_remote_code=True
        )

        # Model dimensions (IndicTrans2 uses encoder_embed_dim, not d_model)
        self.text_dim = self.seq2seq.config.encoder_embed_dim
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.seq2seq.config.pad_token_id
        )

        # === FREEZING STRATEGY ===
        # Get encoder reference
        if hasattr(self.seq2seq, 'model') and hasattr(self.seq2seq.model, 'encoder'):
            self._encoder = self.seq2seq.model.encoder
        else:
            self._encoder = self.seq2seq.get_encoder()

        if freeze_encoder:
            print("  Freezing IndicTrans2 encoder...")
            for param in self._encoder.parameters():
                param.requires_grad = False

        if freeze_decoder_layers > 0:
            if hasattr(self.seq2seq, 'model') and hasattr(self.seq2seq.model, 'decoder'):
                decoder = self.seq2seq.model.decoder
            else:
                decoder = self.seq2seq.get_decoder()

            decoder_layers = (
                decoder.layers if hasattr(decoder, 'layers')
                else decoder.block if hasattr(decoder, 'block')
                else []
            )
            num_layers = len(decoder_layers)
            freeze_count = min(freeze_decoder_layers, num_layers)
            print(f"  Freezing bottom {freeze_count}/{num_layers} decoder layers...")
            for i in range(freeze_count):
                for param in decoder_layers[i].parameters():
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

        # 8a: learned soft alignment between decoder output positions and
        # melody notes. Produces an [B, T_out, N] matrix where each row is a
        # probability distribution over notes, used by cluster_loss and
        # openness_reward to combine token-level and note-level features.
        print("  ✓ Initializing learned alignment module (8a)...")
        self.aligner = LearnedAlignment(
            text_dim=self.text_dim,
            melody_dim=melody_hidden,
            attn_dim=256,
            num_heads=4,
            dropout=0.1,
        )

        # === LOSS ===
        # Pass tokenizer so PentathlonLoss can build the vowel-bearing-token
        # mask needed for the differentiable syllable/rhythm losses.
        self.loss_fn = PentathlonLoss(
            tokenizer=self.tokenizer,
            pad_token_id=self.pad_token_id,
        )

        # === PARAMETER SUMMARY ===
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"✓ MCNST initialized (Two-Stage, IndicTrans2)")
        print(f"  Text dim:   {self.text_dim}")
        print(f"  Melody dim: {melody_hidden}")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"  Frozen params:    {frozen:,} ({100*frozen/total:.1f}%)")

    def _fix_sinusoidal_buffers(self):
        """Rebuild sinusoidal positional embedding buffers.

        transformers 5.x's init_weights/initialize_weights corrupts
        non-persistent buffers in custom (remote-code) modules.
        Unconditionally regenerate all sinusoidal pos-embed buffers.
        """
        for module in self.seq2seq.modules():
            if hasattr(module, 'make_weights') and hasattr(module, 'weights'):
                module.make_weights(
                    module.weights.size(0),
                    module.embedding_dim,
                    module.padding_idx,
                )

    # ================================================================
    # PREPROCESSING HELPERS
    # ================================================================

    def preprocess_src(self, texts):
        """Preprocess English source text(s) for IndicTrans2.

        IndicTrans2 tokenizer expects: 'src_lang tgt_lang text'
        """
        if isinstance(texts, str):
            texts = [texts]
        return [f"{self.SRC_LANG} {self.TGT_LANG} {t}" for t in texts]

    def postprocess_tgt(self, texts):
        """Postprocess Hindi target text(s) from IndicTrans2."""
        if isinstance(texts, str):
            texts = [texts]
        return texts

    def tokenize_tgt(self, text):
        """Tokenize Hindi target text using IndicTrans2 target vocabulary.

        IndicTrans2 has separate src/tgt SPM models and vocabularies.
        Must switch tokenizer to target mode before encoding Hindi.
        """
        self.tokenizer._switch_to_target_mode()
        ids = self.tokenizer.encode(
            text, return_tensors="pt",
            truncation=True, max_length=128
        )
        self.tokenizer._switch_to_input_mode()
        return ids

    # ================================================================
    # CORE HELPERS
    # ================================================================

    def _encode_and_fuse(self, input_ids, melody_features):
        """
        Encode text with IndicTrans2 encoder, encode melody, fuse them.

        Args:
            input_ids:       [batch, seq_len] — tokenized text
            melody_features: [batch, num_notes, 5]

        Returns:
            encoder_outputs: BaseModelOutput with fused last_hidden_state
            attention_mask:  [batch, seq_len]
            attn_weights:    [batch, seq_len, num_notes] fusion attention map
            melody_encoded:  [batch, num_notes, melody_dim] raw melody
                             features (needed by the 8a alignment module
                             as attention keys/values)
            melody_mask:     [batch, num_notes] True = padding
        """
        attention_mask = (input_ids != self.pad_token_id).long()

        # 1. Encode text with IndicTrans2 encoder (frozen)
        encoder_outputs = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 2. Ensure melody is long enough for CNN kernel (min 3 notes)
        min_notes = 3
        if melody_features.size(1) < min_notes:
            pad = min_notes - melody_features.size(1)
            melody_features = F.pad(melody_features, (0, 0, 0, pad))

        # 3. Encode melody hierarchically
        melody_encoded = self.melody_encoder(melody_features)

        # 4. Create melody padding mask (True = ignore position)
        melody_mask = (melody_features.sum(dim=-1) == 0)

        # 5. Fuse: text attends to melody
        fused_features, attn_weights = self.fusion(
            text_features=encoder_outputs.last_hidden_state,
            melody_features=melody_encoded,
            melody_mask=melody_mask
        )

        # 6. Replace encoder output — decoder now sees melody-aware representations
        encoder_outputs.last_hidden_state = fused_features

        return encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask

    # ================================================================
    # TRAINING
    # ================================================================

    def forward(self, input_ids, melody_features, labels=None,
                tgt_syllables=None, num_notes=None,
                stress_pattern=None, beat_pattern=None, **kwargs):
        """
        Training forward pass.

        English → IndicTrans2 encoder (frozen) → fuse with melody → decoder → Hindi
        """
        # Encode English + fuse with melody
        encoder_outputs, attention_mask, attn_weights, melody_encoded, melody_mask = \
            self._encode_and_fuse(input_ids, melody_features)

        if labels is not None:
            # === TRAINING MODE ===
            outputs = self.seq2seq(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,   # 8a: needed for alignment
                return_dict=True
            )

            translation_loss = outputs.loss
            logits = outputs.logits

            # 8a: compute soft alignment from decoder's last hidden state
            # to the raw melody features. Produces [B, T_out, N] matrix.
            # decoder_hidden_states is a tuple; [-1] is the last layer.
            decoder_last_hidden = outputs.decoder_hidden_states[-1]
            alignment = self.aligner(
                decoder_hidden=decoder_last_hidden,
                melody_encoded=melody_encoded,
                melody_mask=melody_mask,
            )

            # Pass all available signals to PentathlonLoss.
            # NB: with the gradient-fixed loss, the syllable target comes from
            # `num_notes` (melody note count) and the expected syllable count
            # is derived differentiably from `logits`. tgt_syllables is still
            # accepted for backward compat but not used by the new path.
            loss_kwargs = dict(
                translation_loss=translation_loss,
                logits=logits,
                labels=labels,
                melody_features=melody_features,
                alignment=alignment,   # 8a
            )
            if num_notes is not None:
                loss_kwargs['num_notes'] = num_notes.float()
            if stress_pattern is not None:
                loss_kwargs['stress_pattern'] = stress_pattern
                loss_kwargs['beat_pattern'] = (
                    beat_pattern if beat_pattern is not None
                    else melody_features[:, :, 4]
                )

            total_loss, loss_dict = self.loss_fn(**loss_kwargs)
            return total_loss, loss_dict
        else:
            # === INFERENCE MODE (get logits) ===
            return self.seq2seq(
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
        """
        generation_kwargs.setdefault('repetition_penalty', 2.5)
        generation_kwargs.setdefault('no_repeat_ngram_size', 2)
        # IndicTrans2's remote code has a bug where to_legacy_cache()
        # returns tuples with None entries, crashing beam search.
        # Disabling KV cache avoids the issue (slower but correct).
        generation_kwargs.setdefault('use_cache', False)

        if use_diverse_beam:
            num_beams = generation_kwargs.get('num_beams', 6)
            generation_kwargs.setdefault('num_beam_groups', min(3, num_beams))
            generation_kwargs.setdefault('diversity_penalty', 1.0)
            generation_kwargs['num_beams'] = num_beams

        encoder_outputs, attention_mask, _, _, _ = self._encode_and_fuse(
            input_ids, melody_features
        )

        generated = self.seq2seq.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **generation_kwargs
        )

        return generated

    def generate_best_candidate(self, input_ids, melody_features,
                                num_candidates=6, **generation_kwargs):
        """
        Generate diverse candidates and rerank by self-perplexity.
        """
        gen_kwargs = dict(generation_kwargs)
        gen_kwargs['num_beams'] = num_candidates
        gen_kwargs['num_return_sequences'] = num_candidates
        gen_kwargs.setdefault('repetition_penalty', 2.5)
        gen_kwargs.setdefault('no_repeat_ngram_size', 2)

        encoder_outputs, attention_mask, _, _, _ = self._encode_and_fuse(
            input_ids, melody_features
        )

        all_ids = self.seq2seq.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **gen_kwargs
        )

        # Score each candidate by decoder NLL
        best_score = float('inf')
        best_idx = 0
        enc_hidden = encoder_outputs.last_hidden_state

        with torch.no_grad():
            for i in range(all_ids.size(0)):
                cand = all_ids[i].unsqueeze(0)
                outputs = self.seq2seq(
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
        Two-stage generation:
        Stage 1: Plain IndicTrans2 translation (no melody)
        Stage 2: Re-encode draft, fuse with melody, re-decode
        """
        # STAGE 1: Semantic translation (melody-agnostic)
        with torch.no_grad():
            draft_ids = self.seq2seq.generate(
                input_ids=input_ids,
                max_length=generation_kwargs.get('max_length', 50),
                num_beams=generation_kwargs.get('num_beams', 5),
                use_cache=False,
            )

        # STAGE 2: Musical adaptation
        generation_kwargs.setdefault('use_cache', False)
        encoder_outputs, attention_mask, _, _, _ = self._encode_and_fuse(
            draft_ids, melody_features
        )

        adapted_ids = self.seq2seq.generate(
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
# SYLLABLE-SOFT BEAM SEARCH  (post-generation ranking)
# ====================================================================

class SyllableConstrainedBeamSearch:
    """
    Soft syllable guidance via post-generation ranking.

    Runs standard beam search to produce N complete candidates,
    then re-ranks by a joint score:
        score = model_log_prob
                - syllable_penalty * |syl_count - target| / target
                + rhyme_bonus  (if target_rhyme provided)
    """

    def __init__(self, model, tokenizer, max_syllable_deviation=2):
        self.model = model
        self.tokenizer = tokenizer
        self.max_deviation = max_syllable_deviation

    def generate(self, input_ids, melody_features, target_syllables,
                 num_beams=8, target_rhyme=None, rhyme_weight=2.0,
                 syllable_penalty=3.0):
        # Encode + fuse melody ONCE
        with torch.no_grad():
            encoder_outputs, attention_mask, _, _, _ = self.model._encode_and_fuse(
                input_ids, melody_features
            )

        # Generate N candidates
        with torch.no_grad():
            all_ids = self.model.seq2seq.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max(50, target_syllables * 3),
                repetition_penalty=2.5,
                no_repeat_ngram_size=2,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequences = all_ids.sequences
        seq_scores = all_ids.sequences_scores

        # Score each candidate
        best_score = float('-inf')
        best_ids = sequences[0]
        best_syl = 0

        for i in range(sequences.size(0)):
            text = self.tokenizer.decode(sequences[i], skip_special_tokens=True)
            # Postprocess if IndicProcessor available
            processed = self.model.postprocess_tgt(text)
            text = processed[0] if processed else text

            syl_count = count_hindi_syllables(text)
            syl_dev = abs(syl_count - target_syllables) / max(target_syllables, 1)
            model_score = seq_scores[i].item()

            rhyme_bonus = 0.0
            if target_rhyme is not None:
                rhyme_bonus = rhyme_similarity(text, target_rhyme) * rhyme_weight

            joint_score = model_score - syllable_penalty * syl_dev + rhyme_bonus

            if joint_score > best_score:
                best_score = joint_score
                best_ids = sequences[i]
                best_syl = syl_count

        return best_ids.unsqueeze(0), best_syl
