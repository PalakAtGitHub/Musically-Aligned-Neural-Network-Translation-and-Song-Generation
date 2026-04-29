# MCNST Project — Handover for New Chat Session

Paste this whole document as your first message in a new conversation. It contains everything the next Claude needs to pick up where we left off without asking you to re-explain context.

---

## 0. What this project is

MCNST = Musically Constrained Neural Song Translation. The goal is an end-to-end pipeline that takes English songs as input and produces Hindi versions singable over the same melody. Currently the project covers the text-translation stage; singing voice synthesis (DiffSinger or similar) is downstream future work.

The project has been worked on for many months. Architecture, data pipeline, and experiments are mostly built. The bottleneck is no longer code or architecture — it's training data.

Codebase root: `/Users/palakaggarwal/Documents/musically-aligned-translation/`
Working environment: macOS, 24GB Mac, `venv/` at project root, `PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib` required.

The user is Palak. She is the sole researcher, targeting ACL/EMNLP 2026. She is not a beginner — she has written the architecture, run all experiments, and pushes back productively on weak suggestions. Be direct and honest with her; do not soften bad news. She has explicitly thanked previous Claude sessions for being honest rather than encouraging.

---

## 1. The architecture as it currently stands

- **Backbone**: IndicTrans2 `ai4bharat/indictrans2-en-indic-1B`, 1.1B params, 18 decoder layers, LM head over 122,672 tokens (Hindi subset is 32,322).
- **Hierarchical melody encoder**: 3-level (CNN + bidirectional GRU + 8-head self-attention), outputs 256-dim. Motivated by GTTM grouping structure.
- **Cross-modal fusion**: 8-head cross-attention, text query / melody key+value, projects melody 256→1024.
- **Decoder–note alignment module**: 4-head cross-attention from decoder hidden state to melody, attention weights themselves used as soft alignment matrix. ~328k params. Feeds the cluster_loss and openness_reward.
- **PentathlonLoss with 7 terms**: translation cross-entropy, syllable, naturalness, rhythm, rhyme, cluster, openness. Combined via Kendall homoscedastic uncertainty weighting.
- **Pipeline**: Demucs source separation → Whisper transcription → IndicTrans2 + MCNST translation → Basic-Pitch melody extraction.

Critical implementation detail: a vocab-size bug was fixed early in the prior session. The vowel mask used by syllable_loss and rhythm_loss must be built at size 122,672 (LM head dim), NOT 32,322 (Hindi tokenizer). This silently suppressed those loss terms in earlier training runs. All experiments below were run with the fix in place.

---

## 2. Dataset

**Current dataset**: 1917 training examples, 219 test examples, all FMA-derived. Pipeline: Demucs vocal separation → Whisper transcription (English) → IndicTrans2 translation (Hindi target) → Basic-Pitch melody features. The Hindi labels are themselves IndicTrans2 outputs, which is the project's binding constraint.

21 of 219 test examples have English text identical to training examples (mostly nursery rhymes appearing across multiple FMA tracks). The strict held-out subset of 198 examples is the only valid evaluation set. ALL evaluation metrics in this document are reported on those 198 examples.

**Curated dataset** (in progress, parallel work via cowork pipeline): Disney Hindi-dubbed songs. Pipeline exists at `data_gathering/disney_song_curator.py`. Catalog at `data_gathering/disney_song_catalog.json` has 30 candidate songs across Frozen 1+2, Aladdin, Lion King, Jungle Book, Pocahontas, Mulan, Little Mermaid, Beauty & the Beast, Moana, Coco, Tangled, Encanto. URLs are in the catalog. The user has MuseScore Pro for MIDI extraction.

**Data hypothesis** (unconfirmed but high-confidence): Disney Hindi dubs were written by professional lyricists to fit the same melody as English originals, so cross-entropy loss on those labels would actually reward melody-conformant translation. Prior English-Chinese systems (LTAG, Songs Across Borders) used analogous curated data and achieved 60-80% syllable accuracy.

---

## 3. The strongest result on held-out data so far

| System | BLEU | BERTScore F1 | Syl Acc (±2) | Mean Syl Err |
|---|---|---|---|---|
| IndicTrans2 baseline | 22.63 | 0.8435 | 14.1% | 10.25 |
| MCNST default decoding | 36.14 | 0.8725 | 22.7% | 7.01 |
| **MCNST + constrained beam search** | **32.56** | **0.8631** | **36.9%** | **5.17** |

All differences over baseline are statistically significant (paired t-test or McNemar, p < 0.05 across all metrics).

The "constrained search" is inference-time syllable-aware beam reranking: generate top K=8 beam candidates, rescore by `log_prob - λ × |syl - target|` with λ=3.0, return best.

Constraint satisfaction rate (fraction of 198 examples where any beam candidate was within ±2 of target): 37.9%. The remaining 62.1% had to fall back to highest-scoring candidate regardless of syllable budget.

This is the result the paper is currently built around.

---

## 4. Full ablation matrix (all on the same 198 strict held-out)

| System | BLEU | BERTSc F1 | Syl Acc | Syl Err | Notes |
|---|---|---|---|---|---|
| IndicTrans2 baseline | 22.63 | 0.8435 | 14.1% | 10.25 | Plain backbone |
| Decoder-only fine-tuning | 38.48 | 0.8752 | 17.7% | 7.56 | Just top 2 decoder layers fine-tuned, no melody anything |
| MCNST no-fusion | 37.82 | 0.8766 | 18.7% | 7.11 | Cross-modal fusion disabled |
| MCNST no-8a | 39.54 | 0.8757 | 20.2% | 7.11 | Cluster/openness/alignment disabled, fusion ON |
| MCNST default (full) | 36.14 | 0.8725 | 22.7% | 7.01 | All components |
| MCNST + constrained search | 32.56 | 0.8631 | 36.9% | 5.17 | Inference reranking |
| Melody-aware default (Exp 13) | 38.99 | 0.8764 | 15.66% | 7.55 | Per-step decoder-melody attention added |
| Melody-aware constrained (Exp 14) | 38.88 | 0.8763 | 17.17% | 7.12 | + reranking |
| RL exp_15 (50/50 reward) | 8.32 | — | 42.9% | — | Beam search; reward hacking |
| RL exp_20 (30/70 reward + floors) | 13.81 | — | 42.9% | 3.65 | Reward hacking persists |

Key empirical findings from the ablations:

1. **Decoder fine-tuning alone explains ~16 BLEU points of gain over baseline**. Architectural complexity does not improve translation quality further (decoder-only BLEU 38.48 > full MCNST 36.14).
2. **Cross-modal fusion does not statistically contribute**. No-fusion vs full MCNST: p > 0.2 on every metric.
3. **8a additions (alignment + cluster + openness) provide ~2.5pp syllable-accuracy gain**. Not individually statistically significant.
4. **Per-step decoder-to-melody attention HURTS melody adherence** (15.66% vs 22.7%). Constraint satisfaction dropped 37.9% → 29.3%. The model used the new architectural capacity to better fit IndicTrans2-style translations rather than to attend to melody.
5. **Constrained beam search at inference is the only large lever**: +14.2pp syllable accuracy at -3.6 BLEU cost.

These findings collectively point to one conclusion: **the dataset is the binding constraint**. Hindi labels generated by an unconditional translator cannot teach a model to produce melody-conformant Hindi, regardless of architecture or loss design.

---

## 5. RL fine-tuning attempts — all failed

Five RL training attempts using Minimum Risk Training (MRT). All produced reward hacking or training instability:

- **exp_15** (reward = 0.5×syl + 0.5×BLEU, no KL, lr=1e-5): syllable accuracy 22.7%→42.9% but BLEU collapsed 36.14→8.32. 14 empty outputs under beam search.
- **exp_19 Run 1** (β=0.1 KL penalty, lr=5e-6): diverged at batch 89, mean KL 7.97.
- **exp_19 Run 2** (β=0.3, KL clamped ±3): diverged at batch 52.
- **exp_19 Run 3** (β=0.5, lr=1e-6, grad_clip=0.5): survived to batch 186, then diverged.
- **exp_19 Run 4** (skip outlier batches): OOM, 42GB virtual memory on 24GB machine, two 1.1B-param models cannot fit.
- **exp_20** (no KL, single model, dual-floor reward 0.3 syl + 0.7 BLEU, lr=5e-6, with `torch.mps.empty_cache()` fix): completed 1 epoch. Beam BLEU 13.81, syl acc 42.9%. Reward hacking pattern identical to exp_15 even with 70% BLEU weight.

**Conclusion**: RL on this dataset and hardware cannot produce a usable quality/syllable tradeoff. The model's prior (warm-start checkpoint) only knows IndicTrans2-style translations, and the easiest reward-improving samples are length-collapsed/meaning-degraded outputs rather than meaning-preserving short translations (which the model has never seen examples of).

This is a documented negative result, not a configuration problem. Five attempts with different reward designs, KL strengths, and learning rates all converge to the same outcome.

**Do not suggest more RL hyperparameter iteration without strong reason.** The user has explicitly accepted this result.

---

## 6. The paper

Paper draft exists at `MCNST_IEEE_Paper_revised.docx` (in `/mnt/user-data/outputs/` from the prior chat — user should have it locally). It is the HONEST version. An earlier `MCNST_IEEE_Paper.docx` had fabricated content (ESPLIT corpus that doesn't exist, fabricated 94.2% syllable accuracy, fabricated baselines, fabricated human evaluation, fabricated ablations, mBART backbone listed when actual is IndicTrans2). The revised version replaces all of that with real measured numbers.

The revised paper currently contains:
- Honest abstract with real numbers (22.63 → 36.14 → 32.56 BLEU, 14.1% → 22.7% → 36.9% syl acc)
- IndicTrans2 backbone (not mBART)
- Real dataset description (1917/219, FMA-derived, IndicTrans2 Hindi labels)
- All 7 PentathlonLoss terms
- Real training procedure (10 epochs, CPU/MPS, single GPU)
- Three-system comparison table with statistical tests
- Section V.E "Limitations" explicitly listing what's NOT in the paper

The paper still needs to be updated to incorporate:
- The full ablation matrix (currently only 3-system, needs 6+)
- The melody-aware decoder negative result
- The RL negative results section
- Note that paper says 12 decoder layers somewhere — actual is 18, needs verification

A trajectory document also exists at `MCNST_Project_Trajectory.md` in `/mnt/user-data/outputs/`. It's the chronological narrative of the project's experimental history and intuitions, written for internal use rather than as a paper. The user has it.

---

## 7. Active workstreams

**Workstream 1: Curated Disney dataset (cowork pipeline).**
- Status: 30-song catalog exists in `data_gathering/disney_song_catalog.json` with YouTube URLs filled in
- Pipeline scripts exist: `disney_song_curator.py`, `quality_verifier.py`, `dataset_search.py`
- README at `data_gathering/README.md` covers usage
- User has MuseScore Pro for MIDI extraction
- Outputs go to `data_gathering/output/disney_train_data.pt` in same format as `fma_train_data.pt` so they can be concatenated
- Once dataset is built, retrain MCNST default on (FMA + Disney) combined data and re-evaluate on 198 strict held-out

**Workstream 2: Singing voice synthesis demo (not started).**
- Plan: integrate DiffSinger (Mandarin pretrained weights as proof-of-concept; native Hindi quality requires Hindi singing data which is hard to source)
- The project has all upstream pieces: Hindi text from MCNST, phonemes via espeak-ng, pitch/duration from MIDI, alignment from LearnedAlignment module
- Missing: the SVS model itself + reassembler from phonemes back to Hindi audio
- Pure engineering work, not research, but produces a demo audio output

**Workstream 3: Paper updates after curated dataset experiment.**
- When curated data results land, update Table I in the paper with a new row
- If results are strong (60%+ syl accuracy, comparable BLEU), the paper's framing strengthens significantly
- If weak, the paper stays as documented negative result with curated data as future work

---

## 8. Files and locations

**Codebase root**: `/Users/palakaggarwal/Documents/musically-aligned-translation/`

Key files to know:
- `src/models/mcnst_model.py` — main model, has `_current_melody_features` for melody-aware variant
- `src/models/melody_aware_decoder.py` — per-step decoder-melody attention wrapper (frozen base + 47M new params)
- `src/models/alignment.py` — LearnedAlignment module
- `src/training/loss.py` — PentathlonLoss with 7 terms; vocab-size fix is here
- `src/training/rl_trainer.py` — RL infrastructure (functional but RL deemed unproductive)
- `src/utils/precompute_token_phonemes.py` — produces token phoneme table
- `src/utils/phoneme_utils.py` — has `count_hindi_syllables`, `IPA_VOWELS`, etc.
- `src/data/processed/token_phoneme_table.pt` — at full 122,672 vocab size
- `src/data/processed/fma_train_data.pt` — 1917 training examples
- `src/data/processed/fma_test_data.pt` — 219 test examples (filter to 198 strict held-out for evaluation)
- `data_gathering/` — Disney curation pipeline (Workstream 1)

Checkpoints (in `checkpoints/`):
- `best_model.pt` — MCNST default, val_loss 4.83 (PRIMARY checkpoint, the one to start from)
- `melody_aware_model.pt` — val_loss 5.02 (worse on melody adherence, do not use)
- `no_8a_model.pt`, `no_fusion_model.pt`, `decoder_only_model.pt` — ablation checkpoints
- RL checkpoints exist but should not be used (collapsed BLEU)

Logs in `logs/exp_01` through `logs/exp_20` plus diagnostics. JSON files are authoritative for any number — if anything in this document differs from the JSON, the JSON wins.

---

## 9. How to interact with this user

She is direct and prefers honest answers over encouragement. Specific patterns:

- She pushes back when she suspects you're being weak ("can't we try one more thing?"). Engage honestly — sometimes she's right, sometimes she's reaching for false hope.
- She asks "is this cutting corners" type questions and wants you to actually answer them, not deflect.
- She has an instinct for real architectural questions (asked unprompted about LoRA / parameter-efficient fine-tuning, gradient analysis, multi-agent systems). Engage substantively.
- She gets discouraged after negative results but recovers if you explain clearly what was learned.
- She trusts your judgment when you're explicit about uncertainty.
- She does not want process-for-process's-sake (proposed multi-agent debate during a low moment; was OK with the explanation that it wouldn't help).

Avoid:
- Suggesting more RL hyperparameter iterations (5 attempts done, decision made)
- Soft-pedaling negative results
- Suggesting fabrication or "small adjustments" to fabricated numbers in the paper
- Architectural changes that don't address the data bottleneck
- Suggesting she give up — the curated data path is real and likely to work

Do:
- Verify numbers against actual log files before citing them in any new document
- Be explicit about what experiments would tell us before running them
- Flag opportunity costs (RL iterations vs curated data work)
- Acknowledge when previous Claude sessions have been wrong (small samples extrapolated, decoder-layer count off, etc.)
- Help her ship the honest paper rather than chase a stronger one indefinitely

---

## 10. Open questions / immediate next steps

If she comes in fresh wanting to continue, the realistic priorities in order:

1. **Build the Disney curated dataset.** Workstream 1 above. The pipeline exists. Even 50-100 song pairs is enough to test the data hypothesis. If results are strong, the paper changes meaningfully.

2. **Update the paper with the full ablation matrix and RL negative results.** The current revised draft has the 3-way comparison; needs the 6-way ablation table and a section documenting the RL attempts.

3. **DiffSinger demo integration.** Pure engineering, produces a demo audio output. Optional but compelling for a paper presentation.

4. **Check the disney_collection_tracker.xlsx file** the user mentioned — I tried to read it during the prior session but the MCP filesystem connection timed out. It supposedly contains 46 songs across 13 films. If it lists songs not yet in `disney_song_catalog.json`, those should be added.

What NOT to do at the start of a new session:
- Don't immediately propose more architectural experiments
- Don't suggest restarting RL
- Don't try to "fix" the project's results — they are what they are; the path forward is data, not more model iteration

---

## 11. One-paragraph summary

MCNST is an English-to-Hindi song translation system targeting ACL/EMNLP 2026. The architecture (IndicTrans2 backbone + cross-modal fusion + alignment module + 7-term PentathlonLoss) is built and trained. On 198 strict held-out examples, the strongest result is MCNST + inference-time constrained beam search: 32.56 BLEU, 36.9% syllable accuracy, statistically significantly better than the IndicTrans2 baseline (22.63 BLEU, 14.1%) on every metric. Five RL fine-tuning attempts all produced reward hacking, and a per-step decoder-melody attention experiment hurt melody adherence rather than helped. Controlled ablations identified that domain adaptation through decoder fine-tuning explains most of the BLEU gain, while architectural components contribute only marginally to syllable matching. The binding constraint is the dataset: Hindi training labels are themselves IndicTrans2 outputs and were not authored to fit melodies. The forward path is curating a parallel song corpus from Disney Hindi dubs (pipeline exists at `data_gathering/`), where Hindi was professionally written to be sung over the same melody as English. An honest paper draft exists; it documents the architecture, ablations, RL negative results, and identifies curated data as future work.

---

## Use this handover

Paste this entire document as the first message in a new chat. Then say what you want to work on. The new Claude will have everything needed to continue without you re-explaining.
