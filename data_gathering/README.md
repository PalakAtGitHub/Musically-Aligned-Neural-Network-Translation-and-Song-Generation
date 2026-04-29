# data_gathering/ — Disney Song Curation Pipeline

Build high-quality musically-aligned Hindi-English translation training data from Disney dubbed songs.

> **READ THIS FIRST.** There are now **two** curation pipelines in this folder:
>
> 1. **`disney_lyrics_curator.py`** — *the current one to use*. Reads ground-truth Hindi lyrics from text files in `lyrics/` (sourced manually from Genius / lyricstranslate / fan wikis). Uses Whisper only for English-side timestamps.
> 2. `disney_song_curator.py` — *the original one, now deprecated*. Whisper-transcribes both Hindi and English audio. The Hindi side produced gibberish on `let_it_go` (see `output/qa_report.json`); kept around as a fallback only.
>
> Use the lyrics curator. Read `lyrics/README.md` for the file format. The rest of this document describes the original pipeline (still mostly accurate for the audio-handling parts that both share).

## Why This Exists

The FMA pipeline (`src/data/fma_data_builder.py`) creates training data where Hindi translations are **machine-generated** by IndicTrans2. That's a ceiling — the model can never learn to produce translations better than its own MT output.

Disney Hindi dubs are **human-translated to fit the melody**. A professional lyricist wrote Hindi lyrics that match the same rhythm, stress, and singability of the English original. This is ground-truth musically-aligned translation.

## Quick Start

```bash
# 1. Install dependencies
pip install yt-dlp openai-whisper demucs basic-pitch librosa transformers

# 2. Fill in YouTube URLs in the catalog
#    Edit data_gathering/disney_song_catalog.json
#    Replace REPLACE_WITH_HINDI_URL / REPLACE_WITH_ENGLISH_URL with real URLs

# 3. Run the pipeline
cd musically-aligned-translation/
python -m data_gathering.disney_song_curator

# 4. Verify quality
python -m data_gathering.quality_verifier

# 5. Search for additional datasets
python -m data_gathering.dataset_search
```

## Files

FilePurpose`disney_song_curator.py`Main pipeline: download → separate → transcribe → extract → align → package`disney_song_catalog.json`Song catalog with YouTube URLs (you fill these in)`quality_verifier.py`Post-curation QA: syllable ratios, melody coverage, duplicates`dataset_search.py`Search for existing academic datasets`DATASET_ETHICS.md`Legal/ethics framing for the paper's data section

## Pipeline Architecture

```
Hindi YouTube URL ──→ yt-dlp ──→ Demucs ──→ Whisper (hi) ──→ Hindi lyrics
                                                                    │
                                                              [timestamp align]
                                                                    │
English YouTube URL ─→ yt-dlp ──→ Demucs ──→ Whisper (en) ──→ English lyrics
                                    │
                                    └──→ Basic-Pitch ──→ melody [N, 5]
                                                              │
                                                    ┌─────────┘
                                                    ▼
                                        IndicTrans2 tokenize
                                                    │
                                                    ▼
                                         fma_train_data.pt format
                                         + QA flags for review
```

## Output Format

Each training example is a dict identical to `fma_data_builder.py` output, with one extra field:

- `source: "disney"` — so you can distinguish Disney data from FMA data

This means you can **directly concatenate** Disney and FMA training data:

```python
fma_data = torch.load("src/data/processed/fma_train_data.pt")
disney_data = torch.load("data_gathering/output/disney_train_data.pt")
combined = fma_data + disney_data
torch.save(combined, "src/data/processed/combined_train_data.pt")
```

## CLI Options

```
python -m data_gathering.disney_song_curator
  --catalog PATH     Song catalog JSON (default: data_gathering/disney_song_catalog.json)
  --max N            Process at most N songs
  --test-ratio 0.15  Held-out test fraction
  --whisper SIZE     tiny|base|small|medium|large (default: base)
  --device DEVICE    auto|mps|cuda|cpu (default: auto)
```
