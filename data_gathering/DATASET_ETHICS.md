# Dataset Ethics & Legal Framing

## For use in the paper's Data Section

### 1. Data Source & Justification

The Disney Hindi-English song translation dataset is constructed from **publicly available YouTube audio** of officially released Disney songs in their Hindi-dubbed and original English versions.

**Why Disney songs specifically:**

- Disney maintains professional Hindi dubs that are *musically aligned* — the Hindi lyrics are written to fit the same melody, rhythm, and emotional arc as the English original.
- This makes them rare examples of **ground-truth musically-aligned translation**, as opposed to machine-translated lyrics that ignore singability.
- The translations are performed by professional lyricists and singers, ensuring high linguistic quality.

### 2. Fair Use Analysis

This dataset is constructed for **non-commercial academic research** and falls within fair use / fair dealing under the following reasoning:

FactorAssessment**Purpose**Non-commercial research, transformative use (short audio segments processed into numerical features, not redistributed as audio)**Nature of work**Published creative works (commercially released songs)**Amount used**Short segments per song, processed into derived features (MIDI-like note arrays, tokenized text) — original audio is not stored or redistributed**Market effect**No substitute for original — dataset contains numerical tensors, not playable audio; research use does not compete with Disney's commercial market

### 3. What We Store vs. What We Don't

**Stored in the dataset (.pt files):**

- Tokenized text (integer IDs, not raw lyrics)
- Melody feature arrays (\[pitch, pitch_class, duration, dur_bin, beat_strength\])
- Syllable counts and stress patterns
- Song name identifiers

**NOT stored or redistributed:**

- Raw audio files (downloaded temporarily, processed, then optionally deleted)
- Complete raw lyrics text (only short segments for QA review)
- YouTube URLs in the training data

### 4. Reproducibility

The catalog file (`disney_song_catalog.json`) contains YouTube URLs which may become unavailable over time. To address this:

- The pipeline supports resume/caching — once processed, derived features are saved and do not require re-downloading.
- We document the exact song versions (movie, year, Hindi voice artist) so researchers can locate equivalent sources.
- The processing pipeline is fully open-source and documented.

### 5. Precedent in the Literature

Several published works in music information retrieval (MIR) and computational musicology use similar approaches:

- **DALI dataset** (Meseguer-Brocal et al., ISMIR 2018): Uses YouTube audio with time-aligned lyrics for research.
- **Lyrics translation** (Ou et al., LREC-COLING 2024): Parallel song lyrics corpus for En-Zh translation with melody alignment.
- **FMA dataset** (Defferrard et al., ISMIR 2017): Large-scale music audio dataset for research, which we also use.

### 6. Suggested Paper Language

> **Data collection.** We construct a parallel Hindi-English song translation corpus from officially dubbed Disney animated films. Unlike machine-translated lyrics, these Hindi versions were authored by professional lyricists to preserve singability, making them natural ground-truth examples of musically-aligned translation. Audio is downloaded from official YouTube releases, processed through a vocal separation and transcription pipeline (Demucs + Whisper), and stored as derived numerical features (tokenized text, melody pitch/duration arrays, syllable counts). Raw audio is not redistributed. All processing code is publicly available at \[repo URL\].

### 7. IRB / Ethics Board

- This research uses publicly available media and does not involve human subjects. No IRB approval is required.
- No personally identifiable information is collected or stored.

### 8. Limitations to Acknowledge

- Whisper transcription of Hindi sung vocals is imperfect — Hindi ASR on singing voice is an unsolved problem. All transcriptions should be treated as approximate and manually verified for key examples.
- Disney's Hindi dubs vary in translation quality across eras (1990s dubs tend to be more literal, 2010s+ dubs more creative).
- The dataset is small (\~20-50 songs) and specific to Disney's style — generalization to Bollywood or other genres requires further work.
