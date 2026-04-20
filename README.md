# Musically-Aligned Neural Network Translation and Song Generation

An end-to-end neural song translation pipeline designed to convert English songs into Hindi while preserving the original melody, rhythm, and singer's timbre.

## System Environment & Scope

> **This directory (`musically-aligned-translation/`) is the sole source of truth for the entire MCNST pipeline.**
>
> All model code, training scripts, evaluation metrics, data pipelines, and audio processing utilities live here. Do not reference, modify, or rely on code in any other directory (e.g. `Song-Translation/`) unless explicitly instructed.

## 🎵 Project Vision

This project is grounded in **Cognitive Neuroscience** (Shared Syntactic Integration Resource Hypothesis) and **Professional Translation Theory** (Peter Low's Pentathlon Principle). It aims to bridge the gap between simple text translation and singable, musically-aligned song adaptation.

### Key Pillars:
1.  **Singability**: Ensuring syllables fit musical phrases comfortably.
2.  **Sense**: Preserving the semantic meaning using mBART-based models.
3.  **Naturalness**: Generating fluent Hindi lyrics.
4.  **Rhythm**: Aligning stressed syllables with musical beats.
5.  **Rhyme**: Incorporating phonetic similarity in endings.

## 🚀 Repository Structure

- `src/`: Core neural architecture, evaluation metrics (SingScore), and audio processing utilities.
- `data/`: Processed datasets, MIDI files, and sample audio.
- `research/`: Detailed documentation on the cognitive and musical theories grounding this architecture.
- `requirements.txt`: List of dependencies to recreate the environment.

## 🛠️ Setup

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv venv`.
3.  Install dependencies: `pip install -r requirements.txt`.

## 🧪 Evaluation: SingScore

The project introduces **SingScore**, a novel evaluation framework designed to validate translation quality specifically for singing, balancing phonetic flow with musical constraints.

---
*Developed by [PalakAtGitHub](https://github.com/PalakAtGitHub)*
