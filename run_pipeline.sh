#!/usr/bin/env bash
# Full MCNST pipeline: train on FMA data → synthesize 3 test songs
set -e
cd "$(dirname "$0")"
PYTHON=venv/bin/python

echo "========================================================"
echo "Step 1/4: Training MCNST on FMA data (MPS GPU)"
echo "========================================================"
$PYTHON -m src.main train \
    --data src/data/processed/fma_train_data.pt \
    --test-data src/data/processed/fma_test_data.pt

echo ""
echo "========================================================"
echo "Step 2/4: Synthesizing 022097 — 'Can be over me'"
echo "========================================================"
$PYTHON -m src.main voice_clone \
    src/data/fma_medium/022/022097.mp3 \
    --lyrics "Can be over me
Before I'm over you"

echo ""
echo "========================================================"
echo "Step 3/4: Synthesizing 132040 — 'Jingle bell rock'"
echo "========================================================"
$PYTHON -m src.main voice_clone \
    src/data/fma_medium/132/132040.mp3 \
    --lyrics "Jingle bell jingle bell jingle bell rock"

echo ""
echo "========================================================"
echo "Step 4/4: Synthesizing 087174 — 'Oh my life'"
echo "========================================================"
$PYTHON -m src.main voice_clone \
    src/data/fma_medium/087/087174.mp3 \
    --lyrics "Oh my life.
Oh my life.
I wonder what's the important thing.
Tell me what's the important thing.
So I know what's the important thing.
So I know what's the important thing."

echo ""
echo "========================================================"
echo "Pipeline complete! Hindi audio files in:"
echo "  src/data/output/"
echo "========================================================"
