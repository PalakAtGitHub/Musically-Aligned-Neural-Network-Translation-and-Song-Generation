# Voice Model Setup (RVC)

The `musically-aligned-translation` pipeline supports integrating Retrieval-based Voice Conversion (RVC) models to map the generated Hindi robotic TTS back to the original singer's timbre and vocal inflections.

## 1. Acquiring an RVC Model
If you do not have a trained model of the target singer, you can use open-source generic models or train your own:
- **Download pre-trained models**: Check community hubs like Weights & Biases or the AI Hub Discord server for `.pth` RVC model files and `.index` files.
- **Train your own**: Use an open-source RVC WebUI. You will need about 10-15 minutes of isolated, clean vocal stems from the target singer.

## 2. Directory Structure
Create a `models/rvc/` directory in this repository and put your `.pth` and `.index` files inside it.
```text
musically-aligned-translation/
└── models/
    └── rvc/
        ├── singer_name.pth
        └── singer_name.index
```

## 3. Using the Model in the Pipeline
When running the synthesize pipeline, pass the path to the `.pth` file using the `--voice-model` argument.

```bash
python -m src.main synthesize song.mp3 \
  --lyrics "english lyrics here" \
  --voice-model "models/rvc/singer_name.pth"
```

## If No Model is Provided
The `--voice-model` argument is optional. If you omit it, the pipeline will gracefully skip Phase 5 (Voice Conversion) and proceed directly to mixing the base TTS vocals with the instrumental.
