# Text2TuneAI - Text-to-Music Deep Learning Pipeline

Generate musical melodies from lyrics using deep learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

**Text2TuneAI** generates musical melodies based on the semantic meaning and emotional context of input lyrics. It uses:
- **BERT** for text understanding (110M parameters)
- **Transformer** for music generation (45M parameters)
- **Hybrid datasets**: DALI (aligned) + Lakh MIDI (patterns)

**User provides lyrics → AI generates contextual melody**

---

## Quick Start

### 1. Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Demo (instant)
```bash
python main.py demo
```

### 3. Setup Datasets
```bash
python scripts/setup_data.py
```

### 4. Train
```bash
python main.py train --epochs 100
```

### 5. Generate Music
```bash
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "A beautiful sunny day filled with joy"
```

---

## Architecture

```
Lyrics Input
    ↓
[BERT Text Encoder] → Semantic + Emotion Features
    ↓
[Cross-Attention Bridge]
    ↓
[Transformer Music Decoder] → Note Sequences
    ↓
MIDI/Audio Output
```

**Components**:
- Text Encoder: BERT-based (768-dim embeddings, emotion classification)
- Music Decoder: 8-layer Transformer (note, duration, velocity prediction)
- Loss Functions: Reconstruction + coherence + emotion + rhythm
- Generation: Autoregressive sampling (temperature, top-k, top-p)

---

## Datasets

### DALI (Primary)
- **5,358 songs** with lyrics-melody alignment
- Word-level and note-level synchronization
- Download: https://github.com/gabolsgabs/DALI

### Lakh MIDI (Secondary)
- **180K+ MIDI files** for music patterns
- Download: http://colinraffel.com/projects/lmd/

### Why Hybrid?
DALI provides precise alignment, Lakh provides scale → best results

---

## Features

✅ End-to-end pipeline (text → MIDI → audio)
✅ Emotion-aware generation
✅ Multi-objective training
✅ PyTorch Lightning (scalable, multi-GPU)
✅ Mixed precision (FP16/BF16)
✅ TensorBoard monitoring
✅ Batch generation
✅ CLI + Python API

---

## Usage

### Training

```bash
# Basic
python main.py train

# Custom
python main.py train --epochs 50 --batch-size 8 --lr 0.0001

# Monitor
tensorboard --logdir logs/
```

### Generation

```bash
# Single
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "Your lyrics here"

# Batch
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text-file lyrics.txt

# With audio
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "Your lyrics" \
  --audio
```

### Python API

```python
from src.generation import load_model_from_checkpoint, MusicGenerator
from src.utils.config import get_config

config = get_config()
model = load_model_from_checkpoint("checkpoints/best.ckpt", config.to_dict())
generator = MusicGenerator(model, config.to_dict())

files = generator.generate_and_save(
    text="A beautiful sunny day filled with joy",
    output_dir="outputs",
    temperature=0.9,
    tempo=120,
    save_midi=True,
    save_audio=True
)
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
# Model
model:
  text_encoder:
    model_name: "bert-base-uncased"
    hidden_size: 768
  music_decoder:
    hidden_size: 512
    num_layers: 8

# Training
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0001
  precision: "16-mixed"

# Generation
generation:
  temperature: 1.0
  top_k: 50
  max_notes: 256
```

---

## Project Structure

```
Text2TuneAI/
├── configs/config.yaml          # Configuration
├── main.py                      # CLI entry point
├── src/
│   ├── data/                    # DALI & Lakh MIDI loaders
│   ├── models/                  # Text encoder, music decoder
│   ├── training/                # PyTorch Lightning trainer
│   ├── generation/              # Music generation
│   └── utils/                   # Config management
├── scripts/setup_data.py        # Dataset setup
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── PROJECT_INFO.md              # Detailed documentation
```

---

## Performance

**Training** (RTX 3090):
- FP32: ~2.5 sec/batch (18GB)
- FP16: ~1.8 sec/batch (12GB)

**Generation**:
- GPU: ~2-3 sec/song
- CPU: ~10 sec/song

**Quality** (after 50 epochs):
- Note accuracy: 65-75%
- Coherent melodies with emotion alignment

---

## Examples

### Happy Melody
```bash
python main.py generate \
  --text "Dancing in the sunshine with friends" \
  --tempo 140 --temperature 0.9
```

### Sad Melody
```bash
python main.py generate \
  --text "The rain falls as tears stream down" \
  --tempo 70 --temperature 0.7
```

---

## Troubleshooting

**Out of Memory**:
```yaml
training:
  batch_size: 8
  precision: "16-mixed"
```

**Audio fails**:
```bash
brew install fluid-synth  # macOS
sudo apt-get install fluidsynth  # Ubuntu
```

**Poor quality**:
- Train longer (50-100 epochs)
- Lower temperature (0.7-0.9)
- Increase coherence weight in config

---

## Documentation

- **README.md** (this file): Quick reference
- **PROJECT_INFO.md**: Complete technical documentation
- **configs/config.yaml**: All configuration options

---

## Technologies

**Core**: PyTorch, PyTorch Lightning, Transformers (Hugging Face)
**Music**: pretty_midi, music21, FluidSynth, librosa
**NLP**: BERT, sentence-transformers, TextBlob

---

## Citation

```bibtex
@software{text2tuneai2024,
  title={Text2TuneAI: Text-to-Music Generation},
  year={2024}
}
```

**Datasets**:
- DALI: Meseguer-Brocal et al., ISMIR 2018
- Lakh MIDI: Colin Raffel, 2016

---

## Contributing

Contributions welcome! Areas:
- Multi-instrument generation
- Chord progressions
- Web interface (Gradio)
- Genre-specific fine-tuning

---

## License

MIT License

---

**Status**: Production-Ready | **Version**: 1.0.0
Built with PyTorch, Transformers, and Music21
