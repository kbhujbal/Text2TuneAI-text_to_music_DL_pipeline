# Text2TuneAI - Quick Start Guide

Get up and running with Text2TuneAI in 5 minutes!

---

## Installation (2 minutes)

```bash
# Clone repository
git clone <your-repo>
cd Text2TuneAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Demo (1 minute)

Run an instant demo with an untrained model:

```bash
python main.py demo
```

This generates 3 sample MIDI files in `outputs/demo/`.

---

## Dataset Setup (Optional - for training)

### Option 1: Automated Setup

```bash
python scripts/setup_data.py
```

Follow the interactive instructions.

### Option 2: Manual Setup

**DALI Dataset** (5.3K songs, ~500MB):
```bash
git clone https://github.com/gabolsgabs/DALI
cd DALI
wget https://github.com/gabolsgabs/DALI/releases/download/v2.0/dali_v2.0.tar.gz
tar -xzf dali_v2.0.tar.gz
mv dali_v2.0/* ../Text2TuneAI/data/raw/dali/annotations/
```

**Lakh MIDI** (Clean version, ~3GB):
```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz -C data/raw/lakh_midi/
```

---

## Training (Simple)

Start training with default settings:

```bash
python main.py train
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/
```

### Quick Training Options

```bash
# Train for 10 epochs only
python main.py train --epochs 10

# Use smaller batch size (if memory issues)
python main.py train --batch-size 8

# Custom learning rate
python main.py train --lr 0.0001
```

---

## Music Generation

### From Command Line

```bash
# Generate from text
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "A beautiful sunny day" \
  --output outputs/my_music

# With custom parameters
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "Dancing all night long" \
  --temperature 0.9 \
  --tempo 140 \
  --max-length 256 \
  --audio  # Requires FluidSynth
```

### From Python

```python
from src.models import Text2TuneModel
from src.generation import MusicGenerator
from src.utils.config import get_config

# Setup
config = get_config()
model = Text2TuneModel(config.to_dict())
generator = MusicGenerator(model, config.to_dict())

# Generate
files = generator.generate_and_save(
    text="Your lyrics here",
    output_dir="outputs",
    filename="my_song",
    save_midi=True
)

print(f"Saved: {files}")
```

---

## Common Commands Cheat Sheet

```bash
# Demo
python main.py demo

# Train
python main.py train
python main.py train --epochs 50 --batch-size 16

# Generate
python main.py generate --text "Your lyrics" --checkpoint checkpoints/best.ckpt

# Generate with audio
python main.py generate --text "Your lyrics" --checkpoint checkpoints/best.ckpt --audio

# Interactive generation
python main.py generate --checkpoint checkpoints/best.ckpt
# (then type your lyrics)

# Batch generation from file
python main.py generate --checkpoint checkpoints/best.ckpt --text-file lyrics.txt
```

---

## File Locations

- **Datasets**: `data/raw/`
- **Checkpoints**: `checkpoints/`
- **Generated music**: `outputs/`
- **Training logs**: `logs/`
- **Configuration**: `configs/config.yaml`

---

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Quick tweaks for better results:

training:
  batch_size: 16          # Reduce if memory issues
  num_epochs: 100         # Increase for better quality
  learning_rate: 0.0001   # Lower for stability

model:
  music_decoder:
    num_layers: 8         # Reduce to 6 for faster training

generation:
  temperature: 1.0        # Lower (0.7-0.9) for more coherent melodies
  top_k: 50              # Lower (30) for more conservative output
```

---

## Troubleshooting

### Out of Memory

```yaml
# In configs/config.yaml
training:
  batch_size: 8
  accumulate_grad_batches: 8
  precision: "16-mixed"
```

### Audio Generation Fails

Install FluidSynth:
```bash
# macOS
brew install fluid-synth

# Ubuntu
sudo apt-get install fluidsynth
```

### Dataset Not Found

```bash
python scripts/setup_data.py
```

---

## Next Steps

1. **Train the model** on your datasets (50+ epochs recommended)
2. **Experiment with parameters** in `configs/config.yaml`
3. **Generate music** from your own lyrics
4. **Fine-tune** on specific genres or styles
5. **Share your results**!

---

## Getting Help

- Read the full [README.md](README.md)
- Check configuration in [configs/config.yaml](configs/config.yaml)
- Open an issue on GitHub

---

## Example Workflow

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download datasets
python scripts/setup_data.py

# 3. Quick test (optional)
python main.py demo

# 4. Train (start small)
python main.py train --epochs 10

# 5. Generate music
python main.py generate \
  --checkpoint checkpoints/text2tune-epoch=09-val_loss=1.234.ckpt \
  --text "Your amazing lyrics here"

# 6. Check outputs
ls outputs/
```

---

**That's it! You're ready to generate music from text!** 🎵

For advanced usage, see the full [README.md](README.md).
