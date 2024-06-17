# Text2TuneAI - Complete Usage Guide

Comprehensive guide for using Text2TuneAI to generate music from text.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Music Generation](#music-generation)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: System Requirements

**Minimum**:
- Python 3.9+
- 8GB RAM
- 10GB disk space

**Recommended**:
- Python 3.10+
- 16GB+ RAM
- CUDA-compatible GPU (8GB+ VRAM)
- 50GB disk space (for datasets)

### Step 2: Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd Text2TuneAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Run demo to verify everything works
python main.py demo
```

If this runs without errors, you're ready to go!

---

## Dataset Setup

### DALI Dataset (Required for Quality Training)

**What it is**: 5,358 songs with lyrics aligned to melody at note level.

**Download**:

```bash
# 1. Clone DALI repository
git clone https://github.com/gabolsgabs/DALI external/DALI

# 2. Download annotations
cd external/DALI
wget https://github.com/gabolsgabs/DALI/releases/download/v2.0/dali_v2.0.tar.gz
tar -xzf dali_v2.0.tar.gz

# 3. Move to project
cd ../..
mkdir -p data/raw/dali/annotations
cp -r external/DALI/dali_v2.0/* data/raw/dali/annotations/
```

**Verify**:
```bash
ls data/raw/dali/annotations/*.gz | wc -l  # Should show ~5300
```

### Lakh MIDI Dataset (Optional but Recommended)

**What it is**: 180K+ MIDI files for learning general music patterns.

**Download** (choose one):

**Option A: Clean Version** (3GB, recommended):
```bash
cd data/raw/lakh_midi
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz
tar -xzf lmd_matched.tar.gz
rm lmd_matched.tar.gz
```

**Option B: Full Version** (25GB):
```bash
cd data/raw/lakh_midi
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
rm lmd_full.tar.gz
```

**Verify**:
```bash
find data/raw/lakh_midi -name "*.mid" | wc -l  # Should show many files
```

### Automated Setup

Use the setup script for guided installation:

```bash
python scripts/setup_data.py
```

---

## Configuration

All settings are in [configs/config.yaml](configs/config.yaml).

### Key Configuration Sections

#### 1. Data Settings

```yaml
data:
  dali:
    root_dir: "data/raw/dali"
    download: true
  lakh:
    root_dir: "data/raw/lakh_midi"
    subset_size: 50000  # Use subset for faster training
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

#### 2. Model Architecture

```yaml
model:
  text_encoder:
    model_name: "bert-base-uncased"  # or "distilbert-base-uncased"
    hidden_size: 768
    freeze_bert: false  # Set true to freeze BERT weights

  music_decoder:
    hidden_size: 512
    num_layers: 8  # Reduce to 6 for faster training
    num_heads: 8
```

#### 3. Training Settings

```yaml
training:
  batch_size: 16  # Reduce if memory issues
  num_epochs: 100
  learning_rate: 0.0001
  precision: "16-mixed"  # FP16 for faster training
  gradient_clip: 1.0
```

#### 4. Generation Settings

```yaml
generation:
  max_notes: 256
  temperature: 1.0  # Lower (0.7-0.9) for coherent melodies
  top_k: 50
  top_p: 0.9
```

### Editing Configuration

Edit `configs/config.yaml` directly, or override via code:

```python
from src.utils.config import get_config

config = get_config()
config.update('training.batch_size', 8)
config.update('model.music_decoder.num_layers', 6)
```

---

## Training

### Basic Training

Start training with default settings:

```bash
python main.py train
```

### Custom Training

```bash
# Train for 50 epochs
python main.py train --epochs 50

# Use batch size 8
python main.py train --batch-size 8

# Custom learning rate
python main.py train --lr 0.00005

# Combine options
python main.py train --epochs 50 --batch-size 8 --lr 0.0001
```

### Training from Python

```python
from src.utils.config import get_config
from src.training import train_model

# Load config
config = get_config()
config_dict = config.to_dict()

# Customize
config.update('training.batch_size', 8)
config.update('training.num_epochs', 50)

# Train
model, trainer = train_model(config.to_dict())
```

### Monitoring Training

**TensorBoard**:
```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 to view:
- Loss curves
- Learning rate
- Gradient norms
- Validation metrics

**Checkpoints**:
- Saved in `checkpoints/`
- Best model: `checkpoints/text2tune-epoch=XX-val_loss=X.XX.ckpt`

### Training Tips

1. **Start small**: Train for 5-10 epochs first to verify everything works
2. **Monitor validation loss**: Should decrease steadily
3. **Watch note accuracy**: Target 65-75% after 50 epochs
4. **Check coherence loss**: Should decrease over time
5. **Save checkpoints**: Every 5-10 epochs

---

## Music Generation

### Quick Generation

```bash
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "A beautiful sunny day filled with joy"
```

### Generation Options

```bash
# Custom parameters
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text "Your lyrics here" \
  --temperature 0.9 \
  --tempo 120 \
  --max-length 256 \
  --top-k 50 \
  --top-p 0.9 \
  --output outputs/my_music \
  --audio  # Generate audio (requires FluidSynth)
```

### Interactive Generation

```bash
# Start interactive mode
python main.py generate --checkpoint checkpoints/best.ckpt

# Then type your lyrics
> A beautiful sunny day
> Filled with joy and happiness
> (press Enter on empty line to finish)
```

### Batch Generation

Create a text file with lyrics (one per line):

```text
A beautiful sunny day filled with joy
The darkness falls and sadness fills my heart
Dancing all night long with endless energy
```

Generate all:

```bash
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text-file lyrics.txt \
  --output outputs/batch
```

### Generation from Python

```python
from src.generation import load_model_from_checkpoint, MusicGenerator
from src.utils.config import get_config

# Load model
config = get_config()
model = load_model_from_checkpoint(
    "checkpoints/best.ckpt",
    config.to_dict()
)

# Create generator
generator = MusicGenerator(model, config.to_dict())

# Generate
files = generator.generate_and_save(
    text="A beautiful sunny day filled with joy",
    output_dir="outputs",
    filename="my_song",
    max_length=256,
    temperature=0.9,
    tempo=120.0,
    save_midi=True,
    save_audio=True
)

print(f"Generated files: {files}")
```

### Advanced Generation

**Multiple variations**:
```python
for i in range(5):
    files = generator.generate_and_save(
        text="Your lyrics",
        output_dir=f"outputs/variation_{i}",
        temperature=1.0 + (i * 0.1)  # Vary temperature
    )
```

**Different tempos**:
```python
tempos = [80, 100, 120, 140, 160]
for tempo in tempos:
    files = generator.generate_and_save(
        text="Your lyrics",
        output_dir=f"outputs/tempo_{tempo}",
        tempo=tempo
    )
```

---

## Advanced Usage

### Fine-tuning on Custom Data

```python
from src.training import Text2TuneLightningModule
from src.data import create_dataloaders

# Load pre-trained model
model = Text2TuneLightningModule.load_from_checkpoint(
    "checkpoints/pretrained.ckpt"
)

# Prepare your custom data
# ... (implement custom dataset loader)

# Fine-tune
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, your_custom_dataloader)
```

### Custom Loss Weights

```yaml
# In configs/config.yaml
training:
  loss_weights:
    reconstruction: 1.0
    emotion_consistency: 0.5  # Increase for emotion emphasis
    musical_coherence: 1.0     # Increase for smoother melodies
    rhythm_consistency: 0.6
    pitch_contour: 0.4
```

### Multi-GPU Training

```yaml
# In configs/config.yaml
training:
  num_gpus: 4
  strategy: "ddp"  # Distributed data parallel
```

Or via command line:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py train
```

### Export to Different Formats

```python
from pretty_midi import PrettyMIDI

# Load MIDI
midi = PrettyMIDI("outputs/my_song.mid")

# Export to MusicXML
midi.write("outputs/my_song.musicxml")

# Transpose
for instrument in midi.instruments:
    for note in instrument.notes:
        note.pitch += 5  # Transpose up 5 semitones

midi.write("outputs/my_song_transposed.mid")
```

---

## Best Practices

### For Best Quality Results

1. **Train Longer**: 50-100 epochs minimum
2. **Use DALI Dataset**: Essential for lyrics-melody alignment
3. **Lower Temperature**: 0.7-0.9 for coherent melodies
4. **Monitor Metrics**: Watch validation loss and note accuracy
5. **Adjust Loss Weights**: Increase coherence for smoother output

### For Faster Training

1. **Reduce Batch Size**: Use 8 instead of 16
2. **Enable Mixed Precision**: `precision: "16-mixed"`
3. **Freeze BERT**: Set `freeze_bert: true`
4. **Fewer Layers**: Use 6 decoder layers instead of 8
5. **Smaller Model**: Use `distilbert-base-uncased`

### For Creative Output

1. **Higher Temperature**: 1.0-1.2
2. **Increase Top-K**: Use 80-100
3. **Vary Parameters**: Generate multiple versions
4. **Experiment**: Try different emotion/tempo settings

---

## Troubleshooting

### Out of Memory

**Solution 1**: Reduce batch size
```yaml
training:
  batch_size: 8
  accumulate_grad_batches: 8  # Compensate
```

**Solution 2**: Enable mixed precision
```yaml
training:
  precision: "16-mixed"
```

**Solution 3**: Reduce sequence length
```yaml
model:
  text_encoder:
    max_seq_length: 256
  music_decoder:
    max_seq_length: 512
```

### Audio Generation Fails

**Install FluidSynth**:

```bash
# macOS
brew install fluid-synth

# Ubuntu
sudo apt-get install fluidsynth libfluidsynth-dev

# Windows
# Download from: https://github.com/FluidSynth/fluidsynth/releases
```

**Download Soundfont**:
```bash
wget https://member.keymusician.com/Member/FluidR3_GM/FluidR3_GM.sf2
mv FluidR3_GM.sf2 data/
```

Update config:
```yaml
generation:
  soundfont: "data/FluidR3_GM.sf2"
```

### Poor Quality Output

**Problem**: Random or incoherent melodies

**Solutions**:
1. Train longer (50+ epochs)
2. Lower temperature (0.7-0.9)
3. Increase `musical_coherence_weight`
4. Use smaller top-k (30-40)
5. Verify training data quality

### Training Slow

**Solutions**:
1. Enable GPU: Check `torch.cuda.is_available()`
2. Enable mixed precision
3. Reduce num_workers if CPU-bound
4. Use smaller model
5. Train on subset first

### Dataset Not Found

**Run setup script**:
```bash
python scripts/setup_data.py
```

**Manual verification**:
```bash
# Check DALI
ls data/raw/dali/annotations/*.gz

# Check Lakh
find data/raw/lakh_midi -name "*.mid" | head -10
```

---

## Example Workflows

### Workflow 1: Quick Test

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Demo
python main.py demo

# 3. Done!
```

### Workflow 2: Training from Scratch

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download datasets
python scripts/setup_data.py

# 3. Train (start small)
python main.py train --epochs 10

# 4. Generate
python main.py generate \
  --checkpoint checkpoints/latest.ckpt \
  --text "Your lyrics"
```

### Workflow 3: Production Training

```bash
# 1. Setup datasets (both DALI and Lakh)
python scripts/setup_data.py

# 2. Configure (edit configs/config.yaml)
# - Set batch_size, epochs, etc.

# 3. Train
python main.py train --epochs 100

# 4. Monitor
tensorboard --logdir logs/

# 5. Generate (use best checkpoint)
python main.py generate \
  --checkpoint checkpoints/best.ckpt \
  --text-file my_lyrics.txt \
  --output outputs/production \
  --audio
```

---

## Parameter Reference

### Temperature

- **0.5-0.6**: Very predictable, conservative
- **0.7-0.8**: Balanced, coherent (recommended)
- **0.9-1.0**: Creative, varied
- **1.1-1.5**: Experimental, unpredictable

### Top-K

- **10-20**: Very conservative
- **30-50**: Balanced (recommended)
- **50-80**: More variety
- **80-100**: Maximum diversity

### Top-P

- **0.7-0.8**: Conservative
- **0.85-0.95**: Balanced (recommended)
- **0.95-1.0**: Maximum creativity

### Tempo

- **60-80 BPM**: Slow, ballad-like
- **80-100 BPM**: Moderate, calm
- **100-120 BPM**: Standard pop tempo
- **120-140 BPM**: Upbeat, energetic
- **140-160 BPM**: Fast, dance-like

---

## Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Project Summary**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Configuration**: [configs/config.yaml](configs/config.yaml)

---

## Getting Help

1. Check this guide
2. Read [README.md](README.md)
3. Check [troubleshooting section](#troubleshooting)
4. Open GitHub issue
5. Review code comments

---

**Happy Music Generation!** 🎵
