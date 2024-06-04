"""
Music Generation Pipeline
Generate music from text using trained Text2Tune model
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

try:
    import pretty_midi
except ImportError:
    warnings.warn("pretty_midi not installed")
    pretty_midi = None

from src.models import Text2TuneModel


class MusicGenerator:
    """Generate music from text using Text2Tune model"""

    def __init__(
        self,
        model: Text2TuneModel,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize music generator

        Args:
            model: Trained Text2Tune model
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        self.generation_config = config.get('generation', {})

    @torch.no_grad()
    def generate_from_text(
        self,
        text: str,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        tempo: float = 120.0,
        key: str = 'C'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate music from text

        Args:
            text: Input lyrics/text
            max_length: Maximum number of notes to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            tempo: Tempo in BPM
            key: Musical key

        Returns:
            Tuple of (notes, durations) as numpy arrays
        """
        if max_length is None:
            max_length = self.generation_config.get('max_notes', 256)

        print(f"Generating music for: '{text[:50]}...'")
        print(f"Parameters: max_length={max_length}, temperature={temperature}, tempo={tempo}")

        # Generate music
        notes, durations = self.model.generate_music(
            texts=text,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Convert to numpy
        notes = notes.cpu().numpy()[0]  # (L,)
        durations = durations.cpu().numpy()[0]  # (L,)

        print(f"Generated {len(notes)} notes")

        return notes, durations

    def notes_to_midi(
        self,
        notes: np.ndarray,
        durations: np.ndarray,
        output_path: str,
        tempo: float = 120.0,
        velocity: int = 80,
        program: int = 0
    ):
        """
        Convert note sequence to MIDI file

        Args:
            notes: Array of MIDI note numbers
            durations: Array of note durations in seconds
            output_path: Output MIDI file path
            tempo: Tempo in BPM
            velocity: Note velocity (0-127)
            program: MIDI program/instrument (0 = Acoustic Grand Piano)
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi required. Install: pip install pretty-midi")

        print(f"Converting to MIDI: {output_path}")

        # Create PrettyMIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Create instrument
        instrument = pretty_midi.Instrument(program=program)

        # Add notes
        current_time = 0.0
        for note_num, duration in zip(notes, durations):
            # Skip invalid notes
            if note_num < 21 or note_num > 108:
                continue

            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=int(note_num),
                start=current_time,
                end=current_time + float(duration)
            )

            instrument.notes.append(note)
            current_time += duration

        # Add instrument to MIDI
        midi.instruments.append(instrument)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))

        print(f"MIDI saved: {output_path}")
        return midi

    def midi_to_audio(
        self,
        midi_path: str,
        output_path: str,
        soundfont: Optional[str] = None,
        sample_rate: int = 44100
    ):
        """
        Convert MIDI to audio using FluidSynth

        Args:
            midi_path: Path to MIDI file
            output_path: Output audio file path
            soundfont: Path to soundfont (.sf2) file
            sample_rate: Audio sample rate
        """
        try:
            import pretty_midi

            print(f"Converting MIDI to audio: {output_path}")

            # Load MIDI
            midi = pretty_midi.PrettyMIDI(midi_path)

            # Synthesize audio
            if soundfont:
                audio = midi.fluidsynth(fs=sample_rate, sf2_path=soundfont)
            else:
                audio = midi.fluidsynth(fs=sample_rate)

            # Save audio
            import soundfile as sf
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sample_rate)

            print(f"Audio saved: {output_path}")

        except Exception as e:
            print(f"Warning: Could not convert to audio: {e}")
            print("Make sure FluidSynth is installed: brew install fluid-synth (macOS)")

    def generate_and_save(
        self,
        text: str,
        output_dir: str,
        filename: str = "generated",
        save_midi: bool = True,
        save_audio: bool = True,
        **generation_kwargs
    ) -> Dict[str, str]:
        """
        Generate music and save to files

        Args:
            text: Input text
            output_dir: Output directory
            filename: Base filename (without extension)
            save_midi: Save MIDI file
            save_audio: Save audio file
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate notes and durations
        notes, durations = self.generate_from_text(text, **generation_kwargs)

        saved_files = {}

        # Save MIDI
        if save_midi:
            midi_path = output_dir / f"{filename}.mid"
            self.notes_to_midi(
                notes,
                durations,
                str(midi_path),
                tempo=generation_kwargs.get('tempo', 120.0)
            )
            saved_files['midi'] = str(midi_path)

            # Save audio from MIDI
            if save_audio:
                audio_path = output_dir / f"{filename}.wav"
                try:
                    self.midi_to_audio(
                        str(midi_path),
                        str(audio_path),
                        soundfont=self.generation_config.get('soundfont')
                    )
                    saved_files['audio'] = str(audio_path)
                except Exception as e:
                    print(f"Could not generate audio: {e}")

        return saved_files

    def batch_generate(
        self,
        texts: List[str],
        output_dir: str,
        **generation_kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate music for multiple texts

        Args:
            texts: List of input texts
            output_dir: Output directory
            **generation_kwargs: Generation parameters

        Returns:
            List of dictionaries with saved file paths
        """
        results = []

        for idx, text in enumerate(texts):
            print(f"\n[{idx + 1}/{len(texts)}] Processing: {text[:50]}...")

            files = self.generate_and_save(
                text,
                output_dir,
                filename=f"generated_{idx:03d}",
                **generation_kwargs
            )

            results.append({
                'text': text,
                'files': files
            })

        print(f"\nBatch generation complete! {len(results)} files generated.")
        return results


def load_model_from_checkpoint(checkpoint_path: str, config: Dict) -> Text2TuneModel:
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary

    Returns:
        Loaded Text2Tune model
    """
    from src.training.trainer import Text2TuneLightningModule

    print(f"Loading model from: {checkpoint_path}")

    # Load Lightning checkpoint
    lightning_module = Text2TuneLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config
    )

    return lightning_module.model


if __name__ == "__main__":
    from src.utils.config import get_config

    print("=" * 80)
    print("TEXT2TUNE AI - Music Generation Demo")
    print("=" * 80)

    # Load config
    config = get_config()
    config_dict = config.to_dict()

    # For demo, create a model (in production, load from checkpoint)
    print("\nCreating model...")
    model = Text2TuneModel(config_dict)

    # Create generator
    generator = MusicGenerator(model, config_dict)

    # Sample texts
    sample_texts = [
        "A beautiful sunny day filled with joy and happiness",
        "The darkness falls as sadness fills my heart",
        "Dancing through the night with endless energy"
    ]

    print("\nGenerating music samples...")

    # Generate for each text
    for idx, text in enumerate(sample_texts):
        print(f"\n{'-' * 80}")
        print(f"Text {idx + 1}: {text}")
        print('-' * 80)

        # Generate
        files = generator.generate_and_save(
            text,
            output_dir="outputs/demo",
            filename=f"sample_{idx}",
            max_length=128,
            temperature=1.0,
            tempo=120.0,
            save_midi=True,
            save_audio=False  # Set to True if FluidSynth is installed
        )

        print(f"Saved files: {files}")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)
