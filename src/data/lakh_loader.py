"""
Lakh MIDI Dataset Loader and Preprocessor
Handles loading and processing of Lakh MIDI dataset for music pattern learning
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import pretty_midi
except ImportError:
    warnings.warn("pretty_midi not installed. Install with: pip install pretty-midi")
    pretty_midi = None

try:
    import mido
except ImportError:
    warnings.warn("mido not installed. Install with: pip install mido")
    mido = None


class LakhMIDILoader:
    """Loader for Lakh MIDI dataset"""

    def __init__(
        self,
        root_dir: str,
        subset_size: Optional[int] = None,
        min_notes: int = 20,
        max_notes: int = 2000,
        note_range: Tuple[int, int] = (21, 108),  # Piano range: A0 to C8
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Lakh MIDI loader

        Args:
            root_dir: Root directory for Lakh MIDI data
            subset_size: Number of files to use (None = all)
            min_notes: Minimum number of notes in a valid MIDI
            max_notes: Maximum number of notes to extract
            note_range: Valid MIDI note range (min, max)
            cache_dir: Directory for caching processed data
        """
        self.root_dir = Path(root_dir)
        self.subset_size = subset_size
        self.min_notes = min_notes
        self.max_notes = max_notes
        self.note_range = note_range
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / "cache"

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.midi_files = []
        self.processed_data = []

    def download_and_setup(self):
        """Setup instructions for Lakh MIDI dataset"""
        print("Setting up Lakh MIDI dataset...")
        print("\nTo download Lakh MIDI Dataset:")
        print("1. Visit: http://colinraffel.com/projects/lmd/")
        print("2. Download 'Lakh MIDI Dataset' (full or clean version)")
        print("3. Extract to:", self.root_dir)
        print("\nNote: The dataset is ~25GB (full) or ~3GB (clean)")
        print("For this project, the 'clean' version is recommended")

        # Check if MIDI files exist
        midi_files = self._scan_midi_files()

        if not midi_files:
            print("\nWARNING: No MIDI files found!")
            print(f"Please download Lakh MIDI dataset to: {self.root_dir}")
            return False

        print(f"\nFound {len(midi_files)} MIDI files")
        return True

    def _scan_midi_files(self) -> List[Path]:
        """
        Scan directory for MIDI files

        Returns:
            List of MIDI file paths
        """
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []

        print(f"Scanning {self.root_dir} for MIDI files...")

        for ext in midi_extensions:
            midi_files.extend(self.root_dir.rglob(f"*{ext}"))

        # Apply subset limit if specified
        if self.subset_size and len(midi_files) > self.subset_size:
            midi_files = midi_files[:self.subset_size]

        self.midi_files = sorted(midi_files)
        return self.midi_files

    def load_midi_files(self) -> List[Dict]:
        """
        Load and process all MIDI files

        Returns:
            List of processed MIDI data dictionaries
        """
        if not self.midi_files:
            self._scan_midi_files()

        print(f"Loading {len(self.midi_files)} MIDI files...")

        processed_data = []
        for idx, midi_file in enumerate(self.midi_files):
            if idx % 100 == 0:
                print(f"Processing: {idx}/{len(self.midi_files)}")

            try:
                data = self._process_midi_file(midi_file)
                if data:
                    processed_data.append(data)
            except Exception as e:
                # Skip corrupted files
                continue

        self.processed_data = processed_data
        print(f"Successfully processed {len(processed_data)} MIDI files")
        return processed_data

    def _process_midi_file(self, midi_path: Path) -> Optional[Dict]:
        """
        Process a single MIDI file and extract musical features

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with processed MIDI data or None if invalid
        """
        if pretty_midi is None:
            raise ImportError("pretty_midi required. Install: pip install pretty-midi")

        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))

            # Extract notes from all instruments
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:  # Skip drum tracks
                    for note in instrument.notes:
                        # Filter by note range
                        if self.note_range[0] <= note.pitch <= self.note_range[1]:
                            notes.append({
                                'pitch': note.pitch,
                                'start': note.start,
                                'end': note.end,
                                'velocity': note.velocity
                            })

            # Sort by start time
            notes = sorted(notes, key=lambda x: x['start'])

            # Filter by note count
            if len(notes) < self.min_notes or len(notes) > self.max_notes:
                return None

            # Extract musical features
            processed = {
                'file_path': str(midi_path),
                'notes': notes,
                'note_sequence': np.array([n['pitch'] for n in notes]),
                'durations': np.array([n['end'] - n['start'] for n in notes]),
                'velocities': np.array([n['velocity'] for n in notes]),
                'timing': np.array([n['start'] for n in notes]),
                'tempo': self._estimate_tempo(midi_data),
                'time_signature': midi_data.time_signature_changes[0]
                                 if midi_data.time_signature_changes else (4, 4),
                'key': self._estimate_key(notes),
                'total_duration': midi_data.get_end_time(),
                'num_notes': len(notes)
            }

            return processed

        except Exception as e:
            # Skip corrupted or invalid MIDI files
            return None

    def _estimate_tempo(self, midi_data: 'pretty_midi.PrettyMIDI') -> float:
        """
        Estimate tempo from MIDI data

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Estimated tempo in BPM
        """
        if midi_data.get_tempo_changes()[1].size > 0:
            # Use first tempo marking
            return float(midi_data.get_tempo_changes()[1][0])
        else:
            # Default tempo
            return 120.0

    def _estimate_key(self, notes: List[Dict]) -> str:
        """
        Estimate musical key from notes (simple version)

        Args:
            notes: List of note dictionaries

        Returns:
            Estimated key (e.g., 'C', 'Am')
        """
        if not notes:
            return 'C'

        # Count pitch classes (0-11)
        pitch_class_counts = np.zeros(12)
        for note in notes:
            pitch_class = note['pitch'] % 12
            pitch_class_counts[pitch_class] += 1

        # Find most common pitch class
        most_common = np.argmax(pitch_class_counts)

        # Simple key mapping (could be improved with key profiles)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return key_names[most_common]

    def get_note_sequences(self) -> List[np.ndarray]:
        """
        Get all note sequences

        Returns:
            List of note sequence arrays
        """
        return [data['note_sequence'] for data in self.processed_data]

    def get_temporal_sequences(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get sequences with timing information

        Returns:
            List of (notes, durations, velocities) tuples
        """
        sequences = []
        for data in self.processed_data:
            sequences.append((
                data['note_sequence'],
                data['durations'],
                data['velocities']
            ))
        return sequences

    def get_musical_features(self) -> List[Dict]:
        """
        Get high-level musical features for each MIDI

        Returns:
            List of feature dictionaries
        """
        features = []
        for data in self.processed_data:
            features.append({
                'tempo': data['tempo'],
                'key': data['key'],
                'num_notes': data['num_notes'],
                'duration': data['total_duration'],
                'avg_pitch': float(np.mean(data['note_sequence'])),
                'pitch_range': float(np.ptp(data['note_sequence'])),
                'avg_duration': float(np.mean(data['durations'])),
                'avg_velocity': float(np.mean(data['velocities']))
            })
        return features

    def extract_patterns(self, pattern_length: int = 8) -> List[np.ndarray]:
        """
        Extract melodic patterns of fixed length

        Args:
            pattern_length: Number of notes in each pattern

        Returns:
            List of note patterns
        """
        patterns = []

        for data in self.processed_data:
            notes = data['note_sequence']

            # Extract sliding windows
            for i in range(len(notes) - pattern_length + 1):
                pattern = notes[i:i + pattern_length]
                patterns.append(pattern)

        return patterns

    def __len__(self) -> int:
        """Get number of processed MIDI files"""
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict:
        """Get processed data by index"""
        return self.processed_data[idx]


if __name__ == "__main__":
    # Test Lakh MIDI loader
    loader = LakhMIDILoader(
        root_dir="data/raw/lakh_midi",
        subset_size=100
    )

    print("Testing Lakh MIDI Loader...")
    print(f"Root directory: {loader.root_dir}")

    # Scan for MIDI files
    midi_files = loader._scan_midi_files()
    print(f"Found {len(midi_files)} MIDI files")

    if midi_files:
        print("\nLoading MIDI files...")
        processed = loader.load_midi_files()
        print(f"Processed {len(processed)} files")

        if processed:
            print("\nSample MIDI data:")
            sample = processed[0]
            print(f"Notes: {sample['num_notes']}")
            print(f"Tempo: {sample['tempo']:.1f} BPM")
            print(f"Key: {sample['key']}")
            print(f"Duration: {sample['total_duration']:.2f}s")
    else:
        print("\nNo MIDI files found. Please download Lakh MIDI dataset.")
