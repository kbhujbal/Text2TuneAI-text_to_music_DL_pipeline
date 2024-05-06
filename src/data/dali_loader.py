"""
DALI Dataset Loader and Preprocessor
Handles loading and processing of DALI dataset for lyrics-melody alignment
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings

try:
    import DALI as dali_code
except ImportError:
    warnings.warn("DALI-dataset not installed. Install with: pip install DALI-dataset")
    dali_code = None


class DALILoader:
    """Loader for DALI dataset with lyrics-melody alignment"""

    def __init__(
        self,
        root_dir: str,
        version: str = "v2.0",
        audio_source: str = "youtube",
        sample_rate: int = 22050,
        max_duration: float = 30.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize DALI loader

        Args:
            root_dir: Root directory for DALI data
            version: DALI version (v1.0 or v2.0)
            audio_source: Audio source (youtube or spotify)
            sample_rate: Audio sample rate
            max_duration: Maximum audio duration in seconds
            cache_dir: Directory for caching processed data
        """
        self.root_dir = Path(root_dir)
        self.version = version
        self.audio_source = audio_source
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / "cache"

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # DALI data structures
        self.dali_data = None
        self.dali_info = None
        self.entries = []

    def download_and_setup(self):
        """Download DALI annotations and setup dataset"""
        print("Setting up DALI dataset...")

        if dali_code is None:
            raise ImportError(
                "DALI-dataset not installed. Install with: pip install DALI-dataset"
            )

        # DALI provides annotations, not audio
        # Audio needs to be downloaded separately using their tools
        annotations_path = self.root_dir / "annotations"
        annotations_path.mkdir(exist_ok=True)

        print(f"DALI annotations should be in: {annotations_path}")
        print("To download DALI:")
        print("1. Clone: git clone https://github.com/gabolsgabs/DALI")
        print("2. Follow their instructions to download annotations")
        print("3. Use their tools to download audio from YouTube/Spotify")

        # Check if annotations exist
        if not list(annotations_path.glob("*.gz")):
            print("\nWARNING: No DALI annotations found!")
            print("Please download DALI dataset first.")
            return False

        return True

    def load_annotations(self) -> List[Dict]:
        """
        Load DALI annotations from files

        Returns:
            List of annotation dictionaries
        """
        annotations_path = self.root_dir / "annotations"

        if not annotations_path.exists():
            raise FileNotFoundError(f"DALI annotations not found at {annotations_path}")

        # Load DALI info file
        info_file = annotations_path / "info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.dali_info = json.load(f)

        # Load annotation files
        annotation_files = list(annotations_path.glob("*.gz"))

        if not annotation_files:
            print(f"No annotation files found in {annotations_path}")
            return []

        print(f"Found {len(annotation_files)} DALI annotation files")

        entries = []
        for ann_file in annotation_files[:100]:  # Limit for initial testing
            try:
                entry = self._load_single_annotation(ann_file)
                if entry:
                    entries.append(entry)
            except Exception as e:
                print(f"Error loading {ann_file}: {e}")

        self.entries = entries
        print(f"Successfully loaded {len(entries)} DALI entries")
        return entries

    def _load_single_annotation(self, annotation_file: Path) -> Optional[Dict]:
        """
        Load a single DALI annotation file

        Args:
            annotation_file: Path to annotation file

        Returns:
            Dictionary with lyrics, notes, and timing information
        """
        # This is a placeholder - actual implementation depends on DALI data format
        # DALI uses gzip-compressed JSON files
        import gzip

        try:
            with gzip.open(annotation_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            # Extract relevant information
            entry = {
                'id': data.get('id', annotation_file.stem),
                'artist': data.get('artist', ''),
                'title': data.get('title', ''),
                'lyrics': self._extract_lyrics(data),
                'notes': self._extract_notes(data),
                'alignment': self._extract_alignment(data),
                'metadata': self._extract_metadata(data)
            }

            return entry

        except Exception as e:
            print(f"Error parsing {annotation_file}: {e}")
            return None

    def _extract_lyrics(self, data: Dict) -> List[Dict]:
        """
        Extract lyrics with timestamps from DALI data

        Args:
            data: DALI annotation data

        Returns:
            List of lyric segments with timing
        """
        lyrics = []

        if 'annotations' in data and 'lines' in data['annotations']:
            for line in data['annotations']['lines']:
                lyrics.append({
                    'text': line.get('text', ''),
                    'start_time': line.get('time', [0])[0] if line.get('time') else 0,
                    'end_time': line.get('time', [0])[-1] if line.get('time') else 0,
                    'words': line.get('words', [])
                })

        return lyrics

    def _extract_notes(self, data: Dict) -> List[Dict]:
        """
        Extract musical notes from DALI data

        Args:
            data: DALI annotation data

        Returns:
            List of notes with pitch, timing, and duration
        """
        notes = []

        if 'annotations' in data and 'notes' in data['annotations']:
            for note in data['annotations']['notes']:
                notes.append({
                    'pitch': note.get('freq', 0),  # Frequency in Hz
                    'midi_note': self._freq_to_midi(note.get('freq', 0)),
                    'start_time': note.get('time', [0])[0] if note.get('time') else 0,
                    'duration': note.get('time', [0, 0])[1] - note.get('time', [0, 0])[0]
                                if len(note.get('time', [])) > 1 else 0
                })

        return notes

    def _extract_alignment(self, data: Dict) -> List[Dict]:
        """
        Extract word-to-note alignment from DALI data

        Args:
            data: DALI annotation data

        Returns:
            List of aligned word-note pairs
        """
        alignment = []

        if 'annotations' in data and 'words' in data['annotations']:
            for word_data in data['annotations']['words']:
                alignment.append({
                    'word': word_data.get('text', ''),
                    'start_time': word_data.get('time', [0])[0],
                    'notes': word_data.get('notes', [])
                })

        return alignment

    def _extract_metadata(self, data: Dict) -> Dict:
        """Extract metadata (genre, mood, tempo, etc.)"""
        return {
            'genre': data.get('metadata', {}).get('genre', ''),
            'language': data.get('metadata', {}).get('language', 'en'),
            'duration': data.get('metadata', {}).get('duration', 0),
            'url': data.get('metadata', {}).get('url', '')
        }

    @staticmethod
    def _freq_to_midi(freq: float) -> int:
        """
        Convert frequency (Hz) to MIDI note number

        Args:
            freq: Frequency in Hz

        Returns:
            MIDI note number (0-127)
        """
        if freq <= 0:
            return 0

        # MIDI note number = 69 + 12 * log2(f / 440)
        midi_note = 69 + 12 * np.log2(freq / 440.0)
        return int(np.clip(np.round(midi_note), 0, 127))

    def get_lyrics_melody_pairs(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Get paired lyrics and melody data for training

        Returns:
            List of (lyrics_text, note_sequence, timing_sequence) tuples
        """
        pairs = []

        for entry in self.entries:
            # Concatenate all lyrics
            full_lyrics = " ".join([line['text'] for line in entry['lyrics']])

            # Extract note sequence
            notes = np.array([note['midi_note'] for note in entry['notes']])
            timings = np.array([
                [note['start_time'], note['duration']] for note in entry['notes']
            ])

            if len(notes) > 0 and len(full_lyrics) > 0:
                pairs.append((full_lyrics, notes, timings))

        return pairs

    def get_word_note_alignment(self) -> List[Dict]:
        """
        Get word-to-note alignment data

        Returns:
            List of alignment dictionaries
        """
        alignments = []

        for entry in self.entries:
            for align in entry['alignment']:
                if align['word'] and align['notes']:
                    alignments.append({
                        'word': align['word'],
                        'notes': align['notes'],
                        'start_time': align['start_time']
                    })

        return alignments

    def __len__(self) -> int:
        """Get number of entries"""
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        """Get entry by index"""
        return self.entries[idx]


if __name__ == "__main__":
    # Test DALI loader
    loader = DALILoader(
        root_dir="data/raw/dali",
        version="v2.0"
    )

    print("Testing DALI Loader...")
    print(f"Root directory: {loader.root_dir}")

    # Try to load annotations
    try:
        entries = loader.load_annotations()
        print(f"Loaded {len(entries)} entries")

        if entries:
            print("\nSample entry:")
            print(f"Artist: {entries[0]['artist']}")
            print(f"Title: {entries[0]['title']}")
            print(f"Lyrics lines: {len(entries[0]['lyrics'])}")
            print(f"Notes: {len(entries[0]['notes'])}")

    except Exception as e:
        print(f"Note: {e}")
        print("DALI dataset needs to be downloaded separately")
