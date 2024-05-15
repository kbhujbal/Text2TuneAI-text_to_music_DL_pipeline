"""
PyTorch Dataset Classes for Text2TuneAI
Handles data loading and batching for training
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from .dali_loader import DALILoader
from .lakh_loader import LakhMIDILoader
from .feature_extraction import TextFeatureExtractor, MusicFeatureExtractor


class Text2MusicDataset(Dataset):
    """
    Combined dataset for text-to-music generation
    Uses DALI for aligned data and Lakh for music patterns
    """

    def __init__(
        self,
        config: Dict,
        split: str = 'train',
        use_cache: bool = True
    ):
        """
        Initialize dataset

        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
            use_cache: Use cached preprocessed data
        """
        self.config = config
        self.split = split
        self.use_cache = use_cache

        # Initialize feature extractors
        self.text_extractor = TextFeatureExtractor(
            model_name=config.get('model', {}).get('text_encoder', {}).get('model_name', 'bert-base-uncased'),
            max_length=config.get('model', {}).get('text_encoder', {}).get('max_seq_length', 512)
        )

        self.music_extractor = MusicFeatureExtractor(
            note_range=tuple(config.get('model', {}).get('music_features', {}).get('note_range', [21, 108]))
        )

        # Dataset configuration
        self.max_notes = config.get('model', {}).get('music_decoder', {}).get('max_seq_length', 1024)
        self.note_vocab_size = config.get('model', {}).get('music_decoder', {}).get('vocab_size', 128)

        # Storage for processed samples
        self.samples = []

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and process data from DALI and Lakh datasets"""
        cache_file = Path(self.config.get('data', {}).get('cache_dir', 'data/cache')) / f'{self.split}_dataset.pkl'

        # Try to load from cache
        if self.use_cache and cache_file.exists():
            print(f"Loading cached dataset from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"Loaded {len(self.samples)} samples from cache")
            return

        print(f"Building {self.split} dataset...")

        # Load DALI data (aligned lyrics-melody)
        dali_samples = self._load_dali_data()

        # Load Lakh MIDI data (music patterns)
        lakh_samples = self._load_lakh_data()

        # Combine samples
        self.samples = dali_samples + lakh_samples

        print(f"Total samples: {len(self.samples)}")

        # Split data
        self.samples = self._split_data(self.samples)

        # Cache processed data
        if self.use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"Cached dataset to {cache_file}")

    def _load_dali_data(self) -> List[Dict]:
        """Load and process DALI dataset"""
        dali_config = self.config.get('data', {}).get('dali', {})

        loader = DALILoader(
            root_dir=dali_config.get('root_dir', 'data/raw/dali'),
            version=dali_config.get('version', 'v2.0'),
            sample_rate=dali_config.get('sample_rate', 22050),
            max_duration=dali_config.get('max_duration', 30.0)
        )

        try:
            entries = loader.load_annotations()
            print(f"Loaded {len(entries)} DALI entries")

            samples = []
            for entry in entries:
                # Extract lyrics
                lyrics = " ".join([line['text'] for line in entry['lyrics']])

                if not lyrics.strip():
                    continue

                # Extract notes
                notes = np.array([note['midi_note'] for note in entry['notes']])
                durations = np.array([note['duration'] for note in entry['notes']])

                if len(notes) == 0:
                    continue

                # Create sample
                sample = {
                    'lyrics': lyrics,
                    'notes': notes[:self.max_notes],  # Truncate if needed
                    'durations': durations[:self.max_notes],
                    'source': 'dali',
                    'metadata': entry.get('metadata', {})
                }

                samples.append(sample)

            print(f"Processed {len(samples)} DALI samples")
            return samples

        except Exception as e:
            print(f"Warning: Could not load DALI data: {e}")
            return []

    def _load_lakh_data(self) -> List[Dict]:
        """Load and process Lakh MIDI dataset"""
        lakh_config = self.config.get('data', {}).get('lakh', {})

        loader = LakhMIDILoader(
            root_dir=lakh_config.get('root_dir', 'data/raw/lakh_midi'),
            subset_size=lakh_config.get('subset_size', 10000),
            max_notes=self.max_notes
        )

        try:
            # Scan for MIDI files
            loader._scan_midi_files()

            if len(loader.midi_files) == 0:
                print("No Lakh MIDI files found")
                return []

            # Process MIDI files
            processed = loader.load_midi_files()
            print(f"Loaded {len(processed)} Lakh MIDI files")

            samples = []
            for data in processed:
                # For Lakh data, we don't have lyrics
                # Use placeholder or generate from music features
                placeholder_lyrics = self._generate_placeholder_lyrics(data)

                sample = {
                    'lyrics': placeholder_lyrics,
                    'notes': data['note_sequence'][:self.max_notes],
                    'durations': data['durations'][:self.max_notes],
                    'source': 'lakh',
                    'metadata': {
                        'tempo': data['tempo'],
                        'key': data['key']
                    }
                }

                samples.append(sample)

            print(f"Processed {len(samples)} Lakh samples")
            return samples

        except Exception as e:
            print(f"Warning: Could not load Lakh data: {e}")
            return []

    def _generate_placeholder_lyrics(self, music_data: Dict) -> str:
        """
        Generate placeholder lyrics based on music features
        (For Lakh MIDI files that don't have lyrics)
        """
        tempo = music_data.get('tempo', 120)
        key = music_data.get('key', 'C')

        # Simple template based on musical features
        if tempo > 140:
            mood = "energetic and fast"
        elif tempo > 100:
            mood = "upbeat and lively"
        elif tempo > 70:
            mood = "calm and steady"
        else:
            mood = "slow and peaceful"

        # This is a placeholder - in production, you might want to:
        # 1. Skip Lakh samples during training
        # 2. Use only for melody pre-training
        # 3. Generate synthetic lyrics with GPT
        placeholder = f"A {mood} melody in the key of {key}"

        return placeholder

    def _split_data(self, samples: List[Dict]) -> List[Dict]:
        """Split data into train/val/test"""
        # Get split ratios
        train_ratio = self.config.get('data', {}).get('train_split', 0.8)
        val_ratio = self.config.get('data', {}).get('val_split', 0.1)

        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        if self.split == 'train':
            return samples[:n_train]
        elif self.split == 'val':
            return samples[n_train:n_train + n_val]
        else:  # test
            return samples[n_train + n_val:]

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Returns:
            Dictionary with tensors:
                - text_input: Tokenized lyrics
                - note_target: Target note sequence
                - duration_target: Target duration sequence
                - mask: Padding mask for notes
        """
        sample = self.samples[idx]

        # Process lyrics
        lyrics = sample['lyrics']
        text_features = self.text_extractor.extract_embeddings(lyrics)

        # Process notes
        notes = sample['notes']
        durations = sample['durations']

        # Pad sequences to max length
        padded_notes = np.zeros(self.max_notes, dtype=np.int64)
        padded_durations = np.zeros(self.max_notes, dtype=np.float32)
        mask = np.zeros(self.max_notes, dtype=np.bool_)

        seq_len = min(len(notes), self.max_notes)
        padded_notes[:seq_len] = notes[:seq_len]
        padded_durations[:seq_len] = durations[:seq_len]
        mask[:seq_len] = True

        # Convert to tensors
        return {
            'text_embedding': torch.FloatTensor(text_features),
            'text': lyrics,  # Keep original text for reference
            'notes': torch.LongTensor(padded_notes),
            'durations': torch.FloatTensor(padded_durations),
            'mask': torch.BoolTensor(mask),
            'seq_len': torch.LongTensor([seq_len]),
            'source': sample['source']
        }


def create_dataloaders(config: Dict, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        config: Configuration dictionary
        batch_size: Batch size (overrides config if provided)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 16)

    # Create datasets
    train_dataset = Text2MusicDataset(config, split='train')
    val_dataset = Text2MusicDataset(config, split='val')
    test_dataset = Text2MusicDataset(config, split='test')

    # System settings
    num_workers = config.get('system', {}).get('num_workers', 4)
    pin_memory = config.get('system', {}).get('pin_memory', True)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\nDataLoader Statistics:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    from src.utils.config import get_config

    print("Testing Text2MusicDataset...")

    config = get_config()
    config_dict = config.to_dict()

    # Create dataset
    dataset = Text2MusicDataset(config_dict, split='train', use_cache=False)

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]

        print("\nSample data:")
        print(f"Text: {sample['text'][:100]}...")
        print(f"Text embedding shape: {sample['text_embedding'].shape}")
        print(f"Notes shape: {sample['notes'].shape}")
        print(f"Durations shape: {sample['durations'].shape}")
        print(f"Sequence length: {sample['seq_len'].item()}")
        print(f"Source: {sample['source']}")
