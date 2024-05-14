"""
Feature Extraction Pipeline for Text and Music
Extracts semantic, emotional, and structural features from lyrics and melodies
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import warnings

# NLP imports
try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
except ImportError:
    warnings.warn("transformers not installed")

try:
    from textblob import TextBlob
except ImportError:
    warnings.warn("textblob not installed")


class TextFeatureExtractor:
    """Extract features from lyrics/text"""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        use_sentiment: bool = True,
        use_emotion: bool = True,
        max_length: int = 512
    ):
        """
        Initialize text feature extractor

        Args:
            model_name: Pre-trained model for embeddings
            use_sentiment: Extract sentiment features
            use_emotion: Extract emotion features
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.use_sentiment = use_sentiment
        self.use_emotion = use_emotion
        self.max_length = max_length

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except:
            print(f"Warning: Could not load {model_name}")
            self.tokenizer = None
            self.model = None

        # Emotion keywords (simple approach)
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased'],
            'sad': ['sad', 'sorrow', 'grief', 'depressed', 'melancholy', 'unhappy'],
            'angry': ['angry', 'furious', 'mad', 'rage', 'upset', 'annoyed'],
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'quiet'],
            'love': ['love', 'romantic', 'affection', 'passion', 'adore', 'cherish'],
            'fear': ['fear', 'scared', 'afraid', 'anxious', 'worried', 'nervous']
        }

    def extract_embeddings(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings from text

        Args:
            text: Input lyrics text

        Returns:
            Embedding vector
        """
        if self.model is None or self.tokenizer is None:
            # Return zero vector if model not available
            return np.zeros(768)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings.squeeze()

    def extract_sentiment(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features

        Args:
            text: Input text

        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_sentiment:
            return {'polarity': 0.0, 'subjectivity': 0.0}

        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment

            return {
                'polarity': float(sentiment.polarity),  # -1 to 1
                'subjectivity': float(sentiment.subjectivity)  # 0 to 1
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}

    def extract_emotion(self, text: str) -> Dict[str, float]:
        """
        Extract emotion scores (simple keyword-based approach)

        Args:
            text: Input text

        Returns:
            Dictionary with emotion scores
        """
        if not self.use_emotion:
            return {emotion: 0.0 for emotion in self.emotion_keywords}

        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            # Count keyword occurrences
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length
            emotion_scores[emotion] = score / max(len(text.split()), 1)

        return emotion_scores

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features (syllable count, word count, etc.)

        Args:
            text: Input text

        Returns:
            Dictionary with linguistic features
        """
        words = text.split()

        features = {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'syllable_count': self._count_syllables(text)
        }

        return features

    def _count_syllables(self, text: str) -> int:
        """
        Simple syllable counter (approximation)

        Args:
            text: Input text

        Returns:
            Approximate syllable count
        """
        # Very simple heuristic: count vowel groups
        vowels = 'aeiouAEIOU'
        syllable_count = 0
        previous_was_vowel = False

        for char in text:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        return max(syllable_count, 1)

    def extract_all_features(self, text: str) -> Dict:
        """
        Extract all text features

        Args:
            text: Input lyrics text

        Returns:
            Dictionary with all features
        """
        features = {
            'embeddings': self.extract_embeddings(text),
            'sentiment': self.extract_sentiment(text),
            'emotion': self.extract_emotion(text),
            'linguistic': self.extract_linguistic_features(text)
        }

        return features


class MusicFeatureExtractor:
    """Extract features from musical notes and sequences"""

    def __init__(
        self,
        note_range: Tuple[int, int] = (21, 108),
        extract_harmony: bool = True
    ):
        """
        Initialize music feature extractor

        Args:
            note_range: Valid MIDI note range
            extract_harmony: Extract harmonic features
        """
        self.note_range = note_range
        self.extract_harmony = extract_harmony

    def extract_pitch_features(self, notes: np.ndarray) -> Dict[str, float]:
        """
        Extract pitch-related features

        Args:
            notes: Array of MIDI note numbers

        Returns:
            Dictionary with pitch features
        """
        if len(notes) == 0:
            return {
                'mean_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_std': 0.0
            }

        features = {
            'mean_pitch': float(np.mean(notes)),
            'median_pitch': float(np.median(notes)),
            'pitch_range': float(np.ptp(notes)),
            'pitch_std': float(np.std(notes)),
            'min_pitch': float(np.min(notes)),
            'max_pitch': float(np.max(notes))
        }

        return features

    def extract_interval_features(self, notes: np.ndarray) -> Dict[str, float]:
        """
        Extract melodic interval features

        Args:
            notes: Array of MIDI note numbers

        Returns:
            Dictionary with interval features
        """
        if len(notes) < 2:
            return {
                'mean_interval': 0.0,
                'interval_std': 0.0,
                'max_interval': 0.0
            }

        # Calculate intervals (differences between consecutive notes)
        intervals = np.diff(notes.astype(float))

        features = {
            'mean_interval': float(np.mean(np.abs(intervals))),
            'interval_std': float(np.std(intervals)),
            'max_interval': float(np.max(np.abs(intervals))),
            'ascending_ratio': float(np.sum(intervals > 0) / len(intervals)),
            'descending_ratio': float(np.sum(intervals < 0) / len(intervals)),
            'static_ratio': float(np.sum(intervals == 0) / len(intervals))
        }

        return features

    def extract_contour_features(self, notes: np.ndarray) -> Dict[str, float]:
        """
        Extract melodic contour features

        Args:
            notes: Array of MIDI note numbers

        Returns:
            Dictionary with contour features
        """
        if len(notes) < 3:
            return {
                'contour_direction': 0.0,
                'contour_complexity': 0.0
            }

        # Calculate overall direction
        direction = (notes[-1] - notes[0]) / max(len(notes), 1)

        # Calculate direction changes (complexity)
        diffs = np.diff(notes.astype(float))
        direction_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        features = {
            'contour_direction': float(direction),  # Overall rising/falling
            'contour_complexity': float(direction_changes / max(len(notes) - 2, 1))
        }

        return features

    def extract_rhythm_features(
        self,
        durations: np.ndarray,
        timings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract rhythmic features

        Args:
            durations: Array of note durations
            timings: Array of note start times (optional)

        Returns:
            Dictionary with rhythm features
        """
        if len(durations) == 0:
            return {
                'mean_duration': 0.0,
                'duration_std': 0.0,
                'rhythm_complexity': 0.0
            }

        features = {
            'mean_duration': float(np.mean(durations)),
            'duration_std': float(np.std(durations)),
            'min_duration': float(np.min(durations)),
            'max_duration': float(np.max(durations))
        }

        # Rhythm complexity (variety in durations)
        unique_durations = len(np.unique(np.round(durations, 2)))
        features['rhythm_complexity'] = unique_durations / max(len(durations), 1)

        # Note density (if timing provided)
        if timings is not None and len(timings) > 1:
            total_time = timings[-1] - timings[0]
            features['note_density'] = len(durations) / max(total_time, 0.1)

        return features

    def extract_pitch_class_distribution(self, notes: np.ndarray) -> np.ndarray:
        """
        Extract pitch class histogram (12-bin for chromatic scale)

        Args:
            notes: Array of MIDI note numbers

        Returns:
            12-dimensional pitch class distribution
        """
        pitch_classes = notes % 12
        distribution = np.zeros(12)

        for pc in pitch_classes:
            distribution[int(pc)] += 1

        # Normalize
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()

        return distribution

    def extract_all_features(
        self,
        notes: np.ndarray,
        durations: Optional[np.ndarray] = None,
        timings: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Extract all music features

        Args:
            notes: Array of MIDI note numbers
            durations: Array of note durations (optional)
            timings: Array of note start times (optional)

        Returns:
            Dictionary with all features
        """
        features = {
            'pitch': self.extract_pitch_features(notes),
            'intervals': self.extract_interval_features(notes),
            'contour': self.extract_contour_features(notes),
            'pitch_class_dist': self.extract_pitch_class_distribution(notes)
        }

        if durations is not None:
            features['rhythm'] = self.extract_rhythm_features(durations, timings)

        return features


if __name__ == "__main__":
    # Test feature extractors
    print("Testing Feature Extractors...")

    # Test text features
    print("\n1. Text Feature Extraction:")
    text_extractor = TextFeatureExtractor()

    sample_lyrics = "I'm so happy and excited to see you again, my love"
    text_features = text_extractor.extract_all_features(sample_lyrics)

    print(f"Sentiment: {text_features['sentiment']}")
    print(f"Emotion scores: {text_features['emotion']}")
    print(f"Linguistic: {text_features['linguistic']}")
    print(f"Embedding shape: {text_features['embeddings'].shape}")

    # Test music features
    print("\n2. Music Feature Extraction:")
    music_extractor = MusicFeatureExtractor()

    sample_notes = np.array([60, 62, 64, 65, 67, 69, 71, 72])  # C major scale
    sample_durations = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0])

    music_features = music_extractor.extract_all_features(
        sample_notes,
        sample_durations
    )

    print(f"Pitch features: {music_features['pitch']}")
    print(f"Interval features: {music_features['intervals']}")
    print(f"Rhythm features: {music_features['rhythm']}")
    print(f"Pitch class distribution: {music_features['pitch_class_dist']}")
