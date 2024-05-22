"""
Text2Tune Main Model
Complete architecture combining text encoder and music decoder
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .text_encoder import TextEncoder, EmotionConditioner
from .music_decoder import MusicDecoder


class Text2TuneModel(nn.Module):
    """
    Complete Text-to-Music generation model
    Combines BERT text encoder with Transformer music decoder
    """

    def __init__(self, config: Dict):
        """
        Initialize Text2Tune model

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config

        # Extract model configurations
        text_config = config.get('model', {}).get('text_encoder', {})
        music_config = config.get('model', {}).get('music_decoder', {})
        conditioning_config = config.get('model', {}).get('conditioning', {})

        # Get hidden sizes
        text_hidden_size = text_config.get('hidden_size', 768)
        music_hidden_size = music_config.get('hidden_size', 512)

        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_config.get('model_name', 'bert-base-uncased'),
            hidden_size=text_hidden_size,
            num_layers=text_config.get('num_layers', 6),
            num_heads=text_config.get('num_heads', 12),
            dropout=text_config.get('dropout', 0.1),
            max_seq_length=text_config.get('max_seq_length', 512),
            freeze_bert=text_config.get('freeze_bert', False),
            use_emotion_head=conditioning_config.get('use_emotion', True)
        )

        # Music decoder
        self.music_decoder = MusicDecoder(
            vocab_size=music_config.get('vocab_size', 128),
            hidden_size=music_hidden_size,
            num_layers=music_config.get('num_layers', 8),
            num_heads=music_config.get('num_heads', 8),
            ff_dim=music_config.get('ff_dim', 2048),
            max_seq_length=music_config.get('max_seq_length', 1024),
            dropout=music_config.get('dropout', 0.1),
            use_duration=True,
            use_velocity=True
        )

        # Bridge layer (project text hidden size to music hidden size)
        if text_hidden_size != music_hidden_size:
            self.bridge = nn.Sequential(
                nn.Linear(text_hidden_size, music_hidden_size),
                nn.LayerNorm(music_hidden_size),
                nn.Dropout(text_config.get('dropout', 0.1))
            )
        else:
            self.bridge = nn.Identity()

        # Optional conditioning modules
        self.use_emotion = conditioning_config.get('use_emotion', True)
        self.use_tempo = conditioning_config.get('use_tempo', True)
        self.use_key = conditioning_config.get('use_key', True)

        if self.use_emotion:
            self.emotion_conditioner = EmotionConditioner(
                num_emotions=6,
                emotion_dim=conditioning_config.get('emotion_dim', 64),
                hidden_size=music_hidden_size
            )

        if self.use_tempo:
            self.tempo_embedding = nn.Sequential(
                nn.Linear(1, conditioning_config.get('tempo_dim', 32)),
                nn.ReLU(),
                nn.Linear(conditioning_config.get('tempo_dim', 32), music_hidden_size)
            )

        if self.use_key:
            # 24 keys (12 major + 12 minor)
            self.key_embedding = nn.Embedding(
                24,
                conditioning_config.get('key_dim', 24)
            )
            self.key_projection = nn.Linear(
                conditioning_config.get('key_dim', 24),
                music_hidden_size
            )

    def forward(
        self,
        texts: Optional[list] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tgt_notes: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        key_ids: Optional[torch.Tensor] = None,
        return_text_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            texts: List of text strings
            input_ids: Tokenized text (alternative to texts)
            attention_mask: Attention mask for text
            tgt_notes: Target note sequence for training
            emotion_ids: Emotion class indices (optional)
            tempo: Tempo values (optional)
            key_ids: Key indices (optional)
            return_text_features: Return intermediate text features

        Returns:
            Dictionary with model outputs
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts,
            return_emotion=self.use_emotion,
            return_sentiment=True
        )

        text_embeddings = text_outputs['embeddings']  # (B, L_text, H_text)
        text_mask = text_outputs['attention_mask']  # (B, L_text)

        # Project to music hidden size
        memory = self.bridge(text_embeddings)  # (B, L_text, H_music)

        # Add conditioning if provided
        if self.use_emotion and emotion_ids is not None:
            emotion_cond = self.emotion_conditioner(emotion_ids)  # (B, H_music)
            # Add to memory as additional context
            emotion_cond = emotion_cond.unsqueeze(1)  # (B, 1, H_music)
            memory = torch.cat([memory, emotion_cond], dim=1)
            # Extend mask
            emotion_mask = torch.ones(
                emotion_ids.size(0), 1,
                dtype=torch.bool,
                device=memory.device
            )
            text_mask = torch.cat([text_mask, emotion_mask], dim=1)

        if self.use_tempo and tempo is not None:
            tempo_cond = self.tempo_embedding(tempo.unsqueeze(-1))  # (B, H_music)
            tempo_cond = tempo_cond.unsqueeze(1)
            memory = torch.cat([memory, tempo_cond], dim=1)
            tempo_mask = torch.ones(
                tempo.size(0), 1,
                dtype=torch.bool,
                device=memory.device
            )
            text_mask = torch.cat([text_mask, tempo_mask], dim=1)

        if self.use_key and key_ids is not None:
            key_embeds = self.key_embedding(key_ids)
            key_cond = self.key_projection(key_embeds).unsqueeze(1)
            memory = torch.cat([memory, key_cond], dim=1)
            key_mask = torch.ones(
                key_ids.size(0), 1,
                dtype=torch.bool,
                device=memory.device
            )
            text_mask = torch.cat([text_mask, key_mask], dim=1)

        # Decode music
        if tgt_notes is not None:
            music_outputs = self.music_decoder(
                tgt=tgt_notes,
                memory=memory,
                memory_mask=text_mask
            )
        else:
            music_outputs = {}

        # Combine outputs
        outputs = {
            **music_outputs,
            'text_sentiment': text_outputs.get('sentiment'),
        }

        if 'emotion_logits' in text_outputs:
            outputs['text_emotion_logits'] = text_outputs['emotion_logits']

        if return_text_features:
            outputs['text_embeddings'] = text_embeddings
            outputs['text_pooled'] = text_outputs['pooled']

        return outputs

    @torch.no_grad()
    def generate_music(
        self,
        texts: str,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        emotion_id: Optional[int] = None,
        tempo: Optional[float] = None,
        key_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate music from text

        Args:
            texts: Input lyrics/text
            max_length: Maximum notes to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            emotion_id: Emotion conditioning
            tempo: Tempo conditioning
            key_id: Key conditioning

        Returns:
            Tuple of (generated_notes, generated_durations)
        """
        self.eval()

        # Encode text
        if isinstance(texts, str):
            texts = [texts]

        # Prepare conditioning
        device = next(self.parameters()).device

        emotion_ids = None
        if emotion_id is not None:
            emotion_ids = torch.tensor([emotion_id], device=device)

        tempo_tensor = None
        if tempo is not None:
            tempo_tensor = torch.tensor([tempo], device=device, dtype=torch.float)

        key_ids = None
        if key_id is not None:
            key_ids = torch.tensor([key_id], device=device)

        # Get text encoding
        text_outputs = self.text_encoder(texts=texts)
        text_embeddings = text_outputs['embeddings']
        text_mask = text_outputs['attention_mask']

        # Project to music space
        memory = self.bridge(text_embeddings)

        # Add conditioning
        if self.use_emotion and emotion_ids is not None:
            emotion_cond = self.emotion_conditioner(emotion_ids).unsqueeze(1)
            memory = torch.cat([memory, emotion_cond], dim=1)
            emotion_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            text_mask = torch.cat([text_mask, emotion_mask], dim=1)

        if self.use_tempo and tempo_tensor is not None:
            tempo_cond = self.tempo_embedding(tempo_tensor.unsqueeze(-1)).unsqueeze(1)
            memory = torch.cat([memory, tempo_cond], dim=1)
            tempo_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            text_mask = torch.cat([text_mask, tempo_mask], dim=1)

        if self.use_key and key_ids is not None:
            key_embeds = self.key_embedding(key_ids)
            key_cond = self.key_projection(key_embeds).unsqueeze(1)
            memory = torch.cat([memory, key_cond], dim=1)
            key_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            text_mask = torch.cat([text_mask, key_mask], dim=1)

        # Generate music
        generated_notes, generated_durations = self.music_decoder.generate(
            memory=memory,
            memory_mask=text_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        return generated_notes, generated_durations


if __name__ == "__main__":
    # Test complete model
    print("Testing Text2Tune Complete Model...")

    from src.utils.config import get_config

    # Load config
    config = get_config()
    config_dict = config.to_dict()

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = Text2TuneModel(config_dict).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n\nTesting forward pass...")
    sample_texts = [
        "I'm so happy to see you again",
        "The sadness fills my heart"
    ]

    batch_size = 2
    seq_len = 32

    tgt_notes = torch.randint(0, 128, (batch_size, seq_len), device=device)

    outputs = model(
        texts=sample_texts,
        tgt_notes=tgt_notes
    )

    print(f"\nOutput shapes:")
    print(f"  Note logits: {outputs['note_logits'].shape}")
    print(f"  Durations: {outputs['durations'].shape}")
    print(f"  Sentiment: {outputs['text_sentiment'].shape}")

    # Test generation
    print("\n\nTesting music generation...")
    test_text = "A beautiful sunny day filled with joy and happiness"

    generated_notes, generated_durations = model.generate_music(
        texts=test_text,
        max_length=64,
        temperature=1.0,
        top_k=50
    )

    print(f"Generated notes shape: {generated_notes.shape}")
    print(f"Generated durations shape: {generated_durations.shape}")
    print(f"Sample notes: {generated_notes[0, :10].tolist()}")
    print(f"Sample durations: {generated_durations[0, :10].tolist()}")

    print("\nTest passed!")
