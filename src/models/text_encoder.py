"""
Text Encoder Module
BERT-based encoder for extracting semantic and emotional features from lyrics
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple


class TextEncoder(nn.Module):
    """
    Text encoder using pre-trained BERT
    Extracts semantic embeddings and emotional features from lyrics
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_seq_length: int = 512,
        freeze_bert: bool = False,
        use_emotion_head: bool = True,
        num_emotions: int = 6
    ):
        """
        Initialize text encoder

        Args:
            model_name: Pre-trained BERT model name
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers to use from BERT
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_length: Maximum input sequence length
            freeze_bert: Whether to freeze BERT weights
            use_emotion_head: Add emotion classification head
            num_emotions: Number of emotion classes
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.use_emotion_head = use_emotion_head

        # Load pre-trained BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Get BERT's hidden size
        bert_hidden_size = self.bert.config.hidden_size

        # Projection layer to match target hidden size
        if bert_hidden_size != hidden_size:
            self.projection = nn.Linear(bert_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()

        # Emotion classification head (optional)
        if use_emotion_head:
            self.emotion_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_emotions)
            )

        # Sentiment regression head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def tokenize(self, texts):
        """
        Tokenize text inputs

        Args:
            texts: List of text strings or single string

        Returns:
            Tokenized inputs
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[list] = None,
        return_emotion: bool = False,
        return_sentiment: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Tokenized input IDs (B, L)
            attention_mask: Attention mask (B, L)
            texts: Raw text strings (alternative to input_ids)
            return_emotion: Return emotion predictions
            return_sentiment: Return sentiment predictions

        Returns:
            Dictionary containing:
                - embeddings: Sequence embeddings (B, L, H)
                - pooled: Pooled representation (B, H)
                - emotion_logits: Emotion predictions (optional)
                - sentiment: Sentiment scores (optional)
        """
        # Tokenize if raw text provided
        if texts is not None:
            tokenized = self.tokenize(texts)
            input_ids = tokenized['input_ids'].to(self.bert.device)
            attention_mask = tokenized['attention_mask'].to(self.bert.device)

        # Get BERT embeddings
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Extract embeddings
        sequence_output = bert_outputs.last_hidden_state  # (B, L, bert_hidden_size)
        pooled_output = bert_outputs.pooler_output  # (B, bert_hidden_size)

        # Project to target hidden size
        sequence_embeddings = self.projection(sequence_output)  # (B, L, H)
        pooled_embeddings = self.projection(pooled_output)  # (B, H)

        # Apply layer norm and dropout
        sequence_embeddings = self.dropout(self.layer_norm(sequence_embeddings))
        pooled_embeddings = self.dropout(self.layer_norm(pooled_embeddings))

        outputs = {
            'embeddings': sequence_embeddings,
            'pooled': pooled_embeddings,
            'attention_mask': attention_mask
        }

        # Emotion prediction
        if return_emotion and self.use_emotion_head:
            emotion_logits = self.emotion_classifier(pooled_embeddings)
            outputs['emotion_logits'] = emotion_logits

        # Sentiment prediction
        if return_sentiment:
            sentiment = self.sentiment_head(pooled_embeddings)
            outputs['sentiment'] = sentiment

        return outputs

    def get_pooled_embedding(self, texts) -> torch.Tensor:
        """
        Get pooled embedding for text(s)

        Args:
            texts: Text string or list of strings

        Returns:
            Pooled embeddings (B, H)
        """
        with torch.no_grad():
            outputs = self.forward(texts=texts)
            return outputs['pooled']

    def get_sequence_embedding(self, texts) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sequence embeddings for text(s)

        Args:
            texts: Text string or list of strings

        Returns:
            Tuple of (embeddings, attention_mask)
        """
        with torch.no_grad():
            outputs = self.forward(texts=texts)
            return outputs['embeddings'], outputs['attention_mask']


class EmotionConditioner(nn.Module):
    """
    Emotion conditioning module
    Projects emotion labels/features to conditioning vectors
    """

    def __init__(
        self,
        num_emotions: int = 6,
        emotion_dim: int = 64,
        hidden_size: int = 512
    ):
        """
        Initialize emotion conditioner

        Args:
            num_emotions: Number of emotion categories
            emotion_dim: Dimension of emotion embeddings
            hidden_size: Target hidden size
        """
        super().__init__()

        self.num_emotions = num_emotions
        self.emotion_dim = emotion_dim

        # Emotion embeddings
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)

        # Project to hidden size
        self.projection = nn.Sequential(
            nn.Linear(emotion_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            emotion_ids: Emotion class indices (B,)

        Returns:
            Conditioning vectors (B, H)
        """
        emotion_embeds = self.emotion_embedding(emotion_ids)
        conditioning = self.projection(emotion_embeds)
        return conditioning


if __name__ == "__main__":
    # Test text encoder
    print("Testing Text Encoder...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create encoder
    encoder = TextEncoder(
        model_name="bert-base-uncased",
        hidden_size=512,
        dropout=0.1,
        use_emotion_head=True
    ).to(device)

    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Test with sample text
    sample_texts = [
        "I'm so happy to see you again, my love",
        "The sadness in my heart will never fade away"
    ]

    print(f"\nInput texts:")
    for text in sample_texts:
        print(f"  - {text}")

    # Forward pass
    outputs = encoder(texts=sample_texts, return_emotion=True, return_sentiment=True)

    print(f"\nOutput shapes:")
    print(f"  Sequence embeddings: {outputs['embeddings'].shape}")
    print(f"  Pooled embeddings: {outputs['pooled'].shape}")
    print(f"  Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"  Sentiment: {outputs['sentiment'].shape}")

    print(f"\nSentiment scores: {outputs['sentiment'].squeeze().tolist()}")

    # Test emotion conditioner
    print("\n\nTesting Emotion Conditioner...")
    conditioner = EmotionConditioner(num_emotions=6, emotion_dim=64, hidden_size=512).to(device)

    emotion_ids = torch.tensor([0, 3], device=device)  # happy, calm
    conditioning = conditioner(emotion_ids)

    print(f"Conditioning vectors shape: {conditioning.shape}")
    print("\nTest passed!")
