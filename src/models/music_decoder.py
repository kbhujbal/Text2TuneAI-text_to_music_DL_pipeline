"""
Music Decoder Module
Transformer-based decoder for generating musical note sequences from text embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MusicDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (with text encoder output)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Target sequence (B, L, H)
            memory: Encoder output (B, L_enc, H)
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask

        Returns:
            Decoded sequence (B, L, H)
        """
        # Self-attention with residual
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention with residual
        cross_output, _ = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_mask,
            need_weights=False
        )
        x = self.norm2(x + self.dropout(cross_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class MusicDecoder(nn.Module):
    """
    Transformer decoder for music generation
    Generates note sequences conditioned on text embeddings
    """

    def __init__(
        self,
        vocab_size: int = 128,  # MIDI note range
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
        use_duration: bool = True,
        use_velocity: bool = True
    ):
        """
        Initialize music decoder

        Args:
            vocab_size: Size of note vocabulary (typically 128 for MIDI)
            hidden_size: Hidden dimension size
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            use_duration: Predict note durations
            use_velocity: Predict note velocities
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.use_duration = use_duration
        self.use_velocity = use_velocity

        # Note embedding
        self.note_embedding = nn.Embedding(vocab_size, hidden_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_length, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            MusicDecoderLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        self.note_head = nn.Linear(hidden_size, vocab_size)

        if use_duration:
            self.duration_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softplus()  # Ensure positive durations
            )

        if use_velocity:
            self.velocity_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 128),  # MIDI velocity range 0-127
            )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass

        Args:
            tgt: Target note sequence (B, L)
            memory: Encoder output (B, L_enc, H)
            tgt_mask: Target attention mask
            memory_mask: Memory key padding mask (B, L_enc)

        Returns:
            Dictionary with predictions:
                - note_logits: Note predictions (B, L, vocab_size)
                - durations: Duration predictions (B, L, 1)
                - velocities: Velocity predictions (B, L, 128)
        """
        batch_size, seq_len = tgt.shape

        # Embed notes
        x = self.note_embedding(tgt)  # (B, L, H)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)

        # Convert memory_mask to correct format for MultiheadAttention
        # MultiheadAttention expects True for positions to mask out
        if memory_mask is not None:
            memory_mask = ~memory_mask  # Invert: True -> False, False -> True

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        # Layer norm
        x = self.layer_norm(x)

        # Output predictions
        outputs = {}

        # Note predictions
        note_logits = self.note_head(x)  # (B, L, vocab_size)
        outputs['note_logits'] = note_logits

        # Duration predictions
        if self.use_duration:
            durations = self.duration_head(x)  # (B, L, 1)
            outputs['durations'] = durations.squeeze(-1)

        # Velocity predictions
        if self.use_velocity:
            velocities = self.velocity_head(x)  # (B, L, 128)
            outputs['velocities'] = velocities

        return outputs

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        start_token: int = 60,  # Middle C
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation

        Args:
            memory: Encoder output (B, L_enc, H)
            memory_mask: Memory attention mask
            max_length: Maximum sequence length to generate
            start_token: Starting note
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            Tuple of (generated_notes, generated_durations)
        """
        batch_size = memory.shape[0]
        device = memory.device

        # Start with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        durations = []

        for _ in range(max_length - 1):
            # Forward pass
            outputs = self.forward(generated, memory, memory_mask=memory_mask)

            # Get logits for last position
            logits = outputs['note_logits'][:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Get duration if available
            if self.use_duration and 'durations' in outputs:
                dur = outputs['durations'][:, -1:].cpu()
                durations.append(dur)

        # Concatenate durations
        if durations:
            durations = torch.cat(durations, dim=1)
        else:
            durations = torch.ones(batch_size, max_length - 1, device=device) * 0.5

        return generated, durations


if __name__ == "__main__":
    # Test music decoder
    print("Testing Music Decoder...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create decoder
    decoder = MusicDecoder(
        vocab_size=128,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        max_seq_length=256,
        dropout=0.1
    ).to(device)

    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    enc_len = 16

    tgt = torch.randint(0, 128, (batch_size, seq_len), device=device)
    memory = torch.randn(batch_size, enc_len, 512, device=device)
    memory_mask = torch.ones(batch_size, enc_len, dtype=torch.bool, device=device)

    print(f"\nInput shapes:")
    print(f"  Target: {tgt.shape}")
    print(f"  Memory: {memory.shape}")
    print(f"  Memory mask: {memory_mask.shape}")

    # Forward pass
    outputs = decoder(tgt, memory, memory_mask=memory_mask)

    print(f"\nOutput shapes:")
    print(f"  Note logits: {outputs['note_logits'].shape}")
    print(f"  Durations: {outputs['durations'].shape}")

    # Test generation
    print("\n\nTesting autoregressive generation...")
    generated_notes, generated_durations = decoder.generate(
        memory,
        memory_mask=memory_mask,
        max_length=64,
        temperature=1.0,
        top_k=50
    )

    print(f"Generated notes shape: {generated_notes.shape}")
    print(f"Generated durations shape: {generated_durations.shape}")
    print(f"Sample notes: {generated_notes[0, :10].tolist()}")

    print("\nTest passed!")
