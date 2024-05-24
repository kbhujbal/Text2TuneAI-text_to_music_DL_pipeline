"""
Custom Loss Functions for Text2TuneAI
Implements multi-objective losses for music generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class Text2TuneLoss(nn.Module):
    """
    Combined loss function for text-to-music generation
    Includes reconstruction, coherence, and consistency losses
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        emotion_consistency_weight: float = 0.3,
        musical_coherence_weight: float = 0.5,
        rhythm_consistency_weight: float = 0.4,
        pitch_contour_weight: float = 0.3,
        label_smoothing: float = 0.1
    ):
        """
        Initialize loss function

        Args:
            reconstruction_weight: Weight for note reconstruction loss
            emotion_consistency_weight: Weight for emotion consistency
            musical_coherence_weight: Weight for musical coherence
            rhythm_consistency_weight: Weight for rhythm consistency
            pitch_contour_weight: Weight for pitch contour smoothness
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.emotion_consistency_weight = emotion_consistency_weight
        self.musical_coherence_weight = musical_coherence_weight
        self.rhythm_consistency_weight = rhythm_consistency_weight
        self.pitch_contour_weight = pitch_contour_weight

        # Cross-entropy with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )

        # MSE for durations
        self.mse_loss = nn.MSELoss(reduction='none')

    def reconstruction_loss(
        self,
        note_logits: torch.Tensor,
        target_notes: torch.Tensor,
        durations_pred: Optional[torch.Tensor],
        durations_target: Optional[torch.Tensor],
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for notes and durations

        Args:
            note_logits: Predicted note logits (B, L, vocab_size)
            target_notes: Target notes (B, L)
            durations_pred: Predicted durations (B, L)
            durations_target: Target durations (B, L)
            mask: Padding mask (B, L)

        Returns:
            Reconstruction loss
        """
        # Note prediction loss
        note_loss = self.ce_loss(
            note_logits.transpose(1, 2),  # (B, vocab_size, L)
            target_notes
        )

        # Apply mask
        note_loss = (note_loss * mask).sum() / mask.sum()

        # Duration loss
        if durations_pred is not None and durations_target is not None:
            duration_loss = self.mse_loss(durations_pred, durations_target)
            duration_loss = (duration_loss * mask).sum() / mask.sum()

            total_loss = note_loss + 0.5 * duration_loss
        else:
            total_loss = note_loss

        return total_loss

    def musical_coherence_loss(
        self,
        note_logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage musically coherent sequences
        Penalize large interval jumps and reward smooth contours

        Args:
            note_logits: Predicted note logits (B, L, vocab_size)
            mask: Padding mask (B, L)

        Returns:
            Coherence loss
        """
        # Get predicted notes (argmax)
        predicted_notes = torch.argmax(note_logits, dim=-1).float()  # (B, L)

        # Compute intervals (differences between consecutive notes)
        intervals = torch.diff(predicted_notes, dim=1)  # (B, L-1)

        # Penalize large jumps (octave or more)
        large_jump_penalty = torch.relu(torch.abs(intervals) - 12.0)  # Penalize jumps > octave

        # Apply mask (shifted for diff operation)
        if mask.size(1) > 1:
            interval_mask = mask[:, 1:]  # (B, L-1)
            large_jump_penalty = (large_jump_penalty * interval_mask).sum() / interval_mask.sum()
        else:
            large_jump_penalty = large_jump_penalty.mean()

        return large_jump_penalty

    def pitch_contour_smoothness(
        self,
        note_logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage smooth pitch contours
        Penalize rapid direction changes

        Args:
            note_logits: Predicted note logits (B, L, vocab_size)
            mask: Padding mask (B, L)

        Returns:
            Smoothness loss
        """
        # Get predicted notes
        predicted_notes = torch.argmax(note_logits, dim=-1).float()  # (B, L)

        if predicted_notes.size(1) < 3:
            return torch.tensor(0.0, device=predicted_notes.device)

        # First derivative (intervals)
        first_diff = torch.diff(predicted_notes, dim=1)  # (B, L-1)

        # Second derivative (acceleration/direction changes)
        second_diff = torch.diff(first_diff, dim=1)  # (B, L-2)

        # Penalize large second derivatives (rapid direction changes)
        smoothness_penalty = torch.abs(second_diff)

        # Apply mask
        if mask.size(1) > 2:
            smooth_mask = mask[:, 2:]  # (B, L-2)
            smoothness_penalty = (smoothness_penalty * smooth_mask).sum() / smooth_mask.sum()
        else:
            smoothness_penalty = smoothness_penalty.mean()

        return smoothness_penalty

    def rhythm_consistency_loss(
        self,
        durations_pred: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage rhythmic consistency
        Reward patterns and penalize extreme variations

        Args:
            durations_pred: Predicted durations (B, L)
            mask: Padding mask (B, L)

        Returns:
            Rhythm consistency loss
        """
        if durations_pred is None:
            return torch.tensor(0.0)

        # Compute duration variations
        duration_diffs = torch.diff(durations_pred, dim=1)  # (B, L-1)

        # Penalize extreme variations
        variation_penalty = torch.abs(duration_diffs)

        # Apply mask
        if mask.size(1) > 1:
            rhythm_mask = mask[:, 1:]
            variation_penalty = (variation_penalty * rhythm_mask).sum() / rhythm_mask.sum()
        else:
            variation_penalty = variation_penalty.mean()

        return variation_penalty

    def emotion_consistency_loss(
        self,
        predicted_emotion: Optional[torch.Tensor],
        target_emotion: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encourage emotion consistency between text and music

        Args:
            predicted_emotion: Predicted emotion from text (B, num_emotions)
            target_emotion: Target emotion labels (B,)

        Returns:
            Emotion consistency loss
        """
        if predicted_emotion is None or target_emotion is None:
            return torch.tensor(0.0)

        emotion_loss = F.cross_entropy(predicted_emotion, target_emotion)
        return emotion_loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss

        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary

        Returns:
            Dictionary with loss components and total loss
        """
        # Extract predictions
        note_logits = outputs['note_logits']
        durations_pred = outputs.get('durations')

        # Extract targets
        target_notes = targets['notes']
        target_durations = targets.get('durations')
        mask = targets['mask']

        # 1. Reconstruction loss
        recon_loss = self.reconstruction_loss(
            note_logits,
            target_notes,
            durations_pred,
            target_durations,
            mask
        )

        # 2. Musical coherence loss
        coherence_loss = self.musical_coherence_loss(note_logits, mask)

        # 3. Pitch contour smoothness
        contour_loss = self.pitch_contour_smoothness(note_logits, mask)

        # 4. Rhythm consistency loss
        rhythm_loss = self.rhythm_consistency_loss(durations_pred, mask)

        # 5. Emotion consistency loss (if available)
        emotion_loss = torch.tensor(0.0, device=note_logits.device)
        if 'text_emotion_logits' in outputs and 'emotion' in targets:
            emotion_loss = self.emotion_consistency_loss(
                outputs['text_emotion_logits'],
                targets['emotion']
            )

        # Combine losses
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.musical_coherence_weight * coherence_loss +
            self.pitch_contour_weight * contour_loss +
            self.rhythm_consistency_weight * rhythm_loss +
            self.emotion_consistency_weight * emotion_loss
        )

        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'coherence_loss': coherence_loss,
            'contour_loss': contour_loss,
            'rhythm_loss': rhythm_loss,
            'emotion_loss': emotion_loss
        }


if __name__ == "__main__":
    # Test loss functions
    print("Testing Text2Tune Loss Functions...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create loss function
    criterion = Text2TuneLoss(
        reconstruction_weight=1.0,
        emotion_consistency_weight=0.3,
        musical_coherence_weight=0.5,
        rhythm_consistency_weight=0.4,
        pitch_contour_weight=0.3
    )

    # Create dummy data
    batch_size = 4
    seq_len = 32
    vocab_size = 128

    # Model outputs
    outputs = {
        'note_logits': torch.randn(batch_size, seq_len, vocab_size, device=device),
        'durations': torch.rand(batch_size, seq_len, device=device) * 2.0,
        'text_emotion_logits': torch.randn(batch_size, 6, device=device)
    }

    # Targets
    targets = {
        'notes': torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        'durations': torch.rand(batch_size, seq_len, device=device) * 2.0,
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
        'emotion': torch.randint(0, 6, (batch_size,), device=device)
    }

    # Compute loss
    losses = criterion(outputs, targets)

    print("\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("\nTest passed!")
