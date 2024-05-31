"""
PyTorch Lightning Training Module for Text2TuneAI
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, Optional

from src.models import Text2TuneModel, Text2TuneLoss
from src.data import create_dataloaders


class Text2TuneLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Text2Tune training"""

    def __init__(self, config: Dict):
        """
        Initialize Lightning module

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config
        self.save_hyperparameters(config)

        # Initialize model
        self.model = Text2TuneModel(config)

        # Initialize loss function
        loss_weights = config.get('training', {}).get('loss_weights', {})
        self.criterion = Text2TuneLoss(
            reconstruction_weight=loss_weights.get('reconstruction', 1.0),
            emotion_consistency_weight=loss_weights.get('emotion_consistency', 0.3),
            musical_coherence_weight=loss_weights.get('musical_coherence', 0.5),
            rhythm_consistency_weight=loss_weights.get('rhythm_consistency', 0.4),
            pitch_contour_weight=loss_weights.get('pitch_contour', 0.3)
        )

        # Training config
        self.learning_rate = config.get('training', {}).get('learning_rate', 1e-4)
        self.weight_decay = config.get('training', {}).get('weight_decay', 0.01)
        self.warmup_steps = config.get('training', {}).get('warmup_steps', 1000)

        # Metrics storage
        self.validation_step_outputs = []

    def forward(self, batch):
        """Forward pass"""
        return self.model(
            texts=batch['text'],
            tgt_notes=batch['notes']
        )

    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        outputs = self.model(
            texts=batch['text'],
            tgt_notes=batch['notes'][:, :-1]  # Teacher forcing (exclude last token)
        )

        # Prepare targets
        targets = {
            'notes': batch['notes'][:, 1:],  # Shift by 1 for next-token prediction
            'durations': batch['durations'][:, 1:] if 'durations' in batch else None,
            'mask': batch['mask'][:, 1:]
        }

        # Compute loss
        losses = self.criterion(outputs, targets)

        # Log metrics
        self.log('train/loss', losses['loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/recon_loss', losses['reconstruction_loss'], on_step=False, on_epoch=True)
        self.log('train/coherence_loss', losses['coherence_loss'], on_step=False, on_epoch=True)
        self.log('train/contour_loss', losses['contour_loss'], on_step=False, on_epoch=True)
        self.log('train/rhythm_loss', losses['rhythm_loss'], on_step=False, on_epoch=True)

        return losses['loss']

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Forward pass
        outputs = self.model(
            texts=batch['text'],
            tgt_notes=batch['notes'][:, :-1]
        )

        # Prepare targets
        targets = {
            'notes': batch['notes'][:, 1:],
            'durations': batch['durations'][:, 1:] if 'durations' in batch else None,
            'mask': batch['mask'][:, 1:]
        }

        # Compute loss
        losses = self.criterion(outputs, targets)

        # Log metrics
        self.log('val/loss', losses['loss'], prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', losses['reconstruction_loss'], on_step=False, on_epoch=True)
        self.log('val/coherence_loss', losses['coherence_loss'], on_step=False, on_epoch=True)

        # Store outputs for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': losses['loss'],
            'predictions': outputs['note_logits'].argmax(dim=-1),
            'targets': targets['notes']
        })

        return losses['loss']

    def on_validation_epoch_end(self):
        """Compute epoch-level validation metrics"""
        if not self.validation_step_outputs:
            return

        # Compute accuracy
        all_preds = []
        all_targets = []

        for output in self.validation_step_outputs:
            all_preds.append(output['predictions'])
            all_targets.append(output['targets'])

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # Note accuracy
        accuracy = (preds == targets).float().mean()
        self.log('val/note_accuracy', accuracy, prog_bar=True)

        # Clear outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Scheduler
        scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('training', {}).get('num_epochs', 100)
            )
        elif scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.get('training', {}).get('num_epochs', 100)
            )
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms"""
        if self.global_step % 100 == 0:
            grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.log('train/grad_norm', grad_norm, on_step=True, on_epoch=False)


def train_model(config: Dict):
    """
    Main training function

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("TEXT2TUNE AI - Training Pipeline")
    print("=" * 80)

    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create Lightning module
    print("\nInitializing model...")
    lightning_module = Text2TuneLightningModule(config)

    # Count parameters
    total_params = sum(p.numel() for p in lightning_module.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.get('training', {}).get('checkpoint', {}).get('dirpath', 'checkpoints'),
        filename=config.get('training', {}).get('checkpoint', {}).get('filename', 'text2tune-{epoch:02d}-{val_loss:.4f}'),
        monitor=config.get('training', {}).get('checkpoint', {}).get('monitor', 'val/loss'),
        mode=config.get('training', {}).get('checkpoint', {}).get('mode', 'min'),
        save_top_k=config.get('training', {}).get('checkpoint', {}).get('save_top_k', 3),
        save_last=config.get('training', {}).get('checkpoint', {}).get('save_last', True)
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.get('training', {}).get('early_stopping', {}).get('enabled', True):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=config.get('training', {}).get('early_stopping', {}).get('monitor', 'val/loss'),
            patience=config.get('training', {}).get('early_stopping', {}).get('patience', 15),
            mode=config.get('training', {}).get('early_stopping', {}).get('mode', 'min')
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Logger
    logger = None
    if config.get('logging', {}).get('use_tensorboard', True):
        logger = pl.loggers.TensorBoardLogger(
            save_dir=config.get('logging', {}).get('log_dir', 'logs'),
            name='text2tune'
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.get('training', {}).get('num_epochs', 100),
        accelerator='auto',
        devices=config.get('training', {}).get('num_gpus', 1),
        precision=config.get('training', {}).get('precision', '16-mixed'),
        gradient_clip_val=config.get('training', {}).get('gradient_clip', 1.0),
        accumulate_grad_batches=config.get('training', {}).get('accumulate_grad_batches', 4),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.get('logging', {}).get('log_every_n_steps', 50),
        val_check_interval=1.0,
        num_sanity_val_steps=2
    )

    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("=" * 80)

    return lightning_module, trainer


if __name__ == "__main__":
    from src.utils.config import get_config

    # Load configuration
    config = get_config()
    config_dict = config.to_dict()

    # Train model
    model, trainer = train_model(config_dict)
