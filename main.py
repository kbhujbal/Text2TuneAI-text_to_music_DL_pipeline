"""
Text2TuneAI - Main Entry Point
Quick start script for training and generation
"""

import argparse
from pathlib import Path

from src.utils.config import get_config
from src.training import train_model
from src.generation import MusicGenerator, load_model_from_checkpoint
from src.models import Text2TuneModel


def train(args):
    """Train Text2Tune model"""
    print("=" * 80)
    print("TEXT2TUNE AI - Training Mode")
    print("=" * 80)

    # Load config
    config = get_config(args.config if args.config else None)
    config_dict = config.to_dict()

    # Override config with command-line arguments
    if args.batch_size:
        config.update('training.batch_size', args.batch_size)
    if args.epochs:
        config.update('training.num_epochs', args.epochs)
    if args.lr:
        config.update('training.learning_rate', args.lr)

    # Train
    model, trainer = train_model(config.to_dict())

    print("\nTraining complete!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


def generate(args):
    """Generate music from text"""
    print("=" * 80)
    print("TEXT2TUNE AI - Generation Mode")
    print("=" * 80)

    # Load config
    config = get_config(args.config if args.config else None)
    config_dict = config.to_dict()

    # Load model
    if args.checkpoint:
        print(f"\nLoading model from checkpoint: {args.checkpoint}")
        model = load_model_from_checkpoint(args.checkpoint, config_dict)
    else:
        print("\nCreating new model (untrained - for demo only)")
        print("WARNING: Use --checkpoint for actual generation")
        model = Text2TuneModel(config_dict)

    # Create generator
    generator = MusicGenerator(model, config_dict)

    # Get input text
    if args.text:
        texts = [args.text]
    elif args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Interactive mode
        print("\nEnter lyrics (Ctrl+D or empty line to finish):")
        texts = []
        try:
            while True:
                line = input("> ")
                if not line:
                    break
                texts.append(line)
        except EOFError:
            pass

    if not texts:
        print("No input text provided. Exiting.")
        return

    # Generate for each text
    print(f"\nGenerating music for {len(texts)} text(s)...")

    output_dir = Path(args.output) if args.output else Path("outputs")

    for idx, text in enumerate(texts):
        print(f"\n[{idx + 1}/{len(texts)}] Text: {text[:60]}...")

        files = generator.generate_and_save(
            text,
            output_dir=output_dir,
            filename=f"generated_{idx:03d}",
            max_length=args.max_length,
            temperature=args.temperature,
            tempo=args.tempo,
            top_k=args.top_k,
            top_p=args.top_p,
            save_midi=True,
            save_audio=args.audio
        )

        print(f"Saved: {files}")

    print(f"\n{'=' * 80}")
    print(f"Generation complete! Files saved to: {output_dir}")
    print(f"{'=' * 80}")


def demo(args):
    """Run a quick demo"""
    print("=" * 80)
    print("TEXT2TUNE AI - Demo Mode")
    print("=" * 80)

    config = get_config()
    config_dict = config.to_dict()

    print("\nCreating model...")
    model = Text2TuneModel(config_dict)

    generator = MusicGenerator(model, config_dict)

    # Demo texts
    demo_texts = [
        "A beautiful sunny day filled with joy and happiness",
        "The darkness falls and sadness fills my heart",
        "Dancing all night long with endless energy"
    ]

    print("\nGenerating demo melodies...")
    print("Note: This uses an untrained model, so output will be random.")
    print("Train the model first for meaningful results.\n")

    for idx, text in enumerate(demo_texts):
        print(f"\n{idx + 1}. {text}")
        files = generator.generate_and_save(
            text,
            output_dir="outputs/demo",
            filename=f"demo_{idx}",
            max_length=64,
            temperature=1.0,
            save_audio=False
        )
        print(f"   Saved: {files.get('midi', 'N/A')}")

    print("\n" + "=" * 80)
    print("Demo complete! Check outputs/demo/ for MIDI files")
    print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Text2TuneAI - Generate music from lyrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, help='Learning rate')

    # Generate mode
    gen_parser = subparsers.add_parser('generate', help='Generate music')
    gen_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    gen_parser.add_argument('--text', type=str, help='Input text')
    gen_parser.add_argument('--text-file', type=str, help='File with input texts (one per line)')
    gen_parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    gen_parser.add_argument('--config', type=str, help='Path to config file')
    gen_parser.add_argument('--max-length', type=int, default=256, help='Max notes to generate')
    gen_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    gen_parser.add_argument('--tempo', type=float, default=120.0, help='Tempo (BPM)')
    gen_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=0.9, help='Nucleus sampling threshold')
    gen_parser.add_argument('--audio', action='store_true', help='Generate audio (requires FluidSynth)')

    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Run demo')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'demo':
        demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
