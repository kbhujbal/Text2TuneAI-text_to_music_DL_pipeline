"""
Data Setup Script
Download and prepare datasets for Text2TuneAI
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_config
from src.data import DALILoader, LakhMIDILoader


def setup_dali_dataset(config):
    """Setup DALI dataset"""
    print("=" * 80)
    print("DALI Dataset Setup")
    print("=" * 80)

    dali_config = config.get('data', {}).get('dali', {})

    loader = DALILoader(
        root_dir=dali_config.get('root_dir', 'data/raw/dali'),
        version=dali_config.get('version', 'v2.0'),
        sample_rate=dali_config.get('sample_rate', 22050)
    )

    print("\nDALI Dataset Instructions:")
    print("-" * 80)
    print("1. Clone DALI repository:")
    print("   git clone https://github.com/gabolsgabs/DALI")
    print("\n2. Download DALI annotations:")
    print("   cd DALI")
    print("   wget https://github.com/gabolsgabs/DALI/releases/download/v2.0/dali_v2.0.tar.gz")
    print("   tar -xzf dali_v2.0.tar.gz")
    print("\n3. Move annotations to project:")
    print(f"   mv dali_v2.0/* {loader.root_dir}/annotations/")
    print("\n4. (Optional) Download audio using DALI tools")
    print("   Refer to: https://github.com/gabolsgabs/DALI")
    print("-" * 80)

    # Try to setup
    success = loader.download_and_setup()

    if success:
        print("\nDALI dataset setup successful!")
        try:
            entries = loader.load_annotations()
            print(f"Loaded {len(entries)} DALI entries")
        except Exception as e:
            print(f"Note: {e}")
    else:
        print("\nPlease follow the instructions above to download DALI dataset")

    return success


def setup_lakh_dataset(config):
    """Setup Lakh MIDI dataset"""
    print("\n" + "=" * 80)
    print("Lakh MIDI Dataset Setup")
    print("=" * 80)

    lakh_config = config.get('data', {}).get('lakh', {})

    loader = LakhMIDILoader(
        root_dir=lakh_config.get('root_dir', 'data/raw/lakh_midi'),
        subset_size=lakh_config.get('subset_size', 10000)
    )

    print("\nLakh MIDI Dataset Instructions:")
    print("-" * 80)
    print("1. Visit: http://colinraffel.com/projects/lmd/")
    print("\n2. Download 'Lakh MIDI Dataset' (choose one):")
    print("   - Full version (~25GB): Contains all MIDI files")
    print("   - Clean version (~3GB): Recommended, contains matched/aligned files")
    print("\n3. Download link:")
    print("   wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz")
    print("   or")
    print("   wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz")
    print("\n4. Extract to project:")
    print(f"   tar -xzf lmd_matched.tar.gz -C {loader.root_dir}")
    print("-" * 80)

    # Try to setup
    success = loader.download_and_setup()

    if success:
        print("\nLakh MIDI dataset setup successful!")
        print(f"Found {len(loader.midi_files)} MIDI files")
    else:
        print("\nPlease follow the instructions above to download Lakh MIDI dataset")

    return success


def verify_datasets(config):
    """Verify that datasets are properly set up"""
    print("\n" + "=" * 80)
    print("Verifying Datasets")
    print("=" * 80)

    # Check DALI
    dali_config = config.get('data', {}).get('dali', {})
    dali_root = Path(dali_config.get('root_dir', 'data/raw/dali'))

    print(f"\nDALI dataset path: {dali_root}")
    if dali_root.exists():
        annotations = list(dali_root.glob("**/*.gz"))
        print(f"  Status: {'✓ Found' if annotations else '✗ Not found'}")
        print(f"  Files: {len(annotations)} annotation files")
    else:
        print("  Status: ✗ Directory not found")

    # Check Lakh
    lakh_config = config.get('data', {}).get('lakh', {})
    lakh_root = Path(lakh_config.get('root_dir', 'data/raw/lakh_midi'))

    print(f"\nLakh MIDI dataset path: {lakh_root}")
    if lakh_root.exists():
        midi_files = list(lakh_root.glob("**/*.mid")) + list(lakh_root.glob("**/*.midi"))
        print(f"  Status: {'✓ Found' if midi_files else '✗ Not found'}")
        print(f"  Files: {len(midi_files)} MIDI files")
    else:
        print("  Status: ✗ Directory not found")

    print("\n" + "=" * 80)


def main():
    """Main setup function"""
    print("=" * 80)
    print("TEXT2TUNE AI - Dataset Setup")
    print("=" * 80)

    # Load config
    config = get_config()
    config_dict = config.to_dict()

    # Setup datasets
    print("\nThis script will guide you through downloading the required datasets.")
    print("Note: Datasets are NOT automatically downloaded due to size.")
    print("You will need to manually download them following the instructions.\n")

    input("Press Enter to continue...")

    # Setup DALI
    setup_dali_dataset(config_dict)

    input("\nPress Enter to continue to Lakh MIDI setup...")

    # Setup Lakh
    setup_lakh_dataset(config_dict)

    # Verify
    input("\nPress Enter to verify datasets...")
    verify_datasets(config_dict)

    print("\n" + "=" * 80)
    print("Setup complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Ensure datasets are downloaded and extracted to the correct locations")
    print("2. Run: python -m src.data.dataset to test data loading")
    print("3. Run: python -m src.training.trainer to start training")
    print("=" * 80)


if __name__ == "__main__":
    main()
