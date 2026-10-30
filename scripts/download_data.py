"""CLI Script: Download and verify KU-HAR dataset."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig
from motionguard.data.download import DatasetDownloader
from motionguard.utils.logging import get_logger

logger = get_logger("download_data")


def main():
    logger.info("Initializing KU-HAR dataset download pipeline...")
    config = DataConfig.from_yaml()
    downloader = DatasetDownloader(config=config)
    
    try:
        csv_path = downloader.download(force=False)
        logger.info(f"Dataset successfully downloaded and verified at: {csv_path}")
        print(f"\n[SUCCESS] Dataset ready at: {csv_path}")
    except Exception as e:
        logger.error(f"Download pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
