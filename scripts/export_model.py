"""CLI Script: Export model and metadata to standalone deployment bundle."""

import json
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.utils.logging import get_logger

logger = get_logger("export_model")


def export_bundle(output_dir: Path) -> Path:
    logger.info("Packaging deployment bundle for MotionGuard AI...")
    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir = artifacts_dir / "models"
    metadata_file = models_dir / "best_model_metadata.json"

    if not metadata_file.exists():
        logger.error(f"Cannot export: {metadata_file} missing. Run train_all.py first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_tar = output_dir / "motionguard_deployment_bundle.tar.gz"

    with tarfile.open(bundle_tar, "w:gz") as tar:
        tar.add(str(artifacts_dir), arcname="artifacts")
        tar.add(str(PROJECT_ROOT / "configs" / "inference.yaml"), arcname="configs/inference.yaml")

    logger.info(f"Deployment bundle created at: {bundle_tar} ({bundle_tar.stat().st_size / (1024*1024):.2f} MB)")
    print(f"\n[SUCCESS] Deployment bundle exported to: {bundle_tar}")
    return bundle_tar


def main():
    out = PROJECT_ROOT / "artifacts" / "export"
    export_bundle(out)


if __name__ == "__main__":
    main()
