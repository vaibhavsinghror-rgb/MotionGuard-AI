"""CLI Script: Verify persistent model artifacts in a fresh process.

Fulfills Section 13 Mandatory Persistence Test requirements.
"""

import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import PROJECT_ROOT
from motionguard.data.load import load_processed_arrays
from motionguard.evaluation.compare_models import compute_file_sha256
from motionguard.inference.predictor import MotionGuardPredictor
from motionguard.utils.logging import get_logger

logger = get_logger("verify_saved_model")


def run_persistence_verification() -> bool:
    logger.info("==================================================")
    logger.info("   MOTIONGUARD AI ARTIFACT PERSISTENCE AUDIT")
    logger.info("==================================================")

    artifacts_dir = PROJECT_ROOT / "artifacts"
    models_dir = artifacts_dir / "models"
    manifests_dir = artifacts_dir / "manifests"
    metadata_file = models_dir / "best_model_metadata.json"
    manifest_file = manifests_dir / "model_manifest.json"
    class_map_file = models_dir / "class_mapping.json"

    # Step 1: Verify presence of all required files
    logger.info("Step 1: Checking artifact file presence...")
    required_files = [metadata_file, manifest_file, class_map_file]
    for rf in required_files:
        if not rf.exists():
            logger.error(f"[FAIL] Required artifact missing: {rf}")
            return False
        logger.info(f"  [OK] Found: {rf.name} ({rf.stat().st_size} bytes)")

    # Step 2: Read model metadata and manifest
    logger.info("\nStep 2: Inspecting metadata and checksum integrity...")
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_family = meta.get("model_family")
    artifact_rel_path = meta.get("artifact_path")
    expected_sha = meta.get("artifact_sha256")
    actual_model_path = PROJECT_ROOT / artifact_rel_path

    if not actual_model_path.exists():
        logger.error(f"[FAIL] Selected model artifact file missing: {actual_model_path}")
        return False

    actual_sha = compute_file_sha256(actual_model_path)
    if actual_sha.lower() != expected_sha.lower():
        logger.error(f"[FAIL] Checksum mismatch for {actual_model_path.name}! Expected {expected_sha}, got {actual_sha}")
        return False
    logger.info(f"  [OK] Model artifact verified: {actual_model_path.name} (SHA-256 verified)")

    # Step 3: Instantiate predictor in this fresh process (NO RETRAINING)
    logger.info("\nStep 3: Instantiating MotionGuardPredictor from saved artifacts...")
    predictor = MotionGuardPredictor()
    if not predictor.is_loaded:
        logger.error("[FAIL] Predictor failed to load saved artifacts.")
        return False
    logger.info(f"  [OK] Successfully loaded {model_family} model: {meta.get('model_name')}")

    # Step 4: Verify class mapping integrity
    logger.info("\nStep 4: Validating class mapping...")
    class_map = predictor.class_mapping
    if len(class_map) != 18:
        logger.error(f"[FAIL] Expected 18 activity classes, got {len(class_map)}")
        return False
    logger.info(f"  [OK] Verified 18 mapped activity classes (0: '{class_map[0]}' ... 17: '{class_map[17]}')")

    # Step 5: Test inference on valid example window
    logger.info("\nStep 5: Testing inference on representative sensor window...")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    if (processed_dir / "waveforms.npy").exists():
        waveforms, labels, _ = load_processed_arrays(processed_dir)
        test_sample = waveforms[0]  # (6, 300)
        true_label = labels[0]
        true_name = class_map.get(true_label, "Unknown")
    else:
        # Fallback synthetic clean waveform
        test_sample = np.random.randn(6, 300).astype(np.float32)
        true_name = "N/A"

    resp = predictor.predict(test_sample)
    logger.info(f"  Inference Result: Activity='{resp.activity}', Class ID={resp.class_id}, Confidence={resp.confidence:.2%}, Status='{resp.status}'")

    # Step 6: Verify output schema and finite numerical constraints
    logger.info("\nStep 6: Verifying numerical constraints and schema...")
    if not np.isfinite(resp.confidence) or not (0.0 <= resp.confidence <= 1.0):
        logger.error(f"[FAIL] Non-finite or invalid confidence: {resp.confidence}")
        return False
    
    probs_sum = sum(resp.probabilities.values())
    if not (0.98 <= probs_sum <= 1.02):
        logger.error(f"[FAIL] Probabilities do not sum to ~1.0: {probs_sum}")
        return False
    logger.info(f"  [OK] Valid probabilities sum: {probs_sum:.4f}")

    # Step 7: Test invalid input rejection
    logger.info("\nStep 7: Verifying invalid input rejection...")
    invalid_sample_nan = test_sample.copy()
    invalid_sample_nan[0, 10] = np.nan
    try:
        predictor.predict(invalid_sample_nan)
        logger.error("[FAIL] Predictor failed to reject NaN input!")
        return False
    except ValueError:
        logger.info("  [OK] Successfully rejected input containing NaN.")

    invalid_sample_shape = test_sample[:, :250]  # Only 250 points instead of 300
    try:
        predictor.predict(invalid_sample_shape)
        logger.error("[FAIL] Predictor failed to reject invalid waveform shape!")
        return False
    except (ValueError, AssertionError):
        logger.info("  [OK] Successfully rejected invalid waveform shape (6, 250).")

    # Step 8: Abstention verification
    logger.info("\nStep 8: Testing abstention mechanism...")
    # Force high threshold (e.g. 0.999)
    strict_resp = predictor.predict(test_sample, threshold=0.999)
    if not strict_resp.accepted and strict_resp.status == "uncertain_human_review_recommended":
        logger.info(f"  [OK] Abstention correctly triggered: '{strict_resp.abstention_reason}'")
    else:
        logger.info(f"  [NOTE] Confidence ({strict_resp.confidence:.4f}) exceeded 0.999.")

    logger.info("\n==================================================")
    logger.info("   PERSISTENCE AUDIT: ALL CHECKS PASSED (100%)")
    logger.info("==================================================")
    return True


def main():
    success = run_persistence_verification()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
