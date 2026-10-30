"""CLI Script: Validate KU-HAR dataset schema and integrity."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motionguard.config import DataConfig
from motionguard.data.validate import DataValidator
from motionguard.utils.logging import get_logger

logger = get_logger("validate_data")


def main():
    logger.info("Starting KU-HAR dataset validation...")
    config = DataConfig.from_yaml()
    csv_path = config.raw_dir / config.csv_filename

    if not csv_path.exists():
        # Check if zip exists
        archive_path = config.raw_dir / config.archive_filename
        if archive_path.exists():
            logger.info(f"Extracting {archive_path}...")
            from motionguard.data.download import safe_extract_zip
            safe_extract_zip(archive_path, config.raw_dir)
        else:
            logger.error(f"Dataset CSV not found at {csv_path}. Run download_data.py first.")
            sys.exit(1)

    validator = DataValidator(config=config)
    report = validator.validate_csv(csv_path)

    # Save report
    report_file = PROJECT_ROOT / "data" / "manifests" / "validation_report.json"
    report.save(report_file)

    print("\n" + "=" * 60)
    print("           KU-HAR DATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Validation Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print(f"Row count:         {report.row_count} (Expected: {report.expected_rows})")
    print(f"Column count:      {report.column_count} (Expected: {report.expected_cols})")
    print(f"Classes found:     {report.num_classes_found} ({report.unique_classes})")
    print(f"Missing values:    {report.missing_values_count}")
    print(f"Infinite values:   {report.infinite_values_count}")
    print(f"Duplicate rows:    {report.duplicate_rows_count}")
    print(f"Near-constant ch:  {report.near_constant_channels}")
    if report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for w in report.warnings[:5]:
            print(f"  - {w}")
    if report.schema_errors:
        print(f"Errors ({len(report.schema_errors)}):")
        for e in report.schema_errors:
            print(f"  - {e}")
    print("=" * 60)

    if not report.is_valid:
        logger.error("Dataset validation failed. Please check errors above.")
        sys.exit(1)
    else:
        logger.info("Dataset validation passed all integrity checks.")


if __name__ == "__main__":
    main()
