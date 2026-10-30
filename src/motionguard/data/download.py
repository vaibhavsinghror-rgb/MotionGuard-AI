"""Automated dataset download and verification for MotionGuard AI (KU-HAR)."""

import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from motionguard.config import DataConfig, PROJECT_ROOT
from motionguard.utils.logging import get_logger

logger = get_logger(__name__)


def compute_sha256(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file efficiently."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[Path]:
    """Safely extracts zip archive preventing path traversal (Zip Slip vulnerability)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            # Check for path traversal attempts
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Suspicious path traversal in archive member: {member.filename}")
            
            destination = (target_dir / member_path).resolve()
            if not str(destination).startswith(str(target_dir.resolve())):
                raise ValueError(f"Extracted file path {destination} escapes target directory {target_dir}")
            
            archive.extract(member, target_dir)
            extracted_files.append(destination)
            logger.debug(f"Extracted: {member.filename} -> {destination}")
            
    return extracted_files


def download_file_with_retry(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    max_retries: int = 5,
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """Downloads a file with streaming, exponential backoff, temp file, and checksum check."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + f".tmp_{os.getpid()}")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries} from {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MotionGuard-AI/0.1.0 (KU-HAR Research Downloader)"},
            )
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content_length = response.headers.get("Content-Length")
                total_bytes = int(content_length) if content_length else expected_size
                logger.info(f"Connected. Remote size: {total_bytes} bytes. Streaming to {temp_path.name}...")
                
                downloaded_bytes = 0
                hasher = hashlib.sha256() if expected_sha256 else None
                
                with open(temp_path, "wb") as out_f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if hasher:
                            hasher.update(chunk)
                        
                        if total_bytes:
                            pct = (downloaded_bytes / total_bytes) * 100
                            if downloaded_bytes % (chunk_size * 10) == 0:
                                logger.info(f"Downloaded {downloaded_bytes / (1024*1024):.1f} MB / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%)")

            # Verify size if expected
            if expected_size and downloaded_bytes != expected_size:
                logger.warning(f"Downloaded size ({downloaded_bytes}) does not match expected size ({expected_size})")

            # Verify checksum if expected
            if expected_sha256:
                actual_sha = hasher.hexdigest()
                if actual_sha.lower() != expected_sha256.lower():
                    raise ValueError(f"Checksum mismatch! Expected {expected_sha256}, got {actual_sha}")
                logger.info(f"SHA-256 verified successfully: {actual_sha}")

            # Atomic rename after successful download
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            logger.info(f"Successfully placed downloaded file at {dest_path}")
            return True

        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info(f"Waiting {backoff} seconds before retry...")
                time.sleep(backoff)
            else:
                logger.error(f"All {max_retries} download attempts failed.")
                raise e

    return False


class DatasetDownloader:
    """Manages the download, verification, extraction, and manifest recording of KU-HAR."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig.from_yaml()
        self.raw_dir = self.config.raw_dir
        self.manifest_path = self.config.dataset_manifest
        self.archive_path = self.raw_dir / self.config.archive_filename
        self.csv_path = self.raw_dir / self.config.csv_filename

    def is_cached_and_valid(self) -> bool:
        """Checks if dataset archive or extracted CSV exists and is valid."""
        if self.csv_path.exists() and self.csv_path.stat().st_size > 10_000_000:
            logger.info(f"Found existing valid CSV at {self.csv_path} ({self.csv_path.stat().st_size / (1024*1024):.1f} MB).")
            return True
        if self.archive_path.exists() and self.config.expected_sha256:
            current_hash = compute_sha256(self.archive_path)
            if current_hash.lower() == self.config.expected_sha256.lower():
                logger.info(f"Found existing valid archive at {self.archive_path}.")
                return True
        return False

    def fetch_api_metadata(self) -> Dict[str, Any]:
        """Queries Mendeley API for fresh dataset and file metadata."""
        try:
            req = urllib.request.Request(
                self.config.api_files_endpoint,
                headers={"User-Agent": "MotionGuard-AI/0.1.0", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                files = json.loads(resp.read().decode())
                for f in files:
                    if f.get("filename") == self.config.archive_filename:
                        cd = f.get("content_details", {})
                        return {
                            "filename": f.get("filename"),
                            "size": f.get("size"),
                            "sha256": cd.get("sha256_hash"),
                            "download_url": cd.get("download_url"),
                            "id": f.get("id"),
                        }
        except Exception as e:
            logger.warning(f"Could not query live Mendeley API metadata: {e}. Using configured metadata.")
        return {
            "filename": self.config.archive_filename,
            "size": self.config.expected_size_bytes,
            "sha256": self.config.expected_sha256,
            "download_url": self.config.download_url,
        }

    def download(self, force: bool = False) -> Path:
        """Executes full download and verification pipeline."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        meta = self.fetch_api_metadata()
        download_url = meta.get("download_url") or self.config.download_url
        expected_sha = meta.get("sha256") or self.config.expected_sha256
        expected_size = meta.get("size") or self.config.expected_size_bytes

        if not force and self.is_cached_and_valid():
            if not self.csv_path.exists() and self.archive_path.exists():
                logger.info(f"Extracting archive {self.archive_path}...")
                safe_extract_zip(self.archive_path, self.raw_dir)
            self._write_manifest(verified=True, details=meta)
            return self.csv_path

        logger.info(f"Downloading KU-HAR dataset archive from {download_url}...")
        download_file_with_retry(
            url=download_url,
            dest_path=self.archive_path,
            expected_sha256=expected_sha,
            expected_size=expected_size,
        )

        logger.info(f"Extracting archive {self.archive_path}...")
        safe_extract_zip(self.archive_path, self.raw_dir)

        # Check extracted files
        if not self.csv_path.exists():
            # Search for any csv inside raw_dir
            csv_candidates = list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*/*.csv"))
            if csv_candidates:
                found_csv = csv_candidates[0]
                if found_csv != self.csv_path:
                    shutil.move(str(found_csv), str(self.csv_path))
                logger.info(f"Located primary CSV at {self.csv_path}")
            else:
                raise FileNotFoundError(f"Extracted archive did not contain {self.config.csv_filename}")

        self._write_manifest(verified=True, details=meta)
        return self.csv_path

    def _write_manifest(self, verified: bool, details: Dict[str, Any]) -> None:
        """Writes the dataset manifest according to Section 4.1 specification."""
        file_size = self.csv_path.stat().st_size if self.csv_path.exists() else 0
        csv_sha = compute_sha256(self.csv_path) if self.csv_path.exists() else ""
        archive_sha = compute_sha256(self.archive_path) if self.archive_path.exists() else details.get("sha256", "")

        manifest_data = {
            "dataset_name": self.config.name,
            "doi": self.config.doi,
            "source_page": self.config.source_page,
            "direct_download_url": details.get("download_url", self.config.download_url),
            "dataset_version": "1.0",
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "downloaded_archive_filename": self.config.archive_filename,
            "archive_sha256": archive_sha,
            "primary_csv_filename": self.config.csv_filename,
            "csv_file_size_bytes": file_size,
            "csv_sha256": csv_sha,
            "license": "CC BY 4.0 / Open Access for academic research and modification",
            "citation": "N. Sikder, A.-A. Nahid, 'KU-HAR: An Open Dataset for Heterogeneous Human Activity Recognition', Mendeley Data, V1, doi: 10.17632/45f952y38r.1",
            "verification_status": "VERIFIED" if verified else "UNVERIFIED",
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Dataset manifest written to {self.manifest_path}")
