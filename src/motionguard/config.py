"""Configuration loader and schema definition for MotionGuard AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def _find_project_root() -> Path:
    """Finds project root by searching for pyproject.toml upwards."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return current


PROJECT_ROOT = _find_project_root()


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Loads a YAML file safely."""
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class DataConfig:
    name: str = "KU-HAR"
    doi: str = "10.17632/45f952y38r.1"
    source_page: str = "https://data.mendeley.com/datasets/45f952y38r/1"
    api_files_endpoint: str = "https://data.mendeley.com/api/datasets/45f952y38r/files"
    download_url: str = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/f2b31500-b2a4-48e6-be0b-ebf621c18d42/file_downloaded"
    archive_filename: str = "3.Time_domain_subsamples.zip"
    csv_filename: str = "Time_domain_subsamples.csv"
    expected_sha256: str = "bb804c1338f75cff72e9714f086e0b548a8002baf74a2c6e74a422846e0851fb"
    expected_size_bytes: int = 123005404
    sampling_rate_hz: int = 100
    window_seconds: float = 3.0
    window_samples: int = 300
    num_channels: int = 6
    channels: List[str] = field(default_factory=lambda: ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    num_classes: int = 18
    classes: Dict[int, str] = field(default_factory=dict)
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    manifests_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "manifests")
    dataset_manifest: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "manifests" / "dataset_manifest.json")
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "DataConfig":
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "data.yaml"
        raw = load_yaml(yaml_path)
        ds = raw.get("dataset", {})
        paths = raw.get("paths", {})
        split = raw.get("split", {})
        classes = {int(k): str(v) for k, v in ds.get("classes", {}).items()}

        return cls(
            name=ds.get("name", "KU-HAR"),
            doi=ds.get("doi", "10.17632/45f952y38r.1"),
            source_page=ds.get("source_page", ""),
            api_files_endpoint=ds.get("api_files_endpoint", ""),
            download_url=ds.get("download_url", ""),
            archive_filename=ds.get("archive_filename", "3.Time_domain_subsamples.zip"),
            csv_filename=ds.get("csv_filename", "Time_domain_subsamples.csv"),
            expected_sha256=ds.get("expected_sha256", ""),
            expected_size_bytes=ds.get("expected_size_bytes", 0),
            sampling_rate_hz=ds.get("sampling_rate_hz", 100),
            window_seconds=ds.get("window_seconds", 3.0),
            window_samples=ds.get("window_samples", 300),
            num_channels=ds.get("num_channels", 6),
            channels=ds.get("channels", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]),
            num_classes=ds.get("num_classes", 18),
            classes=classes,
            raw_dir=PROJECT_ROOT / paths.get("raw_dir", "data/raw"),
            processed_dir=PROJECT_ROOT / paths.get("processed_dir", "data/processed"),
            manifests_dir=PROJECT_ROOT / paths.get("manifests_dir", "data/manifests"),
            dataset_manifest=PROJECT_ROOT / paths.get("dataset_manifest", "data/manifests/dataset_manifest.json"),
            train_ratio=split.get("train_ratio", 0.70),
            val_ratio=split.get("val_ratio", 0.15),
            test_ratio=split.get("test_ratio", 0.15),
            random_seed=split.get("random_seed", 42),
        )


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "cuda"
    classical_rf: Dict[str, Any] = field(default_factory=dict)
    classical_gb: Dict[str, Any] = field(default_factory=dict)
    classical_svm: Dict[str, Any] = field(default_factory=dict)
    dl_config: Dict[str, Any] = field(default_factory=dict)
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "models")
    preprocessors_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "preprocessors")
    calibration_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "calibration")
    manifests_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts" / "manifests")

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "TrainingConfig":
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "training.yaml"
        raw = load_yaml(yaml_path)
        classical = raw.get("classical", {})
        persistence = raw.get("persistence", {})

        return cls(
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
            classical_rf=classical.get("random_forest", {}),
            classical_gb=classical.get("gradient_boosting", {}),
            classical_svm=classical.get("svm", {}),
            dl_config=raw.get("deep_learning", {}),
            artifacts_dir=PROJECT_ROOT / persistence.get("artifacts_dir", "artifacts"),
            models_dir=PROJECT_ROOT / persistence.get("models_dir", "artifacts/models"),
            preprocessors_dir=PROJECT_ROOT / persistence.get("preprocessors_dir", "artifacts/preprocessors"),
            calibration_dir=PROJECT_ROOT / persistence.get("calibration_dir", "artifacts/calibration"),
            manifests_dir=PROJECT_ROOT / persistence.get("manifests_dir", "artifacts/manifests"),
        )


@dataclass
class EvaluationConfig:
    min_val_macro_f1: float = 0.85
    target_val_macro_f1: float = 0.90
    min_val_accuracy: float = 0.85
    max_inference_latency_ms: float = 50.0
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1"])
    calibration_method: str = "temperature_scaling"
    calibration_bins: int = 15
    default_confidence_threshold: float = 0.70
    evaluated_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "experiments" / "results" / "evaluation")

    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "EvaluationConfig":
        if yaml_path is None:
            yaml_path = PROJECT_ROOT / "configs" / "evaluation.yaml"
        raw = load_yaml(yaml_path)
        qg = raw.get("quality_gates", {})
        cal = raw.get("calibration", {})
        abst = raw.get("abstention", {})
        rep = raw.get("reporting", {})

        return cls(
            min_val_macro_f1=qg.get("min_val_macro_f1", 0.85),
            target_val_macro_f1=qg.get("target_val_macro_f1", 0.90),
            min_val_accuracy=qg.get("min_val_accuracy", 0.85),
            max_inference_latency_ms=qg.get("max_inference_latency_ms", 50.0),
            metrics=raw.get("metrics", ["accuracy", "balanced_accuracy", "macro_f1"]),
            calibration_method=cal.get("method", "temperature_scaling"),
            calibration_bins=cal.get("n_bins", 15),
            default_confidence_threshold=abst.get("default_confidence_threshold", 0.70),
            evaluated_thresholds=abst.get("evaluated_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9]),
            output_dir=PROJECT_ROOT / rep.get("output_dir", "experiments/results/evaluation"),
        )
