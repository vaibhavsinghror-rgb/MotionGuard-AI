"""Data validation schemas for MotionGuard AI Inference API."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class WindowInput(BaseModel):
    """Input payload for a single 3-second HAR window (6 channels x 300 samples)."""
    # Accept either 2D array (6, 300) or dict of channel lists
    channels: List[List[float]] = Field(
        ...,
        description="6 sensor channels x 300 samples [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional custom confidence acceptance threshold.",
    )

    @field_validator("channels")
    @classmethod
    def validate_channels_shape(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 6:
            raise ValueError(f"Expected exactly 6 sensor channels, got {len(v)}")
        for i, ch in enumerate(v):
            if len(ch) != 300:
                raise ValueError(f"Channel {i} has length {len(ch)}, expected exactly 300 samples (3s at 100Hz)")
            # Check for non-finite values
            for val in ch:
                if val is None or val != val:  # NaN check
                    raise ValueError(f"Channel {i} contains NaN value")
        return v


class PredictionResponse(BaseModel):
    activity: str = Field(..., description="Recognized human activity name")
    class_id: int = Field(..., description="Class integer ID (0-17)")
    confidence: float = Field(..., description="Calibrated confidence score between 0 and 1")
    accepted: bool = Field(..., description="Whether prediction meets confidence threshold")
    status: str = Field(..., description="'prediction_available' or 'uncertain_human_review_recommended'")
    probabilities: Dict[str, float] = Field(..., description="Calibrated probabilities per activity class")
    model_version: str = Field(default="1.0.0")
    abstention_reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    calibrated: bool = False
    device: str = "cpu"
