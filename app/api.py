"""FastAPI REST Inference Service for MotionGuard AI."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from motionguard.inference.predictor import MotionGuardPredictor
from motionguard.inference.schemas import HealthResponse, PredictionResponse, WindowInput

app = FastAPI(
    title="MotionGuard AI Inference Service",
    description="Confidence-Aware Wearable Sensor Human Activity Recognition and Movement Analysis API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
_predictor_instance: Optional[MotionGuardPredictor] = None


def get_predictor() -> Optional[MotionGuardPredictor]:
    global _predictor_instance
    if _predictor_instance is None:
        try:
            _predictor_instance = MotionGuardPredictor()
        except Exception:
            _predictor_instance = None
    return _predictor_instance


@app.on_event("startup")
def startup_event():
    get_predictor()


@app.get("/health", response_model=HealthResponse)
def health_check():
    pred = get_predictor()
    if pred and pred.is_loaded:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=pred.model_metadata.get("model_name"),
            calibrated=pred.calibrator is not None and pred.calibrator.is_fitted,
            device=str(pred.device),
        )
    return HealthResponse(
        status="model_not_loaded",
        model_loaded=False,
        model_name=None,
        calibrated=False,
        device="cpu",
    )


@app.get("/model")
def get_model_info():
    pred = get_predictor()
    if not pred or not pred.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts have not been generated yet. Please run training pipeline.",
        )
    return {
        "metadata": pred.model_metadata,
        "classes": pred.class_mapping,
        "calibrated": pred.calibrator is not None and pred.calibrator.is_fitted,
        "temperature": pred.calibrator.temperature if pred.calibrator else 1.0,
        "default_threshold": pred.default_confidence_threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_window(payload: WindowInput):
    pred = get_predictor()
    if not pred or not pred.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not ready. Model artifacts missing.",
        )

    try:
        response = pred.predict(payload.channels, threshold=payload.confidence_threshold)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(windows: List[WindowInput]):
    pred = get_predictor()
    if not pred or not pred.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not ready. Model artifacts missing.",
        )
    results = []
    for w in windows:
        results.append(pred.predict(w.channels, threshold=w.confidence_threshold))
    return results
