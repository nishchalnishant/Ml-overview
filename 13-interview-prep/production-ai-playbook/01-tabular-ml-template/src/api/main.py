from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import time

from core.config import settings
from core.logger import setup_logging, logger
from core.exceptions import ModelLoadError, ModelInferenceError
from api.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest, 
    BatchPredictionResponse,
    HealthResponse
)
from models.predict import ModelPredictor

# Initialize app and configuration
setup_logging(settings.app.env)
app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="Production Tabular ML Inference Service"
)

# Global model instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Loads the model into memory on API startup."""
    global predictor
    try:
        predictor = ModelPredictor(settings.app.model_path)
    except Exception as e:
        logger.error("Startup failed: Model could not be loaded", error=str(e))
        # We don't raise here so the API can start and report unhealthy via /health

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness and Readiness probe."""
    if predictor is None or predictor.model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "model_loaded": False, "version": settings.app.version}
        )
    return {"status": "healthy", "model_loaded": True, "version": settings.app.version}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Online inference endpoint for a single record."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    start_time = time.perf_counter()
    try:
        pred, prob = predictor.predict(request.model_dump())
        latency = (time.perf_counter() - start_time) * 1000
        
        logger.info("Prediction successful", latency_ms=latency, prediction=pred)
        
        return PredictionResponse(
            prediction=pred,
            probability=prob,
            model_version=settings.app.version
        )
    except ModelInferenceError as e:
        logger.error("Prediction error", detail=e.message)
        raise HTTPException(status_code=400, detail=e.message)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch inference endpoint for high throughput."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    start_time = time.perf_counter()
    try:
        features_list = [req.model_dump() for req in request.instances]
        results = predictor.predict_batch(features_list)
        
        responses = [
            PredictionResponse(prediction=pred, probability=prob, model_version=settings.app.version)
            for pred, prob in results
        ]
        
        latency = (time.perf_counter() - start_time) * 1000
        logger.info("Batch prediction successful", batch_size=len(features_list), latency_ms=latency)
        
        return BatchPredictionResponse(predictions=responses)
    except ModelInferenceError as e:
        raise HTTPException(status_code=400, detail=e.message)
