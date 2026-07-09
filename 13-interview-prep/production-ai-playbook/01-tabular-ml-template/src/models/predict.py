import joblib
import pandas as pd
import numpy as np
from typing import Any
from pathlib import Path
from core.config import settings
from core.exceptions import ModelLoadError, ModelInferenceError
from core.logger import logger

class ModelPredictor:
    def __init__(self, model_path: str = settings.app.model_path):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Loads the serialized model artifact."""
        try:
            path = Path(self.model_path)
            if not path.exists():
                logger.warning(f"Model file not found at {path}. Model will not be loaded.")
                return
            
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully", path=str(self.model_path))
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise ModelLoadError(f"Could not load model from {self.model_path}: {str(e)}")
            
    def predict(self, features: dict[str, Any]) -> tuple[int, float]:
        """Runs single instance prediction."""
        if self.model is None:
            raise ModelLoadError("Model is not initialized.")
            
        try:
            # Convert to DataFrame to maintain feature names for XGBoost/Scikit-learn
            df = pd.DataFrame([features])
            
            # Predict probability
            prob = float(self.model.predict_proba(df)[0][1])
            pred = int(self.model.predict(df)[0])
            
            return pred, prob
        except Exception as e:
            logger.error("Inference failed", payload=features, error=str(e))
            raise ModelInferenceError("Model prediction failed", payload=features)
            
    def predict_batch(self, batch_features: list[dict[str, Any]]) -> list[tuple[int, float]]:
        """Runs batch prediction for high throughput."""
        if self.model is None:
            raise ModelLoadError("Model is not initialized.")
            
        try:
            df = pd.DataFrame(batch_features)
            probs = self.model.predict_proba(df)[:, 1].tolist()
            preds = self.model.predict(df).tolist()
            
            return list(zip(preds, probs))
        except Exception as e:
            logger.error("Batch inference failed", batch_size=len(batch_features), error=str(e))
            raise ModelInferenceError("Batch model prediction failed")
