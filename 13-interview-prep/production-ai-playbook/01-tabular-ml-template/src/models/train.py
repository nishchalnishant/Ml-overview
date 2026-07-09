import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path

from core.config import settings
from core.logger import logger

def build_pipeline() -> Pipeline:
    """Constructs the sklearn/XGBoost training pipeline."""
    # Note: In a real system, you'd pull Categorical encoders, 
    # StandardScaler, etc., based on settings.data.categorical_features
    
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    clf = xgb.XGBClassifier(
        max_depth=settings.model.max_depth,
        learning_rate=settings.model.learning_rate,
        n_estimators=settings.model.n_estimators,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def train_and_save_model(X_train: pd.DataFrame, y_train: pd.Series, output_path: str = settings.app.model_path):
    """Trains the pipeline and serializes it."""
    logger.info("Starting model training...", n_samples=len(X_train))
    
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    logger.info("Training complete. Serializing model...")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    
    logger.info("Model saved successfully", path=output_path)

if __name__ == "__main__":
    # Mock data for template compilation purposes
    import numpy as np
    
    # Setup mock data based on our features
    features = settings.data.features
    X_mock = pd.DataFrame(np.random.rand(100, len(features)), columns=features)
    y_mock = pd.Series(np.random.randint(0, 2, 100))
    
    # Train
    train_and_save_model(X_mock, y_mock)
