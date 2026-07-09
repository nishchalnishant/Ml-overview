import os
import yaml
from pydantic_settings import BaseSettings
from pydantic import Field

class ModelConfig(BaseSettings):
    type: str = "xgboost"
    max_depth: int = 5
    learning_rate: float = 0.1
    n_estimators: int = 100

class DataConfig(BaseSettings):
    features: list[str] = ["age", "income", "credit_score"]
    target: str = "default_status"
    categorical_features: list[str] = []

class AppConfig(BaseSettings):
    name: str = "Tabular ML Inference API"
    version: str = "0.1.0"
    model_path: str = Field(default="models/model.joblib", alias="MODEL_PATH")
    env: str = Field(default="development", alias="APP_ENV")

class Settings:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        self.model = ModelConfig(**raw_config.get("model", {}))
        self.data = DataConfig(**raw_config.get("data", {}))
        self.app = AppConfig(**raw_config.get("app", {}))

# Global settings instance
settings = Settings()
