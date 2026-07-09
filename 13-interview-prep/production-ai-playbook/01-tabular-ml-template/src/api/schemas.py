from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    age: float = Field(..., description="Age of the applicant", ge=18, le=100)
    income: float = Field(..., description="Annual income", ge=0)
    credit_score: float = Field(..., description="Credit score", ge=300, le=850)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000.0,
                "credit_score": 720.0
            }
        }

class BatchPredictionRequest(BaseModel):
    instances: list[PredictionRequest]

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary class prediction (0 or 1)")
    probability: float = Field(..., description="Probability of class 1", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Version of the model used")

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
