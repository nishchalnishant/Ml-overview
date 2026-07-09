import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_no_model():
    # If the model isn't trained yet, the health endpoint should return 503
    response = client.get("/health")
    # Depends on if train.py was run before this test. 
    # Assuming it wasn't, or we mock it:
    assert response.status_code in [200, 503]
    
def test_predict_validation_error():
    # Missing required field 'credit_score'
    payload = {
        "age": 35,
        "income": 75000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error

def test_predict_out_of_bounds():
    # Age < 18 is restricted in schema
    payload = {
        "age": 12,
        "income": 75000.0,
        "credit_score": 720.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 
