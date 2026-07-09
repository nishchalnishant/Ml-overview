# Production Tabular ML Template

This repository contains a production-ready template for deploying classical/tabular machine learning models (e.g., XGBoost, Scikit-Learn, LightGBM).

## Features
- **FastAPI**: Asynchronous, high-performance web framework.
- **Pydantic**: Strict data validation for all incoming inference requests.
- **Docker**: Ready to build and deploy via `docker-compose`.
- **Structlog**: JSON structured logging for Datadog/ELK integration.
- **Pytest**: Unit testing framework.
- **Config Management**: Dynamic configs via `pydantic-settings` and YAML.

## Quickstart

1. **Train a dummy model:**
```bash
python src/models/train.py
```

2. **Run the API locally:**
```bash
uvicorn src.api.main:app --reload
```

3. **Or run via Docker:**
```bash
docker-compose up --build
```

4. **Test the endpoint:**
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 35, "income": 75000.0, "credit_score": 720.0}'
```

## Structure
- `src/api/`: FastAPI web server and routing.
- `src/core/`: Application settings, logging, and exceptions.
- `src/models/`: Training and inference wrapper classes.
- `tests/`: Pytest suite.
