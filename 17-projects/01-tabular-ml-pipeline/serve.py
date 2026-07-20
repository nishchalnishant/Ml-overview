"""Minimal production-serving stub: a FastAPI app around the saved sklearn
pipeline, plus a batch-scoring entrypoint for the same model.

Covers the "scaling in production" phase at the code level: a stateless
prediction endpoint (horizontally scalable behind a load balancer — no
per-request state, model loaded once at startup) and a batch path for
nightly scoring jobs. Talk through the surrounding system (feature store,
monitoring, canary rollout) verbally; this file is the serving code itself.

Run:
    uvicorn serve:app --workers 4
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "model.joblib"

app = FastAPI(title="churn-scoring")
_model = None  # loaded once per worker process, not per-request


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class ScoreRequest(BaseModel):
    age: int
    tenure_months: int
    monthly_charge: float
    total_charge: float
    num_support_calls: int
    contract_type: str
    payment_method: str
    internet_service: str
    tech_support: str


class ScoreResponse(BaseModel):
    churn_probability: float


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    model = get_model()
    row = pd.DataFrame([req.model_dump()])
    proba = model.predict_proba(row)[:, 1][0]
    return ScoreResponse(churn_probability=float(proba))


@app.get("/health")
def health():
    # Liveness/readiness probe for the load balancer / orchestrator.
    return {"status": "ok", "model_loaded": _model is not None}


def batch_score(input_csv: str, output_csv: str, batch_size: int = 10_000) -> None:
    """Nightly batch-scoring path: same model artifact, chunked to bound memory
    on large tables instead of loading the whole file at once."""
    model = get_model()
    chunks = pd.read_csv(input_csv, chunksize=batch_size)
    header_written = False
    for chunk in chunks:
        proba = model.predict_proba(chunk)[:, 1]
        chunk = chunk.assign(churn_probability=proba)
        chunk.to_csv(output_csv, mode="a", header=not header_written, index=False)
        header_written = True


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        batch_score(sys.argv[1], sys.argv[2])
        print(f"Wrote {sys.argv[2]}")
    else:
        print("Usage: python serve.py <input_csv> <output_csv>   (batch mode)")
        print("       uvicorn serve:app --workers 4               (online mode)")
