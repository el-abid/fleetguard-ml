"""
Micro-service HTTP pour remplacer le node "Agent LLM" dans le workflow Abafusion.
Lancement local : uvicorn app:app --host 0.0.0.0 --port 8000
Endpoint : POST /predict-severity
Body attendu : le meme JSON que celui envoye auparavant a l'Agent LLM
Reponse : {"severity": int, "analyse": str}
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from fleetguard_predict import predict_severity

app = FastAPI(title="FleetGuard Severity Model")


class Payload(BaseModel):
    class Config:
        extra = "allow"  # accepte tous les champs, meme structure que l'ancien Agent LLM


@app.post("/predict-severity")
def predict(payload: Dict[str, Any]):
    return predict_severity(payload)


@app.get("/health")
def health():
    return {"status": "ok"}
