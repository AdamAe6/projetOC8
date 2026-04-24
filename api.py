import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os

# Charger le modèle au démarrage
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/features.txt")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    FEATURES = [line.strip() for line in f.readlines()]

app = FastAPI(title="API Scoring Crédit", description="Prédisez le score d'un client.")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != len(FEATURES):
        raise HTTPException(status_code=400, detail=f"Il faut {len(FEATURES)} features.")
    X = np.array(data.features).reshape(1, -1)
    proba = float(model.predict_proba(X)[0, 1])
    return {"score": proba}

@app.get("/")
def root():
    return {"message": "API de scoring opérationnelle. Utilisez /predict."}
