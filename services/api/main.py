# services/api/main.py
import os, json, pickle
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
MODEL_JOBLIB = MODELS_DIR / "model.joblib"
MODEL_PKL    = MODELS_DIR / "model.pkl"
META_PATH    = MODELS_DIR / "meta.json"

if not META_PATH.exists():
    raise RuntimeError(f"meta.json yok: {META_PATH}")

def load_model():
    # 1) joblib varsa onu yükle
    if MODEL_JOBLIB.exists():
        import joblib
        return joblib.load(MODEL_JOBLIB)
    # 2) yoksa pickle dene
    if MODEL_PKL.exists():
        try:
            with open(MODEL_PKL, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            with open(MODEL_PKL, "rb") as f:
                head = f.read(16)
            raise RuntimeError(f"model.pkl okunamadı: {e} | first16={head!r}")
    raise RuntimeError("MODEL bulunamadı: /models altinda ne model.joblib ne model.pkl var")

model = load_model()
meta  = json.loads(META_PATH.read_text(encoding="utf-8"))
W     = int(meta.get("window", 30))

class PredictIn(BaseModel):
    window: list[list[float]]
    horizon: int = 1
    features: list[str] | None = None

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "W": W}

@app.post("/predict")
def predict(body: PredictIn):
    X = np.array(body.window, dtype=np.float32)
    if X.ndim == 2:
        X = X.reshape(1, -1)
    y = float(model.predict(X)[0])
    return {"soc_pred": y, "horizon": body.horizon, "window": W, "features": body.features or []}
