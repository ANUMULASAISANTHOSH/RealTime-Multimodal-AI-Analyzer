import os
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.emotion_service import EmotionInferenceService

app = FastAPI(title="Emotion Recognition API", version="1.0.0")
service: Optional[EmotionInferenceService] = None


@app.on_event("startup")
def startup_event() -> None:
    global service
    model_path = os.getenv("EMOTION_MODEL_PATH")
    try:
        service = EmotionInferenceService(model_path=model_path)
    except Exception as exc:  # pragma: no cover - defensive startup handling
        service = None
        app.state.startup_error = str(exc)


@app.get("/health")
def health() -> dict:
    if service is None:
        return {"status": "degraded", "model_loaded": False, "error": getattr(app.state, "startup_error", "Model unavailable")}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Emotion model is not available")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded")

    try:
        return service.predict_from_image(image)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive API error handling
        raise HTTPException(status_code=500, detail=str(exc)) from exc
