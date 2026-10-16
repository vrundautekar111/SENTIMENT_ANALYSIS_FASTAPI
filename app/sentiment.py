
from pathlib import Path

import joblib

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml"
    / "sentiment_model.joblib"
)

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                "Model not found. Run: python ml/train_model.py"
            )
        _model = joblib.load(MODEL_PATH)

    return _model


def predict_sentiment(text: str) -> dict:
    model = get_model()
    prediction = str(model.predict([text])[0])

    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))

    return {
        "text": text,
        "sentiment": prediction,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            str(label): round(float(prob) * 100, 2)
            for label, prob in zip(model.classes_, probabilities)
        },
    }
