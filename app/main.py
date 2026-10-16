
from collections import Counter
from io import StringIO

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.schemas import SentimentRequest, BatchSentimentRequest
from app.sentiment import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="NLP sentiment analysis dashboard",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Demo-only history: resets whenever the server restarts.
history = []


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/predict")
def predict(payload: SentimentRequest):
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    try:
        result = predict_sentiment(text)
        history.insert(0, result)
        del history[100:]
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/batch")
def predict_batch(payload: BatchSentimentRequest):
    results = []

    try:
        for text in payload.texts:
            cleaned = text.strip()

            if not cleaned:
                continue

            result = predict_sentiment(cleaned)
            results.append(result)
            history.insert(0, result)

        del history[100:]
        return {"results": results, "total": len(results)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/stats")
def get_stats():
    counts = Counter(item["sentiment"] for item in history)
    total = len(history)

    return {
        "total": total,
        "positive": counts["positive"],
        "negative": counts["negative"],
        "neutral": counts["neutral"],
        "positive_pct": round(counts["positive"] / total * 100, 1)
        if total else 0,
        "negative_pct": round(counts["negative"] / total * 100, 1)
        if total else 0,
        "neutral_pct": round(counts["neutral"] / total * 100, 1)
        if total else 0,
    }


@app.get("/api/history")
def get_history():
    return {"results": history}


@app.get("/api/export")
def export_csv():
    df = pd.DataFrame(history)

    if df.empty:
        df = pd.DataFrame(
            columns=["text", "sentiment", "confidence"]
        )
    else:
        df = df[["text", "sentiment", "confidence"]]

    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                'attachment; filename="sentiment_results.csv"'
        },
    )
