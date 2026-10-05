import io
import pickle
import re
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from matplotlib.figure import Figure
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from wordcloud import WordCloud

app = FastAPI(title="Sentiment API")

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handling: keep the same {"error": "..."} shape the Flask app returned
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": "Invalid request body"})


# ---------------------------------------------------------------------------
# Request schemas (fields are Optional so we can return our own 400 messages)
# ---------------------------------------------------------------------------
class CommentWithTimestamp(BaseModel):
    text: str
    timestamp: str


class PredictTimestampsRequest(BaseModel):
    comments: Optional[List[CommentWithTimestamp]] = None


class PredictRequest(BaseModel):
    comments: Optional[List[str]] = None


class ChartRequest(BaseModel):
    sentiment_counts: Optional[Dict[str, Any]] = None


class WordcloudRequest(BaseModel):
    comments: Optional[List[str]] = None


class TrendRequest(BaseModel):
    sentiment_data: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r"\n", " ", comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        comment = " ".join(w for w in comment.split() if w not in stop_words)

        lemmatizer = WordNetLemmatizer()
        comment = " ".join(lemmatizer.lemmatize(w) for w in comment.split())

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, vectorizer_path: str):
    """Load the trained model and the TF-IDF vectorizer."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")


def png_response(fig: Figure, **savefig_kwargs) -> Response:
    """Render a matplotlib Figure to a PNG HTTP response."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", **savefig_kwargs)
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Routes
# NOTE: these are plain `def` (not `async def`) on purpose. The work is
# CPU-bound/blocking, so FastAPI runs them in a threadpool instead of
# blocking the event loop.
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return "Welcome to our flask api"


@app.post("/predict_with_timestamps")
def predict_with_timestamps(payload: PredictTimestampsRequest):
    if not payload.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in payload.comments]
        timestamps = [item.timestamp for item in payload.comments]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        dense = transformed.toarray()

        predictions = [str(p) for p in model.predict(dense).tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]


@app.post("/predict")
def predict(payload: PredictRequest):
    comments = payload.comments
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        dense = transformed.toarray()

        predictions = model.predict(dense).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]


@app.post("/generate_chart")
def generate_chart(payload: ChartRequest):
    sentiment_counts = payload.sentiment_counts
    if not sentiment_counts:
        raise HTTPException(status_code=400, detail="No sentiment counts provided")

    try:
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]  # Blue, Gray, Red

        # Use the object-oriented Figure API (no global pyplot state),
        # which is safe when requests run in multiple threads.
        fig = Figure(figsize=(6, 6))
        ax = fig.subplots()
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        ax.axis("equal")

        return png_response(fig, transparent=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post("/generate_wordcloud")
def generate_wordcloud(payload: WordcloudRequest):
    comments = payload.comments
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph")
def generate_trend_graph(payload: TrendRequest):
    sentiment_data = payload.sentiment_data
    if not sentiment_data:
        raise HTTPException(status_code=400, detail="No sentiment data provided")

    try:
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        # Monthly counts per sentiment ('M' -> 'ME' on pandas >= 2.2)
        monthly_counts = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns exist and are ordered
        monthly_percentages = monthly_percentages.reindex(columns=[-1, 0, 1], fill_value=0)

        colors = {-1: "red", 0: "gray", 1: "green"}

        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()

        for value in [-1, 0, 1]:
            ax.plot(
                monthly_percentages.index,
                monthly_percentages[value],
                marker="o",
                linestyle="-",
                label=sentiment_labels[value],
                color=colors[value],
            )

        ax.set_title("Monthly Sentiment Percentage Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Percentage of Comments (%)")
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        ax.legend()
        fig.tight_layout()

        return png_response(fig)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)