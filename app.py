import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

import io
import logging
import pickle
import re
import threading
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from wordcloud import WordCloud

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# pyplot keeps global state and is not thread-safe. The extension requests the
# pie chart and trend graph at the same time, so plotting is serialized.
plot_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class PredictWithTimestampsRequest(BaseModel):
    comments: Optional[List[Dict[str, Any]]] = None


class PredictRequest(BaseModel):
    comments: Optional[List[str]] = None


class GenerateChartRequest(BaseModel):
    sentiment_counts: Optional[Dict[str, Any]] = None


class GenerateWordcloudRequest(BaseModel):
    comments: Optional[List[str]] = None


class GenerateTrendGraphRequest(BaseModel):
    sentiment_data: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
LEMMATIZER = WordNetLemmatizer()


def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join(w for w in comment.split() if w not in STOP_WORDS)
        comment = ' '.join(LEMMATIZER.lemmatize(w) for w in comment.split())
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("https://dagshub.com/MitadruMridha05/Youtube_Sentiment_Analysis.mlflow")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")


def predict_sentiments(comments: List[str]) -> list:
    """Preprocess -> TF-IDF -> DataFrame with named columns -> model.predict.

    The MLflow model was logged with a named-column schema (one float64 column
    per vocabulary word), so a bare NumPy array fails schema enforcement.
    """
    preprocessed = [preprocess_comment(c) for c in comments]
    transformed = vectorizer.transform(preprocessed)
    features = pd.DataFrame(
        transformed.toarray().astype("float64"),
        columns=vectorizer.get_feature_names_out(),
    )
    return model.predict(features).tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get('/', response_class=HTMLResponse)
def home():
    return "Welcome to our FastAPI sentiment API"


@app.post('/predict_with_timestamps')
def predict_with_timestamps(data: PredictWithTimestampsRequest):
    comments_data = data.comments

    if not comments_data:
        return JSONResponse(content={"error": "No comments provided"}, status_code=400)

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        predictions = [str(p) for p in predict_sentiments(comments)]
    except Exception as e:
        logger.error(f"Error in /predict_with_timestamps: {e}")
        return JSONResponse(content={"error": f"Prediction failed: {str(e)}"}, status_code=500)

    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]
    return JSONResponse(content=response)


@app.post('/predict')
def predict(data: PredictRequest):
    comments = data.comments

    if not comments:
        return JSONResponse(content={"error": "No comments provided"}, status_code=400)

    try:
        predictions = predict_sentiments(comments)
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        return JSONResponse(content={"error": f"Prediction failed: {str(e)}"}, status_code=500)

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return JSONResponse(content=response)


@app.post('/generate_chart')
def generate_chart(data: GenerateChartRequest):
    try:
        sentiment_counts = data.sentiment_counts

        if not sentiment_counts:
            return JSONResponse(content={"error": "No sentiment counts provided"}, status_code=400)

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        with plot_lock:
            plt.figure(figsize=(6, 6))
            plt.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=140,
                textprops={'color': 'w'},
            )
            plt.axis('equal')

            img_io = io.BytesIO()
            plt.savefig(img_io, format='PNG', transparent=True)
            plt.close()

        return Response(content=img_io.getvalue(), media_type='image/png')
    except Exception as e:
        logger.error(f"Error in /generate_chart: {e}")
        return JSONResponse(content={"error": f"Chart generation failed: {str(e)}"}, status_code=500)


@app.post('/generate_wordcloud')
def generate_wordcloud(data: GenerateWordcloudRequest):
    try:
        comments = data.comments

        if not comments:
            return JSONResponse(content={"error": "No comments provided"}, status_code=400)

        text = ' '.join(preprocess_comment(c) for c in comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        return Response(content=img_io.getvalue(), media_type='image/png')
    except Exception as e:
        logger.error(f"Error in /generate_wordcloud: {e}")
        return JSONResponse(content={"error": f"Word cloud generation failed: {str(e)}"}, status_code=500)


@app.post('/generate_trend_graph')
def generate_trend_graph(data: GenerateTrendGraphRequest):
    try:
        sentiment_data = data.sentiment_data

        if not sentiment_data:
            return JSONResponse(content={"error": "No sentiment data provided"}, status_code=400)

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # 'ME' = month end (pandas >= 2.2); 'M' on older versions
        try:
            monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        except ValueError:
            monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for value in [-1, 0, 1]:
            if value not in monthly_percentages.columns:
                monthly_percentages[value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        with plot_lock:
            plt.figure(figsize=(12, 6))
            for value in [-1, 0, 1]:
                plt.plot(
                    monthly_percentages.index,
                    monthly_percentages[value],
                    marker='o',
                    linestyle='-',
                    label=sentiment_labels[value],
                    color=colors[value],
                )

            plt.title('Monthly Sentiment Percentage Over Time')
            plt.xlabel('Month')
            plt.ylabel('Percentage of Comments (%)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
            plt.legend()
            plt.tight_layout()

            img_io = io.BytesIO()
            plt.savefig(img_io, format='PNG')
            plt.close()

        return Response(content=img_io.getvalue(), media_type='image/png')
    except Exception as e:
        logger.error(f"Error in /generate_trend_graph: {e}")
        return JSONResponse(content={"error": f"Trend graph generation failed: {str(e)}"}, status_code=500)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=5000, reload=True)