import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Equivalent of Flask's app.logger
logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Request body schemas (fields are Optional so the original
# "No ... provided" 400 checks still behave exactly as before)
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


# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment



#Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
     # Set MLflow tracking URI to your server
     mlflow.set_tracking_uri("https://dagshub.com/MitadruMridha05/Youtube_Sentiment_Analysis.mlflow")  # Replace with your MLflow tracking URI
     client = MlflowClient()
     model_uri = f"models:/{model_name}/{model_version}"
     model = mlflow.pyfunc.load_model(model_uri)
     with open(vectorizer_path, 'rb') as file:
         vectorizer = pickle.load(file)

     return model, vectorizer



'''def load_model(model_path, vectorizer_path):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        return model, vectorizer
    except Exception as e:
        raise


# Initialize the model and vectorizer
model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")'''

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

@app.get('/', response_class=HTMLResponse)
def home():
    return "Welcome to our flask api"



@app.post('/predict_with_timestamps')
def predict_with_timestamps(data: PredictWithTimestampsRequest):
    comments_data = data.comments

    if not comments_data:
        return JSONResponse(content={"error": "No comments provided"}, status_code=400)

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array

        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return JSONResponse(content={"error": f"Prediction failed: {str(e)}"}, status_code=500)

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return JSONResponse(content=response)



@app.post('/predict')
def predict(data: PredictRequest):
    comments = data.comments
    print("i am the comment: ",comments)
    print("i am the comment type: ",type(comments))

    if not comments:
        return JSONResponse(content={"error": "No comments provided"}, status_code=400)

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()  # Convert to dense array

        # Make predictions
        predictions = model.predict(dense_comments).tolist()  # Convert to list

        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return JSONResponse(content={"error": f"Prediction failed: {str(e)}"}, status_code=500)

    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return JSONResponse(content=response)



@app.post('/generate_chart')
def generate_chart(data: GenerateChartRequest):
    try:
        sentiment_counts = data.sentiment_counts

        if not sentiment_counts:
            return JSONResponse(content={"error": "No sentiment counts provided"}, status_code=400)

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
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

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
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

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return Response(content=img_io.getvalue(), media_type='image/png')
    except Exception as e:
        logger.error(f"Error in /generate_trend_graph: {e}")
        return JSONResponse(content={"error": f"Trend graph generation failed: {str(e)}"}, status_code=500)



if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host='0.0.0.0', port=5000, reload=True)  # reload=True is the FastAPI/uvicorn equivalent of Flask's debug=True