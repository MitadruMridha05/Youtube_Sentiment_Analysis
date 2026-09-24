import pandas as pd
import numpy as np
import os
import pickle
import yaml
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

# Logging Configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_root_directory() -> str:
    """Get root directory (two levels above this file: <root>/src/model/)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(file_path)
        # Fill NaNs only in the text column (filling '' into a numeric column raises on new pandas)
        df['CommentText'] = df['CommentText'].fillna('')
        logger.debug('Data loaded and text NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the training data is safe to model:
    - CommentText is a non-empty string
    - Sentiment is numeric (fillna('') in load_data can turn missing labels into '')
    """
    try:
        df = df.copy()
        df['CommentText'] = df['CommentText'].astype(str).str.strip()
        df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

        before = len(df)
        df = df[(df['CommentText'] != '') & df['Sentiment'].notna()].reset_index(drop=True)
        df['Sentiment'] = df['Sentiment'].astype(int)

        logger.debug("Cleaned training data: %d -> %d rows", before, len(df))
        if df.empty:
            raise ValueError("Training data is empty after cleaning.")
        return df
    except Exception as e:
        logger.error("Error while cleaning training data: %s", e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        x_train = train_data['CommentText'].values
        y_train = train_data["Sentiment"].values

        x_train_tfidf = vectorizer.fit_transform(x_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {x_train_tfidf.shape}")

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with ngrams and data transformed')
        return x_train_tfidf, y_train
    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def train_lgbm(x_train, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
            objective="multiclass",
            metric="multi_logloss",
            class_weight="balanced",   # handles imbalance (is_unbalance is binary-only, so removed)
            reg_alpha=0.1,
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            verbose=-1
        )
        best_model.fit(x_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error("Error during LightGBM model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('LightGBM model saved to %s', file_path)
    except Exception as e:
        logger.error("Error during saving LightGBM model: %s", e)
        raise


def main():
    try:
        root_dir = get_root_directory()

        # 1. Load parameters
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        mb_params = params['model_building']

        max_features = int(mb_params['max_features'])
        ngram_range = tuple(mb_params['ngram_range'])   # YAML gives a list; sklearn needs a tuple
        learning_rate = float(mb_params['learning_rate'])
        max_depth = int(mb_params['max_depth'])
        n_estimators = int(mb_params['n_estimators'])

        # 2. Load + clean training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_preprocessed.csv'))
        train_data = clean_train_data(train_data)

        # 3. TF-IDF
        x_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # 4. Train
        best_model = train_lgbm(x_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # 5. Save
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

        logger.debug("Model building pipeline completed successfully")
        print("Model building completed. Saved lgbm_model.pkl and tfidf_vectorizer.pkl")

    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        raise


if __name__ == '__main__':
    main()