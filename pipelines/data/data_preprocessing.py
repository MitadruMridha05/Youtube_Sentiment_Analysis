import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging Configuration
logger = logging.getLogger("data_ingestion")
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

nltk.download('wordnet')
nltk.download('stopwords')


# Define a preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformation to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', '', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'no', 'however', 'yet'}
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe"""
    try:
        df["clean_comment"] = df["CommentText"].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_preprocessed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_preprocessed.csv"), index=False)
        logger.debug(f"Preprocessed data saved to {interim_data_path}")

    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise


def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Fetch the raw data
        raw_data_path = os.path.join('data', 'raw')
        train_data = pd.read_csv(os.path.join(raw_data_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(raw_data_path, 'test.csv'))
        logger.debug("Raw data loaded successfully")

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
        logger.debug("Data preprocessing completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

        