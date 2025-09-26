"""
Preprocessing utilities for NLP Disaster Tweets project.
"""

import re
import html
from ftfy import fix_text


def clean_text_basic(text: str) -> str:
    """
    Basic text cleaning for traditional ML models like TF-IDF + Naive Bayes.
    Removes punctuation, numbers, and converts to lowercase.
    """
    # Fix broken unicode errors
    text = fix_text(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text_transformer(text: str) -> str:
    """
    Text cleaning for transformer models.
    Keeps hashtags, mentions, emojis for better context.
    """
    # Fix broken unicode errors
    text = fix_text(text)

    # Replace links with [URL] placeholder
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(df, method='basic'):
    """
    Preprocess the dataframe by cleaning text and combining columns.

    Args:
        df: Pandas DataFrame with columns 'text', 'keyword', 'location'
        method: 'basic' for traditional ML, 'transformer' for transformer models

    Returns:
        Preprocessed DataFrame
    """
    # Fill missing values
    df = df.copy()
    df['keyword'] = df['keyword'].fillna('')
    df['location'] = df['location'].fillna('')

    # Combine text, keyword, location
    df['combined_text'] = df['text'] + " " + df['keyword'] + " " + df['location']

    # Apply cleaning
    if method == 'basic':
        df['combined_text'] = df['combined_text'].apply(clean_text_basic)
    elif method == 'transformer':
        df['combined_text'] = df['combined_text'].apply(clean_text_transformer)
    else:
        raise ValueError("Method must be 'basic' or 'transformer'")

    # Add text length feature
    df['text_length'] = df['text'].apply(len)

    return df