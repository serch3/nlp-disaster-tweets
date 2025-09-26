"""
Naive Bayes model for disaster tweet classification using TF-IDF.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from .preprocessing import preprocess_data


def train_naive_bayes_model(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Train and evaluate Naive Bayes model with TF-IDF vectorization.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV

    Returns:
        Trained model and vectorizer
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess
    train_df = preprocess_data(train_df, method='basic')
    test_df = preprocess_data(test_df, method='basic')

    # Features and labels
    y = train_df['target']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english',
        max_features=10000
    )

    X_text = vectorizer.fit_transform(train_df['combined_text'])

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Grid search for best alpha
    param_grid = {'alpha': [0.1, 0.3, 0.5, 1.0, 2.0]}
    grid = GridSearchCV(MultinomialNB(), param_grid=param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_alpha = grid.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")

    # Final model
    model = MultinomialNB(alpha=best_alpha)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    return model, vectorizer


def get_feature_importance(model, vectorizer, top_n=10):
    """
    Extract and display top positive and negative words.

    Args:
        model: Trained MultinomialNB model
        vectorizer: Fitted TfidfVectorizer
        top_n: Number of top words to show
    """
    feature_names = vectorizer.get_feature_names_out()
    log_probabilities = model.feature_log_prob_
    coefficients = log_probabilities[1] - log_probabilities[0]

    # Get top positive and negative words
    top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_negative_idx = np.argsort(coefficients)[:top_n]

    top_positive_words = [(feature_names[i], coefficients[i]) for i in top_positive_idx]
    top_negative_words = [(feature_names[i], coefficients[i]) for i in top_negative_idx]

    print("Top Positive Words (Most indicative of disaster):")
    for word, score in top_positive_words:
        print(f"{word}: {score:.4f}")

    print("\nTop Negative Words (Most indicative of non-disaster):")
    for word, score in top_negative_words:
        print(f"{word}: {score:.4f}")

    # Plot
    positive_words, positive_scores = zip(*top_positive_words)
    negative_words, negative_scores = zip(*top_negative_words)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.barh(positive_words[::-1], positive_scores[::-1], color='green')
    plt.title('Top Positive Words')
    plt.xlabel('Log Probability Difference')

    plt.subplot(1, 2, 2)
    plt.barh(negative_words[::-1], negative_scores[::-1], color='red')
    plt.title('Top Negative Words')
    plt.xlabel('Log Probability Difference')

    plt.tight_layout()
    plt.show()


def predict_and_save(model, vectorizer, test_path='data/test.csv', output_file='submission_nb.csv'):
    """
    Make predictions on test data and save submission file.

    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        test_path: Path to test CSV
        output_file: Output filename
    """
    test_df = pd.read_csv(test_path)
    test_df = preprocess_data(test_df, method='basic')

    X_test = vectorizer.transform(test_df['combined_text'])
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": predictions
    })

    submission.to_csv(output_file, index=False)
    print(f"Submission saved as {output_file}")


if __name__ == "__main__":
    # Train model
    model, vectorizer = train_naive_bayes_model()

    # Show feature importance
    get_feature_importance(model, vectorizer)

    # Make predictions
    predict_and_save(model, vectorizer)