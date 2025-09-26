"""
Transformer model for disaster tweet classification using DistilRoBERTa.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import preprocess_data


def compute_metrics(pred):
    """Compute accuracy and F1 score for evaluation."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1
    }


class TweetDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for tweets.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def train_transformer_model(train_path='data/train.csv', test_path='data/test.csv', model_name="distilroberta-base"):
    """
    Train and evaluate transformer model.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        model_name: HuggingFace model name

    Returns:
        Trained trainer and tokenizer
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess
    train_df = preprocess_data(train_df, method='transformer')
    test_df = preprocess_data(test_df, method='transformer')

    # Split training data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['combined_text'].tolist(),
        train_df['target'].tolist(),
        test_size=0.2,
        random_state=404
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Create datasets
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        seed=35,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_strategy="epoch"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

    # Evaluate
    eval_result = trainer.evaluate()
    print("Validation results:", eval_result)

    return trainer, tokenizer


def predict_and_save(trainer, tokenizer, test_path='data/test.csv', output_file='submission_transformer.csv'):
    """
    Make predictions on test data and save submission file.

    Args:
        trainer: Trained Trainer
        tokenizer: Tokenizer
        test_path: Path to test CSV
        output_file: Output filename
    """
    test_df = pd.read_csv(test_path)
    test_df = preprocess_data(test_df, method='transformer')

    # Create test dataset
    test_dataset = TweetDataset(test_df['combined_text'].tolist(), labels=None, tokenizer=tokenizer)

    # Predict
    test_outputs = trainer.predict(test_dataset)
    predictions = np.argmax(test_outputs.predictions, axis=1)

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "target": predictions
    })

    submission.to_csv(output_file, index=False)
    print(f"Submission saved as {output_file}")


if __name__ == "__main__":
    # Train model
    trainer, tokenizer = train_transformer_model()

    # Make predictions
    predict_and_save(trainer, tokenizer)