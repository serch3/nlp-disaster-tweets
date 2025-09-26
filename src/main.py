"""
Main script for running disaster tweet classification models.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from naive_bayes_model import train_naive_bayes_model, get_feature_importance, predict_and_save as predict_nb
from transformer_model import train_transformer_model, predict_and_save as predict_transformer


def main():
    parser = argparse.ArgumentParser(description='Disaster Tweet Classification')
    parser.add_argument('--model', choices=['nb', 'transformer'], default='nb',
                       help='Model to use: nb (Naive Bayes) or transformer')
    parser.add_argument('--train', action='store_true',
                       help='Train the model')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions on test data')
    parser.add_argument('--train-path', default='data/train.csv',
                       help='Path to training data')
    parser.add_argument('--test-path', default='data/test.csv',
                       help='Path to test data')
    parser.add_argument('--output', default=None,
                       help='Output file for predictions')

    args = parser.parse_args()

    if not args.train and not args.predict:
        print("Please specify --train or --predict")
        return

    if args.model == 'nb':
        if args.train:
            print("Training Naive Bayes model...")
            model, vectorizer = train_naive_bayes_model(args.train_path, args.test_path)
            get_feature_importance(model, vectorizer)

        if args.predict:
            print("Making predictions with Naive Bayes...")
            # For prediction only, we'd need to load saved model
            # For now, assume training was done
            model, vectorizer = train_naive_bayes_model(args.train_path, args.test_path)
            output_file = args.output or 'submission_nb.csv'
            predict_nb(model, vectorizer, args.test_path, output_file)

    elif args.model == 'transformer':
        if args.train:
            print("Training Transformer model...")
            trainer, tokenizer = train_transformer_model(args.train_path, args.test_path)

        if args.predict:
            print("Making predictions with Transformer...")
            # For prediction only, we'd need to load saved model
            trainer, tokenizer = train_transformer_model(args.train_path, args.test_path)
            output_file = args.output or 'submission_transformer.csv'
            predict_transformer(trainer, tokenizer, args.test_path, output_file)


if __name__ == "__main__":
    main()